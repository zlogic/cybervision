#include "pointtriangulator.h"

#include <Reconstruction/options.h>

#include <limits>
#include <QSize>
#include <QVector2D>
#include <QSet>

#define EIGEN_NO_EXCEPTIONS
#include <Eigen/Dense>
#include <Eigen/SVD>

namespace cybervision{

	PointTriangulator::PointTriangulator(QObject *parent) : QObject(parent){
		result= RESULT_OK;
	}

	QList<QVector3D> PointTriangulator::getPoints3D()const{ return Points3D; }
	PointTriangulator::TriangulationResult PointTriangulator::getResult()const{ return result; }

	QList<PointTriangulator::StereopairPosition> PointTriangulator::computePose(const Eigen::Matrix3d& F){
		emit sgnStatusMessage("Estimating pose...");

		QList<StereopairPosition> RTList= computeRT(F);
		//Output R,T matrices to log

		QString RT_str;
		for(QList<StereopairPosition>::const_iterator i= RTList.begin();i!=RTList.end();i++){
			RT_str.append(QString("R%1T\n").arg(QString(""),40));
			for(int j=0;j<3;j++){
				qreal T_value_i= j==0? i->T(0,0): (j==1?i->T(1,0):i->T(2,0));//Extract i-th value from QVector3D
				QString matrix_row= QString("%1 %2 %3 %4\n").arg(i->R(j,0),6,'g',6).arg(i->R(j,1),6,'g',6).arg(i->R(j,2),6,'g',6).arg(T_value_i,6,'g',6);
				RT_str.append(matrix_row);
			}
		}
		emit sgnLogMessage(QString("Resulting camera poses\n").append(RT_str));
		return RTList;
	}


	Eigen::Matrix3d PointTriangulator::computeRT_rzfunc(double angle) const{
		Eigen::Matrix3d result;
		result.fill(0.0);
		result(0,1)= angle>=0?1.0:(-1.0);
		result(1,0)= -(angle>=0?1.0:(-1.0));
		result(2,2)= 1.0;
		return result;
	}

	Eigen::MatrixXd PointTriangulator::kronecker(const Eigen::MatrixXd A, const Eigen::MatrixXd &B) const{
		Eigen::MatrixXd result(A.rows()*B.rows(),A.cols()*B.cols());

		// http://mathworld.wolfram.com/KroneckerProduct.html
		for(Eigen::MatrixXd::Index i=0;i<A.rows();i++)
			for(Eigen::MatrixXd::Index j=0;j<A.cols();j++)
				for(Eigen::MatrixXd::Index k=0;k<B.rows();k++)
					for(Eigen::MatrixXd::Index l=0;l<B.cols();l++)
						result(B.rows()*i+k,B.cols()*j+l)= A(i,j)*B(k,l);

		return result;
	}

	Eigen::MatrixXd PointTriangulator::leastSquares(const Eigen::MatrixXd A, const Eigen::MatrixXd &B) const{
		Eigen::JacobiSVD<Eigen::MatrixXd,Eigen::FullPivHouseholderQRPreconditioner> svd(A, Eigen::ComputeFullV|Eigen::ComputeFullU);

		Eigen::MatrixXd V= svd.matrixV();
		Eigen::MatrixXd U= svd.matrixU();
		Eigen::VectorXd S= svd.singularValues();
		Eigen::MatrixXd Sd(A.cols(),A.rows());
		for(Eigen::MatrixXd::Index i=0;i<Sd.rows();i++){
			for(Eigen::MatrixXd::Index j=0;j<Sd.cols();j++){
				if(i==j && S(i)>(Options::constraintsThreshold*S(0)))
					Sd(i,j)= 1/S(i);
				else
					Sd(i,j)= 0;
			}
		}
		Eigen::MatrixXd result= V*Sd*(U.transpose())*B;

		return result;
	}


	Eigen::Matrix3d PointTriangulator::computeCameraMatrix(const QSize& imageSize)const{
		Eigen::Matrix3d K;
		K.fill(0.0);
		K(0,0)= Options::scaleFocalDistance;//Focal distance X
		K(1,1)= Options::scaleFocalDistance;//Focal distance Y
		K(0,2)= -imageSize.width();//Optical center X
		K(1,2)= -imageSize.height();//Optical center Y
		K(2,2)= 1;
		return K;
	}

	QList<PointTriangulator::StereopairPosition> PointTriangulator::computeRT(Eigen::Matrix3d Essential_matrix) const{
		QList<PointTriangulator::StereopairPosition> RTList;
		QList<double> pi_values;
		pi_values<<M_PI_2<<-M_PI_2;

		//Project into essential space
		if(false){
			Eigen::JacobiSVD<Eigen::Matrix3d,Eigen::FullPivHouseholderQRPreconditioner> svd(Essential_matrix, Eigen::ComputeFullV|Eigen::ComputeFullU);
			Eigen::Matrix3d U= svd.matrixU(), V=svd.matrixV();
			Eigen::Vector3d Sigma=svd.singularValues();

			double Sigma_value= 1;//(Sigma(0,0)+Sigma(1,0))/2.0;
			Sigma.fill(0.0);
			Sigma(0,0)= Sigma_value,Sigma(1,0) = Sigma_value;

			Essential_matrix= U*(Sigma.asDiagonal())*(V.transpose());
		}

		Eigen::JacobiSVD<Eigen::Matrix3d,Eigen::FullPivHouseholderQRPreconditioner> svd(Essential_matrix, Eigen::ComputeFullV|Eigen::ComputeFullU);
		Eigen::Matrix3d U= svd.matrixU(), V=svd.matrixV();
		Eigen::Vector3d Sigma= svd.singularValues();

		for(QList<double>::const_iterator i= pi_values.begin();i!=pi_values.end();i++){
			double PI_R= *i;
			Eigen::Matrix3d R= U*(computeRT_rzfunc(PI_R).transpose())*(V.transpose());
			//if(R(0,0)*(R(1,1)*R(2,2)-R(1,2)*R(2,1))-R(0,1)*(R(1,0)*R(2,2)-R(1,2)*R(2,0))+R(0,2)*(R(1,0)*R(2,1)-R(1,1)*R(2,0))<0)
			//	R=R*(-1.0);//Probably unnecessary
			for(QList<double>::const_iterator j= pi_values.begin();j!=pi_values.end();j++){
				double PI_T=*j;

				Eigen::Vector3d T_unhatted;

				/*
				Eigen::Matrix3d T= U*computeRT_rzfunc(PI_T)*Sigma*(U.transposed());
				T_unhatted(0,0)= (T(2,1)-T(1,2))/2;
				T_unhatted(1,0)= (T(0,2)-T(2,0))/2;
				T_unhatted(2,0)= (T(1,0)-T(0,1))/2;
				*/

				T_unhatted(0,0)=U(0,2)*(PI_T>=0?1:-1);
				T_unhatted(1,0)=U(1,2)*(PI_T>=0?1:-1);
				T_unhatted(2,0)=U(2,2)*(PI_T>=0?1:-1);

				StereopairPosition RT;
				RT.R= R, RT.T= T_unhatted;
				RTList.push_back(RT);
			}
		}

		return RTList;
	}


	QList<QVector3D> PointTriangulator::compute3DPoints(const SortedKeypointMatches&matches,const QList<StereopairPosition>& RTList){
		emit sgnStatusMessage("Performing 3D triangulation...");

		double best_min_depth=std::numeric_limits<double>::infinity(),best_max_depth=-std::numeric_limits<double>::infinity();

		StereopairPosition bestPosition=RTList.first();
		bool found=false;
		//Find best R/T pair
		for(QList<StereopairPosition>::const_iterator it1=RTList.begin();it1!=RTList.end();it1++){
			QList<QVector3D> Points3d= computeTriangulatedPoints(matches,it1->R,it1->T,false);

			//Search for maximum depth in order to evaluate and select best configuration
			double max_depth= -std::numeric_limits<double>::infinity(), min_depth=std::numeric_limits<double>::infinity();
			for(QList<QVector3D>::const_iterator it2=Points3d.begin();it2!=Points3d.end();it2++){
				if(it2->z()>max_depth)
					max_depth= it2->z();
				if(it2->z()<min_depth)
					min_depth= it2->z();
			}

			if(min_depth>0 && max_depth-min_depth>best_max_depth-best_min_depth){
				best_min_depth= min_depth;
				best_max_depth= max_depth;
				bestPosition= *it1;
				found=true;
			}

			emit sgnLogMessage(QString("Minimum depth is %1, maximum depth is %2").arg(min_depth).arg(max_depth));
		}

		if(best_min_depth<0)
			emit sgnLogMessage(QString("Warning: minimum depth is %1, less than zero!").arg(best_min_depth));

		if(best_min_depth==std::numeric_limits<double>::infinity())
			emit sgnLogMessage(QString("Warning: minimum depth is less than zero!"));

		return computeTriangulatedPoints(matches,bestPosition.R,bestPosition.T,true);
	}

	QList<QVector3D> PointTriangulator::computeTriangulatedPoints(const SortedKeypointMatches&matches,const Eigen::Matrix3d&R,const Eigen::Vector3d& T,bool normalizeCameras){
		Eigen::Matrix<double,3,4> P1,P2;
		P1.fill(0.0);
		for(int i=0;i<3;i++)
			P1(i,i)= 1;

		P2.fill(0.0);
		for(int i=0;i<3;i++)
			for(int j=0;j<3;j++)
				P2(i,j)= R(i,j);

		for(int i=0;i<3;i++)
			P2(i,3)= T(i,0);

		//Camera matrix
		if(normalizeCameras){
			P1=camera_K*P1;
			P2=camera_K*P2;
		}

		QList<QVector3D> resultPoints;

		Eigen::Matrix4d A;

		for(SortedKeypointMatches::const_iterator it=matches.begin();it!=matches.end();it++){
			QPointF x1= it.value().a, x2= it.value().b;
			for(int i=0;i<4;i++){
				A(0,i)= x1.x()*P1(2,i)-P1(0,i);
				A(1,i)= x1.y()*P1(2,i)-P1(1,i);
				A(2,i)= x2.x()*P2(2,i)-P2(0,i);
				A(3,i)= x2.y()*P2(2,i)-P2(1,i);
			}

			//Normalise rows of A
			/*
			for(int i=0;i<4;i++){
				double row_norm=0;
				for(int j=0;j<4;j++)
					row_norm+= A(i,j)*A(i,j);

				row_norm=sqrt(row_norm);
				for(int j=0;j<4;j++)
					A(i,j)/= row_norm;
			}
			*/


			Eigen::JacobiSVD<Eigen::Matrix4d,Eigen::FullPivHouseholderQRPreconditioner> svd(A, Eigen::ComputeFullV);
			Eigen::Matrix4d  V=svd.matrixV();
			Eigen::Vector4d Sigma= svd.singularValues();
			//Search for min column
			size_t Sigma_min_index=0;
			for(size_t i=1;i<4;i++)
				Sigma_min_index= Sigma(i,0)<Sigma(Sigma_min_index,0)?i:Sigma_min_index;

			//Sigma_min_index=3;//DZ 02.12.2010 seems Sigma=0 gives the best results. Sigma=1 gives a rougher surface.
			Eigen::Vector4d V_col_min= V.block(0,Sigma_min_index,4,1);

			QVector3D resultPoint(
					x1.x(),x1.y(),
					//V_col_min(0,0),V_col_min(1,0),
					V_col_min(2,0));

			resultPoints.push_back(resultPoint);
		}

		return resultPoints;
	}


	bool PointTriangulator::triangulatePoints(const SortedKeypointMatches&matches,const Eigen::Matrix3d& F,const QSize& imageSize){
		Points3D.clear();
		result= RESULT_OK;

		camera_K= computeCameraMatrix(imageSize);
		Eigen::Matrix3d Essential_matrix_projected=camera_K.transpose()*F*camera_K;

		//Estimate pose
		QList<PointTriangulator::StereopairPosition> RTList= computePose(Essential_matrix_projected);
		if(RTList.empty()){
			result= RESULT_POSE_UNDETERMINED;
			return false;
		}

		//Triangulate points
		Points3D= compute3DPoints(matches,RTList);
		if(Points3D.empty()){
			result= RESULT_TRIANGULATION_ERROR;
			return false;
		}
		return true;
	}

	bool PointTriangulator::triangulatePoints(const QList<cybervision::KeypointMatch>&matches,qreal angle){
		Points3D.clear();
		result= RESULT_OK;
		camera_K= Eigen::Matrix3d();

		QPointF centerA(0,0),centerB(0,0);
		for(QList<cybervision::KeypointMatch>::const_iterator i=matches.begin();i!=matches.end();i++){
			centerA.setX(centerA.x()+i->a.x());
			centerA.setY(centerA.y()+i->a.y());
			centerB.setX(centerB.x()+i->b.x());
			centerB.setY(centerB.y()+i->b.y());
		}
		centerA.setX(centerA.x()/matches.size());
		centerA.setY(centerA.y()/matches.size());
		centerB.setX(centerB.x()/matches.size());
		centerB.setY(centerB.y()/matches.size());

		Eigen::MatrixXd W(4,matches.size());
		{
			size_t j=0;
			for(QList<cybervision::KeypointMatch>::const_iterator i=matches.begin();i!=matches.end();i++,j++){
				W(0,j)=i->a.x()-centerA.x();
				W(1,j)=i->a.y()-centerA.y();
				W(2,j)=i->b.x()-centerB.x();
				W(3,j)=i->b.y()-centerB.y();
			}
		}


		//Decompose W into X*M
		Eigen::MatrixXd X,M;
		{
			Eigen::JacobiSVD<Eigen::MatrixXd,Eigen::FullPivHouseholderQRPreconditioner> svd(W, Eigen::ComputeFullV|Eigen::ComputeFullU);

			Eigen::MatrixXd V= svd.matrixV();
			Eigen::MatrixXd U= svd.matrixU();
			Eigen::MatrixXd Sigma= svd.singularValues().asDiagonal();

			Sigma= Sigma.block(0,0,3,3).eval();
			for(int i=0;i<3;i++)
				Sigma(i,i)= sqrt(Sigma(i,i));

			X= (Sigma*(V.block(0,0,V.rows(),3).transpose())).transpose();
			M= U.block(0,0,U.rows(),3)*Sigma;
		}

		//Compute matrix A for M*A*A^-1*X decomposition
		Eigen::MatrixXd Q;
		{
			//First, create the equation system (tomasiTr92Text.pdf page 14)
			//Use http://en.wikipedia.org/wiki/Kronecker_product#Matrix_equations for
			//transforming this into a linear equation system

			Eigen::MatrixXd A(6,9);
			A.row(0)= kronecker(M.block(0,0,1,M.cols()),M.block(0,0,1,M.cols()));
			A.row(1)= kronecker(M.block(1,0,1,M.cols()),M.block(1,0,1,M.cols()));
			A.row(2)= kronecker(M.block(2,0,1,M.cols()),M.block(2,0,1,M.cols()));
			A.row(3)= kronecker(M.block(3,0,1,M.cols()),M.block(3,0,1,M.cols()));
			A.row(4)= kronecker(M.block(0,0,1,M.cols()),M.block(1,0,1,M.cols()));
			A.row(5)= kronecker(M.block(2,0,1,M.cols()),M.block(3,0,1,M.cols()));

			Eigen::MatrixXd B(6,1);
			B<< 1,1,1,1,0,0;

			Eigen::MatrixXd Qv= leastSquares(A,B);
			Q= Eigen::MatrixXd(3,3);
			Q.col(0)= Qv.block(0,0,3,1);
			Q.col(1)= Qv.block(3,0,3,1);
			Q.col(2)= Qv.block(6,0,3,1);

			//N. Higham, "Computing a Nearest Symmetric Positive Semidefinite Matrix"
			//vincentToolbox - sfm\metricUpgrade.m
			Q= (Q+Q.transpose()).eval()/2;

			Eigen::JacobiSVD<Eigen::MatrixXd,Eigen::FullPivHouseholderQRPreconditioner> Qsvd(Q,Eigen::ComputeFullU);
			Eigen::MatrixXd QsvdU= Qsvd.matrixU();
			Eigen::MatrixXd QsvdS= QsvdU.transpose()*Q*QsvdU;

			for(Eigen::MatrixXd::Index i=0;i<QsvdS.cols() && i<QsvdS.rows();i++){
				if(QsvdS(i,i)<0)
					QsvdS(i,i)= 0;
				else
					QsvdS(i,i)= sqrt(QsvdS(i,i));
			}

			Q= QsvdU*QsvdS;
		}

		M= M*Q;
		X= (Q.inverse()*(X.transpose())).transpose();

		//Compute image-wide scale
		qreal scale=0;
		{
			Eigen::VectorXd Z_multiplier(3);
			Z_multiplier<< 0,0,tan(angle*M_PI/180);
			Eigen::MatrixXd Z_curr= X*Z_multiplier;
			Eigen::MatrixXd Z_disp(Z_curr.rows(),Z_curr.cols());

			Eigen::MatrixXd deltaX= W.block(0,0,1,W.cols())-W.block(2,0,1,W.cols());
			Eigen::MatrixXd deltaY= W.block(1,0,1,W.cols())-W.block(3,0,1,W.cols());

			for(Eigen::MatrixXd::Index i=0;i<W.cols();i++)
				Z_disp(i)= sqrt(deltaX(0,i)*deltaX(0,i)+deltaY(0,i)*deltaY(0,i));
			Eigen::MatrixXd scale_matrix= leastSquares(Z_curr,Z_disp);
			scale= scale_matrix(0,0);
		}

		Eigen::MatrixXd M1= M.block(0,0,2,2);

		//Final processing for points
		for(int i=0;i<X.rows();i++){
			//Project points with matrix M to remove rotation
			Eigen::MatrixXd projectedXY= M1*(X.block(i,0,1,2).transpose());
			//double scale= sqrt(projectedXY(0,0)*projectedXY(0,0)+projectedXY(1,0)*projectedXY(1,0))/sqrt(X(i,0)*X(i,0)+X(i,1)*X(i,1));
			QVector3D resultPoint(projectedXY(0,0),projectedXY(1,0),scale*X(i,2));
			//QVector3D resultPoint(X(i,0),X(i,1),X(i,2)*scale);
			Points3D.push_back(resultPoint);
		}

		if(Points3D.empty()){
			result= RESULT_TRIANGULATION_ERROR;
			return false;
		}
		return true;
	}
}

bool cybervision::PointTriangulator::triangulatePoints(const SortedKeypointMatches&matches,qreal angle,bool filterPeaks){
	QList<cybervision::KeypointMatch> matches_values= matches.values();
	bool ok= triangulatePoints(matches_values,angle);
	if(!ok/* || !filterPeaks*/)
		return ok;
	{
		QSet<int> peaks= findPeaks(Points3D);
		QList<cybervision::KeypointMatch> matches_values_no_peaks;
		for(int i=0;i<matches_values.size();i++){
			if(!peaks.contains(i))
				matches_values_no_peaks<<matches_values[i];
		}
		matches_values= matches_values_no_peaks;
	}
	return triangulatePoints(matches_values,angle);
}

QSet<int> cybervision::PointTriangulator::findPeaks(const QList<QVector3D> &points) const{
	QSet<int> discardedPoints;
	QList<qreal> values_x,values_y;

	//Find min/max values
	//Fill the first cell
	{
		qreal minX= points.begin()->x(), minY= points.begin()->y(), maxX= points.begin()->x(), maxY= points.begin()->y();
		for(QList<QVector3D>::const_iterator it=points.begin();it!=points.end();it++){
			minX= qMin(minX,it->x());
			minY= qMin(minY,it->y());
			maxX= qMax(maxX,it->x());
			maxY= qMax(maxY,it->y());
		}
		values_x<<minX<<maxX;
		values_y<<minY<<maxY;
	}

	for(int resolution=0;resolution<Options::gridResolution;resolution++){
		//Increase grid density (steps)
		for(QList<qreal>::iterator it=values_x.begin();it!=values_x.end()-1;){
			//Add X-middle
			qreal min_x= *it,max_x= *(it+1);
			qreal middle_x=(min_x+max_x)/2;
			it= values_x.insert(it+1,middle_x)+1;
		}
		for(QList<qreal>::iterator jt=values_y.begin();jt!=values_y.end()-1;){
			//Add Y-middle
			qreal min_y= *jt,max_y= *(jt+1);
			qreal middle_y=(min_y+max_y)/2;
			jt= values_y.insert(jt+1,middle_y)+1;
		}
		//Iterate through grid

		#pragma omp parallel
		for(QList<qreal>::const_iterator it=values_x.begin();it!=values_x.end()-1;it++){
			qreal min_x= *it,max_x= *(it+1);
			#pragma omp single nowait
			{
				for(QList<qreal>::const_iterator jt=values_y.begin();jt!=values_y.end()-1;jt++){
					qreal min_y= *jt,max_y= *(jt+1);
					//Create filter for peaks
					QVector2D min(min_x,min_y);
					QVector2D max(max_x,max_y);
					QVector2D middle= min+(max-min)/2;
					QList<qreal> Zp;
					for(QList<QVector3D>::const_iterator kt=points.begin();kt!=points.end();kt++){
						qreal distance= (QVector2D(*kt)-middle).length();
						if(distance <= Options::gridPeakFilterRadius*(max-middle).length())
							Zp<< kt->z();
					}
					qSort(Zp);
					qreal median,
							Zmax= !Zp.isEmpty()?Zp.at(Zp.size()-1):0,
							Zmin= !Zp.isEmpty()?Zp.at(0):0;

					if(Zp.isEmpty())
						continue;
					else
						median= Zp.size()%2==1 ?
									Zp[(Zp.size()-1)/2] :
									((Zp[Zp.size()/2-1]+Zp[Zp.size()/2])/2.0);

					if(Zp.size()>=3){
						for(int i=0;i<Zp.size()/2;i++)
							if(abs(Zp.at(i)-median)>Options::gridPeakSize*abs(Zp.at(i+1)-median))
								Zmin= Zp.at(i+1);
							else break;

						for(int i=Zp.size()-1;i>Zp.size()/2;i--)
							if(abs(Zp.at(i)-median)>Options::gridPeakSize*abs(Zp.at(i-1)-median))
								Zmax= Zp.at(i-1);
							else break;
					}
					Zp.clear();

					for(int k=0;k<points.length();k++){
						if((points[k].z()>=Zmin) && (points[k].z()<=Zmax))
							continue;

						if(points[k].x()>=min.x() && points[k].x()<=max.x() && points[k].y()>=min.y() && points[k].y()<=max.y()){
							#pragma omp critical
							{
								discardedPoints.insert(k);
							}
						}
					}
				}
			}
		}

	}
	return discardedPoints;
}
