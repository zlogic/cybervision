#include "pointtriangulator.h"

#include <Reconstruction/options.h>

#include <Eigen/Dense>
#include <Eigen/SVD>
#include <limits>
#include <QSize>

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
			Eigen::JacobiSVD<Eigen::Matrix3d> svd(Essential_matrix, Eigen::ComputeFullV|Eigen::ComputeFullU);
			Eigen::Matrix3d U= svd.matrixU(), V=svd.matrixV();
			Eigen::Vector3d Sigma=svd.singularValues();

			double Sigma_value= 1;//(Sigma(0,0)+Sigma(1,0))/2.0;
			Sigma.fill(0.0);
			Sigma(0,0)= Sigma_value,Sigma(1,0) = Sigma_value;

			Essential_matrix= U*(Sigma.asDiagonal())*(V.transpose());
		}

		Eigen::JacobiSVD<Eigen::Matrix3d> svd(Essential_matrix, Eigen::ComputeFullV|Eigen::ComputeFullU);
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


			Eigen::JacobiSVD<Eigen::Matrix4d> svd(A, Eigen::ComputeFullV);
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

	bool PointTriangulator::triangulatePoints(const SortedKeypointMatches&matches){
		Points3D.clear();
		result= RESULT_OK;
		camera_K= Eigen::Matrix3d();

		QPointF centerA(0,0),centerB(0,0);
		for(SortedKeypointMatches::const_iterator i=matches.begin();i!=matches.end();i++){
			centerA.setX(centerA.x()+i.value().a.x());
			centerA.setY(centerA.y()+i.value().a.y());
			centerB.setX(centerB.x()+i.value().b.x());
			centerB.setY(centerB.y()+i.value().b.y());
		}
		centerA.setX(centerA.x()/matches.size());
		centerA.setY(centerA.y()/matches.size());
		centerB.setX(centerB.x()/matches.size());
		centerB.setY(centerB.y()/matches.size());

		Eigen::MatrixXd W(4,matches.size());
		{
			size_t j=0;
			for(SortedKeypointMatches::const_iterator i=matches.begin();i!=matches.end();i++,j++){
				W(0,j)=i.value().a.x()-centerA.x();
				W(1,j)=i.value().a.y()-centerA.y();
				W(2,j)=i.value().b.x()-centerB.x();
				W(3,j)=i.value().b.y()-centerB.y();
			}
		}

		Eigen::JacobiSVD<Eigen::MatrixXd> svd(W, Eigen::ComputeThinV);
		Eigen::MatrixXd X(matches.size(),3);
		Eigen::MatrixXd V= svd.matrixV();

		if(Options::triangulationMode==Options::TRIANGULATION_PARALLEL_V){
			//Method 1 (simpler)
			X= V.block(0,0,V.rows(),3);
		}else if(Options::triangulationMode==Options::TRIANGULATION_PARALLEL_SV) {
			//Method 2 (S*V')'
			Eigen::MatrixXd S= svd.singularValues().asDiagonal();
			S= S.block(0,0,3,3);
			V= V.block(0,0,V.rows(),3);
			X= (S*(V.transpose())).transpose();
		}


		for(int i=0;i<X.rows();i++){
			QVector3D resultPoint(X(i,0),X(i,1),X(i,2));
			Points3D.push_back(resultPoint);
		}

		if(Points3D.empty()){
			result= RESULT_TRIANGULATION_ERROR;
			return false;
		}
		return true;
	}
}
