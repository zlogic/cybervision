#include "pointtriangulator.h"

#include <Reconstruction/options.h>
#include <Reconstruction/svd.h>
#include <limits>
#include <QSize>

namespace cybervision{

	PointTriangulator::PointTriangulator(QObject *parent) : QObject(parent){
		result= RESULT_OK;
	}

	QList<QVector3D> PointTriangulator::getPoints3D()const{ return Points3D; }
	PointTriangulator::TriangulationResult PointTriangulator::getResult()const{ return result; }

	QList<PointTriangulator::StereopairPosition> PointTriangulator::computePose(const QGenericMatrix<3,3,double>& F){
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


	QGenericMatrix<3,3,double> PointTriangulator::computeRT_rzfunc(double angle) const{
		QGenericMatrix<3,3,double> result;
		result.fill(0.0);
		result(0,1)= angle>=0?1.0:(-1.0);
		result(1,0)= -(angle>=0?1.0:(-1.0));
		result(2,2)= 1.0;
		return result;
	}


	QGenericMatrix<3,3,double> PointTriangulator::computeCameraMatrix(const QSize& imageSize)const{
		QGenericMatrix<3,3,double> K;
		K.fill(0.0);
		K(0,0)= Options::scaleFocalDistance;//Focal distance X
		K(1,1)= Options::scaleFocalDistance;//Focal distance Y
		K(0,2)= -imageSize.width();//Optical center X
		K(1,2)= -imageSize.height();//Optical center Y
		K(2,2)= 1;
		return K;
	}

	QList<PointTriangulator::StereopairPosition> PointTriangulator::computeRT(QGenericMatrix<3,3,double> Essential_matrix) const{
		QList<PointTriangulator::StereopairPosition> RTList;
		QList<double> pi_values;
		pi_values<<M_PI_2<<-M_PI_2;

		//Project into essential space
		if(false){
			SVD<3,3,double> svd(Essential_matrix);
			QGenericMatrix<3,3,double> U= svd.getU(),Sigma=svd.getSigma(), V=svd.getV();

			double Sigma_value= 1;//(Sigma(0,0)+Sigma(1,1))/2.0;
			Sigma.fill(0.0);
			Sigma(0,0)= Sigma_value,Sigma(1,1) = Sigma_value;

			Essential_matrix= U*Sigma*(V.transposed());
		}

		SVD<3,3,double> svd(Essential_matrix);
		QGenericMatrix<3,3,double> U= svd.getU(),Sigma=svd.getSigma(), V=svd.getV();

		for(QList<double>::const_iterator i= pi_values.begin();i!=pi_values.end();i++){
			double PI_R= *i;
			QGenericMatrix<3,3,double> R= U*(computeRT_rzfunc(PI_R).transposed())*(V.transposed());
			//if(R(0,0)*(R(1,1)*R(2,2)-R(1,2)*R(2,1))-R(0,1)*(R(1,0)*R(2,2)-R(1,2)*R(2,0))+R(0,2)*(R(1,0)*R(2,1)-R(1,1)*R(2,0))<0)
			//	R=R*(-1.0);//Probably unnecessary
			for(QList<double>::const_iterator j= pi_values.begin();j!=pi_values.end();j++){
				double PI_T=*j;

				QGenericMatrix<1,3,double> T_unhatted;

				/*
				QGenericMatrix<3,3,double> T= U*computeRT_rzfunc(PI_T)*Sigma*(U.transposed());
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

	QList<QVector3D> PointTriangulator::computeTriangulatedPoints(const SortedKeypointMatches&matches,const QGenericMatrix<3,3,double>&R,const QGenericMatrix<1,3,double>& T,bool normalizeCameras){
		QGenericMatrix<4,3,double> P1,P2;
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

		QGenericMatrix<4,4,double> A;

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

			SVD<4,4,double> svd(A);
			QGenericMatrix<4,4,double> Sigma= svd.getSigma();
			QGenericMatrix<4,4,double> V= svd.getV();
			//Search for min column
			size_t Sigma_min_index=0;
			for(size_t i=1;i<4;i++)
				Sigma_min_index= Sigma(i,i)<Sigma(Sigma_min_index,Sigma_min_index)?i:Sigma_min_index;

			//Sigma_min_index=3;//DZ 02.12.2010 seems Sigma=0 gives the best results. Sigma=1 gives a rougher surface.
			QGenericMatrix<1,4,double> V_col_min;
			for(int i=0;i<4;i++)
				V_col_min(i,0)= V(i,Sigma_min_index);

			QVector3D resultPoint(
					x1.x(),x1.y(),
					//V_col_min(0,0),V_col_min(1,0),
					V_col_min(2,0));

			resultPoints.push_back(resultPoint);
		}

		return resultPoints;
	}


	bool PointTriangulator::triangulatePoints(const SortedKeypointMatches&matches,const QGenericMatrix<3,3,double>& F,const QSize& imageSize){
		Points3D.clear();
		result= RESULT_OK;

		camera_K= computeCameraMatrix(imageSize);
		QGenericMatrix<3,3,double> Essential_matrix_projected=camera_K.transposed()*F*camera_K;

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
		camera_K= QGenericMatrix<3,3,double>();

		for(SortedKeypointMatches::const_iterator it=matches.begin();it!=matches.end();it++){
			QPointF x1= it.value().a, x2=it.value().b;
			QVector3D resultPoint(
					x1.x(),x1.y(),
					sqrt((x1.x()-x2.x())*(x1.x()-x2.x())+(x1.y()-x2.y())*(x1.y()-x2.y())));

			Points3D.push_back(resultPoint);
		}

		if(Points3D.empty()){
			result= RESULT_TRIANGULATION_ERROR;
			return false;
		}
		return true;
	}
}
