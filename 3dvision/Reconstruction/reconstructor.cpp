#include "reconstructor.h"

#include <Reconstruction/svd.h>
#include <SIFT/siftgateway.h>

#include <limits>
#include <ctime>

#include <QImage>

namespace cybervision{


	bool Reconstructor::KeypointMatch::operator==(const KeypointMatch&m)const{ return a==m.a && b==m.b; }

	Reconstructor::Reconstructor(){
		srand(time(NULL));
	}

	Reconstructor::SortedKeypointMatches Reconstructor::extractMatches(const QString& filename1,const QString& filename2){
		emit sgnStatusMessage("Detecting SIFT keypoints...");
		emit sgnLogMessage("Starting SIFT keypoint detection");
		emit sgnLogMessage(QString("Loading images %1 and %2").arg(filename1).arg(filename2));
		QImage img1(filename1),img2(filename2);
		QList <SIFT::Keypoint> keypoints1,keypoints2;
		{
			SIFT::Extractor extractor;
			emit sgnLogMessage(QString("Extracting keypoints from %1").arg(filename1));
			keypoints1= extractor.extract(img1);
			emit sgnLogMessage(QString("Extracting keypoints from %2").arg(filename2));
			keypoints2= extractor.extract(img2);
		}

		emit sgnStatusMessage("Matching SIFT keypoints...");
		emit sgnLogMessage(QString("Matching keypoints from %1 to %2").arg(filename1).arg(filename2));
		SortedKeypointMatches matches;
		//match first image with second
		for(QList<SIFT::Keypoint>::const_iterator it1= keypoints1.begin();it1!=keypoints1.end();it1++){
			KeypointMatch match;
			float minDistance= std::numeric_limits<float>::infinity();
			for(QList<SIFT::Keypoint>::const_iterator it2= keypoints2.begin();it2!=keypoints2.end();it2++){
				float distance= it1->distance(*it2);
				if(distance<minDistance){
					minDistance= distance;
					match.a= QPointF(it1->getX(),it1->getY());
					match.b= QPointF(it2->getX(),it2->getY());
				}
			}
			if(minDistance!=std::numeric_limits<float>::infinity() && minDistance<Options::MaxKeypointDistance){
				if(!matches.contains(minDistance,match))
					matches.insert(minDistance,match);
			}
		}
		emit sgnLogMessage(QString("Matching keypoints from %1 to %2").arg(filename2).arg(filename1));
		//match second image with first
		for(QList<SIFT::Keypoint>::const_iterator it2= keypoints2.begin();it2!=keypoints2.end();it2++){
			KeypointMatch match;
			float minDistance= std::numeric_limits<float>::infinity();
			for(QList<SIFT::Keypoint>::const_iterator it1= keypoints1.begin();it1!=keypoints1.end();it1++){
				float distance= it2->distance(*it1);
				if(distance<minDistance){
					minDistance= distance;
					match.a= QPointF(it1->getX(),it1->getY());
					match.b= QPointF(it2->getX(),it2->getY());
				}
			}
			if(minDistance!=std::numeric_limits<float>::infinity() && minDistance<Options::MaxKeypointDistance){
				if(!matches.contains(minDistance,match))
					matches.insert(minDistance,match);
			}
		}

		emit sgnLogMessage(QString("Found %1 keypoint matches").arg(matches.size()));
		return matches;
	}

	QList<Reconstructor::StereopairPosition> Reconstructor::computePose(SortedKeypointMatches& matches){
		emit sgnStatusMessage("Estimating pose...");

		//Increase speed with precomputed lists
		QList<float> matches_keys= matches.uniqueKeys();
		int matches_keys_max_i=0;
		for(int i=0;i<matches_keys.size();i++)
			if(matches_keys.at(i)<=Options::ReliableDistance)
				matches_keys_max_i= i;
			else
				break;
		matches_keys_max_i=std::max(matches_keys_max_i,8);

		//Compute centroid and scaling factor (normalise2dpts)
		QPointF centroidA,centroidB;
		double scalingA, scalingB;
		{
			double sumAX=0,sumAY=0,sumBX=0,sumBY=0;
			for(SortedKeypointMatches::const_iterator it1= matches.begin();it1!=matches.end();it1++){
				sumAX+= it1.value().a.x(), sumBX+= it1.value().b.x();
				sumAY+= it1.value().a.y(), sumBY+= it1.value().b.y();
			}
			centroidA.setX(sumAX/matches.size()), centroidB.setX(sumBX/matches.size());
			centroidA.setY(sumAY/matches.size()), centroidB.setY(sumBY/matches.size());
			double distA=0,distB=0;
			for(SortedKeypointMatches::const_iterator it1= matches.begin();it1!=matches.end();it1++){
				double dAX= it1.value().a.x()-centroidA.x(), dAY= it1.value().a.y()-centroidA.y();
				double dBX= it1.value().b.x()-centroidB.x(), dBY= it1.value().b.y()-centroidB.y();
				distA+= sqrt(dAX*dAX+dAY*dAY);
				distB+= sqrt(dBX*dBX+dBY*dBY);
			}
			distA/= matches.size();
			distB/= matches.size();
			scalingA= sqrt(2)/distA;
			scalingB= sqrt(2)/distB;
		}

		//Use the RANSAC algorithm to estimate camera poses

		KeypointMatches best_consensus_set;
		double best_error= std::numeric_limits<float>::infinity();
		QGenericMatrix<3,3,double> best_E;

		//TODO: convert this to OpenMP
		for(int i=0;i<Options::RANSAC_k;i++){
			if(i%(Options::RANSAC_k/10)==0)
				emit sgnLogMessage(QString("RANSAC %1% complete").arg((i*100)/Options::RANSAC_k));

			//Extract RANSAC_n random values
			KeypointMatches consensus_set;
			SortedKeypointMatches master_consensus_set;

			while(consensus_set.size()<Options::RANSAC_n){
				//Generate random distance
				int rand_number= (matches_keys_max_i*(long long)rand())/RAND_MAX;
				float random_distance= matches_keys.at(std::min(rand_number,matches_keys.size()-1));
				//Find points at generated distance
				QList<KeypointMatch> it1_values= matches.values(random_distance);
				//Find a match on a random position
				int random_pos= (it1_values.size()*(long long)rand())/RAND_MAX;
				QPair<float,KeypointMatch> new_match(
					random_distance,
					it1_values.at(std::min(it1_values.size()-1,random_pos))
				);
				if(!master_consensus_set.contains(new_match.first,new_match.second)){
					master_consensus_set.insert(new_match.first,new_match.second);
					consensus_set.push_back(new_match);
				}
			}

			double error=0;
			//Compute E from the random values
			QGenericMatrix<3,3,double> E=computeEssentialMatrix(consensus_set,centroidA,centroidB,scalingA,scalingB);

			//Expand consensus set
			for(SortedKeypointMatches::const_iterator it1= matches.begin();it1!=matches.end();it1++){
				//Check error
				double current_error= computeEssentialMatrixError(E,it1.value(),centroidA,centroidB,scalingA,scalingB);
				//Check if match exists in master consensus set
				if(master_consensus_set.contains(it1.key(),it1.value())){
					//Add to global error
					error+= current_error;
					continue;//Match was found in master consensus set, ignore
				}
				if(current_error<Options::RANSAC_t){
					//Add to global error
					error+= current_error;
					consensus_set.push_back(QPair<float,KeypointMatch>(it1.key(),it1.value()));
				}
			}

			if(consensus_set.size()>Options::RANSAC_d){
				//Error was already computed, normalize it
				error/= consensus_set.size();
				if(consensus_set.size()>best_consensus_set.size() || (consensus_set.size()==best_consensus_set.size() && error<best_error)){
					best_consensus_set= consensus_set;
					best_error= error;
					best_E= E;
				}
			}
		}

		if(best_consensus_set.empty()){
			emit sgnLogMessage("No RANSAC consensus found");
			return QList<Reconstructor::StereopairPosition>();
		}else{
			emit sgnLogMessage(QString("RANSAC consensus found, error=%1, size=%2").arg(best_error,0,'g',4).arg(best_consensus_set.size()));
		}

		//Update matches
		matches.clear();
		for(KeypointMatches::const_iterator it=best_consensus_set.begin();it!=best_consensus_set.end();it++)
			matches.insert(it->first,it->second);

		QList<StereopairPosition> RTList= computeRT(best_E);
		//Output R,T matrices to log
		/*
		QString RT_str;
		for(QList<StereopairPosition>::const_iterator i= RTList.begin();i!=RTList.end();i++){
			RT_str.append("R%1T\n").arg(QString(""),40);
			for(int j=0;j<3;j++){
				qreal T_value_i= j==0? i->T.x(): (j==1?i->T.y():i->T.z());//Extract i-th value from QVector3D
				QString matrix_row= QString("%1 %2 %3 %4\n").arg(i->R(j,0),6,'g',6).arg(i->R(j,1),6,'g',6).arg(i->R(j,2),6,'g',6).arg(T_value_i,6,'g',6);
				RT_str.append(matrix_row);
			}
		}
		emit sgnLogMessage(QString("Resulting camera poses\n").append(RT_str));
		*/
		return RTList;
	}


	QGenericMatrix<3,3,double> Reconstructor::computeEssentialMatrix(const KeypointMatches& matches,const QPointF& centroidA,const QPointF& centroidB,double scalingA,double scalingB){
		if(matches.size()!=8){
			emit sgnLogMessage(QString("Wrong consensus set size (%1), should be %2").arg(matches.size()).arg(8));
			/*
			QGenericMatrix<3,3,float> badResult;
			badResult.fill(0.0);
			return badResult;
			*/
		}
		//Create the Chi matrix
		QGenericMatrix<9,8,double> chi;
		{
			int i=0;
			for(KeypointMatches::const_iterator it= matches.begin();it!=matches.end();it++,i++){
				qreal x1= it->second.a.x(), x2=it->second.b.x(), y1=it->second.a.y(), y2=it->second.b.y(), z1=1, z2=1;
				//Normalize2dpts
				//x1= (x1-centroidA.x())*scalingA, x2= (x2-centroidB.x())*scalingB;
				//y1= (y1-centroidA.y())*scalingA, y2= (y2-centroidB.y())*scalingB;
				//Kronecker product
				chi(i,0)= x1*x2, chi(i,1)= x1*y2, chi(i,2)= x1*z2;
				chi(i,3)= y1*x2, chi(i,4)= y1*y2, chi(i,5)= y1*z2;
				chi(i,6)= z1*x2, chi(i,7)= z1*y2, chi(i,8)= z1*z2;
			}
		}
		//Compute V_chi from the SVD decomposition
		QGenericMatrix<3,3,double> E;
		{
			SVD<9,8,double> svd(chi);
			QGenericMatrix<9,9,double> V_chi= svd.getV();
			//Get and unstack E
			E(0,0)=V_chi(0,8), E(1,0)=V_chi(1,8), E(2,0)=V_chi(2,8);
			E(0,1)=V_chi(3,8), E(1,1)=V_chi(4,8), E(2,1)=V_chi(5,8);
			E(0,2)=V_chi(6,8), E(1,2)=V_chi(7,8), E(2,2)=V_chi(8,8);
		}
		//"Normalize" E
		//(Project into essential space (do we need this?))
		{
			SVD<3,3,double> svd(E);

			QGenericMatrix<3,3,double> Sigma_new;
			Sigma_new.fill(0.0);
			Sigma_new(0,0)= 1.0, Sigma_new(1,1)= 1.0;

			QGenericMatrix<3,3,double> Sigma= svd.getSigma();
			Sigma(2,2)= 0.0;

			E= svd.getU()*Sigma*(svd.getV().transposed());
		}
		//De-normalize2dpts
		/*
		QGenericMatrix<3,3,double> T1;
		T1.fill(0.0);
		T1(0,0)= scalingA, T1(1,1)= scalingA, T1(2,2)= 1;
		T1(0,2)= -scalingA*centroidA.x(), T1(1,2)= -scalingA*centroidA.y();
		QGenericMatrix<3,3,double> T2;
		T2.fill(0.0);
		T2(0,0)= scalingB, T1(1,1)= scalingB, T1(2,2)= 1;
		T2(0,2)= -scalingB*centroidB.x(), T1(1,2)= -scalingB*centroidB.y();
		*/
		return E;

		//return T2.transposed()*E*T1;
	}

	double Reconstructor::computeEssentialMatrixError(const QGenericMatrix<3,3,double>&E, const KeypointMatch& match,const QPointF& centroidA,const QPointF& centroidB,double scalingA,double scalingB) const{
		QGenericMatrix<1,3,double> x1; x1(0,0)=match.a.x(), x1(1,0)=match.a.y(), x1(2,0)=1;
		QGenericMatrix<1,3,double> x2; x2(0,0)=match.b.x(), x2(1,0)=match.b.y(), x2(2,0)=1;
		QGenericMatrix<1,1,double> x2tEx1= x2.transposed()*E*x1;

		QGenericMatrix<1,3,double> Ex1=E*x1,Etx2=E.transposed()*x2;
		return fabs(x2tEx1(0,0)*x2tEx1(0,0)/(Ex1(0,0)*Ex1(0,0)+Ex1(1,0)*Ex1(1,0)+Etx2(0,0)*Etx2(0,0)+Etx2(1,0)*Etx2(1,0)));
	}


	QGenericMatrix<3,3,double> Reconstructor::computeRT_rzfunc(double angle) const{
		QGenericMatrix<3,3,double> result;
		result.fill(0.0);
		result(0,1)= angle>=0?1:(-1);
		result(1,0)= -(angle>=0?1:(-1));
		result(2,2)= 1.0;
		return result;
	}

	QList<Reconstructor::StereopairPosition> Reconstructor::computeRT(const QGenericMatrix<3,3,double>&Essential_matrix) const{
		SVD<3,3,double> svd(Essential_matrix);

		QList<Reconstructor::StereopairPosition> RTList;
		QList<double> pi_values;
		pi_values<<M_PI_2<<-M_PI_2;

		QGenericMatrix<3,3,double> U= svd.getU(),Ut= U.transposed(),Sigma=svd.getSigma(), V=svd.getV(),Vt=V.transposed();

		Sigma.fill(0.0); Sigma(0,0)= 1.0, Sigma(1,1)= 1.0;
		for(QList<double>::const_iterator i= pi_values.begin();i!=pi_values.end();i++){
			double PI_R= *i;
			QGenericMatrix<3,3,double> R= U*(computeRT_rzfunc(PI_R).transposed())*Vt;
			for(QList<double>::const_iterator j= pi_values.begin();j!=pi_values.end();j++){
				double PI_T=*j;
				QGenericMatrix<3,3,double> T= U*computeRT_rzfunc(PI_T)*Sigma*Ut;
				QVector3D T_unhatted(
						(T(2,1)+T(1,2))/2,
						(T(0,2)+T(2,0))/2,
						(T(1,0)+T(0,1))/2
					);
				StereopairPosition RT;
				RT.R= R, RT.T= T_unhatted;
				RTList.push_back(RT);
			}
		}

		return RTList;
	}


	QList<QVector3D> Reconstructor::compute3DPoints(const SortedKeypointMatches&matches,const QList<StereopairPosition>& RTList){
		emit sgnStatusMessage("Performing 3D triangulation...");

		double best_max_depth= -std::numeric_limits<double>::infinity(), best_min_depth=std::numeric_limits<double>::infinity();
		QList<QVector3D> best_Points3d;
		for(QList<StereopairPosition>::const_iterator it1=RTList.begin();it1!=RTList.end();it1++){
			QList<QVector3D> Points3d= computeTriangulatedPoints(matches,it1->R,it1->T);

			//Search for maximum depth in order to evaluate and select best configuration
			double max_depth= -std::numeric_limits<double>::infinity(), min_depth=std::numeric_limits<double>::infinity();
			for(QList<QVector3D>::const_iterator it2=Points3d.begin();it2!=Points3d.end();it2++){
				if(it2->z()>max_depth)
					max_depth= it2->z();
				if(it2->z()<min_depth)
					min_depth= it2->z();
			}

			if(max_depth>best_max_depth){
				best_max_depth= max_depth;
				best_min_depth= min_depth;
				best_Points3d= Points3d;
			}

			emit sgnLogMessage(QString("Minimum depth is %1").arg(min_depth));

			if(min_depth>0)
				break;
		}

		if(best_max_depth<0)
			emit sgnLogMessage(QString("Warning: minimum depth is %1, less than zero!").arg(best_min_depth));
		return best_Points3d;
	}

	QList<QVector3D> Reconstructor::computeTriangulatedPoints(const SortedKeypointMatches&matches,const QGenericMatrix<3,3,double>&R,const QVector3D& T){
		QGenericMatrix<4,3,double> P1,P2;
		P1.fill(0.0);
		for(int i=0;i<3;i++)
			P1(i,i)= 1;

		P2.fill(0.0);
		for(int i=0;i<3;i++)
			for(int j=0;j<3;j++)
				P2(i,j)= R(i,j);
		P2(0,3)= T.x(), P2(1,3)= T.y(), P2(2,3)= T.z();

		//Search for maximums
		double max_x=0, max_y=0;
		for(SortedKeypointMatches::const_iterator it=matches.begin();it!=matches.end();it++){
			if(it.value().a.x()>max_x)
				max_x= it.value().a.x();
			if(it.value().b.x()>max_x)
				max_x= it.value().b.x();
			if(it.value().a.y()>max_y)
				max_y= it.value().a.y();
			if(it.value().b.y()>max_y)
				max_y= it.value().b.y();
		}

		max_x*=10, max_y*=10;
		QList<QVector3D> resultPoints;

		QGenericMatrix<3,4,double> A;
		for(SortedKeypointMatches::const_iterator it=matches.begin();it!=matches.end();it++){
			QPointF x1= it.value().a, x2= it.value().b;
			for(int i=0;i<3;i++){
				A(0,i)= (x1.x()+max_x)*P1(i,2)-P1(i,0);
				A(1,i)= (x1.y()+max_y)*P1(i,2)-P1(i,1);
				A(2,i)= (x2.x()+max_x)*P2(i,2)-P2(i,0);
				A(3,i)= (x2.y()+max_y)*P2(i,2)-P2(i,1);
			}
			SVD<3,4,double> svd(A);
			QGenericMatrix<3,3,double> Sigma= svd.getSigma();
			QGenericMatrix<3,3,double> V= svd.getV();
			QGenericMatrix<1,3,double> V_col3;
			for(int i=0;i<3;i++)
				V_col3(i,0)= V(i,2);
			QGenericMatrix<1,3,double> X= Sigma*V_col3;


			//emit sgnLogMessage(QString("X=%1 %2 %3").arg(X(0,0)).arg(X(1,0)).arg(X(2,0)));
			QVector3D resultPoint(x1.x(),x1.y(),X(2,0));

			resultPoints.push_back(resultPoint);
		}

		return resultPoints;
	}

	bool Reconstructor::run(const QString& filename1,const QString& filename2){
		//Extract and sort matches by distance
		SortedKeypointMatches matches= extractMatches(filename1,filename2);
		if(matches.isEmpty()){
			errorString= "No matches found";
			return false;
		}
		if(matches.size()<Options::MinMatches){
			errorString= QString("Not enough matches (%1), need at least %2").arg(matches.size()).arg(Options::MinMatches);
			return false;
		}
		//Estimate camera poses
		QList<Reconstructor::StereopairPosition> RTList= Reconstructor::computePose(matches);
		if(RTList.empty()){
			errorString= "Error when estimating pose";
			return false;
		}

		//Triangulate points
		QList<QVector3D> Points3D= compute3DPoints(matches,RTList);
		if(Points3D.empty()){
			errorString= "Error during 3D triangulation";
			return false;
		}
		//log points to console
		QString pointsStr="A=[";
		for(QList<QVector3D>::const_iterator it2=Points3D.begin();it2!=Points3D.end();it2++)
			pointsStr.append(QString("%1 %2 %3%4").arg(it2->x()).arg(it2->y()).arg(it2->z()).arg(it2!=(Points3D.end()-1)?";":""));
		pointsStr.append("];");

		emit sgnLogMessage(QString(pointsStr));
		return true;
	}

	//Getters
	bool Reconstructor::isOk()const{ return !errorString.isNull()&&!errorString.isEmpty(); }
	QString Reconstructor::getErrorString()const{ return errorString; }
}
