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

	bool Reconstructor::computePose(SortedKeypointMatches& matches){
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

		//Use the RANSAC algorithm to estimate camera poses

		KeypointMatches best_consensus_set;
		double best_error= std::numeric_limits<float>::infinity();
		QGenericMatrix<3,3,double> best_E;

		//TODO: convert shit to OpenMP
		for(int i=0;i<Options::RANSAC_k;i++){
			if(i%50==0)
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
			QGenericMatrix<3,3,double> E=computeEssentialMatrix(consensus_set);
			//Expand consensus set
			for(SortedKeypointMatches::const_iterator it1= matches.begin();it1!=matches.end();it1++){
				//Check error
				float current_error= computeEssentialMatrixError(E,it1.value());
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
			return false;
		}else{
			emit sgnLogMessage(QString("RANSAC consensus found, error=%1, size=%2").arg(best_error,0,'g',4).arg(best_consensus_set.size()));
		}
		return true;
	}


	QGenericMatrix<3,3,double> Reconstructor::computeEssentialMatrix(const KeypointMatches& matches){
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
		{
			SVD<3,3,double> svd(E);
			QGenericMatrix<3,3,double> Sigma_new;
			Sigma_new.fill(0.0);
			Sigma_new(0,0)= 1.0, Sigma_new(1,1)= 1.0;
			E= svd.getU()*Sigma_new*(svd.getV().transposed());
		}
		return E;
	}

	float Reconstructor::computeEssentialMatrixError(const QGenericMatrix<3,3,double>&E, const KeypointMatch& match) const{
		QGenericMatrix<1,3,double> x1; x1(0,0)=match.a.x(), x1(1,0)=match.a.y(), x1(2,0)=1;
		QGenericMatrix<3,1,double> x2; x2(0,0)=match.b.x(), x2(0,1)=match.b.y(), x2(0,2)=1;
		QGenericMatrix<1,1,double> result= x2*E*x1;
		return fabs(result(0,0));
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
		if(!Reconstructor::computePose(matches)){
			errorString= "Error when estimating pose";
			return false;
		}
		return true;
	}

	//Getters
	bool Reconstructor::isOk()const{ return !errorString.isNull()&&!errorString.isEmpty(); }
	QString Reconstructor::getError()const{ return errorString; }
}
