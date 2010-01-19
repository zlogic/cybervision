#include "reconstructor.h"

#include <siftgateway.h>

#include <limits>
#include <ctime>

#include <QImage>

namespace cybervision{


	bool Reconstructor::KeypointMatch::operator==(const KeypointMatch&m)const{ return a==m.a && b==m.b; }

	Reconstructor::Reconstructor(){
		srand(time(NULL));
	}


	Reconstructor::SortedKeypointMatches Reconstructor::extractMatches(const QString& filename1,const QString& filename2)const{
		QImage img1(filename1),img2(filename2);
		QList <SIFT::Keypoint> keypoints1,keypoints2;
		{
			SIFT::Extractor extractor;
			keypoints1= extractor.extract(img1);
			keypoints2= extractor.extract(img2);
		}

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
				QList<KeypointMatch>& dst= matches[minDistance];
				if(!dst.contains(match))
					dst.append(match);
			}
		}
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
				QList<KeypointMatch>& dst= matches[minDistance];
				if(!dst.contains(match))
					dst.append(match);
			}
		}
		return matches;
	}

	bool Reconstructor::computePose(SortedKeypointMatches& matches){
		//Use the RANSAC algorithm to estimate camera poses

		KeypointMatches best_consensus_set;
		for(int i=0;i<Options::RANSAC_k;i++){
			//Extract RANSAC_n random values
			KeypointMatches consensus_set;
			while(consensus_set.size()<Options::RANSAC_n){
				//Generate random distance
				float rand_number= (float)rand()/(float)RAND_MAX;//[0..1]
				float rand_min= matches.begin().key();
				float rand_max= std::min((matches.end()-1).key(),Options::ReliableDistance);
				float random_distance= rand_min+(rand_max-rand_min)*rand_number;
				//Find points at generated distance
				SortedKeypointMatches::const_iterator it1= matches.lowerBound(random_distance);
				if(it1==matches.end())
					continue;
				//Find a match on a random position
				int random_pos= (rand()*it1.value().size())/RAND_MAX;
				random_pos= std::min(it1.value().size(),random_pos);
				consensus_set.push_back(QPair<float,KeypointMatch>(
						it1.key(),
						it1.value().at(random_pos)
					)
				);
			}
		}
		return true;
	}

	bool Reconstructor::run(const QString& filename1,const QString& filename2){
		//Extract and sort matches by distance
		SortedKeypointMatches matches= extractMatches(filename1,filename2);
		if(matches.isEmpty()){
			errorString= QT_TR_NOOP("No matches found");
			return false;
		}
		if(matches.size()<Options::MinMatches){
			errorString= QString(QT_TR_NOOP("Not enough matches (%1), need at least %2")).arg(matches.size()).arg(Options::MinMatches);
			return false;
		}
		return true;
	}

}
