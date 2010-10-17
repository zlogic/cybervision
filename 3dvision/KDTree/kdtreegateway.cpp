#include "kdtreegateway.h"
#include "kdtree.hpp"
#include <Reconstruction/options.h>
#include <SIFT/siftgateway.h>

namespace KDTree{

	//Used for comparing two SIFT keypoints by libkdtree++
	inline double sift_coordinate(const SIFT::Keypoint& t, size_t k) { return t[k]; }

	typedef KDTree<128,SIFT::Keypoint,std::pointer_to_binary_function<const SIFT::Keypoint&,size_t,double> > KDTreeType;
	KDTreeGateway::KDTreeGateway(double maxKeypointDistance,size_t bbf_steps){
		this->maxKeypointDistance= maxKeypointDistance;
		this->bbf_steps= bbf_steps;
	}

	cybervision::SortedKeypointMatches KDTreeGateway::matchKeypoints(const QList<SIFT::Keypoint>& matched_keypoints,const QList<SIFT::Keypoint>& stationary_keypoints){
		cybervision::SortedKeypointMatches matches;

		//Create K-D tree for stationary_keypoints
		KDTreeType tree(std::ptr_fun(sift_coordinate));
		for(QList<SIFT::Keypoint>::const_iterator it=stationary_keypoints.begin();it!=stationary_keypoints.end();it++){
			tree.insert(*it);
		}
		tree.optimize();

		for(QList<SIFT::Keypoint>::const_iterator it=matched_keypoints.begin();it!=matched_keypoints.end();it++){
			std::pair<KDTreeType::iterator,double> found_match= tree.find_nearest(*it,maxKeypointDistance,bbf_steps);
			if(found_match.first!=tree.end()){
				cybervision::KeypointMatch match;
				match.a= QPointF(found_match.first->getX(),found_match.first->getY());
				match.b= QPointF(it->getX(),it->getY());
				matches.insert(found_match.second,match);
			}
		}
		return matches;
	}

}
