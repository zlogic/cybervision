#ifndef KTTREEGATEWAY_H
#define KTTREEGATEWAY_H

#include <QList>
#include <QMultiMap>
#include <QPointF>
#include <limits>
#include <Reconstruction/pointmatch.h>

namespace SIFT{

class Keypoint;

}

namespace KDTree{

class KDTreeGateway
{
protected:
	double maxKeypointDistance;
	size_t bbf_steps;
public:
	KDTreeGateway(double MaxKeypointDistance=std::numeric_limits<float>::infinity(),size_t bbf_steps=400);
	//Uses K-D tree to find matches from stationary_keypoints for every keypoint in matched_keypoints
	cybervision::SortedKeypointMatches matchKeypoints(const QList<SIFT::Keypoint>& matched_keypoints,const QList<SIFT::Keypoint>& stationary_keypoints);
};

}

#endif // KTTREEGATEWAY_H
