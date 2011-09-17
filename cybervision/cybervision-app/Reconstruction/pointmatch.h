#ifndef POINTMATCH_H
#define POINTMATCH_H

#include <QPointF>
#include <QList>
#include <QPair>
#include <QMultiMap>

namespace cybervision{
	//Class for storing a single patch between two points on different images
	struct KeypointMatch{
		QPointF a;//Coordinates on the first image
		QPointF b;//Coordinates on the second image
		bool operator==(const KeypointMatch&)const;
	};
	//Stores all acceptable keypoint matches
	typedef QList<QPair<float,KeypointMatch> > KeypointMatches;
	typedef QMultiMap<float,KeypointMatch> SortedKeypointMatches;

}

#endif // POINTMATCH_H
