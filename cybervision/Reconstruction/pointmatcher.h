#ifndef POINTMATCHER_H
#define POINTMATCHER_H

#include <QObject>
#include <QSize>
#include <Reconstruction/pointmatch.h>

namespace cybervision{
	/*
	 * Class for detecting and matching of SIFT points. Called by Reconstructor.
	 */
	class PointMatcher : public QObject	{
		Q_OBJECT
	protected:
		//Results
		SortedKeypointMatches matches;
		QSize imageSize;
	public:
		explicit PointMatcher(QObject *parent = 0);

		//Finds and extracts all valid keypoint matches on two images
		bool extractMatches(const QString& filename1,const QString& filename2);

		//Getters
		SortedKeypointMatches getMatches()const;
		QSize getSize()const;
	signals:
		void sgnLogMessage(QString);
		void sgnStatusMessage(QString);
	};
}

#endif // POINTMATCHER_H
