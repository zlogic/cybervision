#ifndef POINTMATCHER_H
#define POINTMATCHER_H

#include <QObject>
#include <QSize>
#include <QImage>
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
	QImage image1,image2;
	double scaleMetadata;
public:
	explicit PointMatcher(QObject *parent = 0);

	//Finds and extracts all valid keypoint matches on two images
	bool extractMatches(const QString& filename1,const QString& filename2);

	//Getters
	SortedKeypointMatches getMatches()const;
	QSize getSize()const;
	const QImage& getImage1()const;
	const QImage& getImage2()const;
	double getScaleMetadata()const;
signals:
	void sgnLogMessage(QString);
	void sgnStatusMessage(QString);
};

}

#endif // POINTMATCHER_H
