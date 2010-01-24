#ifndef RECONSTRUCTOR_H
#define RECONSTRUCTOR_H

#include <QPoint>
#include <QString>
#include <QMap>
#include <QList>
#include <QPair>
#include <QGenericMatrix>

#include <Reconstruction/options.h>

namespace cybervision{

	class Reconstructor:public QObject{
		Q_OBJECT
	protected:
		//Some internal types
		//Class for storing a single patch between two points on different images
		struct KeypointMatch{
			QPointF a;//Coordinates on the first image
			QPointF b;//Coordinates on the second image
			bool operator==(const KeypointMatch&)const;
		};
		//Stores all acceptable keypoint matches
		typedef QList<QPair<float,KeypointMatch> > KeypointMatches;
		typedef QMultiMap<float,KeypointMatch > SortedKeypointMatches;

		//State
		QString errorString;

		//Internal procedures
		//Finds and extracts all valid keypoint matches on two images
		SortedKeypointMatches extractMatches(const QString& filename1,const QString& filename2);
		//Estimates the best pose (R and T matrices) and essential matrix with RANSAC; filters the point list by removing outliers
		bool computePose(SortedKeypointMatches&);

		//Computes the essential matrix from N points
		QGenericMatrix<3,3,double> computeEssentialMatrix(const KeypointMatches&);
		//Computes a keypoint match's error when used with the essential matrix E
		float computeEssentialMatrixError(const QGenericMatrix<3,3,double>&E, const KeypointMatch&) const;
	public:
		Reconstructor();
		bool run(const QString& filename1,const QString& filename2);

		//Getters
		bool isOk()const;
		QString getError()const;


	signals:
		void sgnLogMessage(QString);
		void sgnStatusMessage(QString);
	};

}
#endif // RECONSTRUCTOR_H
