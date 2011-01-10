#ifndef RECONSTRUCTOR_H
#define RECONSTRUCTOR_H

#include <QPoint>
#include <QString>
#include <QMap>
#include <QList>
#include <QPair>
#include <QGenericMatrix>
#include <QVector3D>
#include <QSize>

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
		typedef QMultiMap<float,KeypointMatch> SortedKeypointMatches;
		//Class for storing stereo pair's R and T matrix pairs
		struct StereopairPosition{
			QGenericMatrix<3,3,double> R;
			QGenericMatrix<1,3,double> T;
		};

		//State
		QString errorString;
		QList<QVector3D> Points3D;
		QSize imageSize;

		//Internal procedures

		//Finds and extracts all valid keypoint matches on two images
		SortedKeypointMatches extractMatches(const QString& filename1,const QString& filename2);
		//Converts QPointF to QGenericMatrix
		inline QGenericMatrix<1,3,double> point2vector(const QPointF&)const;
		//Converts QGenericMatrix (vector) to QPointF
		inline QPointF vector2point(const QGenericMatrix<1,3,double>&)const;
		//Normalizes values and computes fundamental matrix
		QGenericMatrix<3,3,double> computeFundamentalMatrix(SortedKeypointMatches& matches);
		//Computes the fundamental matrix with RANSAC; point coordinates should be normalized
		QGenericMatrix<3,3,double> ransacComputeFundamentalMatrix(SortedKeypointMatches& matches,QGenericMatrix<3,3,double> T1,QGenericMatrix<3,3,double> T2);
		//Estimates the best pose (R and T matrices) and essential matrix with RANSAC; filters the point list by removing outliers
		QList<Reconstructor::StereopairPosition> computePose(SortedKeypointMatches&);

		//Computes the fundamental matrix from N points (currently 8-point method is used)
		QGenericMatrix<3,3,double> computeFundamentalMatrix8Point(const KeypointMatches&);
		//Computes a keypoint match's error when used with the fundamental matrix F
		double computeFundamentalMatrixError(const QGenericMatrix<3,3,double>&F, const KeypointMatch&) const;

		//TODO: create derived classes

		//Essential matrix method functions
		//Returns the camera calibration matrix
		QGenericMatrix<3,3,double> computeCameraMatrix()const;
		//Computes possible R and T matrices from essential matrix
		QList<StereopairPosition> computeRT(const QGenericMatrix<3,3,double>&Essential_matrix) const;
		//Helper function to construct Rz matrix for computeRT
		QGenericMatrix<3,3,double> computeRT_rzfunc(double angle)const;

		//Triangulates a point in 3D space
		QList<QVector3D> compute3DPoints(const SortedKeypointMatches&matches,const QList<StereopairPosition>& RTList);
		QList<QVector3D> computeTriangulatedPoints(const SortedKeypointMatches&matches,const QGenericMatrix<3,3,double>&R,const QGenericMatrix<1,3,double>& T,bool normalizeCameras);


		//Fundamental matrix method functions
		//Optimal triangulation method
		QList<QVector3D> compute3DPoints(const SortedKeypointMatches&matches,QGenericMatrix<3,3,double>F);
	public:
		Reconstructor();

		bool run(const QString& filename1,const QString& filename2);

		//Getters
		bool isOk()const;
		QString getErrorString()const;
		QList<QVector3D> get3DPoints()const;
	signals:
		void sgnLogMessage(QString);
		void sgnStatusMessage(QString);
	};

}
#endif // RECONSTRUCTOR_H
