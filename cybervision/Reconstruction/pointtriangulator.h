#ifndef POINTTRIANGULATOR_H
#define POINTTRIANGULATOR_H

#include <QVector3D>
#include <QObject>
#include <QList>
#include <QGenericMatrix>
#include <Reconstruction/pointmatch.h>

namespace cybervision{
	/*
	 * Class for performing pose estimation and triangulation. Called by Reconstructor.
	 */
	class PointTriangulator : public QObject {
		Q_OBJECT
	public:
		//Operation result
		enum TriangulationResult{RESULT_OK=0,RESULT_POSE_UNDETERMINED,RESULT_TRIANGULATION_ERROR};
	protected:
		//Result
		QGenericMatrix<3,3,double> camera_K;
		QList<QVector3D> Points3D;
		TriangulationResult result;

		//Some internal types

		//Class for storing stereo pair's R and T matrix pairs
		struct StereopairPosition{
			QGenericMatrix<3,3,double> R;
			QGenericMatrix<1,3,double> T;
		};

		//Internal procedures

		//Estimates the best pose (R and T matrices) and essential matrix with RANSAC; filters the point list by removing outliers
		QList<PointTriangulator::StereopairPosition> computePose(const QGenericMatrix<3,3,double>& F);

		//Returns the camera calibration matrix
		QGenericMatrix<3,3,double> computeCameraMatrix(const QSize&)const;
		//Computes possible R and T matrices from essential matrix
		QList<StereopairPosition> computeRT(QGenericMatrix<3,3,double> Essential_matrix) const;
		//Helper function to construct Rz matrix for computeRT
		QGenericMatrix<3,3,double> computeRT_rzfunc(double angle)const;

		//Triangulates a point in 3D space
		QList<QVector3D> compute3DPoints(const SortedKeypointMatches&matches,const QList<StereopairPosition>& RTList);
		QList<QVector3D> computeTriangulatedPoints(const SortedKeypointMatches&matches,const QGenericMatrix<3,3,double>&R,const QGenericMatrix<1,3,double>& T,bool normalizeCameras);

	public:
		explicit PointTriangulator(QObject *parent = 0);

		//Performs a complete triangulation with pose estimation (for perspective projection)
		bool triangulatePoints(const SortedKeypointMatches&matches,const QGenericMatrix<3,3,double>& F,const QSize& imageSize);
		//Performs a simplified disparity-based triangulation (for parallel projection)
		bool triangulatePoints(const SortedKeypointMatches&matches);

		//Getters
		QList<QVector3D> getPoints3D()const;
		TriangulationResult getResult()const;
	signals:
		void sgnLogMessage(QString);
		void sgnStatusMessage(QString);
	};
}

#endif // POINTTRIANGULATOR_H
