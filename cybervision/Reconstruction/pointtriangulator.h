#ifndef POINTTRIANGULATOR_H
#define POINTTRIANGULATOR_H

#include <QVector3D>
#include <QObject>
#include <QList>
#include <Eigen/Dense>
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
		Eigen::Matrix3d camera_K;
		QList<QVector3D> Points3D;
		TriangulationResult result;

		//Some internal types

		//Class for storing stereo pair's R and T matrix pairs
		struct StereopairPosition{
			Eigen::Matrix3d R;
			Eigen::Vector3d T;
		};

		//Internal procedures

		//Estimates the best pose (R and T matrices) and essential matrix with RANSAC; filters the point list by removing outliers
		QList<PointTriangulator::StereopairPosition> computePose(const Eigen::Matrix3d& F);

		//Returns the camera calibration matrix
		Eigen::Matrix3d computeCameraMatrix(const QSize&)const;
		//Computes possible R and T matrices from essential matrix
		QList<StereopairPosition> computeRT(Eigen::Matrix3d Essential_matrix) const;
		//Helper function to construct Rz matrix for computeRT
		Eigen::Matrix3d computeRT_rzfunc(double angle)const;
		//Computes Kronecker product
		Eigen::MatrixXd kronecker(const Eigen::MatrixXd A,const Eigen::MatrixXd& B)const;
		//Performs a least-squares solving of A*x=B equation system
		Eigen::MatrixXd leastSquares(const Eigen::MatrixXd A,const Eigen::MatrixXd& B)const;

		//Triangulates a point in 3D space
		QList<QVector3D> compute3DPoints(const SortedKeypointMatches&matches,const QList<StereopairPosition>& RTList);
		QList<QVector3D> computeTriangulatedPoints(const SortedKeypointMatches&matches,const Eigen::Matrix3d&R,const Eigen::Vector3d& T,bool normalizeCameras);

	public:
		explicit PointTriangulator(QObject *parent = 0);

		//Performs a complete triangulation with pose estimation (for perspective projection)
		bool triangulatePoints(const SortedKeypointMatches&matches,const Eigen::Matrix3d& F,const QSize& imageSize);
		//Performs a triangulation without pose estimation (for parallel projection)
		bool triangulatePoints(const SortedKeypointMatches&matches,qreal angle);

		//Getters
		QList<QVector3D> getPoints3D()const;
		TriangulationResult getResult()const;
	signals:
		void sgnLogMessage(QString);
		void sgnStatusMessage(QString);
	};
}

#endif // POINTTRIANGULATOR_H
