#ifndef FUNDAMENTALMATRIX_H
#define FUNDAMENTALMATRIX_H

#include <QObject>
#include <QFileInfo>
#include <Reconstruction/pointmatch.h>

#include <Reconstruction/config.h>
#include <Eigen/Dense>

namespace cybervision{
/*
 * Class for computing the Fundamental matrix with RANSAC. Called by Reconstructor.
 */
class FundamentalMatrix : public QObject {
	Q_OBJECT
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
protected:
	//Results
	Eigen::Matrix3d F,T1,T2;
	SortedKeypointMatches matches;

	//Internal procedures
	//Converts QPointF to Eigen::Vector3d
	inline ALIGN_EIGEN_FUNCTION Eigen::Vector3d point2vector(const QPointF&)const;
	//Converts Eigen::Vector3d to QPointF
	inline QPointF vector2point(const Eigen::Vector3d&)const;

	//Computes the fundamental matrix from N points
	ALIGN_EIGEN_FUNCTION Eigen::Matrix3d computeFundamentalMatrix(const KeypointMatches&);
	//Computes a keypoint match's error when used with the fundamental matrix F
	ALIGN_EIGEN_FUNCTION double computeFundamentalMatrixError(const Eigen::Matrix3d&F, const KeypointMatch&) const;

	//Computes the fundamental matrix with RANSAC, removing any outliers
	ALIGN_EIGEN_FUNCTION Eigen::Matrix3d computeFundamentalMatrix();
public:
	explicit FundamentalMatrix(QObject *parent = 0);

	//Computes the fundamental matrix with RANSAC, removing any outliers. Uses normalization to reduce errors.
	ALIGN_EIGEN_FUNCTION bool computeFundamentalMatrix(const SortedKeypointMatches& matches);

	//Getters
	SortedKeypointMatches getAcceptedMatches()const;
	Eigen::Matrix3d getFundamentalMatrix()const;
	Eigen::Matrix3d getT1()const;
	Eigen::Matrix3d getT2()const;

	//Output functions
	void saveAcceptedMatches(const QFileInfo &target);
signals:
	void sgnLogMessage(QString);
	void sgnStatusMessage(QString);
};
}

#endif // FUNDAMENTALMATRIX_H
