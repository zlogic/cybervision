#ifndef FUNDAMENTALMATRIX_H
#define FUNDAMENTALMATRIX_H

#include <QObject>
#include <Reconstruction/pointmatch.h>
#include <QGenericMatrix>
#include <QFileInfo>

namespace cybervision{
	/*
	 * Class for computing the Fundamental matrix with RANSAC. Called by Reconstructor.
	 */
	class FundamentalMatrix : public QObject {
		Q_OBJECT
	protected:
		//Results
		QGenericMatrix<3,3,double> F,T1,T2;
		SortedKeypointMatches matches;

		//Internal procedures
		//Converts QPointF to QGenericMatrix
		inline QGenericMatrix<1,3,double> point2vector(const QPointF&)const;
		//Converts QGenericMatrix (vector) to QPointF
		inline QPointF vector2point(const QGenericMatrix<1,3,double>&)const;

		//Computes the fundamental matrix from N points
		QGenericMatrix<3,3,double> computeFundamentalMatrix(const KeypointMatches&);
		//Computes a keypoint match's error when used with the fundamental matrix F
		double computeFundamentalMatrixError(const QGenericMatrix<3,3,double>&F, const KeypointMatch&) const;

		//Computes the fundamental matrix with RANSAC, removing any outliers
		QGenericMatrix<3,3,double> computeFundamentalMatrix();
	public:
		explicit FundamentalMatrix(QObject *parent = 0);

		//Computes the fundamental matrix with RANSAC, removing any outliers. Uses normalization to reduce errors.
		bool computeFundamentalMatrix(const SortedKeypointMatches& matches);

		//Getters
		SortedKeypointMatches getAcceptedMatches()const;
		QGenericMatrix<3,3,double> getFundamentalMatrix()const;
		QGenericMatrix<3,3,double> getT1()const;
		QGenericMatrix<3,3,double> getT2()const;

		//Output functions
		void saveAcceptedMatches(const QFileInfo &target);
	signals:
		void sgnLogMessage(QString);
		void sgnStatusMessage(QString);
	};
}

#endif // FUNDAMENTALMATRIX_H
