#ifndef RECONSTRUCTOR_H
#define RECONSTRUCTOR_H

#include <QPoint>
#include <QString>
#include <QMap>
#include <QList>
#include <QPair>
#include <QVector3D>
#include <QSize>

#include <Eigen/Dense>

#include <Reconstruction/options.h>
#include <Reconstruction/pointmatch.h>


namespace cybervision{
	/*
	 * Class for performing 3D reconstruction.
	 * Calls functions of PointMatcher, FundamentalMatrix and PointTriangulation and controls execution of these functions.
	 */
	class Reconstructor:public QObject{
		Q_OBJECT
	protected:
		//State
		QString errorString;
		QList<QVector3D> Points3D;
		QSize imageSize;

	public:
		explicit Reconstructor(QObject *parent);

		bool run(const QString& filename1,const QString& filename2);

		//Getters
		bool isOk()const;
		QString getErrorString()const;
		QList<QVector3D> get3DPoints()const;
		QSize getImageSize()const;
	signals:
		void sgnLogMessage(QString);
		void sgnStatusMessage(QString);
	};

}
#endif // RECONSTRUCTOR_H
