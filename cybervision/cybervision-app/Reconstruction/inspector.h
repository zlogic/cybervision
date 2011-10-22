#ifndef INSPECTOR_H
#define INSPECTOR_H

#include <QObject>
#include <QImage>
#include <Reconstruction/surface.h>

namespace cybervision{
class Inspector : public QObject
{
	Q_OBJECT
protected:
	const cybervision::Surface& surface;

	//Computed cross-section data
	QList<QPointF> crossSection;//profile
	qreal mA,mB,mL;//average line (y=bx+a)
	qreal Ra,Rz,Rmax;//Height params

	//Computation code for roughness analysis
	void computeParams();
public:
	Inspector(const cybervision::Surface&,QObject *parent = 0);

	//Compute cross-section with line
	void updateCrossSection(const QVector3D& start,const QVector3D& end);

	//Create the cross-section image
	QImage renderCrossSection(const QSize& imageSize)const;

	//Returns the cross-section profile
	QList<QPointF> getCrossSection()const;

	//Getters for cross-section roughness parameters
	qreal getRoughnessRa()const;
	qreal getRoughnessRz()const;
	qreal getRoughnessRmax()const;
};
}
#endif // INSPECTOR_H
