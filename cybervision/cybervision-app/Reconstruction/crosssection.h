#ifndef CROSSSECTION_H
#define CROSSSECTION_H

#include <QObject>
#include <QImage>
#include <QVector3D>

namespace cybervision{

class Surface;

/*
 * Class for computing surface cross-section and roughness statistics
 */
class CrossSection : public QObject
{
	Q_OBJECT
protected:
	bool ok;
	//Computed cross-section data
	QList<QPointF> crossSection;//profile
	qreal mA,mB,mL;//average line (y=bx+a)
	qreal Ra,Rz,Rmax;//Height parameters
	qreal S,Sm,tp;//Step parameters

public:
	CrossSection(QObject *parent = 0);
	void operator=(const CrossSection& crossSection);

	//Compute cross-section with line
	void computeCrossSection(const cybervision::Surface&,const QVector3D& start,const QVector3D& end);

	//Computation code for roughness analysis
	void computeParams(int p);

	//Create the cross-section image
	QImage renderCrossSection(const QSize& imageSize)const;

	//Returns the cross-section profile
	QList<QPointF> getCrossSection()const;

	//Returns if cross-section contains valid information
	bool isOk()const;

	//Getters for cross-section roughness parameters
	qreal getRoughnessRa()const;
	qreal getRoughnessRz()const;
	qreal getRoughnessRmax()const;
	qreal getRoughnessS()const;
	qreal getRoughnessSm()const;
	qreal getRoughnessTp()const;
};
}
#endif // CROSSSECTION_H
