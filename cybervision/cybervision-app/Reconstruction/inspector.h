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
public:
	Inspector(const cybervision::Surface&,QObject *parent = 0);

	//Compute cross-section with line
	QList<QPointF> getCrossSection(const QVector3D& start,const QVector3D& end)const;

	//Create the cross-section image
	QImage renderCrossSection(const QList<QPointF>& crossSection,const QSize& imageSize)const;
};
}
#endif // INSPECTOR_H
