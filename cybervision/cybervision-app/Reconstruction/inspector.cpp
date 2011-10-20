#include "inspector.h"

#include <QMultiMap>
#include <QPainter>

#include <limits>

cybervision::Inspector::Inspector(const cybervision::Surface&surface, QObject *parent):QObject(parent),surface(surface) {}

QList<QPointF> cybervision::Inspector::getCrossSection(const QVector3D &start, const QVector3D &end) const{
	QMultiMap<qreal,qreal> intersections;
	QLineF intersectionLine(start.x(),start.y(),end.x(),end.y());

	for(QList<Surface::Triangle>::const_iterator it= surface.triangles.begin();it!=surface.triangles.end();it++){
		QLineF lineAB(surface.points[it->a].coord.x(),surface.points[it->a].coord.y(),surface.points[it->b].coord.x(),surface.points[it->b].coord.y());
		QLineF lineBC(surface.points[it->b].coord.x(),surface.points[it->b].coord.y(),surface.points[it->c].coord.x(),surface.points[it->c].coord.y());
		QLineF lineCA(surface.points[it->c].coord.x(),surface.points[it->c].coord.y(),surface.points[it->a].coord.x(),surface.points[it->a].coord.y());
		QPointF intersectionPoint;
		if(lineAB.intersect(intersectionLine,&intersectionPoint)==QLineF::BoundedIntersection){
			qreal tLine= qAbs(intersectionLine.dx())>qAbs(intersectionLine.dy())?
						(intersectionPoint.x()-intersectionLine.x1())/(intersectionLine.x2()-intersectionLine.x1()):
						(intersectionPoint.y()-intersectionLine.y1())/(intersectionLine.y2()-intersectionLine.y1());
			qreal tPolygon= qAbs(lineAB.dx())>qAbs(lineAB.dy())?
						(intersectionPoint.x()-lineAB.x1())/(lineAB.x2()-(lineAB.x1())):
						(intersectionPoint.y()-lineAB.y1())/(lineAB.y2()-(lineAB.y1()));
			qreal z= surface.points[it->a].coord.z()*tPolygon+surface.points[it->b].coord.z()*(1-tPolygon);
			intersections.insert(tLine,z);
		}
		if(lineBC.intersect(intersectionLine,&intersectionPoint)==QLineF::BoundedIntersection){
			qreal tLine= qAbs(intersectionLine.dx())>qAbs(intersectionLine.dy())?
						(intersectionPoint.x()-intersectionLine.x1())/(intersectionLine.x2()-intersectionLine.x1()):
						(intersectionPoint.y()-intersectionLine.y1())/(intersectionLine.y2()-intersectionLine.y1());
			qreal tPolygon= qAbs(lineBC.dx())>qAbs(lineBC.dy())?
						(intersectionPoint.x()-lineBC.x1())/(lineBC.x2()-(lineBC.x1())):
						(intersectionPoint.y()-lineBC.y1())/(lineBC.y2()-(lineBC.y1()));
			qreal z= surface.points[it->b].coord.z()*tPolygon+surface.points[it->c].coord.z()*(1-tPolygon);
			intersections.insert(tLine,z);
		}
		if(lineCA.intersect(intersectionLine,&intersectionPoint)==QLineF::BoundedIntersection){
			qreal tLine= qAbs(intersectionLine.dx())>qAbs(intersectionLine.dy())?
						(intersectionPoint.x()-intersectionLine.x1())/(intersectionLine.x2()-intersectionLine.x1()):
						(intersectionPoint.y()-intersectionLine.y1())/(intersectionLine.y2()-intersectionLine.y1());
			qreal tPolygon= qAbs(lineCA.dx())>qAbs(lineCA.dy())?
						(intersectionPoint.x()-lineCA.x1())/(lineCA.x2()-(lineCA.x1())):
						(intersectionPoint.y()-lineCA.y1())/(lineCA.y2()-(lineCA.y1()));
			qreal z= surface.points[it->c].coord.z()*tPolygon+surface.points[it->a].coord.z()*(1-tPolygon);
			intersections.insert(tLine,z);
		}
	}
	qreal sum=0;
	int count=0;
	qreal lineLength= intersectionLine.length();

	QList<QPointF> result;
	for(QMap<qreal,qreal>::const_iterator it=intersections.begin();it!=intersections.end();it++){
		sum+= it.value();
		count++;
		if((it+1)==intersections.end() || !qFuzzyCompare(it.key(),(it+1).key())){
			qreal z= sum/(qreal)count;
			QPointF point(it.key()*lineLength,z);
			result.push_back(point);
			sum= 0;
			count= 0;
		}
	}
	return result;
}

QImage cybervision::Inspector::renderCrossSection(const QList<QPointF>& crossSection,const QSize& imageSize) const{
	//Prepare data
	qreal minX= std::numeric_limits<qreal>::infinity(),
			minY= std::numeric_limits<qreal>::infinity(),
			maxX= -std::numeric_limits<qreal>::infinity(),
			maxY= -std::numeric_limits<qreal>::infinity();
	for(QList<QPointF>::const_iterator it=crossSection.begin();it!=crossSection.end();it++){
		minX= qMin(minX,it->x());
		minY= qMin(minY,it->y());
		maxX= qMax(maxX,it->x());
		maxY= qMax(maxY,it->y());
	}
	//Preapare image
	QImage img(imageSize.width(),imageSize.height(),QImage::Format_ARGB32);
	img.fill(0x00000000);

	//Draw lines
	QPainter painter(&img);
	painter.setPen(QColor(0,255,0));
	for(QList<QPointF>::const_iterator it=crossSection.begin();it!=crossSection.end();it++){
		if((it+1)==crossSection.end())
			continue;
		QPoint point1(img.width()*(    it->x()-minX)/(maxX-minX),img.height()*(maxY-    it->y())/(maxY-minY));
		QPoint point2(img.width()*((it+1)->x()-minX)/(maxX-minX),img.height()*(maxY-(it+1)->y())/(maxY-minY));

		point1.setX(qMax(point1.x(),0));
		point1.setY(qMax(point1.y(),0));
		point1.setX(qMin(point1.x(),img.width()-1));
		point1.setY(qMin(point1.y(),img.height()-1));
		point2.setX(qMax(point2.x(),0));
		point2.setY(qMax(point2.y(),0));
		point2.setX(qMin(point2.x(),img.width()-1));
		point2.setY(qMin(point2.y(),img.height()-1));

		painter.drawLine(point1,point2);
	}

	//Display the image
	painter.end();
	return img;
}
