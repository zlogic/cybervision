#ifndef SCULPTOR_H
#define SCULPTOR_H

#include <QList>
#include <QVector3D>
#include <QVector2D>
#include <QPolygonF>
#include <Reconstruction/surface.h>

namespace cybervision{

/*
 * Class for interpolating a set of 3D points into a 3D surface. Also extracts additional info from the 3D point set.
 */
class Sculptor{
protected:
	//Class for storing data of one cell (set of points + average depth + min/max coordinate)
	class CellData{
	public:
		QList<QVector3D> points;
		qreal averageDepth;
		QVector2D min, max;

		CellData();
		CellData(const QList<QVector3D>& points,qreal parentAverageDepth,const QVector2D& min,const QVector2D& max);
		CellData(const CellData&);

		void operator=(const CellData&);
	};


	cybervision::Surface surface;
	QSize imageSize;
	qreal scaleXY,scaleZ;

	//Average points with the same (x,y) values, change scale to better fit result
	QList<QVector3D> filterPoints(const QList<QVector3D>& points);
	//Remove inside triangles by averaging after detecting them
	bool filterTriangles(QList<QVector3D>& points,const QList<Surface::Triangle>& triangles);
	//Increase point density by mapping the points to a grid
	QList<QVector3D> interpolatePointsToGrid(const QList<QVector3D>& points)const;

	//Returns a triangle with computed normal and clockwise vertex placement
	Surface::Triangle createTriangle(const QList<QVector3D>& points,const Surface::PolygonPoint& a, const Surface::PolygonPoint& b, const Surface::PolygonPoint& c)const;
	//Returns a normal for a 3D triangle
	QVector3D calcNormal(const QVector3D& a, const QVector3D& b)const;
	//Returns a normal for a 3D point based on all edges that contain it
	QVector3D calcNormal(const QList<Surface::Triangle>& triangles,const QList<QVector3D>& points,const Surface::PolygonPoint& point)const;
	//Returns a normal for a vector (normal and source vector projections onto XY plane will be collinear)
	QVector3D calcNormal(const QVector3D& vector)const;

	//Code ported from http://local.wasp.uwa.edu.au/~pbourke/papers/triangulate/index.html
	bool delaunayCircumCircle(const QPointF& p,const QPointF& a,const QPointF& b,const QPointF& c,QVector3D* circle=NULL)const;
	//Interpolates points to create surface (using Delaunay triangulation)
	void delaunayTriangulate(const QList<QVector3D>& points);
public:
	Sculptor(const QList<QVector3D>& points=QList<QVector3D>(),QSize imageSize=QSize(),double scaleXY=0.0,double scaleZ=0.0);

	cybervision::Surface getSurface()const;
};

}

#endif // SCULPTOR_H
