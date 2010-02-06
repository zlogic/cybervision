#ifndef SCULPTOR_H
#define SCULPTOR_H

#include <QList>
#include <QVector3D>
#include <QPolygonF>
#include <Reconstruction/surface.h>

namespace cybervision{
	//Class for interpolating a set of 3D points into a 3D surface. Also extracts additional info from the 3D point set.
	class Sculptor{
	protected:

		cybervision::Surface surface;

		QList<QVector3D> filterPoints(const QList<QVector3D>& points);//average points with the same (x,y) values, change scale to better fit result

		//Returns a triangle with computed normal and clockwise vertex placement
		Surface::Triangle createTriangle(const QVector3D& a, const QVector3D& b, const QVector3D& c)const;
		//Returns a normal for a 3D triangle
		QVector3D calcNormal(const QVector3D& a, const QVector3D& b)const;

		//Code ported from http://local.wasp.uwa.edu.au/~pbourke/papers/triangulate/index.html
		bool delaunayCircumCircle(const QPointF& p,const QPointF& a,const QPointF& b,const QPointF& c,QVector3D* circle=NULL)const;
		//Interpolates points to create surface (using Delaunay triangulation)
		void delaunayTriangulate(const QList<QVector3D>& points);
	public:
		Sculptor(const QList<QVector3D>& points=QList<QVector3D>());

		cybervision::Surface getSurface()const;
	};
}

#endif // SCULPTOR_H
