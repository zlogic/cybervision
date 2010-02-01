#ifndef SCULPTOR_H
#define SCULPTOR_H

#include <QList>
#include <QVector3D>

namespace cybervision{

	//Class for interpolating a set of 3D points into a 3D surface. Also extracts additional info from the 3D point set.
	class Sculptor{
	protected:
		//Internal classes
		struct Triangle{
			QVector3D a,b,c;
			QVector3D normal;
		};

		QList<QVector3D> points;


		QList<Triangle> triangles;//Surface data

		//Interpolates points to create surface
		void createSurface(const QList<QVector3D>& points);
		//Returns points belonging to a specific coordinate range
		QList<QVector3D> getPointsInRange(const QList<QVector3D>& points,QVector3D min, QVector3D max, bool ignoreZ=true)const;

		QVector3D calcNormal(const QVector3D& a, const QVector3D& b)const;
	public:
		Sculptor(const QList<QVector3D>& points=QList<QVector3D>());
		//Draws the surface with OpenGL, should be called from paintGL
		void glDraw()const;
	};
}

#endif // SCULPTOR_H
