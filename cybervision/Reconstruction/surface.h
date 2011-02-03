#ifndef SURFACE_H
#define SURFACE_H

#include <QVector3D>
#include <QFileInfo>

namespace cybervision{
	class Surface{
		friend class Sculptor;
	protected:
		//Internal classes
		typedef int PolygonPoint;//Index of point in array. NOT checked!
		class Triangle{
		public:
			PolygonPoint a,b,c;
			QVector3D normal;
		};

		//Surface data in two formats
		QList<Triangle> triangles;
		QList<QVector3D> points;

		//Scale for viewport
		qreal scale;
	public:
		Surface();
		Surface(const Surface&);
		void operator =(const Surface&);
		//Draws the surface with OpenGL, should be called from paintGL
		void glDraw()const;

		//Returns true if surface contains valid data instead of an empty set
		bool isOk() const;

		//Functions for saving image
		void savePoints(QString fileName)const;
		void savePolygons(QString fileName)const;
		void saveCollada(QString fileName)const;
	signals:

	public slots:

	};
}

#endif // SURFACE_H
