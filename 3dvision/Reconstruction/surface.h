#ifndef SURFACE_H
#define SURFACE_H

#include <QVector3D>

namespace cybervision{
	class Surface{
		friend class Sculptor;
	protected:
		//Internal classes
		class Triangle{
		public:
			QVector3D a,b,c;
			QVector3D normal;
		};

		QList<Triangle> triangles;//Surface data
	public:
		Surface();
		Surface(const Surface&);
		void operator =(const Surface&);
		//Draws the surface with OpenGL, should be called from paintGL
		void glDraw()const;
	signals:

	public slots:

	};
}

#endif // SURFACE_H
