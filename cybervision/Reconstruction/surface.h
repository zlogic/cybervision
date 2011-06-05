#ifndef SURFACE_H
#define SURFACE_H

#include <QVector3D>
#include <QFileInfo>

namespace cybervision{
	class Surface{
		friend class Sculptor;
	protected:
		//Internal classes
		typedef int PolygonPoint;//Index of point in array. NOT checked! TODO: remove int limits
		class Triangle{
		public:
			PolygonPoint a,b,c;
			QVector3D normal;
		};

		class Point{
		public :
			QVector3D coord, normal;
		};

		//Surface data in two formats
		QList<Triangle> triangles;
		QList<Point> points;

		//Scale for viewport
		qreal scale;

		//Depth statistics
		qreal medianDepth,minDepth,maxDepth;

		//Image size
		QRectF imageSize;
	public:
		Surface();
		Surface(const Surface&);
		void operator =(const Surface&);
		//Draws the surface with OpenGL, should be called from paintGL
		void glDraw()const;

		//Returns true if surface contains valid data instead of an empty set
		bool isOk() const;

		//Returns point depth statistics (min/max/median)
		qreal getMedianDepth()const;
		qreal getMinDepth()const;
		qreal getMaxDepth()const;

		//Returns source image size
		QRectF getImageSize()const;

		//Returns scale
		qreal getScale()const;

		//Functions for saving image
		void savePoints(QString fileName)const;
		void savePolygons(QString fileName)const;
		void saveCollada(QString fileName)const;
	signals:

	public slots:

	};
}

#endif // SURFACE_H
