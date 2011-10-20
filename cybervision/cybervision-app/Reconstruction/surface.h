#ifndef SURFACE_H
#define SURFACE_H

#include <QVector2D>
#include <QVector3D>
#include <QFileInfo>
#include <QRectF>
#include <QImage>

namespace cybervision{
	class Surface{
		friend class Sculptor;
		friend class Inspector;
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
			QVector2D uv;
		};

		//Surface data in two formats
		QList<Triangle> triangles;
		QList<Point> points;

		//Scale for viewport
		qreal scale;

		//Depth statistics
		qreal medianDepth,baseDepth,minDepth,maxDepth;

		//Image size
		QRectF imageSize;

		//Textures
		QImage image1,image2;
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
		qreal getBaseDepth()const;

		//Returns source image size
		QRectF getImageSize()const;

		//Returns scale
		qreal getScale()const;

		//Returns the textures
		const QImage& getTexture1() const;
		const QImage& getTexture2() const;

		//Sets the textures
		void setTextures(const QImage& image1,const QImage& image2);

		//Functions for saving image
		void savePoints(QString fileName)const;
		void savePolygons(QString fileName)const;
		void saveCollada(QString fileName)const;
		void saveSceneJS(QString fileName)const;
	signals:

	public slots:

	};
}

#endif // SURFACE_H
