#ifndef SURFACE_H
#define SURFACE_H

#include <QVector2D>
#include <QVector3D>
#include <QFileInfo>
#include <QRectF>
#include <QImage>
#include <Qt3DCore/QEntity>

namespace cybervision{
/*
 * Class for storing a reconstructed surface along with its other data. Also contains exporting code.
 */
class Surface{
	friend class Sculptor;
	friend class CrossSection;
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
	float scale;

	//Depth statistics
	float medianDepth,baseDepth,minDepth,maxDepth;

	//Image size
	QRectF imageSize;

	//Textures
	QImage image1,image2;
public:
	Surface();
	Surface(const Surface&);
	void operator =(const Surface&);

	//Returns true if surface contains valid data instead of an empty set
	bool isOk() const;

	//Returns point depth statistics (min/max/median)
	float getMedianDepth()const;
	float getMinDepth()const;
	float getMaxDepth()const;
	float getBaseDepth()const;

	//Returns source image size
	QRectF getImageSize()const;

	//Returns scale
	float getScale()const;

	//Returns the textures
	const QImage& getTexture1() const;
	const QImage& getTexture2() const;

	//Sets the textures
	void setTextures(const QImage& image1,const QImage& image2);

	//Creates and returns a QEntity that can be rendered by Qt3D
	Qt3DCore::QEntity* create3DEntity(Qt3DCore::QEntity* parent)const;

	//Functions for saving image
	void savePoints(QString fileName)const;
	void savePolygons(QString fileName)const;
	void saveCollada(QString fileName)const;
	void saveSceneJS(QString fileName)const;
	void saveCybervision(QString fileName)const;

	//Loads the image from file
	static const Surface fromFile(QString fileName);
signals:

public slots:

};

}

#endif // SURFACE_H
