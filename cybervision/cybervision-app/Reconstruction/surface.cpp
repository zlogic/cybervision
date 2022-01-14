#include <Qt3DCore/QTransform>
#include <Qt3DRender/QGeometryRenderer>
#include <Qt3DRender/QAttribute>
#include <Qt3DRender/QBuffer>
#include <QPainter>
#include <QTextStream>
#include <QLineF>

#include "surface.h"
#include "options.h"
#include <cmath>

namespace cybervision{

Surface::Surface(){
	scale= 1.0;
	medianDepth= 0,baseDepth=0,minDepth= 0,maxDepth= 0;
}

Surface::Surface(const Surface&sc){
	operator =(sc);
}

void Surface::operator =(const Surface&sc){
	this->triangles= sc.triangles;
	this->points= sc.points;
	this->scale= sc.scale;
	this->medianDepth= sc.medianDepth;
	this->baseDepth= sc.baseDepth;
	this->minDepth= sc.minDepth;
	this->maxDepth= sc.maxDepth;
	this->imageSize= sc.imageSize;
	this->image1= sc.image1;
	this->image2= sc.image2;
}

Qt3DCore::QEntity* Surface::create3DEntity(Qt3DCore::QEntity* parent) const{
	const int nVerts = 3 * triangles.size();
	const int elementSize = 3 + 3 + 2;
	const int stride = elementSize * sizeof(float);
	QByteArray vertexBytes;
	vertexBytes.resize(stride * nVerts);
	float* vertices = reinterpret_cast<float*>(vertexBytes.data());

	for(QList<Surface::Triangle>::const_iterator it= triangles.constBegin();it!=triangles.constEnd();it++){
		const Point& pa=points[it->a];
		const Point& pb=points[it->b];
		const Point& pc=points[it->c];

		if(Options::renderNormalsMode== Options::RENDER_NORMALS_TRIANGLE){
			//Use triangle normals
			//Front side
			*vertices++= pa.coord.x(); *vertices++= pa.coord.y(); *vertices++= pa.coord.z();
			*vertices++= it->normal.x(); *vertices++= it->normal.y(); *vertices++= it->normal.z();
			*vertices++= pa.uv.x(); *vertices++= pa.uv.y();

			*vertices++= pb.coord.x(); *vertices++= pb.coord.y(); *vertices++= pb.coord.z();
			*vertices++= it->normal.x(); *vertices++= it->normal.y(); *vertices++= it->normal.z();
			*vertices++= pb.uv.x(); *vertices++= pb.uv.y();

			*vertices++= pc.coord.x(); *vertices++= pc.coord.y(); *vertices++= pc.coord.z();
			*vertices++= it->normal.x(); *vertices++= it->normal.y(); *vertices++= it->normal.z();
			*vertices++= pc.uv.x(); *vertices++= pc.uv.y();
		}else if(Options::renderNormalsMode== Options::RENDER_NORMALS_POINT){
			//Use point normals
			//Front side
			*vertices++= pa.coord.x(); *vertices++= pa.coord.y(); *vertices++= pa.coord.z();
			*vertices++= pa.normal.x(); *vertices++= pa.normal.y(); *vertices++= pa.normal.z();
			*vertices++= pa.uv.x(); *vertices++= pa.uv.y();

			*vertices++= pb.coord.x(); *vertices++= pb.coord.y(); *vertices++= pb.coord.z();
			*vertices++= pb.normal.x(); *vertices++= pb.normal.y(); *vertices++= pb.normal.z();
			*vertices++= pb.uv.x(); *vertices++= pb.uv.y();

			*vertices++= pc.coord.x(); *vertices++= pc.coord.y(); *vertices++= pc.coord.z();
			*vertices++= pc.normal.x(); *vertices++= pc.normal.y(); *vertices++= pc.normal.z();
			*vertices++= pc.uv.x(); *vertices++= pc.uv.y();
		}
	}

	Qt3DCore::QEntity* entity = new Qt3DCore::QEntity(parent);
	Qt3DRender::QGeometryRenderer* renderer = new Qt3DRender::QGeometryRenderer(entity);
	Qt3DRender::QGeometry* geometry = new Qt3DRender::QGeometry(renderer);

	Qt3DRender::QBuffer* vertexBuffer = new Qt3DRender::QBuffer(geometry);
	vertexBuffer->setData(vertexBytes);

	Qt3DRender::QAttribute* positionAttribute = new Qt3DRender::QAttribute(geometry);
	positionAttribute->setAttributeType(Qt3DRender::QAttribute::VertexAttribute);
	positionAttribute->setBuffer(vertexBuffer);
	positionAttribute->setVertexBaseType(Qt3DRender::QAttribute::Float);
	positionAttribute->setVertexSize(3);
	positionAttribute->setByteOffset(0);
	positionAttribute->setByteStride(stride);
	positionAttribute->setCount(nVerts);
	positionAttribute->setName(Qt3DRender::QAttribute::defaultPositionAttributeName());

	Qt3DRender::QAttribute* normalAttribute = new Qt3DRender::QAttribute(geometry);
	normalAttribute->setAttributeType(Qt3DRender::QAttribute::VertexAttribute);
	normalAttribute->setBuffer(vertexBuffer);
	normalAttribute->setVertexBaseType(Qt3DRender::QAttribute::Float);
	normalAttribute->setVertexSize(3);
	normalAttribute->setByteOffset(3 * sizeof(float));
	normalAttribute->setByteStride(stride);
	normalAttribute->setCount(nVerts);
	normalAttribute->setName(Qt3DRender::QAttribute::defaultNormalAttributeName());

	Qt3DRender::QAttribute* texCoordAttribute = new Qt3DRender::QAttribute(geometry);
	texCoordAttribute->setAttributeType(Qt3DRender::QAttribute::VertexAttribute);
	texCoordAttribute->setBuffer(vertexBuffer);
	texCoordAttribute->setVertexBaseType(Qt3DRender::QAttribute::Float);
	texCoordAttribute->setVertexSize(2);
	texCoordAttribute->setByteOffset((3 + 3) * sizeof(float));
	texCoordAttribute->setByteStride(stride);
	texCoordAttribute->setCount(nVerts);
	texCoordAttribute->setName(Qt3DRender::QAttribute::defaultTextureCoordinateAttributeName());

	geometry->addAttribute(positionAttribute);
	geometry->addAttribute(normalAttribute);
	geometry->addAttribute(texCoordAttribute);

	renderer->setGeometry(geometry);
	renderer->setInstanceCount(1);
	renderer->setPrimitiveType(Qt3DRender::QGeometryRenderer::Triangles);
	renderer->setVertexCount(nVerts);
	entity->addComponent(renderer);

	Qt3DCore::QTransform *surfaceTransform = new Qt3DCore::QTransform(entity);
	surfaceTransform->setScale(scale);
	entity->addComponent(surfaceTransform);

	return entity;
}

bool Surface::isOk()const{ return !triangles.empty() && !points.empty() && scale>0.0; }

qreal Surface::getMedianDepth()const{ return medianDepth; }
qreal Surface::getMinDepth()const{ return minDepth; }
qreal Surface::getMaxDepth()const{ return maxDepth; }
qreal Surface::getBaseDepth()const{ return baseDepth; }
QRectF cybervision::Surface::getImageSize() const{ return imageSize; }

qreal Surface::getScale()const{ return scale; }


const QImage& Surface::getTexture1()const{ return image1; }
const QImage& Surface::getTexture2()const{ return image2; }
void Surface::setTextures(const QImage& image1,const QImage& image2){
	this->image1= image1, this->image2= image2;
}

/*
 * Functions for saving image data
 */
void Surface::savePoints(QString fileName)const{
	QFile file(fileName);
	file.open(QIODevice::WriteOnly);

	QTextStream stream(&file);
	for(QList<Point>::const_iterator it= points.begin();it!=points.end();it++)
		stream<<it->coord.x()<<"\t"<<it->coord.y()<<"\t"<<it->coord.z()<<"\r\n";

	file.close();
}
void Surface::savePolygons(QString fileName)const{
	QFile file(fileName);
	file.open(QIODevice::WriteOnly);

	QTextStream stream(&file);
	for(QList<Surface::Triangle>::const_iterator it= triangles.begin();it!=triangles.end();it++){
		const QVector3D& pa=points[it->a].coord;
		const QVector3D& pb=points[it->b].coord;
		const QVector3D& pc=points[it->c].coord;
		stream<<it->normal.x()<<"\t"<<it->normal.y()<<"\t"<<it->normal.z()<<"\t";
		stream<<pa.x()<<"\t"<<pa.y()<<"\t"<<pa.z()<<"\t";
		stream<<pb.x()<<"\t"<<pb.y()<<"\t"<<pb.z()<<"\t";
		stream<<pc.x()<<"\t"<<pc.y()<<"\t"<<pc.z()<<"\r\n";
	}

	file.close();
}
void Surface::saveCollada(QString fileName)const{
	//Read XML template
	QString xmlTemplate;
	{
		QFile templateFile(":/collada/Template.xml");
		templateFile.open(QIODevice::ReadOnly);
		QTextStream stream(&templateFile);
		xmlTemplate= stream.readAll();
	}

	QFile file(fileName);
	file.open(QIODevice::WriteOnly);

	QTextStream stream(&file);

	QString vertexesString,normalsString,textureCoordinatesString,trianglesString;

	QString result=xmlTemplate;
	if(Options::colladaFormat==Options::COLLADA_SHARED_POINTS){
		//Output points, normals and texture coordinates
		for(QList<Point>::const_iterator it= points.begin();it!=points.end();it++){
			QString currentVertexString;
			currentVertexString.append(QString("%1 %2 %3 ").arg(it->coord.x()).arg(it->coord.y()).arg(it->coord.z()));
			QString currentNormalString;
			currentNormalString.append(QString("%1 %2 %3 ").arg(it->normal.x()).arg(it->normal.y()).arg(it->normal.z()));
			QString currentTextureCoordinatesString;
			currentTextureCoordinatesString.append(QString("%1 %2 ").arg(it->uv.x()).arg(it->uv.y()));

			vertexesString.append(currentVertexString);
			normalsString.append(currentNormalString);
			textureCoordinatesString.append(currentTextureCoordinatesString);
		}

		//Output polygons
		for(QList<Surface::Triangle>::const_iterator it= triangles.begin();it!=triangles.end();it++){
			QString currentTriangleString;
			currentTriangleString.append(QString("%1 %2 %3 ").arg(it->a).arg(it->b).arg(it->c));

			trianglesString.append(currentTriangleString);
		}

		result.replace("##[points-array-size]##",QString("%1").arg(points.length()*3));
		result.replace("##[points-count]##",QString("%1").arg(points.length()));
		result.replace("##[normals-array-size]##",QString("%1").arg(points.length()*3));
		result.replace("##[normals-count]##",QString("%1").arg(triangles.length()));
		result.replace("##[texture-coordinates-array-size]##",QString("%1").arg(points.length()*2));
		result.replace("##[texture-coordinates-count]##",QString("%1").arg(points.length()));
		result.replace("##[triangles-count]##",QString("%1").arg(triangles.length()));
	}else if(Options::colladaFormat==Options::COLLADA_INDEPENDENT_POLYGONS){
		PolygonPoint i=0;//Triangle indexes
		for(QList<Surface::Triangle>::const_iterator it= triangles.begin();it!=triangles.end();it++){
			const Point& pa=points[it->a];
			const Point& pb=points[it->b];
			const Point& pc=points[it->c];
			QString currentVertexString;
			currentVertexString.append(QString("%1 %2 %3 ").arg(pa.coord.x()).arg(pa.coord.y()).arg(pa.coord.z()));
			currentVertexString.append(QString("%1 %2 %3 ").arg(pb.coord.x()).arg(pb.coord.y()).arg(pb.coord.z()));
			currentVertexString.append(QString("%1 %2 %3 ").arg(pc.coord.x()).arg(pc.coord.y()).arg(pc.coord.z()));
			QString currentNormalString;
			currentNormalString.append(QString("%1 %2 %3 ").arg(pa.normal.x()).arg(pa.normal.y()).arg(pa.normal.z()));
			currentNormalString.append(QString("%1 %2 %3 ").arg(pb.normal.x()).arg(pb.normal.y()).arg(pb.normal.z()));
			currentNormalString.append(QString("%1 %2 %3 ").arg(pc.normal.x()).arg(pc.normal.y()).arg(pc.normal.z()));
			QString currentTextureCoordinatesString;
			currentTextureCoordinatesString.append(QString("%1 %2 ").arg(pa.uv.x()).arg(pa.uv.y()));
			currentTextureCoordinatesString.append(QString("%1 %2 ").arg(pb.uv.x()).arg(pb.uv.y()));
			currentTextureCoordinatesString.append(QString("%1 %2 ").arg(pc.uv.x()).arg(pc.uv.y()));
			QString currentTriangleString;
			currentTriangleString.append(QString("%1 %2 %3 ").arg(i).arg(i+1).arg(i+2));

			vertexesString.append(currentVertexString);
			normalsString.append(currentNormalString);
			textureCoordinatesString.append(currentTextureCoordinatesString);
			trianglesString.append(currentTriangleString);
			i+=3;
		}

		result.replace("##[points-array-size]##",QString("%1").arg(triangles.length()*9));
		result.replace("##[points-count]##",QString("%1").arg(triangles.length()*3));
		result.replace("##[normals-array-size]##",QString("%1").arg(triangles.length()*9));
		result.replace("##[normals-count]##",QString("%1").arg(triangles.length()*3));
		result.replace("##[texture-coordinates-array-size]##",QString("%1").arg(triangles.length()*6));
		result.replace("##[texture-coordinates-count]##",QString("%1").arg(triangles.length()*3));
		result.replace("##[triangles-count]##",QString("%1").arg(triangles.length()));
	}

	QString fileNameTexture= fileName+".png";

	result.replace("##[points]##",vertexesString);
	result.replace("##[normals]##",normalsString);
	result.replace("##[texture-coordinates]##",textureCoordinatesString);
	result.replace("##[triangles-indexes]##",trianglesString);
	result.replace("##[texture-image-filename]##",QFileInfo(fileNameTexture).fileName());

	if(!image1.save(fileNameTexture,"png"))
		;//TODO:error

	stream<<result;
	file.close();
}
void Surface::saveSceneJS(QString fileName)const{
	//Read SceneJS template
	QString sceneJSTemplate;
	{
		QFile templateFile(":/scenejs/Template.js");
		templateFile.open(QIODevice::ReadOnly);
		QTextStream stream(&templateFile);
		sceneJSTemplate= stream.readAll();
	}

	QFile file(fileName);
	file.open(QIODevice::WriteOnly);

	QTextStream stream(&file);

	QString vertexesString,normalsString,textureCoordinatesString,trianglesString;

	QString result=sceneJSTemplate;
	//Output points and normals
	for(QList<Point>::const_iterator it= points.begin();it!=points.end();it++){
		bool last= it==(points.end()-1);
		QString currentVertexString;
		currentVertexString.append(QString("%1,%2,%3%4").arg(it->coord.x()*scale).arg(it->coord.y()*scale).arg(it->coord.z()*scale).arg(!last?",":""));
		QString currentNormalString;
		currentNormalString.append(QString("%1,%2,%3%4").arg(it->normal.x()).arg(it->normal.y()).arg(it->normal.z()).arg(!last?",":""));
		QString currentTextureCoordinatesString;
		currentTextureCoordinatesString.append(QString("%1,%2%3").arg(it->uv.x()).arg(it->uv.y()).arg(!last?",":""));

		vertexesString.append(currentVertexString);
		normalsString.append(currentNormalString);
		textureCoordinatesString.append(currentTextureCoordinatesString);
	}

	//Output polygons
	for(QList<Surface::Triangle>::const_iterator it= triangles.begin();it!=triangles.end();it++){
		bool last= it==(triangles.end()-1);
		QString currentTriangleString;
		currentTriangleString.append(QString("%1,%2,%3%4").arg(it->a).arg(it->b).arg(it->c).arg(!last?",":""));

		trianglesString.append(currentTriangleString);
	}

	QString fileNameTexture= fileName+".png";

	result.replace("##[points]##",vertexesString);
	result.replace("##[normals]##",normalsString);
	result.replace("##[texture-coordinates]##",textureCoordinatesString);
	result.replace("##[triangles-indexes]##",trianglesString);
	result.replace("##[texture-image-filename]##",QFileInfo(fileNameTexture).fileName());

	if(!image1.save(fileNameTexture,"png"))
		;//TODO:error

	stream<<result;
	file.close();
}

void Surface::saveCybervision(QString fileName) const{
	QFile file(fileName);
	file.open(QIODevice::WriteOnly);
	QDataStream out(&file);

	out<<(int)1;//Image format version

	out<<points.size();
	for(QList<Point>::const_iterator it= points.begin();it!=points.end();it++)
		out<<it->coord<<it->normal<<it->uv;

	out<<triangles.size();
	for(QList<Triangle>::const_iterator it= triangles.begin();it!=triangles.end();it++)
		out<<it->a<<it->b<<it->c<<it->normal;

	out<<scale;
	out<<medianDepth<<baseDepth<<minDepth<<maxDepth;
	out<<imageSize;
	out<<image1<<image2;
	file.close();
}

const Surface Surface::fromFile(QString fileName){
	Surface surface;
	QFile file(fileName);
	file.open(QIODevice::ReadOnly);
	QDataStream in(&file);

	int version;
	in>>version;
	if(version==1){
		int pointsCount;
		in>>pointsCount;
		for(int i=0;i<pointsCount;i++){
			Point point;
			in>>point.coord>>point.normal>>point.uv;
			surface.points<<point;
		}
		int trianglesCount;
		in>>trianglesCount;
		for(int i=0;i<trianglesCount;i++){
			Triangle triangle;
			in>>triangle.a>>triangle.b>>triangle.c>>triangle.normal;
			surface.triangles<<triangle;
		}
		in>>surface.scale;
		in>>surface.medianDepth>>surface.baseDepth>>surface.minDepth>>surface.maxDepth;
		in>>surface.imageSize;
		in>>surface.image1>>surface.image2;
	}

	file.close();
	return surface;
}

}
