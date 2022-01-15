#define _USE_MATH_DEFINES
#include <cmath>

#include "cybervisionviewer.h"

#include <Reconstruction/options.h>

#include <QMutexLocker>
#include <QMouseEvent>
#include <QThread>

#include <limits>

#include <QMatrix4x4>
#include <QScreen>

#include <Qt3DRender/QCamera>
#include <Qt3DRender/QPointLight>
#include <Qt3DRender/QGeometry>
#include <Qt3DRender/QAttribute>
#include <Qt3DRender/QGeometryRenderer>
#include <Qt3DRender/QRenderSettings>
#include <Qt3DRender/QTexture>
#include <Qt3DRender/QParameter>
#include <Qt3DExtras/QForwardRenderer>
#include <Qt3DExtras/QDiffuseSpecularMaterial>
#include <Qt3DExtras/QExtrudedTextMesh>
#include <Qt3DExtras/QCylinderMesh>
#include <Qt3DExtras/QConeMesh>
#include <Qt3DExtras/QCuboidMesh>

CybervisionViewer::CybervisionViewer(): Qt3DExtras::Qt3DWindow(){
	rootEntity= NULL;
	surfaceEntity= NULL;
	surfaceTexture= NULL;
	axesTransform= NULL;
	clickDetector= NULL;
	selectedPointEntity= NULL;
	selectedPointTransform= NULL;

	this->mouseMode= MOUSE_ROTATION;
	this->mousePressed = false;
	this->textureMode= TEXTURE_1;
	this->showGrid= false;
	this->drawingCrossSectionLine= -1;
	clickLocation= QVector3D(std::numeric_limits<qreal>::infinity(),std::numeric_limits<qreal>::infinity(),std::numeric_limits<qreal>::infinity());
	for(int i=0;i<2;i++){
		QPair<QVector3D,QVector3D> crossSectionLine;
		crossSectionLine.first= QVector3D(std::numeric_limits<qreal>::infinity(),std::numeric_limits<qreal>::infinity(),std::numeric_limits<qreal>::infinity());
		crossSectionLine.second= QVector3D(std::numeric_limits<qreal>::infinity(),std::numeric_limits<qreal>::infinity(),std::numeric_limits<qreal>::infinity());
		crossSectionLines<<crossSectionLine;
	}

	initializeScene();
}

CybervisionViewer::~CybervisionViewer(){
	delete rootEntity;
}

void CybervisionViewer::setSurface3D(const cybervision::Surface& surface){
	QVector3D infinity(std::numeric_limits<qreal>::infinity(),std::numeric_limits<qreal>::infinity(),std::numeric_limits<qreal>::infinity());
	clickLocation= infinity;
	for(QList<QPair<QVector3D,QVector3D> >::iterator it=crossSectionLines.begin();it!=crossSectionLines.end();it++){
		it->first= infinity;
		it->second= infinity;
	}
	emit selectedPointUpdated(infinity);
	{
		QMutexLocker lock(&surfaceMutex);
		this->surface= surface;

		if(surfaceEntity!=NULL)
			delete surfaceEntity;
		gridEntities.clear();
		surfaceEntity = surface.create3DEntity(rootEntity);

		addSurfaceMaterial();
		addSelectedPoint();
		addCrossSectionLines();
	}
	updateGrid();
	setTextureMode(textureMode);
}

void CybervisionViewer::addSurfaceMaterial(){
	Qt3DExtras::QDiffuseSpecularMaterial *material = new Qt3DExtras::QDiffuseSpecularMaterial(surfaceEntity);
	if(cybervision::Options::renderShiny)
		material->setShininess(50.0f);
	else
		material->setShininess(0.0f);

	Qt3DRender::QTexture2D *texture2D = new Qt3DRender::QTexture2D();
	surfaceTexture = new SurfaceTexture(texture2D);
	texture2D->addTextureImage(surfaceTexture);
	texture2D->setMagnificationFilter(Qt3DRender::QTexture2D::Linear);
	texture2D->setMinificationFilter(Qt3DRender::QTexture2D::Linear);
	material->setDiffuse(QVariant::fromValue(texture2D));
	material->setTextureScale(1.0f);

	surfaceEntity->addComponent(material);
}

void CybervisionViewer::addSelectedPoint(){
	selectedPointEntity = new Qt3DCore::QEntity(surfaceEntity);
	selectedPointEntity->setEnabled(false);

	Qt3DExtras::QCuboidMesh *mesh = new Qt3DExtras::QCuboidMesh(selectedPointEntity);

	selectedPointTransform = new Qt3DCore::QTransform();
	selectedPointTransform->setScale(cybervision::Options::PointDiameter/surface.getScale());

	Qt3DExtras::QDiffuseSpecularMaterial *material = new Qt3DExtras::QDiffuseSpecularMaterial();
	material->setDiffuse(QColor(0xff,0x99,0x00));

	selectedPointEntity->addComponent(mesh);
	selectedPointEntity->addComponent(material);
	selectedPointEntity->addComponent(selectedPointTransform);
}

void CybervisionViewer::addCrossSectionLines(){
	const int elementSize = 3;
	const int stride = elementSize * sizeof(float);

	crossSectionLineEntities.clear();

	for(QList<QPair<QVector3D,QVector3D> >::const_iterator it=crossSectionLines.constBegin();it!=crossSectionLines.constEnd();it++){
		Qt3DCore::QEntity* entity = new Qt3DCore::QEntity(surfaceEntity);
		Qt3DRender::QGeometryRenderer* renderer = new Qt3DRender::QGeometryRenderer(entity);

		Qt3DRender::QGeometry *geometry = new Qt3DRender::QGeometry(renderer);
		QByteArray bufferBytes;
		bufferBytes.resize(2 * stride);
		float *positions = reinterpret_cast<float*>(bufferBytes.data());

		Qt3DRender::QBuffer *buf = new Qt3DRender::QBuffer(geometry);
		buf->setData(bufferBytes);

		*positions++=it->first.x();
		*positions++=it->first.y();
		*positions++=it->first.z();
		*positions++=it->second.x();
		*positions++=it->second.y();
		*positions++=it->second.z();

		Qt3DRender::QAttribute* positionAttribute = new Qt3DRender::QAttribute(geometry);
		positionAttribute->setAttributeType(Qt3DRender::QAttribute::VertexAttribute);
		positionAttribute->setBuffer(buf);
		positionAttribute->setVertexBaseType(Qt3DRender::QAttribute::Float);
		positionAttribute->setVertexSize(3);
		positionAttribute->setByteOffset(0);
		positionAttribute->setByteStride(stride);
		positionAttribute->setCount(1);
		positionAttribute->setName(Qt3DRender::QAttribute::defaultPositionAttributeName());

		geometry->addAttribute(positionAttribute);

		renderer->setGeometry(geometry);
		renderer->setInstanceCount(1);
		renderer->setPrimitiveType(Qt3DRender::QGeometryRenderer::Lines);
		renderer->setVertexCount(2);
		renderer->setFirstInstance(0);

		Qt3DExtras::QDiffuseSpecularMaterial *material = new Qt3DExtras::QDiffuseSpecularMaterial(entity);
		material->setAmbient(QColor(0xff,0x99,0x00));
		material->setSpecular(QColor(0xff,0x99,0x00));
		material->setDiffuse(QColor(0xff,0x99,0x00));
		material->setShininess(.0f);

		entity->addComponent(renderer);
		entity->addComponent(material);

		buf->setEnabled(false);

		crossSectionLineEntities<<buf;
	}
}

void CybervisionViewer::setMouseMode(MouseMode mouseMode){
	this->mouseMode= mouseMode;
}

void CybervisionViewer::setTextureMode(TextureMode textureMode){
	this->textureMode= textureMode;
	QMutexLocker lock(&surfaceMutex);
	if(surfaceTexture==NULL)
		return;
	if(textureMode== TEXTURE_1)
		surfaceTexture->setImage(surface.getTexture1());
	else if(textureMode== TEXTURE_2)
		surfaceTexture->setImage(surface.getTexture2());
	else
		surfaceTexture->setImage(QImage());
}

void CybervisionViewer::setShowGrid(bool show){
	showGrid= show;
	updateGrid();
}

void CybervisionViewer::setDrawCrossSectionLine(int lineId){
	drawingCrossSectionLine= lineId;
}

const cybervision::Surface& CybervisionViewer::getSurface3D()const{
	return surface;
}

QMutex& CybervisionViewer::getSurfaceMutex(){
	return surfaceMutex;
}

QVector3D CybervisionViewer::getSelectedPoint() const{
	return clickLocation;
}

QPair<QVector3D,QVector3D> CybervisionViewer::getCrossSectionLine(int lineId) const{
	if(lineId<0 || lineId>=crossSectionLines.size()){
		QPair<QVector3D,QVector3D> result(
					QVector3D(std::numeric_limits<qreal>::infinity(),std::numeric_limits<qreal>::infinity(),std::numeric_limits<qreal>::infinity()),
					QVector3D(std::numeric_limits<qreal>::infinity(),std::numeric_limits<qreal>::infinity(),std::numeric_limits<qreal>::infinity())
					);
		return result;
	}
	QPair<QVector3D,QVector3D> result(
				QVector3D(crossSectionLines[lineId].first.x(),crossSectionLines[lineId].first.y(),0),
				QVector3D(crossSectionLines[lineId].second.x(),crossSectionLines[lineId].second.y(),0)
				);
	return result;
}

//Scene-specific stuff

void CybervisionViewer::initializeScene(){
	rootEntity = new Qt3DCore::QEntity();
	setRootEntity(rootEntity);

	//Set up the rendering scene
	defaultFrameGraph()->setClearColor(QColor(0xff,0xff,0xff));
	renderSettings()->setRenderPolicy(Qt3DRender::QRenderSettings::OnDemand);

	//Camera
	Qt3DRender::QCamera *cameraEntity = camera();
	connect(cameraEntity,&Qt3DRender::QCamera::viewMatrixChanged,this,&CybervisionViewer::cameraUpdated,Qt::AutoConnection);
	connect(cameraEntity,&Qt3DRender::QCamera::projectionMatrixChanged,this,&CybervisionViewer::cameraUpdated,Qt::AutoConnection);

	cameraEntity->lens()->setPerspectiveProjection(90.0f, 1.0f, 1.0f, 1000.0f);
	cameraEntity->setPosition(QVector3D(.0f,.0f,15.0f));
	cameraEntity->setUpVector(QVector3D(.0f,1.0f,.0f));
	cameraEntity->setViewCenter(QVector3D(.0f,.0f,.0f));

	//Light options
	Qt3DCore::QEntity *lightEntity = new Qt3DCore::QEntity(cameraEntity);
	Qt3DRender::QPointLight *light = new Qt3DRender::QPointLight(lightEntity);
	light->setColor(QColor(0xff,0xff,0xff));
	light->setIntensity(1);
	lightEntity->addComponent(light);

	clickDetector= new Qt3DRender::QScreenRayCaster(rootEntity);
	clickDetector->setRunMode(Qt3DRender::QAbstractRayCaster::SingleShot);
	clickDetector->setEnabled(false);
	rootEntity->addComponent(clickDetector);

	renderSettings()->pickingSettings()->setPickMethod(Qt3DRender::QPickingSettings::TrianglePicking);
	renderSettings()->pickingSettings()->setPickResultMode(Qt3DRender::QPickingSettings::NearestPick);

	connect(clickDetector,&Qt3DRender::QScreenRayCaster::hitsChanged,this,&CybervisionViewer::hitsChanged,Qt::AutoConnection);

	initializeAxesWidget();
}

Qt3DRender::QGeometry* CybervisionViewer::createLines(const QVector<QVector3D>& lines,Qt3DCore::QNode* parent){
	const int elementSize = 3;
	const int stride = elementSize * sizeof(float);

	Qt3DRender::QGeometry *geometry = new Qt3DRender::QGeometry(parent);
	QByteArray bufferBytes;
	bufferBytes.resize(lines.size() * stride);
	float *positions = reinterpret_cast<float*>(bufferBytes.data());

	Qt3DRender::QBuffer *buf = new Qt3DRender::QBuffer(geometry);
	buf->setData(bufferBytes);

	for(QVector<QVector3D>::const_iterator it = lines.constBegin();it!=lines.constEnd();it++){
		*positions++=it->x();
		*positions++=it->y();
		*positions++=it->z();
	}

	Qt3DRender::QAttribute* positionAttribute = new Qt3DRender::QAttribute(geometry);
	positionAttribute->setAttributeType(Qt3DRender::QAttribute::VertexAttribute);
	positionAttribute->setBuffer(buf);
	positionAttribute->setVertexBaseType(Qt3DRender::QAttribute::Float);
	positionAttribute->setVertexSize(3);
	positionAttribute->setByteOffset(0);
	positionAttribute->setByteStride(stride);
	positionAttribute->setCount(lines.size()/2);
	positionAttribute->setName(Qt3DRender::QAttribute::defaultPositionAttributeName());

	geometry->addAttribute(positionAttribute);

	return geometry;
}

void CybervisionViewer::initializeAxesWidget(){
	float axesLength= 0.7f;
	float arrowLength= 0.1f;

	Qt3DCore::QEntity* entity = new Qt3DCore::QEntity(rootEntity);
	struct AxisPropeties{
		QVector3D direction;
		QString label;
		QColor color;
		AxisPropeties(QVector3D direction,QString label, QColor color): direction(direction),label(label),color(color){}
	};
	QVector<AxisPropeties> axes;
	axes<<AxisPropeties(QVector3D(1.0f,.0f,.0f),tr("x"),QColor(0xff,0x00,0x00));
	axes<<AxisPropeties(QVector3D(.0f,1.0f,.0f),tr("y"),QColor(0x00,0xff,0x00));
	axes<<AxisPropeties(QVector3D(.0f,.0f,1.0f),tr("z"),QColor(0x00,0xff,0xff));

	QFont font("Arial",8);
	for(QVector<AxisPropeties>::const_iterator it=axes.constBegin();it!=axes.constEnd();it++){
		Qt3DCore::QEntity *lineEntity = new Qt3DCore::QEntity(entity);
		Qt3DExtras::QDiffuseSpecularMaterial *material = new Qt3DExtras::QDiffuseSpecularMaterial(entity);
		material->setAmbient(it->color);
		material->setDiffuse(it->color);
		material->setSpecular(it->color);
		material->setShininess(.0f);

		Qt3DExtras::QCylinderMesh *lineMesh = new Qt3DExtras::QCylinderMesh(lineEntity);
		lineMesh->setRadius(0.03);
		lineMesh->setLength(axesLength-arrowLength);
		lineMesh->setRings(3);
		lineMesh->setSlices(10);

		Qt3DCore::QTransform *lineTransform = new Qt3DCore::QTransform(lineEntity);
		QVector3D rotateAround= QVector3D::crossProduct(it->direction,QVector3D(.0f,1.0f,.0f));
		lineTransform->setRotation(QQuaternion::fromAxisAndAngle(rotateAround, -90.0f));
		lineTransform->setTranslation(it->direction * lineMesh->length() / 2.0f);

		lineEntity->addComponent(lineMesh);
		lineEntity->addComponent(lineTransform);
		lineEntity->addComponent(material);

		Qt3DCore::QEntity *arrowEntity = new Qt3DCore::QEntity(entity);
		Qt3DExtras::QConeMesh *arrowMesh = new Qt3DExtras::QConeMesh(arrowEntity);
		arrowMesh->setTopRadius(.0f);
		arrowMesh->setBottomRadius(0.05f);
		arrowMesh->setLength(arrowLength);
		arrowMesh->setRings(3);
		arrowMesh->setSlices(10);

		Qt3DCore::QTransform *arrowTransform = new Qt3DCore::QTransform(arrowEntity);
		arrowTransform->setRotation(lineTransform->rotation());
		arrowTransform->setTranslation(it->direction * (axesLength - arrowMesh->length()/2.0f));

		arrowEntity->addComponent(arrowTransform);
		arrowEntity->addComponent(material);
		arrowEntity->addComponent(arrowMesh);

		Qt3DCore::QEntity* textEntity = new Qt3DCore::QEntity(entity);
		Qt3DExtras::QExtrudedTextMesh *textMesh = new Qt3DExtras::QExtrudedTextMesh(textEntity);
		textMesh->setFont(font);
		textMesh->setDepth(0.0f);
		textMesh->setText(it->label);

		Qt3DCore::QTransform *textTransform = new Qt3DCore::QTransform(textMesh);
		textTransform->setTranslation(it->direction * axesLength * 1.2);
		textTransform->setRotation(lineTransform->rotation());
		textTransform->setScale(0.15);

		textEntity->addComponent(textMesh);
		textEntity->addComponent(textTransform);
		textEntity->addComponent(material);
	}

	axesTransform = new Qt3DCore::QTransform(entity);
	entity->addComponent(axesTransform);
}

void CybervisionViewer::updateCrossSectionLines(){
	for(int i=0;i<crossSectionLines.size();i++){
		const QVector3D& start= crossSectionLines[i].first;
		const QVector3D& end= crossSectionLines[i].second;

		QByteArray bufferBytes;
		bufferBytes.resize(3*2*sizeof(float));
		float *positions = reinterpret_cast<float*>(bufferBytes.data());

		*positions++=start.x();
		*positions++=start.y();
		*positions++=start.z();
		*positions++=end.x();
		*positions++=end.y();
		*positions++=end.z();

		crossSectionLineEntities[i]->setEnabled(start!=QVector3D(std::numeric_limits<qreal>::infinity(),std::numeric_limits<qreal>::infinity(),std::numeric_limits<qreal>::infinity()));
		crossSectionLineEntities[i]->setData(bufferBytes);
	}
}

void CybervisionViewer::addGrid(){
	if(!surface.isOk() || !gridEntities.isEmpty())
		return;

	//Calculate grid steps
	qreal step_x= getOptimalGridStep(surface.getImageSize().left(),surface.getImageSize().right());
	qreal step_y= getOptimalGridStep(surface.getImageSize().top(),surface.getImageSize().bottom());
	qreal step_z= getOptimalGridStep(surface.getMinDepth(),surface.getMaxDepth());
	qreal step_xy= std::max(step_x,step_y);
	step_x= step_xy, step_y= step_xy;
	const int min_x= floor(surface.getImageSize().left()/step_x);
	const int max_x= ceil(surface.getImageSize().right()/step_x);
	const int min_y= floor(surface.getImageSize().top()/step_y);
	const int max_y= ceil(surface.getImageSize().bottom()/step_y);
	const int min_z= floor(surface.getMinDepth()/step_z);
	const int max_z= ceil(surface.getMaxDepth()/step_z);
	const qreal tickSize= 1/surface.getScale();

	gridMin= QVector3D(min_x*step_x*surface.getScale(),min_y*step_y*surface.getScale(),min_z*step_z*surface.getScale());
	gridMax= QVector3D(max_x*step_x*surface.getScale(),max_y*step_y*surface.getScale(),max_z*step_z*surface.getScale());

	//Create grid
	QMap<Corner,QVector<QVector3D> > gridLines;
	for(int i=min_x;i<=max_x;i++) {
		qreal x= step_x*i;

		QVector<QVector3D> lines;

		lines<<QVector3D(x,min_y*step_y,max_z*step_z)<<QVector3D(x,max_y*step_y,max_z*step_z);
		lines<<QVector3D(x,min_y*step_y,min_z*step_z)<<QVector3D(x,min_y*step_y,max_z*step_z);

		lines<<QVector3D(x,min_y*step_y,max_z*step_z)<<QVector3D(x,min_y*step_y-tickSize,max_z*step_z+tickSize);

		gridLines[CORNER_xyZ].append(lines);
		gridLines[CORNER_XyZ].append(lines);

		lines.clear();

		lines<<QVector3D(x,min_y*step_y,max_z*step_z)<<QVector3D(x,max_y*step_y,max_z*step_z);
		lines<<QVector3D(x,max_y*step_y,min_z*step_z)<<QVector3D(x,max_y*step_y,max_z*step_z);

		lines<<QVector3D(x,max_y*step_y,max_z*step_z)<<QVector3D(x,max_y*step_y+tickSize,max_z*step_z+tickSize);

		gridLines[CORNER_xYZ].append(lines);
		gridLines[CORNER_XYZ].append(lines);

		lines.clear();

		lines<<QVector3D(x,min_y*step_y,min_z*step_z)<<QVector3D(x,max_y*step_y,min_z*step_z);
		lines<<QVector3D(x,min_y*step_y,min_z*step_z)<<QVector3D(x,min_y*step_y,max_z*step_z);

		lines<<QVector3D(x,min_y*step_y,min_z*step_z)<<QVector3D(x,min_y*step_y-tickSize,min_z*step_z-tickSize);

		gridLines[CORNER_xyz].append(lines);
		gridLines[CORNER_Xyz].append(lines);

		lines.clear();

		lines<<QVector3D(x,min_y*step_y,min_z*step_z)<<QVector3D(x,max_y*step_y,min_z*step_z);
		lines<<QVector3D(x,max_y*step_y,min_z*step_z)<<QVector3D(x,max_y*step_y,max_z*step_z);

		lines<<QVector3D(x,max_y*step_y,min_z*step_z)<<QVector3D(x,max_y*step_y+tickSize,min_z*step_z-tickSize);

		gridLines[CORNER_xYz].append(lines);
		gridLines[CORNER_XYz].append(lines);
	}
	for(int i=min_y;i<=max_y;i++) {
		qreal y= step_y*i;

		QVector<QVector3D> lines;

		lines<<QVector3D(min_x*step_x,y,max_z*step_z)<<QVector3D(max_x*step_x,y,max_z*step_z);
		lines<<QVector3D(min_x*step_x,y,min_z*step_z)<<QVector3D(min_x*step_x,y,max_z*step_z);

		lines<<QVector3D(max_x*step_x,y,max_z*step_z)<<QVector3D(max_x*step_x+tickSize,y,max_z*step_z+tickSize);

		gridLines[CORNER_xyZ].append(lines);
		gridLines[CORNER_xYZ].append(lines);

		lines.clear();

		lines<<QVector3D(min_x*step_x,y,max_z*step_z)<<QVector3D(max_x*step_x,y,max_z*step_z);
		lines<<QVector3D(max_x*step_x,y,min_z*step_z)<<QVector3D(max_x*step_x,y,max_z*step_z);

		lines<<QVector3D(min_x*step_x,y,max_z*step_z)<<QVector3D(min_x*step_x-tickSize,y,max_z*step_z+tickSize);

		gridLines[CORNER_XyZ].append(lines);
		gridLines[CORNER_XYZ].append(lines);

		lines.clear();

		lines<<QVector3D(min_x*step_x,y,min_z*step_z)<<QVector3D(max_x*step_x,y,min_z*step_z);
		lines<<QVector3D(min_x*step_x,y,min_z*step_z)<<QVector3D(min_x*step_x,y,max_z*step_z);

		lines<<QVector3D(max_x*step_x,y,min_z*step_z)<<QVector3D(max_x*step_x+tickSize,y,min_z*step_z-tickSize);

		gridLines[CORNER_xyz].append(lines);
		gridLines[CORNER_xYz].append(lines);

		lines.clear();

		lines<<QVector3D(min_x*step_x,y,min_z*step_z)<<QVector3D(max_x*step_x,y,min_z*step_z);
		lines<<QVector3D(max_x*step_x,y,min_z*step_z)<<QVector3D(max_x*step_x,y,max_z*step_z);

		lines<<QVector3D(min_x*step_x,y,min_z*step_z)<<QVector3D(min_x*step_x-tickSize,y,min_z*step_z-tickSize);

		gridLines[CORNER_Xyz].append(lines);
		gridLines[CORNER_XYz].append(lines);
	}
	for(int i=min_z;i<=max_z;i++) {
		qreal z= step_z*i;

		QVector<QVector3D> lines;

		lines<<QVector3D(min_x*step_x,min_y*step_y,z)<<QVector3D(max_x*step_x,min_y*step_y,z);
		lines<<QVector3D(min_x*step_x,min_y*step_y,z)<<QVector3D(min_x*step_x,max_y*step_y,z);

		lines<<QVector3D(min_x*step_x,min_y*step_y,z)<<QVector3D(min_x*step_x-tickSize,min_y*step_y-tickSize,z);

		gridLines[CORNER_xyZ].append(lines);
		gridLines[CORNER_xyz].append(lines);

		lines.clear();

		lines<<QVector3D(min_x*step_x,max_y*step_y,z)<<QVector3D(max_x*step_x,max_y*step_y,z);
		lines<<QVector3D(min_x*step_x,min_y*step_y,z)<<QVector3D(min_x*step_x,max_y*step_y,z);

		lines<<QVector3D(min_x*step_x,max_y*step_y,z)<<QVector3D(min_x*step_x-tickSize,max_y*step_y+tickSize,z);

		gridLines[CORNER_xYZ].append(lines);
		gridLines[CORNER_xYz].append(lines);

		lines.clear();

		lines<<QVector3D(min_x*step_x,min_y*step_y,z)<<QVector3D(max_x*step_x,min_y*step_y,z);
		lines<<QVector3D(max_x*step_x,min_y*step_y,z)<<QVector3D(max_x*step_x,max_y*step_y,z);

		lines<<QVector3D(max_x*step_x,min_y*step_y,z)<<QVector3D(max_x*step_x+tickSize,min_y*step_y-tickSize,z);

		gridLines[CORNER_XyZ].append(lines);
		gridLines[CORNER_Xyz].append(lines);

		lines.clear();

		lines<<QVector3D(min_x*step_x,max_y*step_y,z)<<QVector3D(max_x*step_x,max_y*step_y,z);
		lines<<QVector3D(max_x*step_x,min_y*step_y,z)<<QVector3D(max_x*step_x,max_y*step_y,z);

		lines<<QVector3D(max_x*step_x,max_y*step_y,z)<<QVector3D(max_x*step_x+tickSize,max_y*step_y+tickSize,z);

		gridLines[CORNER_XYZ].append(lines);
		gridLines[CORNER_XYz].append(lines);
	}

	Qt3DCore::QEntity *gridEntity = new Qt3DCore::QEntity(surfaceEntity);

	Qt3DExtras::QDiffuseSpecularMaterial *material = new Qt3DExtras::QDiffuseSpecularMaterial(gridEntity);
	material->setAmbient(QColor(0x00,0x00,0x00));
	material->setDiffuse(QColor(0x00,0x00,0x00));
	material->setSpecular(QColor(0x00,0x00,0x00));
	material->setShininess(.0f);

	for(QMap<Corner,QVector<QVector3D> >::const_iterator it=gridLines.constBegin();it!=gridLines.constEnd();it++){
		Qt3DCore::QEntity *cornerGridEntity = new Qt3DCore::QEntity(gridEntity);
		gridEntities.insert(it.key(), cornerGridEntity);

		Qt3DCore::QEntity *linesEntity = new Qt3DCore::QEntity(cornerGridEntity);

		Qt3DRender::QGeometryRenderer* renderer = new Qt3DRender::QGeometryRenderer(linesEntity);
		Qt3DRender::QGeometry* gridGeometry= createLines(it.value(), renderer);

		renderer->setGeometry(gridGeometry);
		renderer->setInstanceCount(1);
		renderer->setPrimitiveType(Qt3DRender::QGeometryRenderer::Lines);
		renderer->setVertexCount(it.value().size());
		renderer->setFirstInstance(0);

		linesEntity->addComponent(renderer);
		linesEntity->addComponent(material);
	}

	QMultiMap<Corner,QPair<QString,QVector3D> > labels;
	for(int i=min_x;i<=max_x;i++) {
		qreal x= step_x*i;
		QString str;
		QTextStream stream(&str);
		//stream.setRealNumberPrecision(1);
		//stream.setRealNumberNotation(QTextStream::ScientificNotation);
		stream<<i*step_x*cybervision::Options::TextUnitScale;
		str= QString(tr("%1 \xC2\xB5m")).arg(str);

		QPair<QString,QVector3D> label;
		label= QPair<QString,QVector3D>(str,QVector3D(x,min_y*step_y-tickSize,max_z*step_z+tickSize));
		labels.insert(CORNER_xyZ, label);
		labels.insert(CORNER_XyZ, label);

		label= QPair<QString,QVector3D>(str,QVector3D(x,max_y*step_y+tickSize,max_z*step_z+tickSize));
		labels.insert(CORNER_xYZ, label);
		labels.insert(CORNER_XYZ, label);

		label= QPair<QString,QVector3D>(str,QVector3D(x,min_y*step_y-tickSize,min_z*step_z-tickSize));
		labels.insert(CORNER_xyz, label);
		labels.insert(CORNER_Xyz, label);

		label= QPair<QString,QVector3D>(str,QVector3D(x,max_y*step_y+tickSize,min_z*step_z-tickSize));
		labels.insert(CORNER_xYz, label);
		labels.insert(CORNER_XYz, label);
	}
	for(int i=min_y;i<=max_y;i++) {
		qreal y= step_y*i;
		QString str;
		QTextStream stream(&str);
		//stream.setRealNumberPrecision(1);
		//stream.setRealNumberNotation(QTextStream::ScientificNotation);
		stream<<i*step_y*cybervision::Options::TextUnitScale;
		str= QString(tr("%1 \xC2\xB5m")).arg(str);

		QPair<QString,QVector3D> label;
		label= QPair<QString,QVector3D>(str,QVector3D(max_x*step_x+tickSize,y,max_z*step_z+tickSize));
		labels.insert(CORNER_xyZ, label);
		labels.insert(CORNER_xYZ, label);

		label= QPair<QString,QVector3D>(str,QVector3D(min_x*step_x-tickSize,y,max_z*step_z+tickSize));
		labels.insert(CORNER_XyZ, label);
		labels.insert(CORNER_XYZ, label);

		label= QPair<QString,QVector3D>(str,QVector3D(max_x*step_x+tickSize,y,min_z*step_z-tickSize));
		labels.insert(CORNER_xyz, label);
		labels.insert(CORNER_xYz, label);

		label= QPair<QString,QVector3D>(str,QVector3D(min_x*step_x-tickSize,y,min_z*step_z-tickSize));
		labels.insert(CORNER_Xyz, label);
		labels.insert(CORNER_XYz, label);
	}
	for(int i=min_z;i<=max_z;i++) {
		qreal z= step_z*i;
		QString str;
		QTextStream stream(&str);
		//stream.setRealNumberPrecision(1);
		//stream.setRealNumberNotation(QTextStream::ScientificNotation);
		stream<<i*step_z*cybervision::Options::TextUnitScale;
		str= QString(tr("%1 \xC2\xB5m")).arg(str);

		QPair<QString,QVector3D> label;
		label= QPair<QString,QVector3D>(str,QVector3D(min_x*step_x-tickSize,min_y*step_y-tickSize,z));
		labels.insert(CORNER_xyZ, label);
		labels.insert(CORNER_xyz, label);

		label= QPair<QString,QVector3D>(str,QVector3D(min_x*step_x-tickSize,max_y*step_y+tickSize,z));
		labels.insert(CORNER_xYZ, label);
		labels.insert(CORNER_xYz, label);

		label= QPair<QString,QVector3D>(str,QVector3D(max_x*step_x+tickSize,min_y*step_y-tickSize,z));
		labels.insert(CORNER_XyZ, label);
		labels.insert(CORNER_Xyz, label);

		label= QPair<QString,QVector3D>(str,QVector3D(max_x*step_x+tickSize,max_y*step_y+tickSize,z));
		labels.insert(CORNER_XYZ, label);
		labels.insert(CORNER_XYz, label);
	}

	QFont font("Arial",7);
	for(QMultiMap<Corner,QPair<QString,QVector3D> >::const_iterator it=labels.constBegin();it!=labels.constEnd();it++){
		Qt3DCore::QEntity* textEntity = new Qt3DCore::QEntity(gridEntities[it.key()]);
		Qt3DExtras::QExtrudedTextMesh *textMesh = new Qt3DExtras::QExtrudedTextMesh(textEntity);
		textMesh->setFont(font);
		textMesh->setDepth(0.0f);
		textMesh->setText(it.value().first);

		Qt3DCore::QTransform *text2DTransform = new Qt3DCore::QTransform(textMesh);
		text2DTransform->setTranslation(it.value().second);
		text2DTransform->setScale(0.25f/surface.getScale());

		textEntity->addComponent(textMesh);
		textEntity->addComponent(text2DTransform);
		textEntity->addComponent(material);
	}
}

void CybervisionViewer::updateGrid(){
	QMutexLocker lock(&surfaceMutex);

	//Workaround: postpone creating grid until it's enabled, otherwise Qt doesn't properly update the QTransform
	if(showGrid && gridEntities.isEmpty())
		addGrid();

	//Get best coordinate pair
	Corner selected_corner= getOptimalCorner(gridMin, gridMax);

	for(QMap<Corner,Qt3DCore::QEntity*>::const_iterator it= gridEntities.constBegin();it!=gridEntities.constEnd();it++)
		it.value()->setEnabled(showGrid && it.key() == selected_corner);
}

qreal CybervisionViewer::getOptimalGridStep(qreal min, qreal max) const{
	qreal delta= max-min;
	qreal exp_x= pow(10.0,floor(log10(delta)));

	//Check if selected scale is too small
	if(delta/exp_x<5)
		exp_x/= 10;

	//Select optimal step
	int max_step_count= 10;
	qreal step_1= exp_x, step_2= exp_x*2, step_5= exp_x*5;
	int step_1_count= ceil(delta/step_1);
	int step_2_count= ceil(delta/step_2);
	//int step_5_count= ceil(delta/step_5);
	if(step_1_count<max_step_count)
		return step_1;
	else if(step_2_count<max_step_count)
		return step_2;
	else return step_5;
}

void CybervisionViewer::mousePressEvent(QMouseEvent *event){
	mousePressed= true;
	lastMousePos= event->pos();
	clickMousePos= event->pos();
	if(drawingCrossSectionLine>=0 && drawingCrossSectionLine<crossSectionLines.size()){
		QVector3D infinityLocation(std::numeric_limits<qreal>::infinity(),std::numeric_limits<qreal>::infinity(),std::numeric_limits<qreal>::infinity());
		QPair<QVector3D,QVector3D> currentCrossSectionLine(infinityLocation,infinityLocation);
		crossSectionLines[drawingCrossSectionLine]= currentCrossSectionLine;
	}
}

void CybervisionViewer::mouseReleaseEvent(QMouseEvent *event){
	//Detect click location
	lastMousePos = event->pos();
	clickDetector->trigger(event->pos());

	if(drawingCrossSectionLine>=0 && drawingCrossSectionLine<crossSectionLines.size()){
		QPair<QVector3D,QVector3D> result= getCrossSectionLine(drawingCrossSectionLine);
		emit crossSectionLineChanged(drawingCrossSectionLine,result.first,result.second);
		drawingCrossSectionLine= -1;
	}
	mousePressed= false;
}

void CybervisionViewer::mouseMoveEvent(QMouseEvent *event){
	if(!mousePressed)
		return;
	int dx= event->pos().x()-lastMousePos.x(), dy=event->pos().y()-lastMousePos.y();

	if(drawingCrossSectionLine>=0 && drawingCrossSectionLine<crossSectionLines.size()){
		clickDetector->trigger(event->pos());
	}else if(mouseMode==MOUSE_ROTATION){
		//Rotation
		if ((event->buttons() & Qt::LeftButton) && event->modifiers()==Qt::NoModifier) {
			camera()->tiltAboutViewCenter(dy);
			camera()->panAboutViewCenter(-dx);
		} else {
			camera()->tiltAboutViewCenter(dy);
			camera()->rollAboutViewCenter(-dx);
		}
	}else if(mouseMode==MOUSE_PANNING){
		//Translation
		if ((event->buttons() & Qt::LeftButton) && event->modifiers()==Qt::NoModifier) {
			camera()->translate(QVector3D(-dx/10.0f,dy/10.0f,0.0f));
		} else {
			camera()->translate(QVector3D(-dx/10.0f,0.0f,-dy/10.0f));
		}
	}
	lastMousePos = event->pos();
}

void CybervisionViewer::cameraUpdated(){
	{
		QMutexLocker lock(&surfaceMutex);
		if(axesTransform==NULL)
			return;
		QVector3D worldPosition = QVector3D(100.0f,100.0f,.7f).unproject(camera()->viewMatrix(), camera()->projectionMatrix(), QRect(0,0, size().width(),size().height()));
		axesTransform->setTranslation(worldPosition);
	}

	updateGrid();
}

void CybervisionViewer::hitsChanged(const Qt3DRender::QAbstractRayCaster::Hits &hits){
	if(hits.empty())
		return;
	const Qt3DRender::QRayCasterHit& hit = hits.first();
	if(hit.entity()!=surfaceEntity)
		return;

	QMutexLocker lock(&surfaceMutex);
	if(!mousePressed && (clickMousePos-lastMousePos).manhattanLength()<=3){
		if(selectedPointEntity!=NULL)
			selectedPointEntity->setEnabled(true);
		if(selectedPointTransform!=NULL)
			selectedPointTransform->setTranslation(hit.localIntersection());
		clickLocation = hit.localIntersection();
		emit selectedPointUpdated(hit.localIntersection()-QVector3D(0,0,surface.getBaseDepth()));
	}else if(drawingCrossSectionLine>=0 && drawingCrossSectionLine<crossSectionLines.size()){
		QVector3D infinity(std::numeric_limits<qreal>::infinity(),std::numeric_limits<qreal>::infinity(),std::numeric_limits<qreal>::infinity());
		QVector3D hitLocation(hit.localIntersection().x(),hit.localIntersection().y(),0);
		if(crossSectionLines[drawingCrossSectionLine].first==infinity)
			crossSectionLines[drawingCrossSectionLine].first= hitLocation;
		else
			crossSectionLines[drawingCrossSectionLine].second= hitLocation;
		updateCrossSectionLines();
	}
}

CybervisionViewer::Corner CybervisionViewer::getOptimalCorner(const QVector3D& min,const QVector3D& max) const{
	if(!showGrid)
		return CORNER_NONE;

	const QMatrix4x4 &transformationMatrix= camera()->viewMatrix();
	const QMatrix4x4 &projectionMatrix= camera()->projectionMatrix();

	QList<QPair<QVector3D,Corner> > corners;
	corners<<QPair<QVector3D,Corner>(QVector3D( min.x(), min.y(), min.z()),CORNER_xyz);
	corners<<QPair<QVector3D,Corner>(QVector3D( min.x(), min.y(), max.z()),CORNER_xyZ);
	corners<<QPair<QVector3D,Corner>(QVector3D( min.x(), max.y(), min.z()),CORNER_xYz);
	corners<<QPair<QVector3D,Corner>(QVector3D( min.x(), max.y(), max.z()),CORNER_xYZ);
	corners<<QPair<QVector3D,Corner>(QVector3D( max.x(), min.y(), min.z()),CORNER_Xyz);
	corners<<QPair<QVector3D,Corner>(QVector3D( max.x(), min.y(), max.z()),CORNER_XyZ);
	corners<<QPair<QVector3D,Corner>(QVector3D( max.x(), max.y(), min.z()),CORNER_XYz);
	corners<<QPair<QVector3D,Corner>(QVector3D( max.x(), max.y(), max.z()),CORNER_XYZ);

	qreal closestDistance= -std::numeric_limits<qreal>::infinity();
	Corner closestCorner= CORNER_NONE;
	for(QList<QPair<QVector3D,Corner> >::const_iterator it=corners.constBegin();it!=corners.constEnd();it++){
		QVector3D projection= projectionMatrix*((transformationMatrix*QVector4D(it->first,-1)).toVector3DAffine());
		if(projection.z()>closestDistance){
			closestDistance= projection.z();
			closestCorner= it->second;
		}
	}
	return closestCorner;
}

QPixmap CybervisionViewer::getScreenshot() const{
	return screen()->grabWindow(winId());
}

SurfaceTexture::SurfaceTexture(Qt3DCore::QNode *parent): Qt3DRender::QPaintedTextureImage(parent){
}

void SurfaceTexture::setImage(const QImage& texture){
	this->texture= texture;
	if(texture==QImage()){
		this->texture= QImage(1,1,QImage::Format_Mono);
		QPainter painter(&this->texture);
		painter.fillRect(0,0,1,1,QColor(0xff,0xff,0xff));
	}
	setSize(this->texture.size());
	update();
}

void SurfaceTexture::paint(QPainter *painter){
	if(texture==QImage()){
		return;
	}
	painter->drawImage(0,0,texture.mirrored());
}
