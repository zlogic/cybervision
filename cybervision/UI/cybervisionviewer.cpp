#include "cybervisionviewer.h"

#include <QMutexLocker>
#include <QMouseEvent>
#include <QThread>

#define _USE_MATH_DEFINES
#include <cmath>

CybervisionViewer::CybervisionViewer(QWidget *parent): QGLWidget(parent){
	vpTranslation= QVector3D(0,0,-15);
}


void CybervisionViewer::setSurface3D(const cybervision::Surface& surface){
	{
		QMutexLocker lock(&surfaceMutex);
		this->surface= surface;
	}
	updateGL();
}

const cybervision::Surface& CybervisionViewer::getSurface3D()const{
	return surface;
}

//OpenGL-specific stuff

void CybervisionViewer::initializeGL(){
	// Set up the rendering context, define display lists etc.:
	glClearColor(1.0, 1.0, 1.0, 0.0);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);
	glShadeModel(GL_SMOOTH);
	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);
	//static GLfloat lightPosition[4] = { 0.5, 5.0, 7.0, 1.0 };
	static GLfloat light0Position[4] = { 5.0f, 5.0f, 10.0f, 1.0f };
	glLightfv(GL_LIGHT0, GL_POSITION, light0Position);
}

void CybervisionViewer::resizeGL(int w, int h){
	// setup viewport, projection etc.:
	glViewport(0, 0, w, h);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(90,(float)w/(float)h,1,  1000000);
	glMatrixMode(GL_MODELVIEW);
}

void CybervisionViewer::paintGL(){
	// draw the scene:
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glLoadIdentity();
	glTranslatef(vpTranslation.x(), vpTranslation.y(), vpTranslation.z());
	glRotatef(vpRotation.x(), 1.0, 0.0, 0.0);
	glRotatef(vpRotation.y(), 0.0, 1.0, 0.0);
	glRotatef(vpRotation.z(), 0.0, 0.0, 1.0);

	{
		QMutexLocker lock(&surfaceMutex);
		surface.glDraw();
	}
}

float CybervisionViewer::normalizeAngle(float angle) const{
	if(angle>360.0F)
		angle-=floor(angle/360.0F)*360.0F;
	if(angle<-360.0F)
		angle-=floor(angle/360.0F)*360.0F;
	return angle;
}

void CybervisionViewer::mousePressEvent(QMouseEvent *event){
	lastMousePos= event->pos();
}

void CybervisionViewer::mouseMoveEvent(QMouseEvent *event){
	//event->
	int dx= event->pos().x()-lastMousePos.x(), dy=event->pos().y()-lastMousePos.y();

	if(event->modifiers()==Qt::NoModifier){
		//Rotation
		if (event->buttons() & Qt::LeftButton) {
			vpRotation.setX(normalizeAngle(vpRotation.x() + dy));
			vpRotation.setY(normalizeAngle(vpRotation.y() + dx));
			updateGL();
		} else if (event->buttons() & Qt::RightButton) {
			vpRotation.setX(normalizeAngle(vpRotation.x() + dy));
			vpRotation.setZ(normalizeAngle(vpRotation.z() + dx));
			updateGL();
		}
	}else if(event->modifiers()==Qt::ShiftModifier){
		//Translation
		if (event->buttons() & Qt::LeftButton) {
			vpTranslation.setX(vpTranslation.x() + dx/10.0F);
			vpTranslation.setY(vpTranslation.y() - dy/10.0F);
			updateGL();
		} else if (event->buttons() & Qt::RightButton) {
			vpTranslation.setX(vpTranslation.x() + dx/10.0F);
			vpTranslation.setZ(vpTranslation.z() - dy/10.0F);
			updateGL();
		}
	}
	lastMousePos = event->pos();
}
