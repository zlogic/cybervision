#include "cybervisionviewer.h"

#include <QMutexLocker>
#include <QMouseEvent>
#include <QThread>

#define _USE_MATH_DEFINES
#include <cmath>

#include <Reconstruction/options.h>

CybervisionViewer::CybervisionViewer(QWidget *parent): QGLWidget(parent){
	vpTranslation= QVector3D(0,0,-15);
	this->mouseMode= MOUSE_ROTATION;
}


void CybervisionViewer::setSurface3D(const cybervision::Surface& surface){
	{
		QMutexLocker lock(&surfaceMutex);
		this->surface= surface;
	}
	updateGL();
}

void CybervisionViewer::setMouseMode(MouseMode mouseMode){
	this->mouseMode= mouseMode;
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
	static GLfloat light0Ambiance[4] = { 0.2f, 0.2f, 0.2f, 0.2f };
	glLightfv(GL_LIGHT0, GL_POSITION, light0Position);
	glLightfv(GL_LIGHT0,GL_AMBIENT,light0Ambiance);

	if(cybervision::Options::renderShiny){
		static GLfloat mat_specular[] = { 1.0, 1.0, 1.0, 1.0 };
		static GLfloat mat_shininess[] = { 50.0 };
		glMaterialfv(GL_FRONT, GL_SPECULAR, mat_specular);
		glMaterialfv(GL_FRONT, GL_SHININESS, mat_shininess);
	}
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

	if(mouseMode==MOUSE_ROTATION){
		//Rotation
		if ((event->buttons() & Qt::LeftButton) && event->modifiers()==Qt::NoModifier) {
			vpRotation.setX(normalizeAngle(vpRotation.x() + dy));
			vpRotation.setY(normalizeAngle(vpRotation.y() + dx));
			updateGL();
		} else {
			vpRotation.setX(normalizeAngle(vpRotation.x() + dy));
			vpRotation.setZ(normalizeAngle(vpRotation.z() + dx));
			updateGL();
		}
	}else if(mouseMode==MOUSE_PANNING){
		//Translation
		if ((event->buttons() & Qt::LeftButton) && event->modifiers()==Qt::NoModifier) {
			vpTranslation.setX(vpTranslation.x() + dx/10.0F);
			vpTranslation.setY(vpTranslation.y() - dy/10.0F);
			updateGL();
		} else {
			vpTranslation.setX(vpTranslation.x() + dx/10.0F);
			vpTranslation.setZ(vpTranslation.z() - dy/10.0F);
			updateGL();
		}
	}
	lastMousePos = event->pos();
}
