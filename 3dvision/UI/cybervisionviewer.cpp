#include "cybervisionviewer.h"

#include <QMutexLocker>
#include <QMouseEvent>

#define _USE_MATH_DEFINES
#include <cmath>

CybervisionViewer::CybervisionViewer(QWidget *parent): QGLWidget(parent){
	vpTranslation= QVector3D(0,0,-10);
}


void CybervisionViewer::setPoints3D(const QList<QVector3D>&points){
	{
		QMutexLocker lock(&pointsMutex);
		this->points= points;
	}
	updateGL();
}

//OpenGL-specific stuff

void CybervisionViewer::initializeGL(){
	// Set up the rendering context, define display lists etc.:
	glClearColor(0.0, 0.0, 0.0, 0.0);
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
	int side = qMin(w, h);
	//glViewport((w-side)/2, (h-side)/2, side, side);
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
	glRotatef(vpRotation.x(), 1.0, 0.0, 0.0);
	glRotatef(vpRotation.y(), 0.0, 1.0, 0.0);
	glRotatef(vpRotation.z(), 0.0, 0.0, 1.0);
	glTranslatef(vpTranslation.x(), vpTranslation.y(), vpTranslation.z());


	glBegin(GL_LINES);
	glVertex3f(10.0e-1f, 10.0e-1f, 0.0e-1f); // origin of the FIRST line
	glVertex3f(20.0e-1f, 14.0e-1f, 5.0e-1f); // ending point of the FIRST line
	glVertex3f(12.0e-1f, 17.0e-1f, 10.0e-1f); // origin of the SECOND line
	glVertex3f(24.0e-1f, 12.0e-1f, 5.0e-1f); // ending point of the SECOND line
	glEnd( );

	{
		QMutexLocker lock(&pointsMutex);
		glBegin(GL_POLYGON);
		for(QList<QVector3D>::const_iterator it= points.begin();it!=points.end();it++){
			glVertex3f(it->x(),it->y(),it->z()*5e6F);
		}
		glEnd();
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
