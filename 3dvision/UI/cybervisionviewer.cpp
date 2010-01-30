#include "cybervisionviewer.h"

#include <QMutexLocker>

CybervisionViewer::CybervisionViewer(QWidget *parent): QGLWidget(parent){}


void CybervisionViewer::setPoints3D(const QList<QVector3D>&points){
	{
		QMutexLocker lock(&pointsMutex);
		this->points= points;
	}
	updateGL();
}


void CybervisionViewer::initializeGL(){
	// Set up the rendering context, define display lists etc.:
	glClearColor(0.8, 0.0, 0.0, 0.0);
	glEnable(GL_DEPTH_TEST);
}

void CybervisionViewer::resizeGL(int w, int h){
	// setup viewport, projection etc.:
	glViewport(0, 0, (GLint)w, (GLint)h);
	//glFrustum(...);
}

void CybervisionViewer::paintGL(){
	// draw the scene:
	/*
	...
	glRotatef(...);
	glMaterialfv(...);
	glBegin(GL_QUADS);
	glVertex3f(...);
	glVertex3f(...);
	...
	glEnd();
	...
	*/
}
