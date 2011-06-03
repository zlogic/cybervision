#include "cybervisionviewer.h"

#include <Reconstruction/options.h>

#include <QMutexLocker>
#include <QMouseEvent>
#include <QThread>

#define _USE_MATH_DEFINES
#include <cmath>

#include <GL/glext.h>


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


	//Line smoothing
	glEnable(GL_LINE_SMOOTH);
	glEnable (GL_BLEND);
	glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glHint (GL_LINE_SMOOTH_HINT, GL_NICEST);

	//static GLfloat lightPosition[4] = { 0.5, 5.0, 7.0, 1.0 };
	static GLfloat light0Position[4] = { 5.0f, 5.0f, 10.0f, 1.0f };
	static GLfloat light0Ambiance[4] = { 0.2f, 0.2f, 0.2f, 0.2f };
	glLightfv(GL_LIGHT0, GL_POSITION, light0Position);
	glLightfv(GL_LIGHT0,GL_AMBIENT,light0Ambiance);
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

	drawGrid(surface.getMaxDepth()*surface.getScale());
}
void CybervisionViewer::drawGrid(qreal z){
	//if(!surface.isOk())
	//	return;
	//glDisable(GL_DEPTH_TEST);
	glDisable(GL_LIGHTING);
	glColor3f(.0,.0,.0);
	glBegin(GL_LINES);
	for(int i=-5;i<=5;i++) {
		glVertex3f(i*2,-10,z);
		glVertex3f(i*2,11,z);
		glVertex3f(-10,i*2,z);
		glVertex3f(10,i*2,z);
	};
	glEnd();
	glEnable(GL_LIGHTING);

	//Change materials
	GLfloat backup_mat_specular[4];
	GLfloat backup_mat_shininess[1];
	GLfloat backup_mat_emission[4];
	glGetMaterialfv(GL_FRONT, GL_SPECULAR, backup_mat_specular);
	glGetMaterialfv(GL_FRONT, GL_SHININESS, backup_mat_shininess);
	glGetMaterialfv(GL_FRONT, GL_EMISSION, backup_mat_emission);

	static GLfloat mat_specular[] = { 1.0, 1.0, 1.0, 1.0 };
	static GLfloat mat_shininess[] = { 0.0 };
	static GLfloat mat_emission[] = { 0.0,0.0,0.0,1.0 };
	glMaterialfv(GL_FRONT, GL_SPECULAR, mat_specular);
	glMaterialfv(GL_FRONT, GL_SHININESS, mat_shininess);
	glMaterialfv(GL_FRONT, GL_EMISSION, mat_emission);

	//Draw checks
	for(int i=-5;i<=5;i+=1) {
		//Draw text
		QString str= QString("%1").arg(i*2);
		QFont font("Arial",64);
		font.setStyleStrategy(QFont::NoAntialias);
		int border= 2;
		QImage fontBmp(QFontMetrics(font).width(str)+border*2,QFontMetrics(font).height()+border*2,QImage::Format_ARGB32_Premultiplied);
		fontBmp.fill(Qt::transparent);
		QPainter painter(&fontBmp);
		painter.fillRect(0,0,fontBmp.width(),fontBmp.height(),QBrush(Qt::white));
		painter.setPen(Qt::black);
		painter.setFont(font);
		painter.drawText(QRect(border,border,fontBmp.width()-border*2,fontBmp.height()-border*2),str);

		//Create texture
		glEnable(GL_TEXTURE_2D);
		GLuint texture= bindTexture(fontBmp);

		//Trilinear filter
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, 30);


		glBegin(GL_QUADS);

		//Create text surface & texture
		qreal divider=70;
		glNormal3d(0, 0, 1);
		glTexCoord2f(0, 0);
		glVertex3f(-fontBmp.width()/(2*divider)+i*2, 12, z);
		glTexCoord2f(1, 0);
		glVertex3f(fontBmp.width()/(2*divider)+i*2, 12, z);
		glTexCoord2f(1, 1);
		glVertex3f(fontBmp.width()/(2*divider)+i*2, fontBmp.height()/divider+12, z);
		glTexCoord2f(0, 1);
		glVertex3f(-fontBmp.width()/(2*divider)+i*2, fontBmp.height()/divider+12, z);
		glEnd();

		deleteTexture(texture);
		/*
		qglColor(QColor(0,0,0));
		renderText(i*2,11.5,0,str,font);
		*/
		glDisable(GL_TEXTURE_2D);
	}

	glMaterialfv(GL_FRONT, GL_SPECULAR, backup_mat_specular);
	glMaterialfv(GL_FRONT, GL_SHININESS, backup_mat_shininess);
	glMaterialfv(GL_FRONT, GL_EMISSION, backup_mat_emission);

	//glEnable(GL_DEPTH_TEST);
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
