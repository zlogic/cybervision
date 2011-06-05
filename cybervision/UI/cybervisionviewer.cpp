#include "cybervisionviewer.h"

#include <Reconstruction/options.h>

#include <QMutexLocker>
#include <QMouseEvent>
#include <QThread>

#define _USE_MATH_DEFINES
#include <cmath>

#include <QMatrix4x4>
#include <GL/glext.h>


CybervisionViewer::CybervisionViewer(QWidget *parent): QGLWidget(parent){
	vpTranslation= QVector3D(0,0,-15);
	this->mouseMode= MOUSE_ROTATION;
	this->showGrid= false;
	glFarPlane= 1000000;
	glNearPlane= 1;
	glAspectRatio= 1;
	glFOV= 90;
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

void CybervisionViewer::setShowGrid(bool show){
	showGrid= show;
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


	//Line smoothing
	/*
 glEnable(GL_LINE_SMOOTH);
 glEnable (GL_BLEND);
 glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
 glHint (GL_LINE_SMOOTH_HINT, GL_NICEST);
 */

	//static GLfloat lightPosition[4] = { 0.5, 5.0, 7.0, 1.0 };
	static GLfloat light0Position[4] = { 5.0f, 5.0f, 10.0f, 1.0f };
	static GLfloat light0Ambiance[4] = { 0.2f, 0.2f, 0.2f, 0.2f };
	glLightfv(GL_LIGHT0, GL_POSITION, light0Position);
	glLightfv(GL_LIGHT0,GL_AMBIENT,light0Ambiance);
}

void CybervisionViewer::resizeGL(int w, int h){
	// setup viewport, projection etc.:
	glAspectRatio= (float)w/(float)h;
	glViewport(0, 0, w, h);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(glFOV,glAspectRatio,glNearPlane,glFarPlane);
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
		drawGrid();
	}
}
void CybervisionViewer::drawGrid(){
	if(!surface.isOk() || !showGrid)
		return;

	//Calculate grid steps
	qreal step_x= getOptimalGridStep(surface.getImageSize().left(),surface.getImageSize().right());
	qreal step_y= getOptimalGridStep(surface.getImageSize().top(),surface.getImageSize().bottom());
	qreal step_z= getOptimalGridStep(surface.getMinDepth(),surface.getMaxDepth());
	qreal step_xy= qMax(step_x,step_y);
	step_x= step_xy, step_y= step_xy;
	int min_x= floor(surface.getImageSize().left()/step_x);
	int max_x= ceil(surface.getImageSize().right()/step_x);
	int min_y= floor(surface.getImageSize().top()/step_y);
	int max_y= ceil(surface.getImageSize().bottom()/step_y);
	int min_z= floor(surface.getMinDepth()/step_z);
	int max_z= ceil(surface.getMaxDepth()/step_z);

	//glDisable(GL_DEPTH_TEST);

	//Get best coordinate pair
	Show_Planes selected_planes= getOptimalGridPlanes();

	//Draw grid
	glDisable(GL_LIGHTING);
	glColor3f(.0,.0,.0);
	glBegin(GL_LINES);
	for(int i=min_x;i<=max_x;i++) {
		qreal x= step_x*i*surface.getScale();
		if((selected_planes & SHOW_FRONT)){
			glVertex3f(x,min_y*step_y*surface.getScale()-1,max_z*step_z*surface.getScale());
			glVertex3f(x,max_y*step_y*surface.getScale(),max_z*step_z*surface.getScale());
		}
		if((selected_planes & SHOW_BACK)){
			glVertex3f(x,min_y*step_y*surface.getScale()-1,min_z*step_z*surface.getScale());
			glVertex3f(x,max_y*step_y*surface.getScale(),min_z*step_z*surface.getScale());
		}
		if((selected_planes & SHOW_TOP)){
			glVertex3f(x,max_y*step_y*surface.getScale(),min_z*step_z*surface.getScale()-1);
			glVertex3f(x,max_y*step_y*surface.getScale(),max_z*step_z*surface.getScale());
		}
		if((selected_planes & SHOW_BOTTOM)){
			glVertex3f(x,min_y*step_y*surface.getScale(),min_z*step_z*surface.getScale()-1);
			glVertex3f(x,min_y*step_y*surface.getScale(),max_z*step_z*surface.getScale());
		}
	}
	for(int i=min_y;i<=max_y;i++) {
		qreal y= step_y*i*surface.getScale();
		if((selected_planes & SHOW_FRONT)){
			glVertex3f(min_x*step_x*surface.getScale()-1,y,max_z*step_z*surface.getScale());
			glVertex3f(max_x*step_x*surface.getScale(),y,max_z*step_z*surface.getScale());
		}
		if((selected_planes & SHOW_BACK)){
			glVertex3f(min_x*step_x*surface.getScale()-1,y,min_z*step_z*surface.getScale());
			glVertex3f(max_x*step_x*surface.getScale(),y,min_z*step_z*surface.getScale());
		}
		if((selected_planes & SHOW_LEFT)){
			glVertex3f(min_x*step_x*surface.getScale(),y,min_z*step_z*surface.getScale()-1);
			glVertex3f(min_x*step_x*surface.getScale(),y,max_z*step_z*surface.getScale());
		}
		if((selected_planes & SHOW_RIGHT)){
			glVertex3f(max_x*step_x*surface.getScale(),y,min_z*step_z*surface.getScale()-1);
			glVertex3f(max_x*step_x*surface.getScale(),y,max_z*step_z*surface.getScale());
		}
	}
	for(int i=min_z;i<=max_z;i++) {
		qreal z= step_z*i*surface.getScale();
		if((selected_planes & SHOW_TOP)){
			glVertex3f(min_x*step_x*surface.getScale(),max_y*step_y*surface.getScale(),z);
			glVertex3f(max_x*step_x*surface.getScale()+1,max_y*step_y*surface.getScale(),z);
		}
		if((selected_planes & SHOW_BOTTOM)){
			glVertex3f(min_x*step_x*surface.getScale()-1,min_y*step_y*surface.getScale(),z);
			glVertex3f(max_x*step_x*surface.getScale(),min_y*step_y*surface.getScale(),z);
		}
		if((selected_planes & SHOW_LEFT)){
			glVertex3f(min_x*step_x*surface.getScale(),min_y*step_y*surface.getScale(),z);
			glVertex3f(min_x*step_x*surface.getScale(),max_y*step_y*surface.getScale()+1,z);
		}
		if((selected_planes & SHOW_RIGHT)){
			glVertex3f(max_x*step_x*surface.getScale(),min_y*step_y*surface.getScale()-1,z);
			glVertex3f(max_x*step_x*surface.getScale(),max_y*step_y*surface.getScale(),z);
		}
	}
	glEnd();
	glEnable(GL_LIGHTING);

	//Draw labels
	QFont font("Arial",8);
	qglColor(QColor(0,0,0,255));
	for(int i=min_x;i<=max_x;i++) {
		qreal x= step_x*i*surface.getScale();
		QString str= QString("%1").arg(i*step_x);
		if((selected_planes & SHOW_FRONT))
			renderText(x,min_y*step_y*surface.getScale()-2,max_z*step_z*surface.getScale(),str,font);
		if((selected_planes & SHOW_BACK))
			renderText(x,min_y*step_y*surface.getScale()-2,min_z*step_z*surface.getScale(),str,font);
		if((selected_planes & SHOW_TOP))
			renderText(x,max_y*step_y*surface.getScale(),min_z*step_z*surface.getScale()-2,str,font);
		if((selected_planes & SHOW_BOTTOM))
			renderText(x,min_y*step_y*surface.getScale(),min_z*step_z*surface.getScale()-2,str,font);
	}
	for(int i=min_y;i<=max_y;i++) {
		qreal y= step_y*i*surface.getScale();
		QString str= QString("%1").arg(i*step_y);
		if((selected_planes & SHOW_FRONT))
			renderText(min_x*step_x*surface.getScale()-2,y,max_z*step_z*surface.getScale(),str,font);
		if((selected_planes & SHOW_BACK))
			renderText(min_x*step_x*surface.getScale()-2,y,min_z*step_z*surface.getScale(),str,font);
		if((selected_planes & SHOW_LEFT))
			renderText(min_x*step_x*surface.getScale(),y,min_z*step_z*surface.getScale()-2,str,font);
		if((selected_planes & SHOW_RIGHT))
			renderText(max_x*step_x*surface.getScale(),y,min_z*step_z*surface.getScale()-2,str,font);
	}
	for(int i=min_z;i<=max_z;i++) {
		qreal z= step_z*i*surface.getScale();
		QString str= QString("%1").arg(i*step_z);
		if((selected_planes & SHOW_TOP))
			renderText(max_x*step_x*surface.getScale()+2,max_y*step_y*surface.getScale(),z,str,font);
		if((selected_planes & SHOW_BOTTOM))
			renderText(min_x*step_x*surface.getScale()-2,min_y*step_y*surface.getScale(),z,str,font);
		if((selected_planes & SHOW_LEFT))
			renderText(min_x*step_x*surface.getScale(),max_y*step_y*surface.getScale()+2,z,str,font);
		if((selected_planes & SHOW_RIGHT))
			renderText(max_x*step_x*surface.getScale(),min_y*step_y*surface.getScale()-2,z,str,font);
	}

	//glEnable(GL_DEPTH_TEST);
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

CybervisionViewer::Show_Planes CybervisionViewer::getOptimalGridPlanes() const{
	Show_Planes selected_planes= SHOW_NONE;
	{
		QMatrix4x4 transformationMatrix,projectionMatrix;
		transformationMatrix.setToIdentity();
		projectionMatrix.setToIdentity();
		transformationMatrix.translate(vpTranslation);
		transformationMatrix.rotate(vpRotation.x(),1,0,0);
		transformationMatrix.rotate(vpRotation.y(),0,1,0);
		transformationMatrix.rotate(vpRotation.z(),0,0,1);

		projectionMatrix.perspective(glFOV,glAspectRatio,glNearPlane,glFarPlane);


		QList<QPair<QVector3D,Show_Planes> > projections;
		projections<<QPair<QVector3D,Show_Planes>(QVector3D( 0, 0, 1),SHOW_FRONT);
		projections<<QPair<QVector3D,Show_Planes>(QVector3D( 0, 0,-1),SHOW_BACK);
		projections<<QPair<QVector3D,Show_Planes>(QVector3D( 1, 0, 0),SHOW_LEFT);
		projections<<QPair<QVector3D,Show_Planes>(QVector3D(-1, 0, 0),SHOW_RIGHT);
		projections<<QPair<QVector3D,Show_Planes>(QVector3D( 0,-1, 0),SHOW_TOP);
		projections<<QPair<QVector3D,Show_Planes>(QVector3D( 0, 1, 0),SHOW_BOTTOM);
		for(QList<QPair<QVector3D,Show_Planes> >::const_iterator it=projections.begin();it!=projections.end();it++){
			QVector3D start( 0, 0, 0);
			QVector3D end(it->first);
			start= projectionMatrix*((transformationMatrix*QVector4D(start,1)).toVector3DAffine());
			end= projectionMatrix*((transformationMatrix*QVector4D(end,1)).toVector3DAffine());
			QVector3D result= start-end;
			if(QVector3D::dotProduct(result,QVector3D(0,0,1))/result.length()>0.1)
				selected_planes= (CybervisionViewer::Show_Planes)(selected_planes|it->second);
		}
	}

	return selected_planes;
}
