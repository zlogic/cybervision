#include "cybervisionviewer.h"

#include <Reconstruction/options.h>

#include <QMutexLocker>
#include <QMouseEvent>
#include <QThread>

#define _USE_MATH_DEFINES
#include <cmath>
#include <limits>

#include <QMatrix4x4>
#include <GL/glext.h>
#include <GL/glu.h>


CybervisionViewer::CybervisionViewer(QWidget *parent): QGLWidget(parent){
	vpTranslation= QVector3D(0,0,-15);
	this->mouseMode= MOUSE_ROTATION;
	this->textureMode= TEXTURE_1;
	this->showGrid= false;
	glFarPlane= 1000000;
	glNearPlane= 1;
	glAspectRatio= 1;
	glFOV= 90;
	clickLocation= QVector3D(std::numeric_limits<qreal>::infinity(),std::numeric_limits<qreal>::infinity(),std::numeric_limits<qreal>::infinity());
}


void CybervisionViewer::setSurface3D(const cybervision::Surface& surface){
	clickLocation= QVector3D(std::numeric_limits<qreal>::infinity(),std::numeric_limits<qreal>::infinity(),std::numeric_limits<qreal>::infinity());
	{
		QMutexLocker lock(&surfaceMutex);
		this->surface= surface;
		//Convert textures and delete sources since we don't need them anymore
		deleteTexture(textures[0]);
		deleteTexture(textures[1]);
		glDeleteTextures(2,textures);
		glGenTextures(2,textures);
		textures[0]= bindTexture(this->surface.getTexture1());
		textures[1]= bindTexture(this->surface.getTexture2());
		//this->surface.setTextures(QImage(),QImage());
	}
	updateGL();
}

void CybervisionViewer::setMouseMode(MouseMode mouseMode){
	this->mouseMode= mouseMode;
}


void CybervisionViewer::setTextureMode(TextureMode textureMode){
	this->textureMode= textureMode;
	updateGL();
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
	//glEnable(GL_CULL_FACE);
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
	float modelTwoside[] = {GL_TRUE};
	glLightModelfv(GL_LIGHT_MODEL_TWO_SIDE, modelTwoside);

	//Texture options
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT );
	glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT );
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
		if(textureMode!=TEXTURE_NONE)
			glEnable(GL_TEXTURE_2D);
		if(textureMode==TEXTURE_1)
			glBindTexture(GL_TEXTURE_2D, textures[0]);
		if(textureMode==TEXTURE_2)
			glBindTexture(GL_TEXTURE_2D, textures[1]);

		surface.glDraw();

		glDisable(GL_TEXTURE_2D);

		drawGrid();

		//Draw click point
		drawPoint(clickLocation);
	}
}
void CybervisionViewer::drawPoint(const QVector3D& point)const{
	if(point.x()==std::numeric_limits<qreal>::infinity() || point.y()==std::numeric_limits<qreal>::infinity() || point.z()==std::numeric_limits<qreal>::infinity())
		return;
	glDisable(GL_DEPTH_TEST);
	glDisable(GL_LIGHTING);
	glColor3f(1,.0,.0);
	if(cybervision::Options::pointDrawingMode==cybervision::Options::POINT_DRAW_AS_3DCIRCLE){
		double radius=.08;
		int num_segments=36;
		glBegin(GL_POLYGON);
		for(int i=0;i<num_segments;i++){
			double angle = i*M_PI*2/(double)num_segments;
			glVertex3f(point.x()+cos(angle)*radius,point.y()+sin(angle)*radius,point.z());
		}
		glEnd();
	}else if(cybervision::Options::pointDrawingMode==cybervision::Options::POINT_DRAW_AS_POINT){
		glBegin(GL_POINTS);
		glVertex3f(point.x(),point.y(),point.z());
		glEnd();
	}
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_LIGHTING);
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
	Corner selected_corner= getOptimalCorner(
				QVector3D(min_x*step_x*surface.getScale(),min_y*step_y*surface.getScale(),min_z*step_z*surface.getScale()),
				QVector3D(max_x*step_x*surface.getScale(),max_y*step_y*surface.getScale(),max_z*step_z*surface.getScale())
				);

	//Draw grid
	glDisable(GL_LIGHTING);
	glColor3f(.0,.0,.0);
	glBegin(GL_LINES);
	for(int i=min_x;i<=max_x;i++) {
		qreal x= step_x*i*surface.getScale();
		if((selected_corner == CORNER_xyZ) || (selected_corner == CORNER_XyZ)){
			glVertex3f(x,min_y*step_y*surface.getScale(),max_z*step_z*surface.getScale());
			glVertex3f(x,max_y*step_y*surface.getScale(),max_z*step_z*surface.getScale());
			glVertex3f(x,min_y*step_y*surface.getScale(),min_z*step_z*surface.getScale());
			glVertex3f(x,min_y*step_y*surface.getScale(),max_z*step_z*surface.getScale());

			glVertex3f(x,min_y*step_y*surface.getScale(),max_z*step_z*surface.getScale());
			glVertex3f(x,min_y*step_y*surface.getScale()-1,max_z*step_z*surface.getScale()+1);
		}
		if((selected_corner == CORNER_xYZ) || (selected_corner == CORNER_XYZ)){
			glVertex3f(x,min_y*step_y*surface.getScale(),max_z*step_z*surface.getScale());
			glVertex3f(x,max_y*step_y*surface.getScale(),max_z*step_z*surface.getScale());
			glVertex3f(x,max_y*step_y*surface.getScale(),min_z*step_z*surface.getScale());
			glVertex3f(x,max_y*step_y*surface.getScale(),max_z*step_z*surface.getScale());

			glVertex3f(x,max_y*step_y*surface.getScale(),max_z*step_z*surface.getScale());
			glVertex3f(x,max_y*step_y*surface.getScale()+1,max_z*step_z*surface.getScale()+1);
		}
		if((selected_corner == CORNER_xyz) || (selected_corner == CORNER_Xyz)){
			glVertex3f(x,min_y*step_y*surface.getScale(),min_z*step_z*surface.getScale());
			glVertex3f(x,max_y*step_y*surface.getScale(),min_z*step_z*surface.getScale());
			glVertex3f(x,min_y*step_y*surface.getScale(),min_z*step_z*surface.getScale());
			glVertex3f(x,min_y*step_y*surface.getScale(),max_z*step_z*surface.getScale());

			glVertex3f(x,min_y*step_y*surface.getScale(),min_z*step_z*surface.getScale());
			glVertex3f(x,min_y*step_y*surface.getScale()-1,min_z*step_z*surface.getScale()-1);
		}
		if((selected_corner == CORNER_xYz) || (selected_corner == CORNER_XYz)){
			glVertex3f(x,min_y*step_y*surface.getScale(),min_z*step_z*surface.getScale());
			glVertex3f(x,max_y*step_y*surface.getScale(),min_z*step_z*surface.getScale());
			glVertex3f(x,max_y*step_y*surface.getScale(),min_z*step_z*surface.getScale());
			glVertex3f(x,max_y*step_y*surface.getScale(),max_z*step_z*surface.getScale());

			glVertex3f(x,max_y*step_y*surface.getScale(),min_z*step_z*surface.getScale());
			glVertex3f(x,max_y*step_y*surface.getScale()+1,min_z*step_z*surface.getScale()-1);
		}
	}
	for(int i=min_y;i<=max_y;i++) {
		qreal y= step_y*i*surface.getScale();
		if((selected_corner == CORNER_xyZ) || (selected_corner == CORNER_xYZ)){
			glVertex3f(min_x*step_x*surface.getScale(),y,max_z*step_z*surface.getScale());
			glVertex3f(max_x*step_x*surface.getScale(),y,max_z*step_z*surface.getScale());
			glVertex3f(min_x*step_x*surface.getScale(),y,min_z*step_z*surface.getScale());
			glVertex3f(min_x*step_x*surface.getScale(),y,max_z*step_z*surface.getScale());

			glVertex3f(max_x*step_x*surface.getScale(),y,max_z*step_z*surface.getScale());
			glVertex3f(max_x*step_x*surface.getScale()+1,y,max_z*step_z*surface.getScale()+1);
		}
		if((selected_corner == CORNER_XyZ) || (selected_corner == CORNER_XYZ)){
			glVertex3f(min_x*step_x*surface.getScale(),y,max_z*step_z*surface.getScale());
			glVertex3f(max_x*step_x*surface.getScale(),y,max_z*step_z*surface.getScale());
			glVertex3f(max_x*step_x*surface.getScale(),y,min_z*step_z*surface.getScale());
			glVertex3f(max_x*step_x*surface.getScale(),y,max_z*step_z*surface.getScale());

			glVertex3f(min_x*step_x*surface.getScale(),y,max_z*step_z*surface.getScale());
			glVertex3f(min_x*step_x*surface.getScale()-1,y,max_z*step_z*surface.getScale()+1);
		}
		if((selected_corner == CORNER_xyz) || (selected_corner == CORNER_xYz)){
			glVertex3f(min_x*step_x*surface.getScale(),y,min_z*step_z*surface.getScale());
			glVertex3f(max_x*step_x*surface.getScale(),y,min_z*step_z*surface.getScale());
			glVertex3f(min_x*step_x*surface.getScale(),y,min_z*step_z*surface.getScale());
			glVertex3f(min_x*step_x*surface.getScale(),y,max_z*step_z*surface.getScale());

			glVertex3f(max_x*step_x*surface.getScale(),y,min_z*step_z*surface.getScale());
			glVertex3f(max_x*step_x*surface.getScale()+1,y,min_z*step_z*surface.getScale()-1);
		}
		if((selected_corner == CORNER_Xyz) || (selected_corner == CORNER_XYz)){
			glVertex3f(min_x*step_x*surface.getScale(),y,min_z*step_z*surface.getScale());
			glVertex3f(max_x*step_x*surface.getScale(),y,min_z*step_z*surface.getScale());
			glVertex3f(max_x*step_x*surface.getScale(),y,min_z*step_z*surface.getScale());
			glVertex3f(max_x*step_x*surface.getScale(),y,max_z*step_z*surface.getScale());

			glVertex3f(min_x*step_x*surface.getScale(),y,min_z*step_z*surface.getScale());
			glVertex3f(min_x*step_x*surface.getScale()-1,y,min_z*step_z*surface.getScale()-1);
		}
	}
	for(int i=min_z;i<=max_z;i++) {
		qreal z= step_z*i*surface.getScale();
		if((selected_corner == CORNER_xyZ) || (selected_corner == CORNER_xyz)){
			glVertex3f(min_x*step_x*surface.getScale(),min_y*step_y*surface.getScale(),z);
			glVertex3f(max_x*step_x*surface.getScale(),min_y*step_y*surface.getScale(),z);
			glVertex3f(min_x*step_x*surface.getScale(),min_y*step_y*surface.getScale(),z);
			glVertex3f(min_x*step_x*surface.getScale(),max_y*step_y*surface.getScale(),z);

			glVertex3f(min_x*step_x*surface.getScale(),min_y*step_y*surface.getScale(),z);
			glVertex3f(min_x*step_x*surface.getScale()-1,min_y*step_y*surface.getScale()-1,z);
		}
		if((selected_corner == CORNER_xYZ) || (selected_corner == CORNER_xYz)){
			glVertex3f(min_x*step_x*surface.getScale(),max_y*step_y*surface.getScale(),z);
			glVertex3f(max_x*step_x*surface.getScale(),max_y*step_y*surface.getScale(),z);
			glVertex3f(min_x*step_x*surface.getScale(),min_y*step_y*surface.getScale(),z);
			glVertex3f(min_x*step_x*surface.getScale(),max_y*step_y*surface.getScale(),z);

			glVertex3f(min_x*step_x*surface.getScale(),max_y*step_y*surface.getScale(),z);
			glVertex3f(min_x*step_x*surface.getScale()-1,max_y*step_y*surface.getScale()+1,z);
		}
		if((selected_corner == CORNER_XyZ) || (selected_corner == CORNER_Xyz)){
			glVertex3f(min_x*step_x*surface.getScale(),min_y*step_y*surface.getScale(),z);
			glVertex3f(max_x*step_x*surface.getScale(),min_y*step_y*surface.getScale(),z);
			glVertex3f(max_x*step_x*surface.getScale(),min_y*step_y*surface.getScale(),z);
			glVertex3f(max_x*step_x*surface.getScale(),max_y*step_y*surface.getScale(),z);

			glVertex3f(max_x*step_x*surface.getScale(),min_y*step_y*surface.getScale(),z);
			glVertex3f(max_x*step_x*surface.getScale()+1,min_y*step_y*surface.getScale()-1,z);
		}
		if((selected_corner == CORNER_XYZ) || (selected_corner == CORNER_XYz)){
			glVertex3f(min_x*step_x*surface.getScale(),max_y*step_y*surface.getScale(),z);
			glVertex3f(max_x*step_x*surface.getScale(),max_y*step_y*surface.getScale(),z);
			glVertex3f(max_x*step_x*surface.getScale(),min_y*step_y*surface.getScale(),z);
			glVertex3f(max_x*step_x*surface.getScale(),max_y*step_y*surface.getScale(),z);

			glVertex3f(max_x*step_x*surface.getScale(),max_y*step_y*surface.getScale(),z);
			glVertex3f(max_x*step_x*surface.getScale()+1,max_y*step_y*surface.getScale()+1,z);
		}

	}
	glEnd();
	glEnable(GL_LIGHTING);

	//Draw labels
	QFont font("Arial",7);
	qglColor(QColor(0,0,0,255));
	for(int i=min_x;i<=max_x;i++) {
		qreal x= step_x*i*surface.getScale();
		QString str;
		QTextStream stream(&str);
		stream.setRealNumberPrecision(1);
		stream.setRealNumberNotation(QTextStream::ScientificNotation);
		stream<<i*step_x;
		if((selected_corner == CORNER_xyZ) || (selected_corner == CORNER_XyZ))
			renderText(x,min_y*step_y*surface.getScale()-1,max_z*step_z*surface.getScale()+1,str,font);
		if((selected_corner == CORNER_xYZ) || (selected_corner == CORNER_XYZ))
			renderText(x,max_y*step_y*surface.getScale()+1,max_z*step_z*surface.getScale()+1,str,font);
		if((selected_corner == CORNER_xyz) || (selected_corner == CORNER_Xyz))
			renderText(x,min_y*step_y*surface.getScale()-1,min_z*step_z*surface.getScale()-1,str,font);
		if((selected_corner == CORNER_xYz) || (selected_corner == CORNER_XYz))
			renderText(x,max_y*step_y*surface.getScale()+1,min_z*step_z*surface.getScale()-1,str,font);
	}
	for(int i=min_y;i<=max_y;i++) {
		qreal y= step_y*i*surface.getScale();
		QString str;
		QTextStream stream(&str);
		stream.setRealNumberPrecision(1);
		stream.setRealNumberNotation(QTextStream::ScientificNotation);
		stream<<i*step_y;
		if((selected_corner == CORNER_xyZ) || (selected_corner == CORNER_xYZ))
			renderText(max_x*step_x*surface.getScale()+1,y,max_z*step_z*surface.getScale()+1,str,font);
		if((selected_corner == CORNER_XyZ) || (selected_corner == CORNER_XYZ))
			renderText(min_x*step_x*surface.getScale()-1,y,max_z*step_z*surface.getScale()+1,str,font);
		if((selected_corner == CORNER_xyz) || (selected_corner == CORNER_xYz))
			renderText(max_x*step_x*surface.getScale()+1,y,min_z*step_z*surface.getScale()-1,str,font);
		if((selected_corner == CORNER_Xyz) || (selected_corner == CORNER_XYz))
			renderText(min_x*step_x*surface.getScale()-1,y,min_z*step_z*surface.getScale()-1,str,font);
	}
	for(int i=min_z;i<=max_z;i++) {
		qreal z= step_z*i*surface.getScale();
		QString str;
		QTextStream stream(&str);
		stream.setRealNumberPrecision(1);
		stream.setRealNumberNotation(QTextStream::ScientificNotation);
		stream<<i*step_z;
		if((selected_corner == CORNER_xyZ) || (selected_corner == CORNER_xyz))
			renderText(min_x*step_x*surface.getScale()-1,min_y*step_y*surface.getScale()-1,z,str,font);
		if((selected_corner == CORNER_xYZ) || (selected_corner == CORNER_xYz))
			renderText(min_x*step_x*surface.getScale()-1,max_y*step_y*surface.getScale()+1,z,str,font);
		if((selected_corner == CORNER_XyZ) || (selected_corner == CORNER_Xyz))
			renderText(max_x*step_x*surface.getScale()+1,min_y*step_y*surface.getScale()-1,z,str,font);
		if((selected_corner == CORNER_XYZ) || (selected_corner == CORNER_XYz))
			renderText(max_x*step_x*surface.getScale()+1,max_y*step_y*surface.getScale()+1,z,str,font);
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

	//Detect click location
	clickLocation= getClickLocation(event->pos());
	updateGL();
}

QVector3D CybervisionViewer::getClickLocation(const QPointF& location) const{
	//See http://nehe.gamedev.net/article/using_gluunproject/16013/
	GLint viewport[4];
	GLdouble modelview[16];
	GLdouble projection[16];
	GLfloat winX, winY, winZ;
	GLdouble posX, posY, posZ;

	glGetDoublev(GL_MODELVIEW_MATRIX,modelview);
	glGetDoublev(GL_PROJECTION_MATRIX,projection);
	glGetIntegerv(GL_VIEWPORT,viewport);

	winX= (GLfloat)location.x();
	winY= (GLfloat)viewport[3]-(GLfloat)location.y();
	glReadPixels(int(winX),int(winY),1,1,GL_DEPTH_COMPONENT,GL_FLOAT,&winZ);

	gluUnProject(winX,winY,winZ,modelview,projection,viewport,&posX,&posY,&posZ);

	return QVector3D(posX,posY,posZ);
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

CybervisionViewer::Corner CybervisionViewer::getOptimalCorner(const QVector3D& min,const QVector3D& max) const{
	QMatrix4x4 transformationMatrix,projectionMatrix;
	transformationMatrix.setToIdentity();
	projectionMatrix.setToIdentity();
	transformationMatrix.translate(vpTranslation);
	transformationMatrix.rotate(vpRotation.x(),1,0,0);
	transformationMatrix.rotate(vpRotation.y(),0,1,0);
	transformationMatrix.rotate(vpRotation.z(),0,0,1);

	//projectionMatrix.perspective(glFOV,glAspectRatio,glNearPlane,glFarPlane);


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
	for(QList<QPair<QVector3D,Corner> >::const_iterator it=corners.begin();it!=corners.end();it++){
		QVector3D projection= projectionMatrix*((transformationMatrix*QVector4D(it->first,1)).toVector3DAffine());
		if(projection.z()>closestDistance){
			closestDistance= projection.z();
			closestCorner= it->second;
		}
	}
	return closestCorner;
}
