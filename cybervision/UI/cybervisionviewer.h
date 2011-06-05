#ifndef CYBERVISIONVIEWER_H
#define CYBERVISIONVIEWER_H

#include <QGLWidget>
#include <QList>
#include <QVector3D>
#include <QMutex>

#include <Reconstruction/surface.h>

class CybervisionViewer : public QGLWidget{
	Q_OBJECT
public:
	enum MouseMode{MOUSE_ROTATION,MOUSE_PANNING};
protected:
	QMutex surfaceMutex;
	cybervision::Surface surface;

	//Viewport configuration
	QVector3D vpRotation,vpTranslation;
	QPoint lastMousePos;
	MouseMode mouseMode;
	bool showGrid;

	//Opengl constants
	float glFarPlane,glNearPlane,glAspectRatio,glFOV;

	enum Show_Planes{ SHOW_NONE=0,SHOW_FRONT=1,SHOW_BACK=1<<2,SHOW_LEFT=1<<3,SHOW_RIGHT=1<<4,SHOW_TOP=1<<5,SHOW_BOTTOM=1<<6 };

	//Grid functions
	void drawGrid();
	//Returns the optimal scale step for the min/max value pair
	qreal getOptimalGridStep(qreal min,qreal max) const;
	//Returns the optimal set of grid planes
	Show_Planes getOptimalGridPlanes()const;
public:
	CybervisionViewer(QWidget *parent);


	//Getters/setters
	void setSurface3D(const cybervision::Surface&);
	void setMouseMode(MouseMode mouseMode);
	void setShowGrid(bool show);
	const cybervision::Surface& getSurface3D()const;
protected:
	//Inherited opengl stuff
	void initializeGL();
	void resizeGL(int w, int h);
	void paintGL();


	//Rotation/movement with mouse
	float normalizeAngle(float angle)const;
	void mousePressEvent(QMouseEvent *event);
	void mouseMoveEvent(QMouseEvent *event);
};

#endif // CYBERVISIONVIEWER_H
