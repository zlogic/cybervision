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
public:
	CybervisionViewer(QWidget *parent);


	void setSurface3D(const cybervision::Surface&);
	void setMouseMode(MouseMode mouseMode);
	const cybervision::Surface& getSurface3D()const;
protected:
	//Inherited opengl stuff
	void initializeGL();
	void resizeGL(int w, int h);
	void paintGL();

	//Grid functions
	void drawGrid(qreal z);

	//Rotation/movement with mouse
	float normalizeAngle(float angle)const;
	void mousePressEvent(QMouseEvent *event);
	void mouseMoveEvent(QMouseEvent *event);
};

#endif // CYBERVISIONVIEWER_H
