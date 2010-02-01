#ifndef CYBERVISIONVIEWER_H
#define CYBERVISIONVIEWER_H

#include <QGLWidget>
#include <QList>
#include <QVector3D>
#include <QMutex>

#include <Reconstruction/sculptor.h>

class CybervisionViewer : public QGLWidget{
	Q_OBJECT
protected:
	QMutex surfaceMutex;
	cybervision::Sculptor surface;

	//Viewport configuration
	QVector3D vpRotation,vpTranslation;
	QPoint lastMousePos;
public:
	CybervisionViewer(QWidget *parent);

	void setPoints3D(const QList<QVector3D>&);

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
