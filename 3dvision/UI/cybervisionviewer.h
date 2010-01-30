#ifndef CYBERVISIONVIEWER_H
#define CYBERVISIONVIEWER_H

#include <QGLWidget>
#include <QList>
#include <QVector3D>
#include <QMutex>

class CybervisionViewer : public QGLWidget{
	Q_OBJECT
protected:
	QList<QVector3D> points;
	QMutex pointsMutex;
public:
	CybervisionViewer(QWidget *parent);

	void setPoints3D(const QList<QVector3D>&);

protected:
	//Inherited opengl stuff
	 void initializeGL();
	 void resizeGL(int w, int h);
	 void paintGL();
};

#endif // CYBERVISIONVIEWER_H
