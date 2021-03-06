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
	enum TextureMode{TEXTURE_NONE,TEXTURE_1,TEXTURE_2};
protected:
	QMutex surfaceMutex;
	cybervision::Surface surface;
	GLuint textures[2];

	//Viewport configuration
	QVector3D vpRotation,vpTranslation;
	QPoint lastMousePos,clickMousePos;
	MouseMode mouseMode;
	int drawingCrossSectionLine;
	TextureMode textureMode;
	bool showGrid;

	//Opengl constants
	float glFarPlane,glNearPlane,glAspectRatio,glFOV,glViewportWidth,glViewportHeight;

	//Corners. Uppercase letter means max, lowercase means min => x= min_x, X=max_x etc.
	enum Corner{ CORNER_NONE,CORNER_xyz,CORNER_xyZ,CORNER_xYz,CORNER_xYZ,CORNER_Xyz,CORNER_XyZ,CORNER_XYz,CORNER_XYZ };

	//Grid functions
	//Draw the grid
	void drawGrid();
	//Returns the optimal scale step for the min/max value pair
	qreal getOptimalGridStep(qreal min,qreal max) const;
	//Returns the best visible corner
	Corner getOptimalCorner(const QVector3D& min,const QVector3D& max)const;

	//Point selection stuff
	//Selected point
	QVector3D clickLocation;
	QList<QPair<QVector3D,QVector3D> > crossSectionLines;
	//Click detection
	QVector3D getClickLocation(const QPointF&);
	void drawPoint(const QVector3D&)const;
	//Cross-section line
	void drawLine(const QVector3D& start,const QVector3D& end,bool lineSelected=false)const;

	//Axes direction widget
	//Draw the axes
	void drawAxesWidget();
public:
	CybervisionViewer(QWidget *parent);

	//Getters/setters
	void setSurface3D(const cybervision::Surface&);
	void setMouseMode(MouseMode mouseMode);
	void setTextureMode(TextureMode textureMode);
	void setShowGrid(bool show);
	void setDrawCrossSectionLine(int lineId=-1);
	const cybervision::Surface& getSurface3D()const;
	QMutex& getSurfaceMutex();
	QVector3D getSelectedPoint() const;
	QPair<QVector3D,QVector3D> getCrossSectionLine(int lineId)const;
protected:
	//Inherited opengl stuff
	void initializeGL();
	void resizeGL(int w, int h);
	void paintGL();

	//Rotation/movement with mouse
	float normalizeAngle(float angle)const;
	void mousePressEvent(QMouseEvent *event);
	void mouseReleaseEvent(QMouseEvent *event);
	void mouseMoveEvent(QMouseEvent *event);
signals:
	void selectedPointUpdated(QVector3D);
	void crossSectionLineChanged(QVector3D,QVector3D,int lineId);
};

#endif // CYBERVISIONVIEWER_H
