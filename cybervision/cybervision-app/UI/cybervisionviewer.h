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
	QPoint lastMousePos;
	MouseMode mouseMode;
	TextureMode textureMode;
	bool showGrid;

	//Opengl constants
	float glFarPlane,glNearPlane,glAspectRatio,glFOV;

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
	//Click detection
	QVector3D getClickLocation(const QPointF&)const;
	void drawPoint(const QVector3D&)const;
public:
	CybervisionViewer(QWidget *parent);

	//Getters/setters
	void setSurface3D(const cybervision::Surface&);
	void setMouseMode(MouseMode mouseMode);
	void setTextureMode(TextureMode textureMode);
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
