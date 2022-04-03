#ifndef CYBERVISIONVIEWER_H
#define CYBERVISIONVIEWER_H

#include <QList>
#include <QVector3D>
#include <QMutex>
#include <QPainter>

#include <Qt3DCore/QNode>
#include <Qt3DCore/QEntity>
#include <Qt3DCore/QTransform>
#include <Qt3DRender/QPaintedTextureImage>
#include <Qt3DCore/QGeometry>
#include <Qt3DCore/QBuffer>
#include <Qt3DRender/QScreenRayCaster>
#include <Qt3DExtras/Qt3DWindow>

#include <Reconstruction/surface.h>

class SurfaceTexture: public Qt3DRender::QPaintedTextureImage {
protected:
	QImage texture;
public:
	SurfaceTexture(Qt3DCore::QNode *parent = nullptr);
	void setImage(const QImage& image);
	void paint(QPainter *painter);
};

class CybervisionViewer: public Qt3DExtras::Qt3DWindow{
	Q_OBJECT
public:
	enum MouseMode{MOUSE_ROTATION,MOUSE_PANNING};
	enum TextureMode{TEXTURE_NONE,TEXTURE_1,TEXTURE_2};
protected:
	QMutex surfaceMutex;
	cybervision::Surface surface;
	Qt3DCore::QEntity *rootEntity,*surfaceEntity;
	SurfaceTexture *surfaceTexture;
	Qt3DCore::QTransform *axesTransform;
	Qt3DRender::QScreenRayCaster *clickDetector;

	//Viewport configuration
	bool mousePressed;
	QPoint lastMousePos,clickMousePos;
	MouseMode mouseMode;
	int drawingCrossSectionLine;
	TextureMode textureMode;
	bool showGrid;

	//Corners. Uppercase letter means max, lowercase means min => x= min_x, X=max_x etc.
	enum Corner{ CORNER_NONE,CORNER_xyz,CORNER_xyZ,CORNER_xYz,CORNER_xYZ,CORNER_Xyz,CORNER_XyZ,CORNER_XYz,CORNER_XYZ };

	//Surface functions
	void addSurfaceMaterial();
	void addSelectedPoint();
	void addCrossSectionLines();

	//Grid configuration
	QVector3D gridMin,gridMax;
	QMap<Corner,Qt3DCore::QEntity*> gridEntities;

	//Grid functions
	//Generate the grid
	void addGrid();
	//Returns the optimal scale step for the min/max value pair
	qreal getOptimalGridStep(qreal min,qreal max) const;
	//Returns the best visible corner
	Corner getOptimalCorner(const QVector3D& min,const QVector3D& max)const;
	//Update grid visibility
	void updateGrid();

	//Point selection stuff
	//Selected point
	QVector3D clickLocation;
	Qt3DCore::QEntity *selectedPointEntity;
	Qt3DCore::QTransform *selectedPointTransform;
	QList<Qt3DCore::QBuffer*> crossSectionLineEntities;
	QList<QPair<QVector3D,QVector3D> > crossSectionLines;
	//Click detection
	QVector3D getClickLocation(const QPointF&);
	void drawPoint(const QVector3D&)const;
	//Cross-section line updates
	void updateCrossSectionLines();

public:
	CybervisionViewer();
	~CybervisionViewer();

	QPixmap getScreenshot()const;

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
	//3D Scene initialization stuff
	void initializeScene();
	void initializeAxesWidget();

	Qt3DCore::QGeometry* createLines(const QVector<QVector3D>& lines,Qt3DCore::QNode* parent=nullptr);

	//Rotation/movement with mouse
	void mousePressEvent(QMouseEvent *event);
	void mouseReleaseEvent(QMouseEvent *event);
	void mouseMoveEvent(QMouseEvent *event);
signals:
	void selectedPointUpdated(QVector3D);
	void crossSectionLineChanged(int lineId,QVector3D,QVector3D);

private slots:
	void cameraUpdated();
	void hitsChanged(const Qt3DRender::QAbstractRayCaster::Hits &hits);
};

#endif // CYBERVISIONVIEWER_H
