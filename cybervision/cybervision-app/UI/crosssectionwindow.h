#ifndef CROSSSECTIONWINDOW_H
#define CROSSSECTIONWINDOW_H

#include <QDialog>
#include <QGraphicsScene>
#include <QGraphicsView>
#include <Reconstruction/crosssection.h>

namespace Ui {
    class CrossSectionWindow;
}


/*
 * Modified QGraphicsView which emits "resize" events
 */
class CybervisionCrosssectionGraphicsView : public QGraphicsView{
	Q_OBJECT
public:
	CybervisionCrosssectionGraphicsView(QWidget *parent = 0);
	CybervisionCrosssectionGraphicsView(QGraphicsScene *scene, QWidget *parent = 0);
protected:
	void resizeEvent(QResizeEvent *event);
signals:
	void resized();
};

/*
 * Modified QGraphicsScene with customized mouse interaction
 */
class CybervisionCrosssectionScene : public QGraphicsScene{
	Q_OBJECT
public:
	CybervisionCrosssectionScene(QObject *parent = 0);
	CybervisionCrosssectionScene(const QRectF &sceneRect, QObject *parent = 0);
	CybervisionCrosssectionScene(qreal x, qreal y, qreal width, qreal height, QObject *parent = 0);
	void updateSceneData(QList<QGraphicsItem*> measurementLines,const QRect &crossSectionArea);
protected:
	QPointF clickPos;
	QList<QGraphicsItem*> measurementLines;
	QRect crossSectionArea;
	void mouseMoveEvent(QGraphicsSceneMouseEvent *event);
	void mousePressEvent(QGraphicsSceneMouseEvent *event);
signals:
	void measurementLineDragged(qreal x,int id);
};


/*
 * Dialog window for cross-section viewer
 */
class CrossSectionWindow : public QDialog
{
    Q_OBJECT
protected:
public:
    explicit CrossSectionWindow(QWidget *parent = 0);
	~CrossSectionWindow();

	//Updates the displayed cross-section
	void updateCrossSection(const cybervision::CrossSection&);

	//Updates the cross-section image and stats labels
	void updateSurfaceStats();
protected:
	qreal measurementLinePos1,measurementLinePos2;
	qreal sceneScaleX;

	//Sends signal on close
	void closeEvent(QCloseEvent *event);

	//Updates widgets enabled/disabled/visible status
	void updateWidgetStatus();

	//Re-draws the cross-section
	void renderCrossSection();

	//Updates the measurement lines label with the latest height information
	void updateMeasurementLinesLabel();

	//Returns the optimal scale step for the min/max value pair. Imported from CybervisionViewer.
	qreal getOptimalGridStep(qreal min,qreal max) const;
private:
	Ui::CrossSectionWindow *ui;

	cybervision::CrossSection crossSection;

	CybervisionCrosssectionScene crossSectionScene;
signals:
	void closed();
private slots:
	void viewportResized();
	void on_roughnessPSpinBox_valueChanged(int arg1);
	void measurementLineDragged(qreal x,int id);
};

#endif // CROSSSECTIONWINDOW_H
