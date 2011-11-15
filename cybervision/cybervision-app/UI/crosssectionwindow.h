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
	void updateSceneData(const QList<QGraphicsItem*>&,QGraphicsItem *movableCrossSection,const QRect &crossSectionArea);
protected:
	QPointF clickPos,itemPos;
	QList<QGraphicsItem*> measurementLines;
	QGraphicsItem *movableCrossSection;
	QRect crossSectionArea;
	void mouseMoveEvent(QGraphicsSceneMouseEvent *event);
	void mousePressEvent(QGraphicsSceneMouseEvent *event);
signals:
	void measurementLineMoved(qreal x,int id);
	void crossSectionMoved(qreal x);
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
	void updateCrossSection(const cybervision::CrossSection&,int crossSectionId);

	//Updates the cross-section image and stats labels
	void updateCrosssectionStats();
private:
	Ui::CrossSectionWindow *ui;
protected:
	QList<cybervision::CrossSection> crossSections;
	QList<cybervision::CrossSection>::const_iterator nonMovableCrosssection;

	CybervisionCrosssectionScene crossSectionScene;

	qreal measurementLinePos1,measurementLinePos2;
	qreal movableCrossSectionPos;
	qreal sceneScaleX;

	//Sends signal on close
	void closeEvent(QCloseEvent *event);

	//Updates widgets enabled/disabled/visible status
	void updateWidgetStatus();

	//Re-draws the cross-section
	void renderCrossSections();

	//Updates the measurement lines label with the latest height information
	void updateMeasurementLinesLabel();

	//Returns the optimal scale step for the min/max value pair. Imported from CybervisionViewer.
	qreal getOptimalGridStep(qreal min,qreal max) const;
signals:
	void closed();
private slots:
	void viewportResized();
	void on_roughnessPSpinBoxPrimary_valueChanged(int arg1);
	void on_roughnessPSpinBoxSecondary_valueChanged(int arg1);
	void measurementLineMoved(qreal x,int id);
	void crossSectionMoved(qreal x);
};

#endif // CROSSSECTIONWINDOW_H
