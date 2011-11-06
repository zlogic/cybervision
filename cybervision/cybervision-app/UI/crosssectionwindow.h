#ifndef CROSSSECTIONWINDOW_H
#define CROSSSECTIONWINDOW_H

#include <QDialog>
#include <QGraphicsScene>
#include <Reconstruction/crosssection.h>

namespace Ui {
    class CrossSectionWindow;
}

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
	//Sends signal on close
	void closeEvent(QCloseEvent *event);
	void resizeEvent(QResizeEvent *);

	//Updates widgets enabled/disabled/visible status
	void updateWidgetStatus();

	//Re-draws the cross-section
	void renderCrossSection();

	//Returns the optimal scale step for the min/max value pair. Imported from CybervisionViewer.
	qreal getOptimalGridStep(qreal min,qreal max) const;
private:
	Ui::CrossSectionWindow *ui;

	cybervision::CrossSection crossSection;

	QGraphicsScene crossSectionScene;
signals:
	void closed();
private slots:
	void on_crosssectionPSpinBox_valueChanged(int arg1);
};

#endif // CROSSSECTIONWINDOW_H
