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

	void renderCrossSection();
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
