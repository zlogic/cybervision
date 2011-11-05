#ifndef CROSSSECTIONWINDOW_H
#define CROSSSECTIONWINDOW_H

#include <QDialog>
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

	//Updates widgets enabled/disabled/visible status
	void updateWidgetStatus();
protected:
	//Sends signal on close
	void closeEvent(QCloseEvent *event);
private:
	Ui::CrossSectionWindow *ui;

	cybervision::CrossSection crossSection;

	//Updates the cross-section image and stats labels
	void updateSurfaceStats();
signals:
	void closed();
private slots:
	void on_crosssectionPSpinBox_valueChanged(int arg1);
};

#endif // CROSSSECTIONWINDOW_H
