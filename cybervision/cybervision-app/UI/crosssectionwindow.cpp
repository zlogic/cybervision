#include "crosssectionwindow.h"
#include "ui_crosssectionwindow.h"

#include <QCloseEvent>

CrossSectionWindow::CrossSectionWindow(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::CrossSectionWindow)
{
    ui->setupUi(this);
	updateSurfaceStats();
}

CrossSectionWindow::~CrossSectionWindow(){
	delete ui;
}

void CrossSectionWindow::updateCrossSection(const cybervision::CrossSection& crossSection){
	this->crossSection= crossSection;
	updateSurfaceStats();
}


void CrossSectionWindow::updateWidgetStatus(){
	ui->crosssectionPSpinBox->setVisible(crossSection.isOk());
	ui->crosssectionRoughnessParametersLabel->setVisible(crossSection.isOk());
}


void CrossSectionWindow::updateSurfaceStats(){
	crossSection.computeParams(ui->crosssectionPSpinBox->value());
	//Set the cross-section image
	QImage img= crossSection.renderCrossSection(ui->crosssectionImage->size());

	if(!crossSection.isOk()){
		img= QImage(0,0);
	}
	ui->crosssectionImage->setPixmap(QPixmap::fromImage(img));

	//Set the roughness parameters
	QString heightParamsString,stepParamsString;
	if(crossSection.isOk()){
		heightParamsString= QString(tr("Ra= %1 m\nRz= %2 m\nRmax= %3 m"))
				.arg(crossSection.getRoughnessRa())
				.arg(crossSection.getRoughnessRz())
				.arg(crossSection.getRoughnessRmax());
		stepParamsString= QString(tr("S= %1 m\nSm= %2 m\ntp= %3"))
				.arg(crossSection.getRoughnessS())
				.arg(crossSection.getRoughnessSm())
				.arg(crossSection.getRoughnessTp());

	}else{
		heightParamsString= "";
		stepParamsString= "";
	}

	ui->crosssectionPSpinBox->setVisible(crossSection.isOk());
	ui->crosssectionRoughnessParametersLabel->setVisible(crossSection.isOk());
	ui->crosssectionHeightStatsLabel->setText(heightParamsString);
	ui->crosssectionStepStatsLabel->setText(stepParamsString);
}

void CrossSectionWindow::closeEvent(QCloseEvent *event){
	event->accept();
	emit closed();
}

void CrossSectionWindow::on_crosssectionPSpinBox_valueChanged(int arg1){
	updateSurfaceStats();
}
