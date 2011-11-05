#include "crosssectionwindow.h"
#include "ui_crosssectionwindow.h"

#include <QCloseEvent>

#include <limits>

CrossSectionWindow::CrossSectionWindow(QWidget *parent) :
    QDialog(parent),
	ui(new Ui::CrossSectionWindow),
	crossSectionScene(ui->crossSectionViewport)
{
    ui->setupUi(this);
	ui->crossSectionViewport->setUpdatesEnabled(true);
	ui->crossSectionViewport->setScene(&crossSectionScene);

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

	//Render the image
	crossSectionScene.clear();
	renderCrossSection();

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

void CrossSectionWindow::resizeEvent(QResizeEvent* event){
	updateSurfaceStats();
}


void CrossSectionWindow::on_crosssectionPSpinBox_valueChanged(int arg1){
	updateSurfaceStats();
}

void CrossSectionWindow::renderCrossSection(){
	QSize imageSize(ui->crossSectionViewport->viewport()->contentsRect().width(),ui->crossSectionViewport->viewport()->contentsRect().height());
	QList<QPointF> crossSectionPoints= crossSection.getCrossSection();
	//Prepare data
	qreal minX= std::numeric_limits<qreal>::infinity(),
			minY= std::numeric_limits<qreal>::infinity(),
			maxX= -std::numeric_limits<qreal>::infinity(),
			maxY= -std::numeric_limits<qreal>::infinity();
	for(QList<QPointF>::const_iterator it=crossSectionPoints.begin();it!=crossSectionPoints.end();it++){
		minX= qMin(minX,it->x());
		minY= qMin(minY,it->y());
		maxX= qMax(maxX,it->x());
		maxY= qMax(maxY,it->y());
	}


	//Draw lines
	QPainterPath crossSectionPath;
	for(QList<QPointF>::const_iterator it=crossSectionPoints.begin();it!=crossSectionPoints.end();it++){
		QPointF point1(imageSize.width()*(it->x()-minX)/(maxX-minX),imageSize.height()*(maxY-it->y())/(maxY-minY));
		if((it)==crossSectionPoints.begin()){
			crossSectionPath.moveTo(point1);
			continue;
		}

		point1.setX(qMax(point1.x(),0.0));
		point1.setY(qMax(point1.y(),0.0));
		point1.setX(qMin(point1.x(),imageSize.width()-1.0));
		point1.setY(qMin(point1.y(),imageSize.height()-1.0));

		crossSectionPath.lineTo(point1.x(),point1.y());
	}

	QPen penCrosssection(QColor(0xff,0x99,0x00));
	crossSectionScene.addPath(crossSectionPath,penCrosssection);

	//Draw the m-line
	QPen penMline(QColor(0x66,0x66,0x66));
	crossSectionScene.addLine(imageSize.width()*(crossSection.getMLine().x1()-minX)/(maxX-minX),imageSize.height()*(maxY-crossSection.getMLine().y1())/(maxY-minY),
				  imageSize.width()*(crossSection.getMLine().x2()-minX)/(maxX-minX),imageSize.height()*(maxY-crossSection.getMLine().y2())/(maxY-minY),
				  penMline
	);

	crossSectionScene.setSceneRect(crossSectionScene.itemsBoundingRect());
}
