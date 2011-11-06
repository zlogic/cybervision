#include "crosssectionwindow.h"
#include "ui_crosssectionwindow.h"

#include <QCloseEvent>

#define _USE_MATH_DEFINES
#include <cmath>
#include <limits>

#include <QTextStream>
#include <QGraphicsTextItem>

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
	QRect crossSectionArea(50,20,imageSize.width()-50,imageSize.height()-20);

	//Draw the grid & labels
	QPen gridPen(QColor(0x33,0x33,0x33),1,Qt::DotLine);
	{
		//Draw the grid
		qreal stepX= getOptimalGridStep(minX,maxX), stepY= getOptimalGridStep(minY,maxY);
		int gridMinX= floor(minX/stepX);
		int gridMaxX= ceil(maxX/stepX);
		int gridMinY= floor(minY/stepY);
		int gridMaxY= ceil(maxY/stepY);
		minX= gridMinX*stepX, maxX= gridMaxX*stepX, minY= gridMinY*stepY, maxY= gridMaxY*stepY;

		QFont font("Arial",7);
		QFontMetrics fontMetrics(font);
		for(int i=gridMinX;i<=gridMaxX;i++){
			qreal x= crossSectionArea.width()*(stepX*i-minX)/(maxX-minX)+crossSectionArea.x();
			crossSectionScene.addLine(x,0,x,imageSize.height()-1,gridPen);

			QString str;
			QTextStream stream(&str);
			stream.setRealNumberPrecision(1);
			stream.setRealNumberNotation(QTextStream::ScientificNotation);
			stream<<stepX*i<<"  ";
			crossSectionScene.addText(str,font)->setPos(x-fontMetrics.width(str),0);
		}
		for(int i=gridMinY;i<=gridMaxY;i++){
			qreal y= crossSectionArea.height()*(maxY-stepY*i)/(maxY-minY)+crossSectionArea.y();
			crossSectionScene.addLine(0,y,imageSize.width()-1,y,gridPen);

			QString str;
			QTextStream stream(&str);
			stream.setRealNumberPrecision(1);
			stream.setRealNumberNotation(QTextStream::ScientificNotation);
			stream<<stepY*i;
			crossSectionScene.addText(str,font)->setPos(1,y-2);
		}
	}

	//Draw lines
	QPainterPath crossSectionPath;
	for(QList<QPointF>::const_iterator it=crossSectionPoints.begin();it!=crossSectionPoints.end();it++){
		QPointF point1(
					crossSectionArea.width()*(it->x()-minX)/(maxX-minX)+crossSectionArea.x(),
					crossSectionArea.height()*(maxY-it->y())/(maxY-minY)+crossSectionArea.y()
		);
		point1.setX(qMax(point1.x(),(qreal)crossSectionArea.left()));
		point1.setY(qMax(point1.y(),(qreal)crossSectionArea.top()));
		point1.setX(qMin(point1.x(),crossSectionArea.right()-1.0));
		point1.setY(qMin(point1.y(),crossSectionArea.bottom()-1.0));

		if((it)==crossSectionPoints.begin()){
			crossSectionPath.moveTo(point1);
			continue;
		}

		crossSectionPath.lineTo(point1);
	}
	QPen penCrosssection(QColor(0xff,0x99,0x00));
	crossSectionScene.addPath(crossSectionPath,penCrosssection);

	//Draw the m-line
	QPen penMline(QColor(0,0,0),3);
	crossSectionScene.addLine(
				crossSectionArea.width()*(crossSection.getMLine().x1()-minX)/(maxX-minX)+crossSectionArea.x(),
				crossSectionArea.height()*(maxY-crossSection.getMLine().y1())/(maxY-minY)+crossSectionArea.y(),
				crossSectionArea.width()*(crossSection.getMLine().x2()-minX)/(maxX-minX)+crossSectionArea.x(),
				crossSectionArea.height()*(maxY-crossSection.getMLine().y2())/(maxY-minY)+crossSectionArea.y(),
				penMline
	);

	//Draw the t-level
	QPen penTline(QColor(0,0,0),1,Qt::DashDotDotLine);
	crossSectionScene.addLine(
				crossSectionArea.width()*(crossSection.getPLine().x1()-minX)/(maxX-minX)+crossSectionArea.x(),
				crossSectionArea.height()*(maxY-crossSection.getPLine().y1())/(maxY-minY)+crossSectionArea.y(),
				crossSectionArea.width()*(crossSection.getPLine().x2()-minX)/(maxX-minX)+crossSectionArea.x(),
				crossSectionArea.height()*(maxY-crossSection.getPLine().y2())/(maxY-minY)+crossSectionArea.y(),
				penTline
	);
	crossSectionScene.setSceneRect(crossSectionScene.itemsBoundingRect());
}

qreal CrossSectionWindow::getOptimalGridStep(qreal min, qreal max) const{
	//Imported from CybervisionViewer.
	//TODO: make an "Utils" class for such functions?
	qreal delta= max-min;
	qreal exp_x= pow(10.0,floor(log10(delta)));

	//Check if selected scale is too small
	if(delta/exp_x<5)
		exp_x/= 10;

	//Select optimal step
	int max_step_count= 10;
	qreal step_1= exp_x, step_2= exp_x*2, step_5= exp_x*5;
	int step_1_count= ceil(delta/step_1);
	int step_2_count= ceil(delta/step_2);
	//int step_5_count= ceil(delta/step_5);
	if(step_1_count<max_step_count)
		return step_1;
	else if(step_2_count<max_step_count)
		return step_2;
	else return step_5;
}
