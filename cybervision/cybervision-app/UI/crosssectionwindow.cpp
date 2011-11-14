#include "crosssectionwindow.h"
#include "ui_crosssectionwindow.h"

#include <QCloseEvent>
#include <QGraphicsSceneMouseEvent>

#define _USE_MATH_DEFINES
#include <cmath>
#include <limits>

#include <QTextStream>
#include <QGraphicsTextItem>

CrossSectionWindow::CrossSectionWindow(QWidget *parent) :
    QDialog(parent),
	ui(new Ui::CrossSectionWindow),
	crossSectionScene(this)
{
    ui->setupUi(this);
	ui->crosssectionViewport->setUpdatesEnabled(true);
	ui->crosssectionViewport->setScene(&crossSectionScene);

	connect(&crossSectionScene,SIGNAL(measurementLineDragged(qreal,int)),this,SLOT(measurementLineDragged(qreal,int)),Qt::AutoConnection);
	connect(ui->crosssectionViewport,SIGNAL(resized()),this,SLOT(viewportResized()),Qt::AutoConnection);

	updateSurfaceStats();
}

CrossSectionWindow::~CrossSectionWindow(){
	delete ui;
}

void CrossSectionWindow::updateCrossSection(const cybervision::CrossSection& crossSection){
	this->crossSection= crossSection;
	measurementLinePos1= std::numeric_limits<qreal>::quiet_NaN();
	measurementLinePos2= std::numeric_limits<qreal>::quiet_NaN();
	updateSurfaceStats();
}


void CrossSectionWindow::updateWidgetStatus(){
	ui->roughnessGroupBox->setVisible(crossSection.isOk());
	ui->heightGroupBox->setVisible(crossSection.isOk());
}


void CrossSectionWindow::updateSurfaceStats(){
	crossSection.computeParams(ui->roughnessPSpinBox->value());

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

	ui->roughnessGroupBox->setVisible(crossSection.isOk());
	ui->heightGroupBox->setVisible(crossSection.isOk());
	ui->roughnessHeightStatsLabel->setText(heightParamsString);
	ui->roughnessStepStatsLabel->setText(stepParamsString);
}

void CrossSectionWindow::closeEvent(QCloseEvent *event){
	event->accept();
	emit closed();
}

void CrossSectionWindow::viewportResized(){
	updateSurfaceStats();
}

void CrossSectionWindow::on_roughnessPSpinBox_valueChanged(int arg1){
	updateSurfaceStats();
}

void CrossSectionWindow::renderCrossSection(){
	QSize imageSize(ui->crosssectionViewport->viewport()->contentsRect().width(),ui->crosssectionViewport->viewport()->contentsRect().height());
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

	//Draw the level measurement lines
	QPen measurementLinePen(QColor(0,0,0x66),1,Qt::SolidLine);

	if(std::isnan(measurementLinePos1))
		measurementLinePos1= minX;
	if(std::isnan(measurementLinePos2))
		measurementLinePos2= maxX;
	sceneScaleX= (maxX-minX)/crossSectionArea.width();

	QGraphicsItem *measurementLine1= crossSectionScene.addLine(
				crossSectionArea.x()+measurementLinePos1/sceneScaleX,
				crossSectionArea.y(),
				crossSectionArea.x()+measurementLinePos1/sceneScaleX,
				crossSectionArea.bottom(),
				measurementLinePen);
	if(measurementLine1){
		measurementLine1->setCursor(Qt::SplitHCursor);
		measurementLine1->setFlag(QGraphicsItem::ItemIsMovable,true);
	}
	QGraphicsItem *measurementLine2= crossSectionScene.addLine(
				crossSectionArea.x()+measurementLinePos2/sceneScaleX,
				crossSectionArea.y(),
				crossSectionArea.x()+measurementLinePos2/sceneScaleX,
				crossSectionArea.bottom(),
				measurementLinePen);
	if(measurementLine2){
		measurementLine2->setCursor(Qt::SplitHCursor);
		measurementLine2->setFlag(QGraphicsItem::ItemIsMovable,true);
	}

	crossSectionScene.setSceneRect(crossSectionScene.itemsBoundingRect());

	QList<QGraphicsItem*> measurementLines;
	measurementLines<<measurementLine1<<measurementLine2;
	crossSectionScene.updateSceneData(measurementLines,crossSectionArea);

	updateMeasurementLinesLabel();
}


void CrossSectionWindow::updateMeasurementLinesLabel(){
	qreal height1= crossSection.getHeight(measurementLinePos1), height2= crossSection.getHeight(measurementLinePos2);
	qreal deltaHeight= height1-height2;
	QString measurementLineStr= QString(tr("x1= %1 m\nh1= %2 m\nx2= %3 m\nh2= %4 m\nHeight difference= %5 m"))
			.arg(measurementLinePos1)
			.arg(height1)
			.arg(measurementLinePos2)
			.arg(height2)
			.arg(deltaHeight);
	ui->heightMeasurementLabel->setText(measurementLineStr);
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

void CrossSectionWindow::measurementLineDragged(qreal x, int id){
	switch(id){
	case 0:
		measurementLinePos1= x*sceneScaleX;
		break;
	case 1:
		measurementLinePos2= x*sceneScaleX;
		break;
	}
	updateMeasurementLinesLabel();
}

/*
 * CybervisionCrosssectionScene code
 */
CybervisionCrosssectionScene::CybervisionCrosssectionScene(QObject * parent):QGraphicsScene(parent){}
CybervisionCrosssectionScene::CybervisionCrosssectionScene(const QRectF &sceneRect, QObject *parent):QGraphicsScene(sceneRect,parent){}
CybervisionCrosssectionScene::CybervisionCrosssectionScene(qreal x, qreal y, qreal width, qreal height, QObject *parent):QGraphicsScene(x,y,width,height,parent){}

void CybervisionCrosssectionScene::mousePressEvent(QGraphicsSceneMouseEvent *event){
	clickPos= event->scenePos();
	QGraphicsScene::mousePressEvent(event);
}

void CybervisionCrosssectionScene::mouseMoveEvent(QGraphicsSceneMouseEvent *event){
	if(measurementLines.contains(mouseGrabberItem())){
		if(event->type()==QEvent::GraphicsSceneMouseMove){
			QPointF newPos(event->scenePos().x(),clickPos.y());
			if(newPos.x()<crossSectionArea.left())
				newPos.setX(crossSectionArea.left());
			if(newPos.x()>crossSectionArea.right())
				newPos.setX(crossSectionArea.right());
			event->setScenePos(newPos);
			emit measurementLineDragged(newPos.x()-crossSectionArea.x(),measurementLines.indexOf(mouseGrabberItem()));
		}
	}
	QGraphicsScene::mouseMoveEvent(event);
}

void CybervisionCrosssectionScene::updateSceneData(QList<QGraphicsItem*> measurementLines, const QRect &crossSectionArea){
	this->measurementLines= measurementLines;
	this->crossSectionArea= crossSectionArea;
}

/*
 * CybervisionCrosssectionGraphicsView code
 */
CybervisionCrosssectionGraphicsView::CybervisionCrosssectionGraphicsView(QWidget *parent):QGraphicsView(parent){}
CybervisionCrosssectionGraphicsView::CybervisionCrosssectionGraphicsView(QGraphicsScene *scene, QWidget *parent):QGraphicsView(scene,parent){}

void CybervisionCrosssectionGraphicsView::resizeEvent(QResizeEvent *event){
	emit resized();
}
