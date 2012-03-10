#include "crosssectionwindow.h"
#include "ui_crosssectionwindow.h"

#include <QCloseEvent>
#include <QGraphicsSceneMouseEvent>

#define _USE_MATH_DEFINES
#include <cmath>
#include <limits>

#include <QTextStream>
#include <QGraphicsTextItem>

#include <Reconstruction/options.h>

CrossSectionWindow::CrossSectionWindow(QWidget *parent) :
    QDialog(parent),
	ui(new Ui::CrossSectionWindow),
	crossSectionScene(this)
{
    ui->setupUi(this);
	ui->crosssectionViewport->setUpdatesEnabled(true);
	ui->crosssectionViewport->setScene(&crossSectionScene);

#ifdef CYBERVISION_DEMO
	setMaximumSize(530,480);
#endif

	for(int i=0;i<2;i++)
		crossSections<<cybervision::CrossSection();

	connect(&crossSectionScene,SIGNAL(measurementLineMoved(qreal,int)),this,SLOT(measurementLineMoved(qreal,int)),Qt::AutoConnection);
	connect(&crossSectionScene,SIGNAL(crossSectionMoved(qreal)),this,SLOT(crossSectionMoved(qreal)),Qt::AutoConnection);
	connect(ui->crosssectionViewport,SIGNAL(resized()),this,SLOT(viewportResized()),Qt::AutoConnection);

	movableCrossSectionPos= 0;

	updateCrosssectionStats();
}

CrossSectionWindow::~CrossSectionWindow(){
	delete ui;
}

void CrossSectionWindow::updateCrossSection(const cybervision::CrossSection& crossSection,int crossSectionId){
	if(crossSectionId>=0 && crossSectionId<crossSections.size())
		crossSections[crossSectionId]= crossSection;
	measurementLinePos1= std::numeric_limits<qreal>::quiet_NaN();
	measurementLinePos2= std::numeric_limits<qreal>::quiet_NaN();
	movableCrossSectionPos= 0;

	nonMovableCrosssection= crossSections.end();

	updateWidgetStatus();
	updateCrosssectionStats();
}


void CrossSectionWindow::updateWidgetStatus(){
	bool crossSectionsOk= false;
	for(QList<cybervision::CrossSection>::iterator it=crossSections.begin();it!=crossSections.end();it++)
		crossSectionsOk|= it->isOk();
	ui->roughnessGroupBox->setVisible(crossSectionsOk);
	ui->heightGroupBox->setVisible(crossSectionsOk);
	ui->roughnessPSpinBoxPrimary->setVisible(crossSections[0].isOk());
	ui->roughnessPSpinBoxSecondary->setVisible(crossSections[1].isOk());
}


void CrossSectionWindow::updateCrosssectionStats(){
	bool crossSectionsOk= false;
	for(QList<cybervision::CrossSection>::iterator it=crossSections.begin();it!=crossSections.end();it++){
		crossSectionsOk|= it->isOk();
		if(it==crossSections.begin())
			it->computeParams(ui->roughnessPSpinBoxPrimary->value());
		else if(it==(crossSections.begin()+1))
			it->computeParams(ui->roughnessPSpinBoxSecondary->value());
	}

	//Render the image
	crossSectionScene.clear();
	renderCrossSections();

	//Set the roughness parameters
	QList<QString> heightParamsString,stepParamsString;

	for(QList<cybervision::CrossSection>::iterator it=crossSections.begin();it!=crossSections.end();it++){
		int crossSectionId= it-crossSections.begin()+1;
		if(it->isOk()){
			heightParamsString<< QString(trUtf8("Cross-section %1\nRa= %2 \xC2\xB5m\nRz= %3 \xC2\xB5m\nRmax= %4 \xC2\xB5m"))
								 .arg(crossSectionId)
								 .arg(it->getRoughnessRa()*cybervision::Options::TextUnitScale)
								 .arg(it->getRoughnessRz()*cybervision::Options::TextUnitScale)
								 .arg(it->getRoughnessRmax()*cybervision::Options::TextUnitScale);
			stepParamsString<< QString(trUtf8("Cross-section %1\nS= %2 \xC2\xB5m\nSm= %3 \xC2\xB5m\ntp= %4"))
							   .arg(crossSectionId)
							   .arg(it->getRoughnessS()*cybervision::Options::TextUnitScale)
							   .arg(it->getRoughnessSm()*cybervision::Options::TextUnitScale)
							   .arg(it->getRoughnessTp());

		}else{
			heightParamsString<< QString(tr("Cross-section %1 not available")).arg(crossSectionId);
			stepParamsString<< QString(tr("Cross-section %1 not available")).arg(crossSectionId);
		}
	}

	ui->roughnessGroupBox->setVisible(crossSectionsOk);
	ui->heightGroupBox->setVisible(crossSectionsOk);
	ui->roughnessHeightStatsLabelPrimary->setText(heightParamsString[0]);
	ui->roughnessHeightStatsLabelSecondary->setText(heightParamsString[1]);
	ui->roughnessStepStatsLabelPrimary->setText(stepParamsString[0]);
	ui->roughnessStepStatsLabelSecondary->setText(stepParamsString[1]);
}

void CrossSectionWindow::closeEvent(QCloseEvent *event){
	event->accept();
	emit closed();
}

void CrossSectionWindow::viewportResized(){
	updateCrosssectionStats();
}

void CrossSectionWindow::on_roughnessPSpinBoxPrimary_valueChanged(int arg1){
	updateCrosssectionStats();
}

void CrossSectionWindow::on_roughnessPSpinBoxSecondary_valueChanged(int arg1){
	updateCrosssectionStats();
}

void CrossSectionWindow::renderCrossSections(){
	QSize imageSize(ui->crosssectionViewport->viewport()->contentsRect().width(),ui->crosssectionViewport->viewport()->contentsRect().height());
	//Prepare data
	qreal minX= std::numeric_limits<qreal>::infinity(),
			minY= std::numeric_limits<qreal>::infinity(),
			maxX= -std::numeric_limits<qreal>::infinity(),
			maxY= -std::numeric_limits<qreal>::infinity();
	nonMovableCrosssection= crossSections.end();
	for(QList<cybervision::CrossSection>::const_iterator it=crossSections.begin();it!=crossSections.end();it++){
		if(!it->isOk())
			continue;
		qreal currentMinX= std::numeric_limits<qreal>::infinity(),
				currentMinY= std::numeric_limits<qreal>::infinity(),
				currentMaxX= -std::numeric_limits<qreal>::infinity(),
				currentMaxY= -std::numeric_limits<qreal>::infinity();
		QList<QPointF> crossSectionPoints= it->getCrossSection();
		for(QList<QPointF>::const_iterator jt=crossSectionPoints.begin();jt!=crossSectionPoints.end();jt++){
			currentMinX= qMin(currentMinX,jt->x());
			currentMinY= qMin(currentMinY,jt->y());
			currentMaxX= qMax(currentMaxX,jt->x());
			currentMaxY= qMax(currentMaxY,jt->y());
		}
		if((currentMaxX-currentMinX)>(maxX-minX))
			nonMovableCrosssection= it;
		minX= qMin(currentMinX,minX);
		minY= qMin(currentMinY,minY);
		maxX= qMax(currentMaxX,maxX);
		maxY= qMax(currentMaxY,maxY);
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
		font.setHintingPreference(QFont::PreferFullHinting);
		QFontMetrics fontMetrics(font);
		for(int i=gridMinX;i<=gridMaxX;i++){
			qreal x= crossSectionArea.width()*(stepX*i-minX)/(maxX-minX)+crossSectionArea.x();
			crossSectionScene.addLine(x,0,x,imageSize.height()-1,gridPen);

			QString str;
			QTextStream stream(&str);
			//stream.setRealNumberPrecision(1);
			//stream.setRealNumberNotation(QTextStream::ScientificNotation);
			stream<<stepX*i*cybervision::Options::TextUnitScale;
			str= QString(trUtf8("%1 \xC2\xB5m  ")).arg(str);
			crossSectionScene.addText(str,font)->setPos(x-fontMetrics.width(str),0);
		}
		for(int i=gridMinY;i<=gridMaxY;i++){
			qreal y= crossSectionArea.height()*(maxY-stepY*i)/(maxY-minY)+crossSectionArea.y();
			crossSectionScene.addLine(0,y,imageSize.width()-1,y,gridPen);

			QString str;
			QTextStream stream(&str);
			//stream.setRealNumberPrecision(1);
			//stream.setRealNumberNotation(QTextStream::ScientificNotation);
			stream<<stepY*i*cybervision::Options::TextUnitScale;
			str= QString(trUtf8("%1 \xC2\xB5m")).arg(str);
			crossSectionScene.addText(str,font)->setPos(1,y-2);
		}
	}

	//Draw lines
	QGraphicsItem *movableCrossSection= NULL;
	for(QList<cybervision::CrossSection>::const_iterator it=crossSections.begin();it!=crossSections.end();it++){
		if(!it->isOk())
			continue;
		QList<QPointF> crossSectionPoints= it->getCrossSection();
		QPainterPath crossSectionPath;
		//Draw the cross-section
		for(QList<QPointF>::const_iterator jt=crossSectionPoints.begin();jt!=crossSectionPoints.end();jt++){
			QPointF point1(
						crossSectionArea.width()*(jt->x()-minX)/(maxX-minX),
						crossSectionArea.height()*(maxY-jt->y())/(maxY-minY)
			);
			point1.setX(qMax(point1.x(),(qreal)0));
			point1.setY(qMax(point1.y(),(qreal)0));
			point1.setX(qMin(point1.x(),crossSectionArea.width()-1.0));
			point1.setY(qMin(point1.y(),crossSectionArea.height()-1.0));

			if((jt)==crossSectionPoints.begin()){
				crossSectionPath.moveTo(point1);
				continue;
			}

			crossSectionPath.lineTo(point1);
		}
		QPen penCrosssection(QColor(0xff,0x99,0x00));
		QGraphicsItem * crossSectionGraphicsPath= crossSectionScene.addPath(crossSectionPath,penCrosssection);
		if(crossSectionGraphicsPath)
			crossSectionGraphicsPath->setPos(crossSectionArea.x(),crossSectionArea.y());
		if(crossSectionGraphicsPath && it!=nonMovableCrosssection && !movableCrossSection){
			crossSectionGraphicsPath->setPos(crossSectionArea.x()+movableCrossSectionPos/sceneScaleX,crossSectionArea.y());
			crossSectionGraphicsPath->setCursor(Qt::SizeAllCursor);
			crossSectionGraphicsPath->setFlag(QGraphicsItem::ItemIsMovable,true);
			movableCrossSection= crossSectionGraphicsPath;
		}

		//Draw the m-line
		QPen penMline(QColor(0,0,0),3);
		QGraphicsItem *mLine= crossSectionScene.addLine(
					crossSectionArea.width()*(it->getMLine().x1()-minX)/(maxX-minX),
					crossSectionArea.height()*(maxY-it->getMLine().y1())/(maxY-minY),
					crossSectionArea.width()*(it->getMLine().x2()-minX)/(maxX-minX),
					crossSectionArea.height()*(maxY-it->getMLine().y2())/(maxY-minY),
					penMline
		);
		if(mLine)
			mLine->setParentItem(crossSectionGraphicsPath);

		//Draw the t-level
		QPen penTline(QColor(0,0,0),1,Qt::DashDotDotLine);
		QGraphicsItem *tLine= crossSectionScene.addLine(
					crossSectionArea.width()*(it->getPLine().x1()-minX)/(maxX-minX),
					crossSectionArea.height()*(maxY-it->getPLine().y1())/(maxY-minY),
					crossSectionArea.width()*(it->getPLine().x2()-minX)/(maxX-minX),
					crossSectionArea.height()*(maxY-it->getPLine().y2())/(maxY-minY),
					penTline
		);
		if(tLine)
			tLine->setParentItem(crossSectionGraphicsPath);
	}

	//Draw the level measurement lines
	QPen measurementLinePen(QColor(0,0,0x66),3,Qt::SolidLine);

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

	crossSectionScene.setSceneRect(ui->crosssectionViewport->viewport()->contentsRect());

	QList<QGraphicsItem*> measurementLines;
	measurementLines<<measurementLine1<<measurementLine2;
	crossSectionScene.updateSceneData(measurementLines,movableCrossSection,crossSectionArea);

	updateMeasurementLinesLabel();
}


void CrossSectionWindow::updateMeasurementLinesLabel(){
	QList<QString> measurementLineStr;
	for(QList<cybervision::CrossSection>::iterator it=crossSections.begin();it!=crossSections.end();it++){
		qreal deltaX= 0;
		if(it!=nonMovableCrosssection)
			deltaX= -movableCrossSectionPos;
		qreal height1= it->getHeight(measurementLinePos1+deltaX), height2= it->getHeight(measurementLinePos2+deltaX);
		qreal deltaHeight= height1-height2;
		measurementLineStr<< QString(trUtf8("Cross-section %1\nx1= %2 \xC2\xB5m\nh1= %3 \xC2\xB5m\nx2= %4 \xC2\xB5m\nh2= %5 \xC2\xB5m\nHeight difference= %6 \xC2\xB5m"))
				.arg(it-crossSections.begin()+1)
				.arg((measurementLinePos1+deltaX)*cybervision::Options::TextUnitScale)
				.arg(height1*cybervision::Options::TextUnitScale)
				.arg((measurementLinePos2+deltaX)*cybervision::Options::TextUnitScale)
				.arg(height2*cybervision::Options::TextUnitScale)
				.arg(deltaHeight*cybervision::Options::TextUnitScale);
	}

	if(crossSections[0].isOk())
		ui->heightMeasurementLabelPrimary->setText(measurementLineStr[0]);
	if(crossSections[1].isOk())
		ui->heightMeasurementLabelSecondary->setText(measurementLineStr[1]);
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

void CrossSectionWindow::measurementLineMoved(qreal x, int id){
	switch(id){
	case 0:
		measurementLinePos1= x*sceneScaleX;
		break;
	case 1:
		measurementLinePos2= x*sceneScaleX;
		break;
	}
	updateMeasurementLinesLabel();

	ui->crosssectionViewport->viewport()->update();// Fix for Qt <= 4.8 ugly line re-rendering
}

void CrossSectionWindow::crossSectionMoved(qreal x){
	movableCrossSectionPos= x*sceneScaleX;
	updateMeasurementLinesLabel();

	ui->crosssectionViewport->viewport()->update();// Fix for Qt <= 4.8 ugly line re-rendering
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
	if(mouseGrabberItem())
		itemPos= mouseGrabberItem()->scenePos();
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
			emit measurementLineMoved(newPos.x()-crossSectionArea.x(),measurementLines.indexOf(mouseGrabberItem()));
		}
	}else if(movableCrossSection && movableCrossSection==mouseGrabberItem()){
		if(event->type()==QEvent::GraphicsSceneMouseMove){
			QPointF newPos(event->scenePos().x(),clickPos.y());
			if(newPos.x()<crossSectionArea.left()-(itemPos.x()-clickPos.x()))
				newPos.setX(crossSectionArea.left()-(itemPos.x()-clickPos.x()));
			if(newPos.x()>crossSectionArea.right()-mouseGrabberItem()->boundingRect().width()-(itemPos.x()-clickPos.x()))
				newPos.setX(crossSectionArea.right()-mouseGrabberItem()->boundingRect().width()-(itemPos.x()-clickPos.x()));
			event->setScenePos(newPos);
			emit crossSectionMoved(mouseGrabberItem()->scenePos().x()-crossSectionArea.left());
		}
	}
	QGraphicsScene::mouseMoveEvent(event);
}

void CybervisionCrosssectionScene::updateSceneData(const QList<QGraphicsItem*>& measurementLines,QGraphicsItem *movableCrossSection, const QRect &crossSectionArea){
	this->measurementLines= measurementLines;
	this->movableCrossSection= movableCrossSection;
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
