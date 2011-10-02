#include "mainwindow.h"
#include "ui_mainwindow.h"

#include <QFileDialog>
#include <QGraphicsItem>

MainWindow::MainWindow(QWidget *parent)	: QMainWindow(parent), ui(new Ui::MainWindow){
    ui->setupUi(this);
	thread.setUi(this);
	updateWidgetStatus();
	loadDebugPreferences();
}

MainWindow::~MainWindow(){
    delete ui;
}

void MainWindow::updateWidgetStatus(){
	if(ui->imageList->count()==0){
		ui->selectionHintLabel->setVisible(false);
		ui->deleteImageButton->setEnabled(false);
		ui->startProcessButton->setEnabled(false);
	}else if(ui->imageList->count()>1 && ui->imageList->selectedItems().count()!=2){
		ui->selectionHintLabel->setVisible(true);
		ui->startProcessButton->setEnabled(false);
	}else {
		ui->selectionHintLabel->setVisible(false);
		ui->startProcessButton->setEnabled(true);
	}

	ui->deleteImageButton->setEnabled(!ui->imageList->selectedItems().empty());
	ui->saveButton->setEnabled(ui->openGLViewport->getSurface3D().isOk());

	scaleXYValidator.setBottom(0.0);
	scaleZValidator.setBottom(0.0);
	angleValidator.setRange(-360.0,360.0,1000);
	ui->scaleXYEdit->setValidator(&scaleXYValidator);
	ui->scaleZEdit->setValidator(&scaleZValidator);
	ui->angleEdit->setValidator(&angleValidator);
}


void MainWindow::loadDebugPreferences(){
	QStringList arguments= qApp->arguments();
	QString debugParam="-debugfile=";
	QString debugFileName;
	for(QStringList::const_iterator it= arguments.begin();it!=arguments.end();it++){
		if(it->startsWith(debugParam,Qt::CaseInsensitive)){
			debugFileName= it->mid(debugParam.length());
			break;
		}
	}

	if(debugFileName.isNull())
		return;

	QFile debugFile(debugFileName);
	if(debugFile.exists()){
		debugFile.open(QFile::ReadOnly);
		QTextStream stream(&debugFile);

		while(!stream.atEnd()){
			QString line= stream.readLine();
			if(!line.isNull() && !line.isEmpty()){
				QRegExp fileRegexp("^file\\s+(.*)$",Qt::CaseInsensitive);
				QRegExp scaleXYRegexp("^ScaleXY\\s+([-+]?[0-9]*\\.?[0-9]+([eE][-+]?[0-9]+)?)$",Qt::CaseInsensitive);
				QRegExp scaleZRegexp("^ScaleZ\\s+([-+]?[0-9]*\\.?[0-9]+([eE][-+]?[0-9]+)?)$",Qt::CaseInsensitive);
				QRegExp angleRegexp("^Angle\\s+([-+]?[0-9]*\\.?[0-9]+([eE][-+]?[0-9]+)?)$",Qt::CaseInsensitive);
				if(fileRegexp.exactMatch(line) && fileRegexp.capturedTexts().size()>=2){
					QString fileName= fileRegexp.capturedTexts().at(1);
					QString name= QFileInfo(fileName).fileName();
					QListWidgetItem* newItem= new QListWidgetItem(name);
					newItem->setData(32,QDir::convertSeparators(fileName));
					ui->imageList->addItem(newItem);
				}else if(scaleXYRegexp.exactMatch(line) && scaleXYRegexp.capturedTexts().size()>=2){
					ui->scaleXYEdit->setText(scaleXYRegexp.capturedTexts().at(1));
				}else if(scaleZRegexp.exactMatch(line) && scaleZRegexp.capturedTexts().size()>=2){
					ui->scaleZEdit->setText(scaleZRegexp.capturedTexts().at(1));
				}else if(angleRegexp.exactMatch(line) && angleRegexp.capturedTexts().size()>=2){
					ui->angleEdit->setText(angleRegexp.capturedTexts().at(1));
				}
			}
		}
		debugFile.close();
	}
}

void MainWindow::processStarted(){
	ui->startProcessButton->setEnabled(false);
	ui->saveButton->setEnabled(false);
	ui->logTextEdit->clear();
	ui->logTextEdit->setCurrentCharFormat(QTextCharFormat());
}


void MainWindow::processUpdated(QString logMessage,QString statusBarText){
	if(!logMessage.isNull())
		ui->logTextEdit->appendPlainText(logMessage+"\n");
	if(!statusBarText.isNull())
		ui->statusBar->showMessage(statusBarText);

	if(logMessage.isNull() && !statusBarText.isNull())
		ui->logTextEdit->appendPlainText(statusBarText+"\n");
}

void MainWindow::processStopped(QString resultText,cybervision::Surface surface){
	ui->statusBar->clearMessage();
	ui->startProcessButton->setEnabled(true);
	if(!resultText.isNull() && !resultText.isEmpty()){
		ui->saveButton->setEnabled(true);
		ui->logTextEdit->appendHtml("<b>"+resultText+"</b>");
	}

	ui->openGLViewport->setSurface3D(surface);
	updateWidgetStatus();
}


void MainWindow::on_startProcessButton_clicked(){
	QStringList filenames;


	double scaleXY= ui->scaleXYEdit->text().toDouble();
	double scaleZ= ui->scaleZEdit->text().toDouble();
	double angle= ui->angleEdit->text().toDouble();

	QList<QListWidgetItem*> selectedItems= ui->imageList->selectedItems();
	for(QList<QListWidgetItem*>::const_iterator i=selectedItems.begin();i!=selectedItems.end();i++){
		if(*i)
			filenames<<(*i)->data(32).toString();
	}

	thread.reconstruct3DShape(filenames,scaleXY,scaleZ,angle,ui->actionPrefer_scale_from_metadata->isChecked());
}

void MainWindow::on_saveButton_clicked(){
	QStringList formats;
	formats<<"Surface points (*.txt)";
	formats<<"Surface polygons (*.txt)";
	formats<<"PNG image (*.png)";
	formats<<"SceneJS model (*.js)";
	formats<<"Collada model (*.dae)";
	QString filter;
	for(QStringList::const_iterator it=formats.begin();it!=formats.end();it++)
		filter.append(*it+";;");
	QString selectedFilter;
	QString fileName = QFileDialog::getSaveFileName(this,"Save the surface","",filter,&selectedFilter,0);
	if(!fileName.isNull()){
		QFileInfo fileInfo(fileName);
		if(selectedFilter==formats[0]){
			if(fileInfo.suffix()!="txt")
				fileName.append(".txt");
			ui->openGLViewport->getSurface3D().savePoints(fileName);
		}else if(selectedFilter==formats[1]){
			if(fileInfo.suffix()!="txt")
				fileName.append(".txt");
			ui->openGLViewport->getSurface3D().savePolygons(fileName);
		}else if(selectedFilter==formats[2]){
			if(fileInfo.suffix()!="png")
				fileName.append(".png");
			QImage screenshot= ui->openGLViewport->grabFrameBuffer();
			screenshot.save(fileName,"png");
		}else if(selectedFilter==formats[3]){
			if(fileInfo.suffix()!="js")
				fileName.append(".js");
			ui->openGLViewport->getSurface3D().saveSceneJS(fileName);
		}else if(selectedFilter==formats[4]){
			if(fileInfo.suffix()!="dae")
				fileName.append(".dae");
			ui->openGLViewport->getSurface3D().saveCollada(fileName);
		}else{
			ui->statusBar->showMessage("Bad save format selected");
			ui->logTextEdit->appendHtml("<b>Bad save format selected:</b> "+selectedFilter);
		}
	}
}

void MainWindow::on_logDockWidget_visibilityChanged(bool visible){
	ui->actionShowlog->setChecked(visible);
}

void MainWindow::on_actionShowlog_triggered(bool checked){
	ui->logDockWidget->setVisible(checked);
}

void MainWindow::on_addImageButton_clicked(){
	QString filter= "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp);;All files(*.*)";
	QStringList filenames = QFileDialog::getOpenFileNames(this,"Select images to add","",filter,0,0);
	for(QStringList::const_iterator it=filenames.begin();it!=filenames.end();it++){
		QString name= QFileInfo(*it).fileName();
		QListWidgetItem* newItem= new QListWidgetItem(name);
		newItem->setData(32,QDir::convertSeparators(*it));
		ui->imageList->addItem(newItem);
	}

	updateWidgetStatus();
}

void MainWindow::on_deleteImageButton_clicked(){
	QList<QListWidgetItem*> selection= ui->imageList->selectedItems();
	for(QList<QListWidgetItem*>::const_iterator i=selection.begin();i!=selection.end();i++){
		int row= ui->imageList->row(*i);
		if(row>=0){
			QListWidgetItem* deletedItem= ui->imageList->takeItem(row);
			if(deletedItem)
				delete deletedItem;
		}
	}
}
void MainWindow::on_imageList_itemSelectionChanged(){
	updateWidgetStatus();
}


void MainWindow::on_moveToolButton_toggled(bool checked){
	ui->openGLViewport->setMouseMode(checked?CybervisionViewer::MOUSE_PANNING:CybervisionViewer::MOUSE_ROTATION);
}

void MainWindow::on_rotateToolButton_toggled(bool checked){
	ui->openGLViewport->setMouseMode(!checked?CybervisionViewer::MOUSE_PANNING:CybervisionViewer::MOUSE_ROTATION);
}

void MainWindow::on_gridToolButton_toggled(bool checked){
	ui->openGLViewport->setShowGrid(checked);
}



void MainWindow::on_texture1ToolButton_clicked(){
	ui->openGLViewport->setTextureMode(CybervisionViewer::TEXTURE_1);
}

void MainWindow::on_texture2ToolButton_clicked(){
	ui->openGLViewport->setTextureMode(CybervisionViewer::TEXTURE_2);
}

void MainWindow::on_textureNoneToolButton_clicked(){
	ui->openGLViewport->setTextureMode(CybervisionViewer::TEXTURE_NONE);
}

