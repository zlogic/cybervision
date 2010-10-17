#include "mainwindow.h"
#include "ui_mainwindow.h"

#include <QFileDialog>
#include <QGraphicsItem>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent), ui(new Ui::MainWindow){
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
			QString fileName= stream.readLine();
			if(!fileName.isNull() && !fileName.isEmpty()){
				QString name= QFileInfo(fileName).fileName();
				QListWidgetItem* newItem= new QListWidgetItem(name);
				newItem->setData(32,QDir::convertSeparators(fileName));
				ui->imageList->addItem(newItem);
			}
		}
		debugFile.close();
	}
}

void MainWindow::processStarted(){
	ui->startProcessButton->setEnabled(false);
	ui->saveButton->setEnabled(false);
	ui->logTextEdit->clear();
}


void MainWindow::processUpdated(QString logMessage,QString statusBarText){
	if(!logMessage.isNull())
		ui->logTextEdit->appendPlainText(logMessage+"\n");
	if(!statusBarText.isNull())
		ui->statusBar->showMessage(statusBarText);

	if(logMessage.isNull() && !statusBarText.isNull())
		ui->logTextEdit->appendPlainText(statusBarText+"\n");
}

void MainWindow::processStopped(QString resultText,QList<QVector3D> points){
	/*
	*/
	if(!resultText.isNull() && !resultText.isEmpty()){
		ui->saveButton->setEnabled(true);
		ui->logTextEdit->appendHtml("<b>"+resultText+"</b>");
	}

	if(!points.empty())
		thread.surface(points);
	else{
		ui->statusBar->clearMessage();
		ui->startProcessButton->setEnabled(true);
	}
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

	QList<QListWidgetItem*> selectedItems= ui->imageList->selectedItems();
	for(QList<QListWidgetItem*>::const_iterator i=selectedItems.begin();i!=selectedItems.end();i++){
		if(*i)
			filenames<<(*i)->data(32).toString();
	}

	ui->imageList->selectedItems();
	thread.extract(filenames);
}

void MainWindow::on_saveButton_clicked(){
	QStringList formats;
	formats<<"Surface points (*.txt)";
	formats<<"Surface polygons (*.txt)";
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

void MainWindow::on_actionShowlog_toggled(bool checked){
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

