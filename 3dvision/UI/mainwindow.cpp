#include "mainwindow.h"
#include "ui_mainwindow.h"

#include <QFileDialog>
#include <QGraphicsItem>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent), ui(new Ui::MainWindow){
    ui->setupUi(this);
	thread.setUi(this);
}

MainWindow::~MainWindow(){
    delete ui;
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
	ui->statusBar->clearMessage();
	ui->startProcessButton->setEnabled(true);
	if(!resultText.isNull() && !resultText.isEmpty()){
		ui->saveButton->setEnabled(true);
		ui->logTextEdit->appendHtml("<b>"+resultText+"</b>");
	}

	if(!points.empty())
		ui->openGLViewport->setPoints3D(points);
}


void MainWindow::on_startProcessButton_clicked(){
	QString filter= "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp);;All files(*.*)";
	QStringList filenames = QFileDialog::getOpenFileNames(this,"Select images to load","",filter,0,0);
	for(QStringList::iterator it=filenames.begin();it!=filenames.end();it++)
		(*it)=QDir::convertSeparators(*it);

	thread.extract(filenames);
}

void MainWindow::on_saveButton_clicked(){
	/*
	for(QList<QGraphicsItem *>::const_iterator it= ui->graphicsView->items().begin();it!=ui->graphicsView->items().end();it++){
		const QGraphicsPixmapItem* item= qgraphicsitem_cast<QGraphicsPixmapItem*>(ui->graphicsView->items().first());
		if(item){
			QString filter= "Jpeg images (*.jpeg *.jpg);;Png images (*.png);;Tiff images(*.tiff *.tif);;BMP images(*.bmp)";
			QString fileName = QFileDialog::getSaveFileName(this,"Save the SIFT keypoint result","",filter,0,0);

			item->pixmap().save(fileName);
			break;
		}
	}
	*/
}

void MainWindow::on_logDockWidget_visibilityChanged(bool visible){
	ui->actionShowlog->setChecked(visible);
}

void MainWindow::on_actionShowlog_toggled(bool checked){
	ui->logDockWidget->setVisible(checked);
}
