#include "mainwindow.h"
#include "ui_mainwindow.h"

#include "../SIFT/process.h"

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

void MainWindow::on_pushButton_clicked(){
	QString filter= "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp);;All files(*.*)";
	QStringList filenames = QFileDialog::getOpenFileNames(this,"Select images to load","",filter,0,0);
	for(QStringList::iterator it=filenames.begin();it!=filenames.end();it++)
		(*it)=QDir::convertSeparators(*it);

	thread.extract(filenames);
}


void MainWindow::on_pushButton_2_clicked(){
	for(QList<QGraphicsItem *>::const_iterator it= ui->graphicsView->items().begin();it!=ui->graphicsView->items().end();it++){
		const QGraphicsPixmapItem* item= qgraphicsitem_cast<QGraphicsPixmapItem*>(ui->graphicsView->items().first());
		if(item){
			QString filter= "Png images (*.png);;Tiff images(*.tiff *.tif);;Jpeg images (*.jpeg *.jpg);;BMP images(*.bmp)";
			QString fileName = QFileDialog::getSaveFileName(this,"Save the SIFT keypoint result","",filter,0,0);

			item->pixmap().save(fileName);
			break;
		}
	}
}

void MainWindow::processStarted(QString statusBarText){
	ui->statusBar->showMessage(statusBarText);
	ui->pushButton->setEnabled(false);
}

void MainWindow::processStopped(QStringList image_filenames,QImage result){
	ui->statusBar->clearMessage();
	ui->pushButton->setEnabled(true);
	ui->pushButton_2->setEnabled(true);

	//Output image to scene
	QPixmap pixmap= QPixmap::fromImage(result);

	if(!ui->graphicsView->scene())
		ui->graphicsView->setScene(&scene);
	ui->graphicsView->scene()->clear();
	ui->graphicsView->items().clear();
	ui->graphicsView->setSceneRect(0,0,pixmap.width(),pixmap.height());
	ui->graphicsView->scene()->addPixmap(pixmap);
}



ProcessThread::ProcessThread(){mw=NULL;}

void ProcessThread::extract(QStringList image_filenames){
	wait();
	this->image_filenames= image_filenames;

	QObject::connect(this, SIGNAL(processStarted(QString)),
					 mw, SLOT(processStarted(QString)),Qt::AutoConnection);
	QObject::connect(this, SIGNAL(processStopped(QStringList,QImage)),
					 mw, SLOT(processStopped(QStringList,QImage)),Qt::AutoConnection);
	start();
}

void ProcessThread::run(){
	emit processStarted("SIFT processing...");
	Process process;
	if(image_filenames.size()==1)
		emit processStopped(image_filenames,process.run(image_filenames.first()));
	else if(image_filenames.size()==2)
		emit processStopped(image_filenames,process.run(image_filenames.first(),image_filenames.last()));
	else
		emit processStopped(image_filenames,QImage(0,0));
}

void ProcessThread::setUi(MainWindow* mw){
	this->mw=mw;
}

