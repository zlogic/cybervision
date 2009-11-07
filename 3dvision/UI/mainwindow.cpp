#include "mainwindow.h"
#include "ui_mainwindow.h"

#include "../SIFT/process.h"

#include <QFileDialog>
#include <QGraphicsItem>
#include <QSharedPointer>

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
			QString filter= "Jpeg images (*.jpeg *.jpg);;Png images (*.png);;Tiff images(*.tiff *.tif);;BMP images(*.bmp)";
			QString fileName = QFileDialog::getSaveFileName(this,"Save the SIFT keypoint result","",filter,0,0);

			item->pixmap().save(fileName);
			break;
		}
	}
}

void MainWindow::on_pushButton_3_clicked(){
	QString filter= "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp);;All files(*.*)";
	QStringList filenames = QFileDialog::getOpenFileNames(this,"Select images to load","",filter,0,0);
	for(QStringList::iterator it=filenames.begin();it!=filenames.end();it++)
		(*it)=QDir::convertSeparators(*it);


	QString saveFilter= "Combination data (*.txt)";
	QString saveFileName = QFileDialog::getSaveFileName(this,"Save the result","",saveFilter,0,0);

	if(saveFileName.isNull())
		saveFileName="";
	thread.extract(filenames,saveFileName);
}

void MainWindow::processStarted(QString statusBarText){
	ui->statusBar->showMessage(statusBarText);
	ui->pushButton->setEnabled(false);
	ui->pushButton_2->setEnabled(false);
	ui->pushButton_3->setEnabled(false);
}

void MainWindow::processStopped(QStringList image_filenames,QImage result){
	ui->statusBar->clearMessage();
	ui->pushButton->setEnabled(true);
	if(result.width()>0 && result.height()>0)
		ui->pushButton_2->setEnabled(true);
	ui->pushButton_3->setEnabled(true);

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

void ProcessThread::extract(QStringList image_filenames,QString output_filename){
	wait();
	this->image_filenames= image_filenames;
	this->output_filename= output_filename;

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
	else if(image_filenames.size()==2){
		//Prepare variables for output data
		QImage resultImage(0,0);

		QSharedPointer<QTextStream> resultStream;
		QSharedPointer<QFile> outputFile;
		if(!output_filename.isEmpty()){
			outputFile=QSharedPointer<QFile>(new QFile(output_filename));
			if(outputFile->open(QFile::WriteOnly))
				resultStream= QSharedPointer<QTextStream>(new QTextStream(outputFile.data()));
		}
		Process::OutputMode outputMode = output_filename.isEmpty()?Process::PROCESS_OUTPUT_IMAGE:Process::PROCESS_OUTPUT_STRING;
		process.run(image_filenames.first(),image_filenames.last(),outputMode,resultImage,*resultStream.data());
		if(outputFile)
			outputFile->close();
		emit processStopped(image_filenames,resultImage);
	}else
		emit processStopped(image_filenames,QImage(0,0));
}

void ProcessThread::setUi(MainWindow* mw){
	this->mw=mw;
}


