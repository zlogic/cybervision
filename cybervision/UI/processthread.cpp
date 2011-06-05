#include "processthread.h"
#include "mainwindow.h"

#include <Reconstruction/reconstructor.h>
#include <Reconstruction/sculptor.h>

#include <QSharedPointer>
#include <QFile>

ProcessThread::ProcessThread(){mw=NULL;}

void ProcessThread::extract(QStringList image_filenames,qreal scaleXY,qreal scaleZ,qreal angle){
	wait();
	task=TASK_RECONSTRUCTION;
	this->image_filenames= image_filenames;
	this->scaleXY= scaleXY, this->scaleZ= scaleZ, this->angle= angle;

	start();
}

void ProcessThread::surface(QList<QVector3D>points,QSize imageSize,double scaleMetadata){
	wait();
	task=TASK_SURFACE;
	this->points= points;
	this->imageSize= imageSize;

	if(scaleMetadata>0){
		scaleXY= scaleMetadata;
		scaleZ= scaleMetadata;
	}

	start();
}

void ProcessThread::run(){
	switch(task){
	case TASK_RECONSTRUCTION:
		runExtract();
		break;
	case TASK_SURFACE:
		runSurface();
		break;
	}
}

void ProcessThread::runExtract(){
	emit processStarted();
	cybervision::Reconstructor reconstructor(NULL);

	QObject::connect(&reconstructor, SIGNAL(sgnLogMessage(QString)),
					 this, SLOT(sgnLogMessage(QString)),Qt::AutoConnection);
	QObject::connect(&reconstructor, SIGNAL(sgnStatusMessage(QString)),
					 this, SLOT(sgnStatusMessage(QString)),Qt::AutoConnection);

	if(image_filenames.size()==2){
		//Prepare variables for output data

		if(reconstructor.run(image_filenames.first(),image_filenames.last(),angle))
			emit processStopped(QString(),reconstructor.get3DPoints(),reconstructor.getImageSize(),reconstructor.getScaleMetadata());
		else
			emit processStopped(reconstructor.getErrorString());
	}else
		emit processStopped("Need exactly 2 images for reconstruction");
}


void ProcessThread::runSurface(){
	emit processUpdated("Creating 3D surface","Creating 3D surface...");
	cybervision::Sculptor sculptor(points,imageSize,scaleXY,scaleZ);

	points.clear();

	emit processStopped(QString(),sculptor.getSurface());
}


void ProcessThread::setUi(MainWindow* mw){
	if(this->mw){
		QObject::disconnect(this, SIGNAL(processStarted()),
							this->mw, SLOT(processStarted()));
		QObject::disconnect(this, SIGNAL(processStopped(QString,QList<QVector3D>,QSize,double)),
							this->mw, SLOT(processStopped(QString,QList<QVector3D>,QSize,double)));
		QObject::disconnect(this, SIGNAL(processStopped(QString,cybervision::Surface)),
							this->mw, SLOT(processStopped(QString,cybervision::Surface)));
		QObject::disconnect(this, SIGNAL(processUpdated(QString,QString)),
							this->mw, SLOT(processUpdated(QString,QString)));
	}
	this->mw=mw;
	qRegisterMetaType< QList<QVector3D> >("QList<QVector3D>");
	qRegisterMetaType< cybervision::Surface >("cybervision::Surface)");

	QObject::connect(this, SIGNAL(processStarted()),
					 mw, SLOT(processStarted()),Qt::AutoConnection);
	QObject::connect(this, SIGNAL(processStopped(QString,QList<QVector3D>,QSize,double)),
					 mw, SLOT(processStopped(QString,QList<QVector3D>,QSize,double)),Qt::AutoConnection);
	QObject::connect(this, SIGNAL(processStopped(QString,cybervision::Surface)),
					 mw, SLOT(processStopped(QString,cybervision::Surface)),Qt::AutoConnection);
	QObject::connect(this, SIGNAL(processUpdated(QString,QString)),
					 mw, SLOT(processUpdated(QString,QString)),Qt::AutoConnection);
}

//Message passing to MainWindow
void ProcessThread::sgnLogMessage(QString str){
	emit processUpdated(str,QString());
}

void ProcessThread::sgnStatusMessage(QString str){
	emit processUpdated(QString(),str);
}
