#include "processthread.h"
#include "mainwindow.h"

#include <Reconstruction/reconstructor.h>
#include <Reconstruction/sculptor.h>

#include <QFile>
#include <QElapsedTimer>

ProcessThread::ProcessThread(){mw=NULL;}

void ProcessThread::reconstruct3DShape(QStringList image_filenames,double scaleXY,double scaleZ,double angle,bool preferScaleFromMetadata){
	wait();
	this->image_filenames= image_filenames;
	this->scaleXY= scaleXY, this->scaleZ= scaleZ, this->angle= angle;
	this->preferScaleFromMetadata= preferScaleFromMetadata;

	start();
}

void ProcessThread::run(){
	emit processStarted();
	QElapsedTimer stopwatch;
	stopwatch.start();

	cybervision::Reconstructor reconstructor(NULL);

	connect(&reconstructor,&cybervision::Reconstructor::sgnLogMessage,
					 this,&ProcessThread::sgnLogMessage,Qt::AutoConnection);
	connect(&reconstructor,&cybervision::Reconstructor::sgnStatusMessage,
					 this,&ProcessThread::sgnStatusMessage,Qt::AutoConnection);

	bool reconstructor_success= false;
	if(image_filenames.size()==2){
		//Run reconstruction
		reconstructor_success= reconstructor.run(image_filenames.first(),image_filenames.last(),angle);
	}else
		emit processStopped(tr("Need exactly 2 images for reconstruction"));

	if(reconstructor_success){
		//Run surface generation
		double scaleMetadata= reconstructor.getScaleMetadata();
		if(scaleMetadata>0 && preferScaleFromMetadata){
			scaleXY= scaleMetadata;
			scaleZ= scaleMetadata;
		}
		emit processUpdated(tr("Creating 3D surface"),tr("Creating 3D surface..."));
		cybervision::Sculptor sculptor(reconstructor.get3DPoints(),reconstructor.getImageSize(),scaleXY,scaleZ);

		cybervision::Surface surface= sculptor.getSurface();

		surface.setTextures(reconstructor.getImage1(),reconstructor.getImage2());

		//Output time
		{
			qint64 msecs= stopwatch.elapsed();
			qint64 mins= msecs/(1000*60);
			msecs-= mins*(1000*60);
			qint64 secs= msecs/1000;
			msecs-= secs*1000;
			QString timeString= QString(tr("Reconstruction completed in %1:%2.%3"))
					.arg((int)mins,2,10,QChar('0'))
					.arg((int)secs,2,10,QChar('0'))
					.arg((int)msecs,3,10,QChar('0'));
			emit processUpdated(timeString,QString());
		}
		emit processStopped(QString(),surface);
	}else
		emit processStopped(reconstructor.getErrorString());
}

void ProcessThread::setUi(MainWindow* mw){
	if(this->mw){
		disconnect(this,&ProcessThread::processStarted,
							this->mw,&MainWindow::processStarted);
		disconnect(this,&ProcessThread::processStopped,
							this->mw,&MainWindow::processStopped);
		disconnect(this,&ProcessThread::processUpdated,
							this->mw,&MainWindow::processUpdated);
	}
	this->mw=mw;
	qRegisterMetaType< QList<QVector3D> >("QList<QVector3D>");
	qRegisterMetaType< cybervision::Surface >("cybervision::Surface)");

	connect(this,&ProcessThread::processStarted,
					 mw,&MainWindow::processStarted,Qt::AutoConnection);
	connect(this,&ProcessThread::processStopped,
					 mw,&MainWindow::processStopped,Qt::AutoConnection);
	connect(this,&ProcessThread::processUpdated,
					 mw,&MainWindow::processUpdated,Qt::AutoConnection);
}

//Message passing to MainWindow
void ProcessThread::sgnLogMessage(QString str){
	emit processUpdated(str,QString());
}

void ProcessThread::sgnStatusMessage(QString str){
	emit processUpdated(QString(),str);
}
