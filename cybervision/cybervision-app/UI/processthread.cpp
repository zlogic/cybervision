#include "processthread.h"
#include "mainwindow.h"

#include <Reconstruction/reconstructor.h>
#include <Reconstruction/sculptor.h>

#include <QSharedPointer>
#include <QFile>
#include <QElapsedTimer>

ProcessThread::ProcessThread(){mw=NULL;}

void ProcessThread::reconstruct3DShape(QStringList image_filenames,qreal scaleXY,qreal scaleZ,qreal angle,bool preferScaleFromMetadata){
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

	QObject::connect(&reconstructor, SIGNAL(sgnLogMessage(QString)),
					 this, SLOT(sgnLogMessage(QString)),Qt::AutoConnection);
	QObject::connect(&reconstructor, SIGNAL(sgnStatusMessage(QString)),
					 this, SLOT(sgnStatusMessage(QString)),Qt::AutoConnection);

	bool reconstructor_success= false;
	if(image_filenames.size()==2){
		//Run reconstruction
		reconstructor_success= reconstructor.run(image_filenames.first(),image_filenames.last(),angle);
	}else
		emit processStopped(tr("Need exactly 2 images for reconstruction"));

	if(reconstructor_success){
		//Run surface generation
		qreal scaleMetadata= reconstructor.getScaleMetadata();
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
			emit sgnLogMessage(timeString);
		}
		emit processStopped(QString(),surface);
	}else
		emit processStopped(reconstructor.getErrorString());
}

void ProcessThread::setUi(MainWindow* mw){
	if(this->mw){
		QObject::disconnect(this, SIGNAL(processStarted()),
							this->mw, SLOT(processStarted()));
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
