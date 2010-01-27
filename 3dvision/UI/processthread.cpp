#include "processthread.h"
#include "mainwindow.h"

#include <Reconstruction/reconstructor.h>

#include <QSharedPointer>
#include <QFile>

ProcessThread::ProcessThread(){mw=NULL;}

void ProcessThread::extract(QStringList image_filenames,QString output_filename){
	wait();
	this->image_filenames= image_filenames;
	this->output_filename= output_filename;

	start();
}

void ProcessThread::run(){
	emit processStarted();
	cybervision::Reconstructor reconstructor;

	QObject::connect(&reconstructor, SIGNAL(sgnLogMessage(QString)),
					 this, SLOT(sgnLogMessage(QString)),Qt::AutoConnection);
	QObject::connect(&reconstructor, SIGNAL(sgnStatusMessage(QString)),
					 this, SLOT(sgnStatusMessage(QString)),Qt::AutoConnection);

	if(image_filenames.size()==2){
		//Prepare variables for output data

		if(reconstructor.run(image_filenames.first(),image_filenames.last()))
			emit processStopped(QString());
		else
			emit processStopped(reconstructor.getErrorString());
	}else
		emit processStopped("Need exactly 2 images for reconstruction");
}

void ProcessThread::setUi(MainWindow* mw){
	if(this->mw){
		QObject::disconnect(this, SIGNAL(processStarted()),
							this->mw, SLOT(processStarted()));
		QObject::disconnect(this, SIGNAL(processStopped(QString)),
							this->mw, SLOT(processStopped(QString)));
		QObject::disconnect(this, SIGNAL(processUpdated(QString,QString)),
							this->mw, SLOT(processUpdated(QString,QString)));
	}
	this->mw=mw;

	QObject::connect(this, SIGNAL(processStarted()),
					 mw, SLOT(processStarted()),Qt::AutoConnection);
	QObject::connect(this, SIGNAL(processStopped(QString)),
					 mw, SLOT(processStopped(QString)),Qt::AutoConnection);
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
