#ifndef PROCESSTHREAD_H
#define PROCESSTHREAD_H

#include <QThread>
#include <QStringList>
#include <QImage>
#include <QVector3D>
#include <QList>
#include <QSharedPointer>

#include <Reconstruction/surface.h>

class MainWindow;
class ProcessThread :public QThread{
	Q_OBJECT
protected:
	MainWindow *mw;
	enum Task{TASK_RECONSTRUCTION,TASK_SURFACE};
	Task task;

	//For reconstruction task
	QStringList image_filenames;
	double scaleXY, scaleZ;
	QSize imageSize;
	//For extraction task
	QList<QVector3D> points;

	void runExtract();
	void runSurface();
public:
	ProcessThread();
	void setUi(MainWindow *nw=NULL);

	void extract(QStringList image_filenames,double scaleXY,double scaleZ);
	void surface(QList<QVector3D> points,QSize imageSize);
	void run();//the main thread loop
signals:
	void processStarted();
	void processUpdated(QString logMessage,QString statusBarText);
	void processStopped(QString resultText,QList<QVector3D> points=QList<QVector3D>(),QSize imageSize=QSize());
	void processStopped(QString resultText,cybervision::Surface);
private slots:
	void sgnLogMessage(QString);
	void sgnStatusMessage(QString);
};


#endif // PROCESSTHREAD_H
