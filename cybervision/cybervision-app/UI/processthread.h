#ifndef PROCESSTHREAD_H
#define PROCESSTHREAD_H

#include <QThread>
#include <QStringList>
#include <QImage>
#include <QVector3D>
#include <QList>

#include <Reconstruction/surface.h>

class MainWindow;
class ProcessThread :public QThread{
	Q_OBJECT
protected:
	MainWindow *mw;

	//For reconstruction task
	QStringList image_filenames;
	qreal scaleXY, scaleZ,angle;
	QSize imageSize;
	bool preferScaleFromMetadata;

	void runExtract();
	void runSurface();
public:
	ProcessThread();
	void setUi(MainWindow *nw=NULL);

	void reconstruct3DShape(QStringList image_filenames,qreal scaleXY,qreal scaleZ,qreal angle,bool preferScaleFromMetadata=false);
	void run();//the main thread loop
signals:
	void processStarted();
	void processUpdated(QString logMessage,QString statusBarText);
	void processStopped(QString resultText,cybervision::Surface=cybervision::Surface());
private slots:
	void sgnLogMessage(QString);
	void sgnStatusMessage(QString);
};


#endif // PROCESSTHREAD_H
