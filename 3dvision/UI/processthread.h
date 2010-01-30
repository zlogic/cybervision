#ifndef PROCESSTHREAD_H
#define PROCESSTHREAD_H

#include <QThread>
#include <QStringList>
#include <QImage>
#include <QVector3D>
#include <QList>

class MainWindow;
class ProcessThread :public QThread{
	Q_OBJECT
protected:
	MainWindow *mw;
	QStringList image_filenames;
	QString output_filename;
public:
	ProcessThread();
	void setUi(MainWindow *nw=NULL);

	void extract(QStringList image_filenames,QString output_filename="");
	void run();//the main thread loop
signals:
	void processStarted();
	void processUpdated(QString logMessage,QString statusBarText);
	void processStopped(QString resultText,QList<QVector3D> points=QList<QVector3D>());
private slots:
	void sgnLogMessage(QString);
	void sgnStatusMessage(QString);
};


#endif // PROCESSTHREAD_H
