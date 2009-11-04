#ifndef PROCESS_H
#define PROCESS_H

#include <QObject>

#include <QStringList>
#include <QImage>
#include <QTextStream>

class Process:public QObject{
	Q_OBJECT
	protected:
public:
    Process();

	enum OutputMode{PROCESS_OUTPUT_IMAGE=1,PROCESS_OUTPUT_STRING=2};
	bool run(QString filename1,QString filename2,OutputMode,QImage& outputImage,QTextStream& outputString);
	QImage run(QString filename);

signals:
	void processCompleted(QImage img);

};

#endif // PROCESS_H
