#ifndef PROCESS_H
#define PROCESS_H

#include <QObject>

#include <QStringList>
#include <QImage>

class Process:public QObject{
	Q_OBJECT
	protected:
public:
    Process();

	QImage run(QString filename1,QString filename2);
	QImage run(QString filename);

signals:
	void processCompleted(QImage img);

};

#endif // PROCESS_H
