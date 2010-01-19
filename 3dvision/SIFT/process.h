#ifndef PROCESS_H
#define PROCESS_H

#include <QObject>

#include <QStringList>
#include <QImage>
#include <QTextStream>

class Process:public QObject{
	Q_OBJECT
	protected:
	float getDistanceThreshold();
public:
    Process();

	enum OutputMode{PROCESS_OUTPUT_IMAGE=1,PROCESS_OUTPUT_STRING=2};
	//Searches for SIFT matches between two images
	bool run(QString filename1,QString filename2,OutputMode,QImage& outputImage,QTextStream& outputString);
	//Searches for SIFT keypoints and returns an image with the keypoints displayed
	QImage run(QString filename);

signals:
	void processCompleted(QImage img);

};

#endif // PROCESS_H
