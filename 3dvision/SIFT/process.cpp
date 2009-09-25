#include "process.h"
#include <siftgateway.h>

#include <limits>

#include <QMap>
#include <QPainter>
#include <QPair>
//#include <QDebug>

bool operator<(const QPoint p1,const QPoint p2){
	if(p1.y()<p2.y())
		return true;
	else if(p1.y()==p2.y()){
		if(p1.x()<p2.x())
			return true;
		else return false;
	}else return false;
}

typedef QPair<QPoint,QPoint> KeypointMatch;

Process::Process(){}

QImage Process::run(QString filename1,QString filename2){
	QImage img1(filename1),img2(filename2);
	QList <SIFT::Keypoint> keypoints1,keypoints2;
	{
		SIFT::Extractor extractor;
		keypoints1= extractor.extract(img1);
		keypoints2= extractor.extract(img2);
	}

	QMap<float,QList<KeypointMatch> > matchlist;
	//match first image with second
	if(true){
		for(QList<SIFT::Keypoint>::const_iterator it1= keypoints1.begin();it1!=keypoints1.end();it1++){
			KeypointMatch match;
			float minDistance= std::numeric_limits<float>::infinity();
			for(QList<SIFT::Keypoint>::const_iterator it2= keypoints2.begin();it2!=keypoints2.end();it2++){
				float distance= it1->distance(*it2);
				if(distance<minDistance){
					minDistance= distance;
					match= KeypointMatch(QPoint((int)it1->getX(),(int)it1->getY()),QPoint((int)it2->getX(),(int)it2->getY()));
				}
			}
			if(minDistance!=std::numeric_limits<float>::infinity()){
				QList<KeypointMatch>& dst= matchlist[minDistance];
				if(!dst.contains(match))
					dst.append(match);
			}
		}
		//match second image with first
		for(QList<SIFT::Keypoint>::const_iterator it1= keypoints2.begin();it1!=keypoints2.end();it1++){
			KeypointMatch match;
			float minDistance= std::numeric_limits<float>::infinity();
			for(QList<SIFT::Keypoint>::const_iterator it2= keypoints1.begin();it2!=keypoints1.end();it2++){
				float distance= it1->distance(*it2);
				if(distance<minDistance){
					minDistance= distance;
					match= KeypointMatch(QPoint((int)it2->getX(),(int)it2->getY()),QPoint((int)it1->getX(),(int)it1->getY()));
				}
			}
			if(minDistance!=std::numeric_limits<float>::infinity()){
				QList<KeypointMatch>& dst= matchlist[minDistance];
				if(!dst.contains(match))
					dst.append(match);
			}
		}
	}else{
		for(QList<SIFT::Keypoint>::const_iterator it1= keypoints1.begin();it1!=keypoints1.end();it1++){
			for(QList<SIFT::Keypoint>::const_iterator it2= keypoints2.begin();it2!=keypoints2.end();it2++){
				float distance= it1->distance(*it2);
				QList<KeypointMatch>& dst= matchlist[distance];
				KeypointMatch match= KeypointMatch(QPoint((int)it1->getX(),(int)it1->getY()),QPoint((int)it2->getX(),(int)it2->getY()));
				if(!dst.contains(match))
					dst.append(match);
			}
		}
	}

	//Output result
	QImage result(qMax(img1.width(),img2.width()),img1.height()+img2.height(),QImage::Format_RGB32);
	QPainter painter(&result);
	painter.drawImage(0,0,img1);
	painter.drawImage(0,img1.height(),img2);
	painter.setPen(QPen(QColor("orange")));
	int height= img1.height();
	//Draw lines
	for(QMap<float,QList<KeypointMatch> >::const_iterator it= matchlist.begin();it!=matchlist.end();it++){
		if(it.key()>=0.4)
			break;
		for(QList<KeypointMatch>::ConstIterator jt=it.value().begin();jt!=it.value().end();jt++){
			//qDebug()<<"drawing line from"<<(*jt).first<<" to "<<QPoint((*jt).second.x(),(*jt).second.y()+height);
			painter.drawLine((*jt).first,QPoint((*jt).second.x(),(*jt).second.y()+height));
		}
	}

	//Draw points
	painter.setPen(QPen(QColor("yellow")));
	for(QList<SIFT::Keypoint>::const_iterator it= keypoints1.begin();it!=keypoints1.end();it++)
		painter.drawPoint((int)it->getX(),(int)it->getY());
	for(QList<SIFT::Keypoint>::const_iterator it= keypoints2.begin();it!=keypoints2.end();it++)
		painter.drawPoint((int)it->getX(),(int)it->getY()+height);

	painter.end();

	return result;
}

QImage Process::run(QString filename){
	QImage bitmap(filename);
	SIFT::Extractor extractor;
	QList <SIFT::Keypoint> keypoints= extractor.extract(bitmap);

	//Output result
	QPainter painter(&bitmap);
	painter.setPen(QColor("yellow"));
	for(QList<SIFT::Keypoint>::const_iterator it= keypoints.begin();it!=keypoints.end();it++)
		painter.drawPoint((int)it->getX(),(int)it->getY());

	painter.end();

	return bitmap;
}
