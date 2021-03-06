#ifndef SIFTGATEWAY_H
#define SIFTGATEWAY_H

#include <QList>
#include <QImage>
#include <QMutex>
#include <SIFT/msvc-exports.h>

#include <limits>

namespace SIFT{

class DLL_API Keypoint{
	friend class Extractor;
protected:
	int x,y;
	float scale, orientation;
	float descriptor[128];
	Keypoint(int row,int column,float scale,float orientation,const float descriptor[128]);
public:
	Keypoint();
	Keypoint(const Keypoint&);
	void operator =(const Keypoint&);
	bool operator==(const Keypoint&)const;
	float distance(const Keypoint&,const double MaxKeypointDistanceSquared=std::numeric_limits<float>::infinity())const;
	float getX()const;
	float getY()const;
	inline float operator[](size_t i)const{return descriptor[i];}
};

class DLL_API Extractor{
protected:
	static QMutex mutex;
	double SIFTContrastCorrection;
	QImage resizeImage(const QImage& src,int scalingFactor)const;
public:
	Extractor(double SIFTContrastCorrection);
	virtual ~Extractor();
	QList<SIFT::Keypoint> extract(const QImage&)const;
};

}

#endif // SIFTGATEWAY_H
