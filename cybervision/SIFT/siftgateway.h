#ifndef SIFTGATEWAY_H
#define SIFTGATEWAY_H

#include <QList>
#include <QImage>
#include <QMutex>

namespace SIFT{
	class Keypoint{
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
		float distance(const Keypoint&)const;
		float getX()const;
		float getY()const;
		inline float operator[](size_t i)const{return descriptor[i];}
	};

	class Extractor{
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
