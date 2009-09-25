#include "siftgateway.h"
#include "libsiftfast-1.1-src/siftfast.h"
#include <cmath>

//Keypoint class
SIFT::Keypoint::Keypoint(int row,int column,float scale,float orientation,const float descriptor[128]){
	this->x= column, this->y= row;
	this->scale= scale;
	this->orientation= orientation;
	for(int i=0;i<128;i++)
		this->descriptor[i]=descriptor[i];
}

SIFT::Keypoint::Keypoint(){
	x=0,y=0;
	orientation=0;
	scale=0;
	for(int i=0;i<128;i++)
		descriptor[i]=0;
}
SIFT::Keypoint::Keypoint(const Keypoint& keypoint){
	operator=(keypoint);
}

void SIFT::Keypoint::operator =(const Keypoint&keypoint){
	x=keypoint.x, y=keypoint.y;
	orientation= keypoint.orientation;
	scale=keypoint.scale;
	for(int i=0;i<128;i++)
		descriptor[i]=keypoint.descriptor[i];
}

bool SIFT::Keypoint::operator ==(const Keypoint& keypoint)const{
	bool location_equal= x==keypoint.x && y==keypoint.y && orientation==keypoint.orientation && scale==keypoint.scale;
	if(!location_equal)
		return false;
	for(int i=0;i<128;i++)
		if(descriptor[i]!=keypoint.descriptor[i])
			return false;
	return true;
}

float SIFT::Keypoint::distance(const Keypoint& keypoint)const{
	float distance_squared=0;
	for(int i=0;i<128;i++){
		float di= descriptor[i]-keypoint.descriptor[i];
		distance_squared+= di*di;
	}
	return sqrt(distance_squared);
}
float SIFT::Keypoint::getX()const{return x;}
float SIFT::Keypoint::getY()const{return y;}

//Class for accessing the libsiftfast library
QMutex SIFT::Extractor::mutex;//static declaration

SIFT::Extractor::Extractor(){
}

SIFT::Extractor::~Extractor(){
	QMutexLocker locker(&mutex);
	DestroyAllResources();
}

QList<SIFT::Keypoint> SIFT::Extractor::extract(const QImage& sourceImg)const{
	QMutexLocker locker(&mutex);
	QImage img= sourceImg.format()==QImage::Format_RGB32?sourceImg:sourceImg.convertToFormat(QImage::Format_RGB32);
	int width=img.width(), height=img.height();
	Image siftfastImage= CreateImage(height,width);

	//Convert to ZSIFT internal format
	const uchar *imgPixels= img.bits();
	int imgStride= img.bytesPerLine();

	float* siftfastPixels= siftfastImage->pixels;
	int siftfastStride= siftfastImage->stride;
	for(int y=0;y<height;y++){
		for(int x=0;x<width;x++){
			const QRgb pixel= *reinterpret_cast<const QRgb*>(&imgPixels[y*imgStride+x*sizeof(QRgb)]);
			float pixelGray= qGray(pixel);
			pixelGray/=255;

			siftfastPixels[y*siftfastStride+x]= pixelGray;
		}
	}

	img= QImage();

	//Detect SIFT features
	::Keypoint keypoints= GetKeypoints(siftfastImage);
	DestroyAllImages();

	//Return SIFT features
	QList<SIFT::Keypoint> points;
	while(keypoints){
		points.append(SIFT::Keypoint(keypoints->row,keypoints->col,keypoints->scale,keypoints->ori,keypoints->descrip));
		keypoints= keypoints->next;
	}
	FreeKeypoints(keypoints);

	return points;
}
