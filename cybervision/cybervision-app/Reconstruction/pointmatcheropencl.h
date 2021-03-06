#ifdef CYBERVISION_OPENCL
#ifndef POINTMATCHEROPENCL_H
#define POINTMATCHEROPENCL_H

#include <QObject>
#include <QString>
#include <QScopedArrayPointer>
#include <Reconstruction/pointmatch.h>
#include <SIFT/siftgateway.h>

//AMD APP hack for linking errors
//#define __CYGWIN__ 1
//Or patch cl/cl_platform.h by adding "__MINGW32__" to the list of define checks:
//  #if defined(_WIN32) || defined(__CYGWIN__) || defined(__MINGW32__)

#ifdef __MINGW32__
#define __CYGWIN__
#endif
#include <CL/cl.h>

namespace cybervision{
/*
 * Class for matching points with OpenCL. Most of the code is borrowed/recycled from AMD APP SDK.
 */
class PointMatcherOpenCL : public QObject
{
	Q_OBJECT
protected:
	QString kernelStr;

	bool kernelInitialized;
	bool kernelFirstRun;

	//OpenCL kernel paramaters
	cl_uint vectorSize,inputVectorsBufferSize;
	QScopedArrayPointer<cl_float> input1,input2;
	QScopedArrayPointer<cl_float> output;

	//OpenCL stuff
	cl_mem input1Buffer,input2Buffer;
	cl_mem outputBuffer;

	cl_context context;
	QScopedArrayPointer<cl_device_id> devices;
	cl_command_queue commandQueue;

	cl_program program;
	cl_kernel kernel;
	size_t kernelWorkGroupSizeFull;
	size_t kernelWorkGroupSize1,kernelWorkGroupSize2;
	size_t kernelWorkGroupSize2Max;

	//Runs OpenCL for the prepared buffers
	bool CalcDistances();

	//Compute distances with OpenCL GPU and OpenMP CPU
	SortedKeypointMatches CalcDistancesHybrid(const QList<SIFT::Keypoint>& keypoints1,const QList<SIFT::Keypoint>& keypoints2);
	//Compute distances with OpenCL only
	SortedKeypointMatches CalcDistancesOpenCL(const QList<SIFT::Keypoint>& keypoints1,const QList<SIFT::Keypoint>& keypoints2);
public:
	explicit PointMatcherOpenCL(int vectorSize, QObject *parent = 0);
	~PointMatcherOpenCL();

	//OpenCL init/release code. Returns true on success, emits messages and returns false on errors.
	bool InitCL();
	bool ShutdownCL();

	//Returns the best match for a keypoints from image 1 to a list of keypoints in image 2
	SortedKeypointMatches CalcDistances(const QList<SIFT::Keypoint>& keypoints1,const QList<SIFT::Keypoint>& keypoints2);
signals:
	void sgnLogMessage(QString);
public slots:

};
}

#endif // POINTMATCHEROPENCL_H
#else

#include <QObject>
//Dummy class to prevent compiler (Qt moc) warnings
namespace cybervision{
class PointMatcherOpenCL : public QObject
{
	Q_OBJECT
};

}
#endif // CYBERVISION_OPENCL
