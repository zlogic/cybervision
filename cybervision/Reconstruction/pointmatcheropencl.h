#ifndef POINTMATCHEROPENCL_H
#define POINTMATCHEROPENCL_H

#include <QObject>
#include <QString>
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
  Class for matching points with OpenCL. Most of the code is borrowed/recycled from AMD APP SDK.
  */
class PointMatcherOpenCL : public QObject
{
    Q_OBJECT
protected:
	QString kernelStr;

	bool kernelInitialized;
	bool inputVectorsCopied;

	//OpenCL kernel paramaters
	cl_uint vectorSize,inputVectorsBufferSize,inputVectorsCount;
	cl_float *vector;
	cl_float *input;
	cl_float *output;

	//OpenCL stuff
	cl_mem   inputBuffer;
	cl_mem	 outputBuffer;
	cl_mem	 vectorBuffer;

	cl_context          context;
	cl_device_id        *devices;
	cl_command_queue    commandQueue;

	cl_program program;
	cl_kernel  kernel;
	size_t    kernelWorkGroupSize;

	//Runs OpenCL for the prepared buffers
	bool CalcDistances();
public:
	explicit PointMatcherOpenCL(int vectorSize, int maxVectorsCount, QObject *parent = 0);
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
