#ifndef POINTMATCHEROPENCL_H
#define POINTMATCHEROPENCL_H

#include <QObject>
#include <QString>

//AMD APP hack for linking errors
//#define __CYGWIN__ 1
//Or patch cl/cl_platform.h by adding "__MINGW32__" to the list of define checks:
//  #if defined(_WIN32) || defined(__CYGWIN__) || defined(__MINGW32__)
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

	cl_uint width;
	cl_uint *input;
	cl_uint *output;

	cl_mem   inputBuffer;
	cl_mem	 outputBuffer;

	cl_context          context;
	cl_device_id        *devices;
	cl_command_queue    commandQueue;

	cl_program program;
	cl_kernel  kernel;

public:
    explicit PointMatcherOpenCL(QObject *parent = 0);
	~PointMatcherOpenCL();

	//OpenCL init/release code
	bool InitCL();
	bool ShutdownCL();
signals:
	void sgnLogMessage(QString);
public slots:

};
}

#endif // POINTMATCHEROPENCL_H
