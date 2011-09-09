#include "pointmatcheropencl.h"

#include <QFile>
#include <QTextStream>

#include <cstdlib>


namespace cybervision{
	PointMatcherOpenCL::PointMatcherOpenCL(QObject *parent) : QObject(parent){
		//Read kernel from file
		{
			QFile kernelFile(":/opencl/PointMatcherKernel.cl");
			kernelFile.open(QIODevice::ReadOnly);
			QTextStream stream(&kernelFile);
			kernelStr= stream.readAll();
		}

		width				= 256;
		input				= NULL;
		output				= NULL;

		/////////////////////////////////////////////////////////////////
		// Allocate and initialize memory used by host
		/////////////////////////////////////////////////////////////////
		cl_uint sizeInBytes = width * sizeof(cl_uint);
		input = (cl_uint *) malloc(sizeInBytes);
		output = (cl_uint *) malloc(sizeInBytes);
	}
	PointMatcherOpenCL::~PointMatcherOpenCL(){

	}

	bool PointMatcherOpenCL::InitCL(){
		if(input == NULL){
			emit sgnLogMessage(QString("OpenCL Error: Failed to allocate input memory on host"));
			return false;
		}
		if(output == NULL){
			emit sgnLogMessage(QString("OpenCL Error: Failed to allocate output memory on host"));
			return false;
		}

		cl_int status = 0;
		size_t deviceListSize;

		/*
		 * Have a look at the available platforms and pick either
		 * the AMD one if available or a reasonable default.
		 */
		cl_uint numPlatforms;
		cl_platform_id platform = NULL;
		status = clGetPlatformIDs(0, NULL, &numPlatforms);
		if(status != CL_SUCCESS){
			emit sgnLogMessage(QString("OpenCL Error: Getting Platforms. (clGetPlatformsIDs) code=%1").arg(status));
			return false;
		}

		if(numPlatforms > 0)
		{
			cl_platform_id* platforms = new cl_platform_id[numPlatforms];
			status = clGetPlatformIDs(numPlatforms, platforms, NULL);
			if(status != CL_SUCCESS){
				emit sgnLogMessage(QString("OpenCL Error: Getting Platform Ids. (clGetPlatformsIDs) code=%1").arg(status));
				return false;
			}

			for(unsigned int i=0; i < numPlatforms; ++i)
			{
				char pbuff[100];
				status = clGetPlatformInfo(
							platforms[i],
							CL_PLATFORM_VENDOR,
							sizeof(pbuff),
							pbuff,
							NULL);
				if(status != CL_SUCCESS){
					emit sgnLogMessage(QString("OpenCL Error: Getting Platform Info.(clGetPlatformInfo) code=%1").arg(status));
					return false;
				}
				platform = platforms[i];
				if(!strcmp(pbuff, "Advanced Micro Devices, Inc."))
				{
					break;
				}
			}
			delete platforms;
		}

		if(NULL == platform){
			emit sgnLogMessage(QString("OpenCL Error: NULL platform found so Exiting Application."));
			return false;
		}

		/*
		 * If we could find our platform, use it. Otherwise use just available platform.
		 */
		cl_context_properties cps[3] = { CL_CONTEXT_PLATFORM, (cl_context_properties)platform, 0 };

		/////////////////////////////////////////////////////////////////
		// Create an OpenCL context
		/////////////////////////////////////////////////////////////////
		context = clCreateContextFromType(cps,
										  CL_DEVICE_TYPE_CPU,
										  NULL,
										  NULL,
										  &status);
		if(status != CL_SUCCESS){
			emit sgnLogMessage(QString("OpenCL Error: Creating Context. (clCreateContextFromType) code=%1").arg(status));
			return false;
		}

		/* First, get the size of device list data */
		status = clGetContextInfo(context,
								  CL_CONTEXT_DEVICES,
								  0,
								  NULL,
								  &deviceListSize);
		if(status != CL_SUCCESS){
			emit sgnLogMessage(QString("OpenCL Error: Getting Context Info (device list size, clGetContextInfo) code=%1").arg(status));
			return false;
		}

		/////////////////////////////////////////////////////////////////
		// Detect OpenCL devices
		/////////////////////////////////////////////////////////////////
		devices = (cl_device_id *)malloc(deviceListSize);
		if(devices == 0){
			emit sgnLogMessage(QString("OpenCL Error: No devices found."));
			return false;
		}

		/* Now, get the device list data */
		status = clGetContextInfo(
					 context,
					 CL_CONTEXT_DEVICES,
					 deviceListSize,
					 devices,
					 NULL);
		if(status != CL_SUCCESS){
			emit sgnLogMessage(QString("OpenCL Error: Getting Context Info (device list, clGetContextInfo) code=%1").arg(status));
			return false;
		}

		/////////////////////////////////////////////////////////////////
		// Create an OpenCL command queue
		/////////////////////////////////////////////////////////////////
		commandQueue = clCreateCommandQueue(
						   context,
						   devices[0],
						   0,
						   &status);
		if(status != CL_SUCCESS){
			emit sgnLogMessage(QString("OpenCL Error: Creating Command Queue. (clCreateCommandQueue) code=%1").arg(status));
			return false;
		}

		/////////////////////////////////////////////////////////////////
		// Create OpenCL memory buffers
		/////////////////////////////////////////////////////////////////
		inputBuffer = clCreateBuffer(
						  context,
						  CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
						  sizeof(cl_uint) * width,
						  input,
						  &status);
		if(status != CL_SUCCESS){
			emit sgnLogMessage(QString("OpenCL Error: clCreateBuffer (inputBuffer) code=%1").arg(status));
			return false;
		}

		outputBuffer = clCreateBuffer(
						   context,
						   CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
						   sizeof(cl_uint) * width,
						   output,
						   &status);
		if(status != CL_SUCCESS){
			emit sgnLogMessage(QString("OpenCL Error: clCreateBuffer (outputBuffer) code=%1").arg(status));
			return false;
		}


		/////////////////////////////////////////////////////////////////
		// Load CL file, build CL program object, create CL kernel object
		/////////////////////////////////////////////////////////////////
		std::string  sourceStr = kernelStr.toStdString();
		const char * source    = sourceStr.c_str();
		size_t sourceSize[]    = { strlen(source) };

		program = clCreateProgramWithSource(
					  context,
					  1,
					  &source,
					  sourceSize,
					  &status);
		if(status != CL_SUCCESS){
			emit sgnLogMessage(QString("OpenCL Error: Loading Binary into cl_program (clCreateProgramWithSource) code=%1").arg(status));
			return false;
		}

		/* create a cl program executable for all the devices specified */
		status = clBuildProgram(program, 1, devices, NULL, NULL, NULL);
		if(status != CL_SUCCESS){
			emit sgnLogMessage(QString("OpenCL Error: Building Program (clBuildProgram) code=%1").arg(status));
			return false;
		}

		/* get a kernel object handle for a kernel with the given name */
		kernel = clCreateKernel(program, "hello", &status);
		if(status != CL_SUCCESS){
			emit sgnLogMessage(QString("OpenCL Error: Creating Kernel from program. (clCreateKernel) code=%1").arg(status));
			return false;
		}

		return true;
	}
	bool PointMatcherOpenCL::ShutdownCL(){
		cl_int status;

		status = clReleaseKernel(kernel);
		if(status != CL_SUCCESS){
			emit sgnLogMessage(QString("OpenCL Error: In clReleaseKernel code=%1").arg(status));
			return false;
		}
		status = clReleaseProgram(program);
		if(status != CL_SUCCESS){
			emit sgnLogMessage(QString("OpenCL Error: In clReleaseProgram code=%1").arg(status));
			return false;
		}
		status = clReleaseMemObject(inputBuffer);
		if(status != CL_SUCCESS){
			emit sgnLogMessage(QString("OpenCL Error: In clReleaseMemObject (inputBuffer) code=%1").arg(status));
			return false;
		}
		status = clReleaseMemObject(outputBuffer);
		if(status != CL_SUCCESS){
			emit sgnLogMessage(QString("OpenCL Error: In clReleaseMemObject (outputBuffer) code=%1").arg(status));
			return false;
		}
		status = clReleaseCommandQueue(commandQueue);
		if(status != CL_SUCCESS){
			emit sgnLogMessage(QString("OpenCL Error: In clReleaseCommandQueue code=%1").arg(status));
			return false;
		}
		status = clReleaseContext(context);
		if(status != CL_SUCCESS){
			emit sgnLogMessage(QString("OpenCL Error: In clReleaseContext code=%1").arg(status));
			return false;
		}

		return true;
	}
}
