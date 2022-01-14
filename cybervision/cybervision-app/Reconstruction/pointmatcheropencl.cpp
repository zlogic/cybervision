#ifdef CYBERVISION_OPENCL

#include "pointmatcheropencl.h"

#include <Reconstruction/options.h>

#include <QFile>
#include <QVector>
#include <QSet>
#include <QTextStream>
#include <QList>

#include <cstdlib>
#include <cmath>


namespace cybervision{
PointMatcherOpenCL::PointMatcherOpenCL(int vectorSize, QObject *parent) : QObject(parent){
	//Read kernel from file
	{
		QFile kernelFile(":/opencl/PointMatcherKernel.cl");
		kernelFile.open(QIODevice::ReadOnly);
		QTextStream stream(&kernelFile);
		kernelStr= stream.readAll();
	}

	kernelInitialized= false;
	kernelFirstRun= true;

	//Prepare buffers
	kernelWorkGroupSizeFull= (Options::keypointMatchingMode==Options::KEYPOINT_MATCHING_OPENCL_CPU)?128:1024;
	kernelWorkGroupSize1= kernelWorkGroupSizeFull;
	kernelWorkGroupSize2= kernelWorkGroupSizeFull;
	kernelWorkGroupSize2Max= 16;

	//Get optimal vector size
	this->vectorSize= vectorSize;
	inputVectorsBufferSize= 1024;


	/////////////////////////////////////////////////////////////////
	// Allocate and initialize memory used by host
	/////////////////////////////////////////////////////////////////
	input1 = new cl_float[vectorSize * inputVectorsBufferSize];
	input2 = new cl_float[vectorSize * inputVectorsBufferSize];
	output = new cl_float[inputVectorsBufferSize*inputVectorsBufferSize];
}
PointMatcherOpenCL::~PointMatcherOpenCL(){
	if(input1!=NULL)
		delete[] input1;
	if(input2!=NULL)
		delete[] input2;
	if(output!=NULL)
		delete[] output;
}

bool PointMatcherOpenCL::InitCL(){
	kernelInitialized= false;

	if(input1 == NULL){
		emit sgnLogMessage(QString(tr("OpenCL Error: Failed to allocate input1 memory on host")));
		return false;
	}
	if(input2 == NULL){
		emit sgnLogMessage(QString(tr("OpenCL Error: Failed to allocate input2 memory on host")));
		return false;
	}
	if(output == NULL){
		emit sgnLogMessage(QString(tr("OpenCL Error: Failed to allocate output memory on host")));
		return false;
	}

	cl_int status = 0;
	size_t deviceListSize;

	/*
	 * Have a look at the available platforms and pick one with the most capacity.
	 */
	cl_uint numPlatforms;
	cl_platform_id platform = NULL;
	status = clGetPlatformIDs(0, NULL, &numPlatforms);
	if(status != CL_SUCCESS){
		emit sgnLogMessage(QString(tr("OpenCL Error: Getting Platforms. (clGetPlatformsIDs) code=%1")).arg(status));
		return false;
	}

	QString bestPlatformVendor;
	int bestPlatformSpeed = 0;
	if(numPlatforms > 0)
	{
		QScopedArrayPointer<cl_platform_id> platforms(new cl_platform_id[numPlatforms]);
		status = clGetPlatformIDs(numPlatforms, platforms.data(), NULL);
		if(status != CL_SUCCESS){
			emit sgnLogMessage(QString(tr("OpenCL Error: Getting Platform Ids. (clGetPlatformsIDs) code=%1")).arg(status));
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
				emit sgnLogMessage(QString(tr("OpenCL Error: Getting Platform Info.(clGetPlatformInfo) code=%1")).arg(status));
				continue;
			}

			/*
			 * Find if the platform has a better device than already selected.
			 */
			cl_uint numDevices;
			status = clGetDeviceIDs(
						platforms[i],
						CL_DEVICE_TYPE_GPU,
						sizeof(cl_uint),
						NULL,
						&numDevices);
			if(status != CL_SUCCESS){
				emit sgnLogMessage(QString(tr("OpenCL Error: Getting Device ID.(clGetDeviceIDs) code=%1")).arg(status));
				continue;
			}
			QScopedArrayPointer<cl_device_id> devices(new cl_device_id[numDevices]);
			status = clGetDeviceIDs(
						platforms[i],
						CL_DEVICE_TYPE_GPU,
						numDevices,
						devices.data(),
						NULL);
			if(status != CL_SUCCESS){
				emit sgnLogMessage(QString(tr("OpenCL Error: Getting Device IDs.(clGetDeviceIDs) code=%1")).arg(status));
				continue;
			}
			for(unsigned int j=0;j<numDevices;j++){
				cl_uint maxComputeUnits;
				status = clGetDeviceInfo(
							devices[j],
							CL_DEVICE_MAX_COMPUTE_UNITS,
							sizeof(cl_uint),
							&maxComputeUnits,
							NULL);
				if(status != CL_SUCCESS){
					emit sgnLogMessage(QString(tr("OpenCL Error: Getting Device Info.(clGetDeviceInfo CL_DEVICE_MAX_COMPUTE_UNITS) code=%1")).arg(status));
					continue;
				}

				cl_uint maxClockFrequency;
				status = clGetDeviceInfo(
							devices[j],
							CL_DEVICE_MAX_CLOCK_FREQUENCY,
							sizeof(cl_uint),
							&maxClockFrequency,
							NULL);
				if(status != CL_SUCCESS){
					emit sgnLogMessage(QString(tr("OpenCL Error: Getting Device Info.(clGetDeviceInfo CL_DEVICE_MAX_CLOCK_FREQUENCY) code=%1")).arg(status));
					continue;
				}

				int speed = maxComputeUnits*maxClockFrequency;
				if(speed>bestPlatformSpeed){
					platform = platforms[i];
					bestPlatformVendor = QLatin1String(pbuff);
					bestPlatformSpeed = speed;
				}
			}
		}
	}

	if(NULL == platform){
		emit sgnLogMessage(QString(tr("OpenCL Error: NULL platform found")));
		return false;
	}else{
		emit sgnLogMessage(QString(tr("OpenCL information: Selected platform: %1.")).arg(bestPlatformVendor));
	}

	/*
	 * If we could find our platform, use it. Otherwise use just available platform.
	 */
	cl_context_properties cps[3] = { CL_CONTEXT_PLATFORM, (cl_context_properties)platform, 0 };

	/////////////////////////////////////////////////////////////////
	// Create an OpenCL context
	/////////////////////////////////////////////////////////////////
	context = clCreateContextFromType(cps,
									  (Options::keypointMatchingMode==Options::KEYPOINT_MATCHING_OPENCL_GPU || Options::keypointMatchingMode==Options::KEYPOINT_MATCHING_OPENCL_HYBRID)
									  ?CL_DEVICE_TYPE_GPU:CL_DEVICE_TYPE_CPU,
									  NULL,
									  NULL,
									  &status);
	if(status != CL_SUCCESS){
		emit sgnLogMessage(QString(tr("OpenCL Error: Creating Context. (clCreateContextFromType) code=%1")).arg(status));
		return false;
	}

	/* First, get the size of device list data */
	status = clGetContextInfo(context,
							  CL_CONTEXT_DEVICES,
							  0,
							  NULL,
							  &deviceListSize);
	if(status != CL_SUCCESS){
		emit sgnLogMessage(QString(tr("OpenCL Error: Getting Context Info (device list size, clGetContextInfo) code=%1")).arg(status));
		return false;
	}

	/////////////////////////////////////////////////////////////////
	// Detect OpenCL devices
	/////////////////////////////////////////////////////////////////
	devices.reset(new cl_device_id[deviceListSize]);
	if(devices == 0){
		emit sgnLogMessage(QString(tr("OpenCL Error: No devices found.")));
		return false;
	}

	/* Now, get the device list data */
	status = clGetContextInfo(
				context,
				CL_CONTEXT_DEVICES,
				deviceListSize,
				devices.data(),
				NULL);
	if(status != CL_SUCCESS){
		emit sgnLogMessage(QString(tr("OpenCL Error: Getting Context Info (device list, clGetContextInfo) code=%1")).arg(status));
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
		emit sgnLogMessage(QString(tr("OpenCL Error: Creating Command Queue. (clCreateCommandQueue) code=%1")).arg(status));
		return false;
	}

	/////////////////////////////////////////////////////////////////
	// Create OpenCL memory buffers
	/////////////////////////////////////////////////////////////////
	input1Buffer = clCreateBuffer(
				context,
				CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
				sizeof(cl_float) * vectorSize * inputVectorsBufferSize,
				input1,
				&status);
	input2Buffer = clCreateBuffer(
				context,
				CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
				sizeof(cl_float) * vectorSize * inputVectorsBufferSize,
				input2,
				&status);
	if(status != CL_SUCCESS){
		emit sgnLogMessage(QString(tr("OpenCL Error: clCreateBuffer (inputBuffer) code=%1")).arg(status));
		return false;
	}

	outputBuffer = clCreateBuffer(
				context,
				CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
				sizeof(cl_float) * inputVectorsBufferSize * inputVectorsBufferSize,
				output,
				&status);
	if(status != CL_SUCCESS){
		emit sgnLogMessage(QString(tr("OpenCL Error: clCreateBuffer (outputBuffer) code=%1")).arg(status));
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
		emit sgnLogMessage(QString(tr("OpenCL Error: Loading Binary into cl_program (clCreateProgramWithSource) code=%1")).arg(status));
		return false;
	}

	QString qOptions= QString("-D VECTOR_SIZE=%1 -D MAX_THREADS_SECOND_DIMENSION=%2").arg(vectorSize).arg(kernelWorkGroupSize2Max);

	/* create a cl program executable for all the devices specified */
	status = clBuildProgram(program, 1, devices.data(), qOptions.toStdString().c_str(), NULL, NULL);
	if(status != CL_SUCCESS){
		QVector<char> build_log;

		size_t log_size;
		// First call to know the proper size
		clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
		build_log.reserve(log_size+1);
		// Second call to get the log
		clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, log_size, build_log.data(), NULL);
		build_log[log_size] = '\0';

		emit sgnLogMessage(QString(tr("OpenCL Error: Building Program (clBuildProgram) code=%1, build log: \'%2\'")).arg(status).arg(build_log.data()));
		return false;
	}

	/* get a kernel object handle for a kernel with the given name */
	kernel = clCreateKernel(program, "computeDistances", &status);
	if(status != CL_SUCCESS){
		emit sgnLogMessage(QString(tr("OpenCL Error: Creating Kernel from program. (clCreateKernel) code=%1")).arg(status));
		return false;
	}

	kernelInitialized= true;
	return true;
}
bool PointMatcherOpenCL::ShutdownCL(){
	if(!kernelInitialized)
		return true;

	kernelInitialized= false;

	cl_int status;

	status = clReleaseKernel(kernel);
	if(status != CL_SUCCESS){
		emit sgnLogMessage(QString(tr("OpenCL Error: In clReleaseKernel code=%1")).arg(status));
		return false;
	}
	status = clReleaseProgram(program);
	if(status != CL_SUCCESS){
		emit sgnLogMessage(QString(tr("OpenCL Error: In clReleaseProgram code=%1")).arg(status));
		return false;
	}
	status = clReleaseMemObject(input1Buffer);
	if(status != CL_SUCCESS){
		emit sgnLogMessage(QString(tr("OpenCL Error: In clReleaseMemObject (input1Buffer) code=%1")).arg(status));
		return false;
	}
	status = clReleaseMemObject(input2Buffer);
	if(status != CL_SUCCESS){
		emit sgnLogMessage(QString(tr("OpenCL Error: In clReleaseMemObject (input2Buffer) code=%1")).arg(status));
		return false;
	}
	status = clReleaseMemObject(outputBuffer);
	if(status != CL_SUCCESS){
		emit sgnLogMessage(QString(tr("OpenCL Error: In clReleaseMemObject (outputBuffer) code=%1")).arg(status));
		return false;
	}
	status = clReleaseCommandQueue(commandQueue);
	if(status != CL_SUCCESS){
		emit sgnLogMessage(QString(tr("OpenCL Error: In clReleaseCommandQueue code=%1")).arg(status));
		return false;
	}
	status = clReleaseContext(context);
	if(status != CL_SUCCESS){
		emit sgnLogMessage(QString(tr("OpenCL Error: In clReleaseContext code=%1")).arg(status));
		return false;
	}

	return true;
}

bool PointMatcherOpenCL::CalcDistances(){
	cl_int   status;
	cl_event events[4];

	size_t globalThreads[2];
	size_t localThreads[2];

	globalThreads[0] = inputVectorsBufferSize;
	globalThreads[1] = inputVectorsBufferSize;
	localThreads[0] = kernelWorkGroupSize1;
	localThreads[1] = kernelWorkGroupSize2;

	if(kernelFirstRun){
		/* Check group size against kernelWorkGroupSize */
		status = clGetKernelWorkGroupInfo(kernel,
										  devices[0],
										  CL_KERNEL_WORK_GROUP_SIZE,
										  sizeof(size_t),
										  &kernelWorkGroupSizeFull,
										  0);
		if(status != CL_SUCCESS){
			emit sgnLogMessage(QString(tr("OpenCL Error: clGetKernelWorkGroupInfo failed, code=%1")).arg(status));
			return false;
		}

		if((cl_uint)(localThreads[0]*localThreads[1]) > kernelWorkGroupSizeFull){
			kernelWorkGroupSize2= std::min(kernelWorkGroupSize2,kernelWorkGroupSize2Max);
			kernelWorkGroupSize1= kernelWorkGroupSizeFull/kernelWorkGroupSize2;

			emit sgnLogMessage(QString(tr("OpenCL information: Reducing group size to fit hardware limits. Group Size specified: %1. Max Group Size supported on the kernel: %2. Changing the group size to %3x%4.")).arg(localThreads[0]).arg(kernelWorkGroupSizeFull).arg(kernelWorkGroupSize1).arg(kernelWorkGroupSize2));

			localThreads[0] = kernelWorkGroupSize1;
			localThreads[1] = kernelWorkGroupSize2;
		}
		kernelFirstRun= false;
	}

	if(Options::keypointMatchingMode==Options::KEYPOINT_MATCHING_OPENCL_GPU || Options::keypointMatchingMode==Options::KEYPOINT_MATCHING_OPENCL_HYBRID){
		/* Enqueue clEnqueueWriteBuffer for input1 buffer */
		status = clEnqueueWriteBuffer(
					commandQueue,
					input1Buffer,
					CL_FALSE,
					0,
					sizeof(cl_float) * vectorSize * inputVectorsBufferSize,
					input1,
					0,
					NULL,
					&events[0]);
		if(status != CL_SUCCESS){
			emit sgnLogMessage(QString(tr("OpenCL Error: clEnqueueWriteBuffer (input1Buffer) failed, code=%1")).arg(status));
			return false;
		}

		/* wait for the write buffer call to finish execution */
		status = clWaitForEvents(1, &events[0]);
		if(status != CL_SUCCESS){
			emit sgnLogMessage(QString(tr("OpenCL Error: clWaitForEvents (0) failed, code=%1")).arg(status));
			return false;
		}
		status = clReleaseEvent(events[0]);
		if(status != CL_SUCCESS){
			emit sgnLogMessage(QString(tr("OpenCL Error: clReleaseEvents (0) failed, code=%1")).arg(status));
			return false;
		}

		/* Enqueue clEnqueueWriteBuffer for input2 buffer */
		status = clEnqueueWriteBuffer(
					commandQueue,
					input2Buffer,
					CL_FALSE,
					0,
					sizeof(cl_float) * vectorSize * inputVectorsBufferSize,
					input2,
					0,
					NULL,
					&events[1]);
		if(status != CL_SUCCESS){
			emit sgnLogMessage(QString(tr("OpenCL Error: clEnqueueWriteBuffer (input2Buffer) failed, code=%1")).arg(status));
			return false;
		}

		/* wait for the write buffer call to finish execution */
		status = clWaitForEvents(1, &events[1]);
		if(status != CL_SUCCESS){
			emit sgnLogMessage(QString(tr("OpenCL Error: clWaitForEvents (1) failed, code=%1")).arg(status));
			return false;
		}
		status = clReleaseEvent(events[1]);
		if(status != CL_SUCCESS){
			emit sgnLogMessage(QString(tr("OpenCL Error: clReleaseEvents (1) failed, code=%1")).arg(status));
			return false;
		}
	}
	/*** Set appropriate arguments to the kernel ***/
	status = clSetKernelArg(
				kernel,
				0,
				sizeof(cl_mem),
				(void *)&outputBuffer);
	if(status != CL_SUCCESS){
		emit sgnLogMessage(QString(tr("OpenCL Error: clSetKernelArg failed. (outputBuffer), code=%1")).arg(status));
		return false;
	}

	status = clSetKernelArg(
				kernel,
				1,
				sizeof(cl_mem),
				(void *)&input1Buffer);
	if(status != CL_SUCCESS){
		emit sgnLogMessage(QString(tr("OpenCL Error: clSetKernelArg failed. (input1Buffer), code=%1")).arg(status));
		return false;
	}

	status = clSetKernelArg(
				kernel,
				2,
				sizeof(cl_mem),
				(void *)&input2Buffer);
	if(status != CL_SUCCESS){
		emit sgnLogMessage(QString(tr("OpenCL Error: clSetKernelArg failed. (input2Buffer), code=%1")).arg(status));
		return false;
	}

	/*
	 * Enqueue a kernel run call.
	 */
	status = clEnqueueNDRangeKernel(
				commandQueue,
				kernel,
				2,
				NULL,
				globalThreads,
				localThreads,
				0,
				NULL,
				&events[2]);

	if(status != CL_SUCCESS){
		emit sgnLogMessage(QString(tr("OpenCL Error: clEnqueueNDRangeKernel failed, code=%1")).arg(status));
		return false;
	}


	/* wait for the kernel call to finish execution */
	status = clWaitForEvents(1, &events[2]);
	if(status != CL_SUCCESS){
		emit sgnLogMessage(QString(tr("OpenCL Error: clWaitForEvents (2) failed, code=%1")).arg(status));
		return false;
	}

	status = clReleaseEvent(events[2]);
	if(status != CL_SUCCESS){
		emit sgnLogMessage(QString(tr("OpenCL Error: clReleaseEvents (2) failed, code=%1")).arg(status));
		return false;
	}

	/* Enqueue readBuffer*/
	status = clEnqueueReadBuffer(
				commandQueue,
				outputBuffer,
				CL_FALSE,
				0,
				inputVectorsBufferSize * inputVectorsBufferSize * sizeof(cl_float),
				output,
				0,
				NULL,
				&events[3]);
	if(status != CL_SUCCESS){
		emit sgnLogMessage(QString(tr("OpenCL Error: clEnqueueReadBuffer failed, code=%1")).arg(status));
		return false;
	}

	/* Wait for the read buffer to finish execution */
	status = clWaitForEvents(1, &events[3]);
	if(status != CL_SUCCESS){
		emit sgnLogMessage(QString(tr("OpenCL Error: clWaitForEvents (3) failed, code=%1")).arg(status));
		return false;
	}

	status = clReleaseEvent(events[3]);
	if(status != CL_SUCCESS){
		emit sgnLogMessage(QString(tr("OpenCL Error: clReleaseEvents (3) failed, code=%1")).arg(status));
		return false;
	}

	return true;
}

SortedKeypointMatches PointMatcherOpenCL::CalcDistances(const QList<SIFT::Keypoint>& keypoints1,const QList<SIFT::Keypoint>& keypoints2){
	if(Options::keypointMatchingMode==Options::KEYPOINT_MATCHING_OPENCL_HYBRID)
		return CalcDistancesHybrid(keypoints1,keypoints2);
	else if(Options::keypointMatchingMode==Options::KEYPOINT_MATCHING_OPENCL_GPU)
		return CalcDistancesOpenCL(keypoints1,keypoints2);
	else if(Options::keypointMatchingMode==Options::KEYPOINT_MATCHING_OPENCL_CPU)
		return CalcDistancesOpenCL(keypoints1,keypoints2);
	else
		return SortedKeypointMatches();
}

SortedKeypointMatches PointMatcherOpenCL::CalcDistancesHybrid(const QList<SIFT::Keypoint>& keypoints1,const QList<SIFT::Keypoint>& keypoints2){
	SortedKeypointMatches result;

	bool gpuBusy= false;
	bool openCLFailed= false;

	QVector<QSet<int> > keypoints1Matches(keypoints1.size(),QSet<int>());
	QVector<QSet<int> > keypoints2Matches(keypoints2.size(),QSet<int>());


	//Compute distances for all keypoints from keypoints1, then choose the minumum distance
	#pragma omp parallel
	for(int x1=0;x1<keypoints1.size();x1+=inputVectorsBufferSize){
		#pragma omp single nowait
		{
			for(int x2=0;x2<keypoints2.size();x2+=inputVectorsBufferSize){
				if(!openCLFailed)
				{
					bool useGPU=false;
					#pragma omp critical
					{
						if(!gpuBusy && (Options::keypointMatchingMode==Options::KEYPOINT_MATCHING_OPENCL_GPU || Options::keypointMatchingMode==Options::KEYPOINT_MATCHING_OPENCL_HYBRID)){
							useGPU= true;
							gpuBusy= true;
						}
					}

					if(useGPU && !kernelInitialized){
						if(!InitCL()){
							openCLFailed= true;
							useGPU= false;
						}
					}

					if(useGPU){
						for(int i=0;(i<(int)inputVectorsBufferSize) && ((x1+i)<keypoints1.size());i++){
							const SIFT::Keypoint& keypoint1= keypoints1[x1+i];
							for(cl_uint j=0;j<vectorSize;j++)
								input1[i*vectorSize+j]= keypoint1[j];
						}

						//Prepare matches set from second image
						for(int i=0;(i<(int)inputVectorsBufferSize) && ((x2+i)<keypoints2.size());i++){
							const SIFT::Keypoint& keypoint2= keypoints2[x2+i];
							for(cl_uint j=0;j<vectorSize;j++)
								input2[i*vectorSize+j]= keypoint2[j];
						}
						//Run OpenCL procedure
						if(!CalcDistances())
							openCLFailed= true;

						//Extract matches
						for(int i=0;(i<(int)inputVectorsBufferSize) && ((x1+i)<keypoints1.size());i++){
							float minDistance= std::numeric_limits<float>::infinity();
							int min_j=0;
							for(int j=0;(j<(int)inputVectorsBufferSize) && ((x2+j)<keypoints2.size());j++){
								float distance= output[inputVectorsBufferSize*i+j];
								if(distance<minDistance){
									minDistance=distance;
									min_j= j;
								}
							}
							#pragma omp critical
							{
								keypoints1Matches[x1+i].insert(x2+min_j);
							}
						}
						for(int i=0;(i<(int)inputVectorsBufferSize) && ((x2+i)<keypoints2.size());i++){
							float minDistance= std::numeric_limits<float>::infinity();
							int min_j=0;
							for(int j=0;(j<(int)inputVectorsBufferSize) && ((x1+j)<keypoints1.size());j++){
								float distance= output[inputVectorsBufferSize*j+i];
								if(distance<minDistance){
									minDistance=distance;
									min_j= j;
								}
							}
							#pragma omp critical
							{
								keypoints2Matches[x2+i].insert(x1+min_j);
							}
						}
						gpuBusy= false;
					}else{
						QVector<float> outputCpu(inputVectorsBufferSize*inputVectorsBufferSize);

						for(int i=0;(i<(int)inputVectorsBufferSize) && ((x1+i)<keypoints1.size());i++){
							for(int j=0;(j<(int)inputVectorsBufferSize) && ((x2+j)<keypoints2.size());j++){
								outputCpu[inputVectorsBufferSize*i+j]= keypoints1[x1+i].distance(keypoints2[x2+j]);
							}
						}

						for(int i=0;(i<(int)inputVectorsBufferSize) && ((x1+i)<keypoints1.size());i++){
							float minDistance= std::numeric_limits<float>::infinity();
							int min_j=0;
							for(int j=0;(j<(int)inputVectorsBufferSize) && ((x2+j)<keypoints2.size());j++){
								float distance= outputCpu[inputVectorsBufferSize*i+j];
								if(distance<minDistance){
									minDistance=distance;
									min_j=j;
								}
							}
							#pragma omp critical
							{
								keypoints1Matches[x1+i].insert(x2+min_j);
							}
						}
						for(int i=0;(i<(int)inputVectorsBufferSize) && ((x2+i)<keypoints2.size());i++){
							float minDistance= std::numeric_limits<float>::infinity();
							int min_j=0;
							for(int j=0;(j<(int)inputVectorsBufferSize) && ((x1+j)<keypoints1.size());j++){
								float distance= outputCpu[inputVectorsBufferSize*j+i];
								if(distance<minDistance){
									minDistance=distance;
									min_j=j;
								}
							}
							#pragma omp critical
							{
								keypoints2Matches[x2+i].insert(x1+min_j);
							}
						}
					}
				}
			}
		}
	}

	if(openCLFailed)
		return SortedKeypointMatches();

	for(int i=0;i<keypoints1Matches.size();i++){
		const SIFT::Keypoint& keypoint1= keypoints1[i];
		float minDistance= std::numeric_limits<float>::infinity();
		int min_j=0;
		{
			const QSet<int>& keypoint1Distances= keypoints1Matches[i];
			if(keypoint1Distances.empty())
				continue;
			for(QSet<int>::const_iterator it=keypoint1Distances.begin();it!=keypoint1Distances.end();it++){
				float distance= keypoint1.distance(keypoints2[*it]);
				if(distance<minDistance){
					minDistance= distance;
					min_j= *it;
				}
			}
		}
		if(minDistance<Options::MaxKeypointDistance){
			const SIFT::Keypoint& keypoint2= keypoints2[min_j];
			KeypointMatch match;
			match.a= QPointF(keypoint1.getX(),keypoint1.getY());
			match.b= QPointF(keypoint2.getX(),keypoint2.getY());

			//Add point
			if(!result.contains(minDistance,match))
				result.insert(minDistance,match);
		}
	}

	for(int i=0;i<keypoints2Matches.size();i++){
		const SIFT::Keypoint& keypoint2= keypoints2[i];
		float minDistance= std::numeric_limits<float>::infinity();
		int min_j=0;
		{
			const QSet<int>& keypoint2Distances= keypoints2Matches[i];
			if(keypoint2Distances.empty())
				continue;
			for(QSet<int>::const_iterator it=keypoint2Distances.begin();it!=keypoint2Distances.end();it++){
				float distance= keypoint2.distance(keypoints1[*it]);
				if(distance<minDistance){
					minDistance= distance;
					min_j= *it;
				}
			}
		}
		if(minDistance<Options::MaxKeypointDistance){
			const SIFT::Keypoint& keypoint1= keypoints1[min_j];
			KeypointMatch match;
			match.a= QPointF(keypoint1.getX(),keypoint1.getY());
			match.b= QPointF(keypoint2.getX(),keypoint2.getY());

			//Add point
			if(!result.contains(minDistance,match))
				result.insert(minDistance,match);
		}
	}

	return result;
}


SortedKeypointMatches PointMatcherOpenCL::CalcDistancesOpenCL(const QList<SIFT::Keypoint>& keypoints1,const QList<SIFT::Keypoint>& keypoints2){
	if(!kernelInitialized)
		if(!InitCL())
			return SortedKeypointMatches();

	SortedKeypointMatches result;

	QVector<QSet<int> > keypoints1Matches(keypoints1.size(),QSet<int>());
	QVector<QSet<int> > keypoints2Matches(keypoints2.size(),QSet<int>());
	for(int x1=0;x1<keypoints1.size();x1+=inputVectorsBufferSize){
		//Prepare matches set from first image
		for(int i=0;(i<(int)inputVectorsBufferSize) && ((x1+i)<keypoints1.size());i++){
			const SIFT::Keypoint& keypoint1= keypoints1[x1+i];
			for(cl_uint j=0;j<vectorSize;j++)
				input1[i*vectorSize+j]= keypoint1[j];
		}
		for(int x2=0;x2<keypoints2.size();x2+=inputVectorsBufferSize){
			//Prepare matches set from second image
			for(int i=0;(i<(int)inputVectorsBufferSize) && ((x2+i)<keypoints2.size());i++){
				const SIFT::Keypoint& keypoint2= keypoints2[x2+i];
				for(cl_uint j=0;j<vectorSize;j++)
					input2[i*vectorSize+j]= keypoint2[j];
			}
			//Run OpenCL procedure
			if(!CalcDistances())
				return SortedKeypointMatches();

			//Extract matches
			for(int i=0;(i<(int)inputVectorsBufferSize) && ((x1+i)<keypoints1.size());i++){
				float minDistance= std::numeric_limits<float>::infinity();
				int min_j=0;
				for(int j=0;(j<(int)inputVectorsBufferSize) && ((x2+j)<keypoints2.size());j++){
					float distance= output[inputVectorsBufferSize*i+j];
					if(distance<minDistance){
						minDistance=distance;
						min_j=j;
					}
				}
				keypoints1Matches[x1+i].insert(x2+min_j);
			}
			for(int i=0;(i<(int)inputVectorsBufferSize) && ((x2+i)<keypoints2.size());i++){
				float minDistance= std::numeric_limits<float>::infinity();
				int min_j=0;
				for(int j=0;(j<(int)inputVectorsBufferSize) && ((x1+j)<keypoints1.size());j++){
					float distance= output[inputVectorsBufferSize*j+i];
					if(distance<minDistance){
						minDistance=distance;
						min_j=j;
					}
				}
				keypoints2Matches[x2+i].insert(x1+min_j);
			}
		}
	}

	for(int i=0;i<keypoints1Matches.size();i++){
		const SIFT::Keypoint& keypoint1= keypoints1[i];
		float minDistance= std::numeric_limits<float>::infinity();
		int min_j=0;
		{
			const QSet<int>& keypoint1Distances= keypoints1Matches[i];
			if(keypoint1Distances.empty())
				continue;
			for(QSet<int>::const_iterator it=keypoint1Distances.begin();it!=keypoint1Distances.end();it++){
				float distance= keypoint1.distance(keypoints2[*it]);
				if(distance<minDistance){
					minDistance= distance;
					min_j= *it;
				}
			}
		}
		if(minDistance<Options::MaxKeypointDistance){
			const SIFT::Keypoint& keypoint2= keypoints2[min_j];
			KeypointMatch match;
			match.a= QPointF(keypoint1.getX(),keypoint1.getY());
			match.b= QPointF(keypoint2.getX(),keypoint2.getY());

			//Add point
			if(!result.contains(minDistance,match))
				result.insert(minDistance,match);
		}
	}

	for(int i=0;i<keypoints2Matches.size();i++){
		const SIFT::Keypoint& keypoint2= keypoints2[i];
		float minDistance= std::numeric_limits<float>::infinity();
		int min_j=0;
		{
			const QSet<int>& keypoint2Distances= keypoints2Matches[i];
			if(keypoint2Distances.empty())
				continue;
			for(QSet<int>::const_iterator it=keypoint2Distances.begin();it!=keypoint2Distances.end();it++){
				float distance= keypoint2.distance(keypoints1[*it]);
				if(distance<minDistance){
					minDistance= distance;
					min_j= *it;
				}
			}
		}
		if(minDistance<Options::MaxKeypointDistance){
			const SIFT::Keypoint& keypoint1= keypoints1[min_j];
			KeypointMatch match;
			match.a= QPointF(keypoint1.getX(),keypoint1.getY());
			match.b= QPointF(keypoint2.getX(),keypoint2.getY());

			//Add point
			if(!result.contains(minDistance,match))
				result.insert(minDistance,match);
		}
	}

	return result;
}
}

#endif // CYBERVISION_OPENCL
