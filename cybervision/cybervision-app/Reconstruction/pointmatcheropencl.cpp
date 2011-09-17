#ifdef CYBERVISION_OPENCL

#include "pointmatcheropencl.h"

#include <Reconstruction/options.h>

#include <QFile>
#include <QVector>
#include <QTextStream>
#include <QList>

#include <cstdlib>


namespace cybervision{
	PointMatcherOpenCL::PointMatcherOpenCL(int vectorSize, int maxVectorsCount, QObject *parent) : QObject(parent){
		//Read kernel from file
		{
			QFile kernelFile(":/opencl/PointMatcherKernel.cl");
			kernelFile.open(QIODevice::ReadOnly);
			QTextStream stream(&kernelFile);
			kernelStr= stream.readAll();
		}

		kernelInitialized= false;
		inputVectorsCopied= false;
		kernelFirstRun= true;

		//Prepare buffers
		kernelWorkGroupSize= (Options::keypointMatchingMode==Options::KEYPOINT_MATCHING_OPENCL_CPU)?128:1024;

		//Get optimal vector size
		{
			int workRounding= (Options::keypointMatchingMode==Options::KEYPOINT_MATCHING_OPENCL_CPU)?1024:512;
			maxVectorsCount= (maxVectorsCount/workRounding + 1)*workRounding;
			maxVectorsCount= qMax(maxVectorsCount,(int)kernelWorkGroupSize);
		}

		this->vectorSize= vectorSize;
		inputVectorsBufferSize= maxVectorsCount;


		/////////////////////////////////////////////////////////////////
		// Allocate and initialize memory used by host
		/////////////////////////////////////////////////////////////////
		input.reset(new cl_float[vectorSize * inputVectorsBufferSize]);
		vector.reset(new cl_float[vectorSize]);
		output.reset(new cl_float[inputVectorsBufferSize]);
	}
	PointMatcherOpenCL::~PointMatcherOpenCL(){
	}

	bool PointMatcherOpenCL::InitCL(){
		kernelInitialized= false;

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
										  Options::keypointMatchingMode==Options::KEYPOINT_MATCHING_OPENCL_GPU?CL_DEVICE_TYPE_GPU:CL_DEVICE_TYPE_CPU,
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
		devices.reset(new cl_device_id[deviceListSize]);
		if(devices == 0){
			emit sgnLogMessage(QString("OpenCL Error: No devices found."));
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
						  CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
						  sizeof(cl_float) * vectorSize * inputVectorsBufferSize,
						  input.data(),
						  &status);
		if(status != CL_SUCCESS){
			emit sgnLogMessage(QString("OpenCL Error: clCreateBuffer (inputBuffer) code=%1").arg(status));
			return false;
		}

		outputBuffer = clCreateBuffer(
						   context,
						   CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
						   sizeof(cl_float) * inputVectorsBufferSize,
						   output.data(),
						   &status);
		if(status != CL_SUCCESS){
			emit sgnLogMessage(QString("OpenCL Error: clCreateBuffer (outputBuffer) code=%1").arg(status));
			return false;
		}

		vectorBuffer = clCreateBuffer(
						   context,
						   CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
						   sizeof(cl_float) * vectorSize,
						   vector.data(),
						   &status);
		if(status != CL_SUCCESS){
			emit sgnLogMessage(QString("OpenCL Error: clCreateBuffer (vectorBuffer) code=%1").arg(status));
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

		QString qOptions= QString("-D VECTOR_SIZE=%1").arg(vectorSize);

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

			emit sgnLogMessage(QString("OpenCL Error: Building Program (clBuildProgram) code=%1, build log: \'%2\'").arg(status).arg(build_log.data()));
			return false;
		}

		/* get a kernel object handle for a kernel with the given name */
		kernel = clCreateKernel(program, "computeDistances", &status);
		if(status != CL_SUCCESS){
			emit sgnLogMessage(QString("OpenCL Error: Creating Kernel from program. (clCreateKernel) code=%1").arg(status));
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

	bool PointMatcherOpenCL::CalcDistances(){
		cl_int   status;
		cl_event events[4];

		size_t globalThreads[1];
		size_t localThreads[1];

		globalThreads[0] = inputVectorsBufferSize;
		localThreads[0] = kernelWorkGroupSize;

		if(kernelFirstRun){
			/* Check group size against kernelWorkGroupSize */
			status = clGetKernelWorkGroupInfo(kernel,
											  devices[0],
											  CL_KERNEL_WORK_GROUP_SIZE,
											  sizeof(size_t),
											  &kernelWorkGroupSize,
											  0);
			if(status != CL_SUCCESS){
				emit sgnLogMessage(QString("OpenCL Error: clGetKernelWorkGroupInfo failed, code=%1").arg(status));
				return false;
			}

			if((cl_uint)(localThreads[0]) > kernelWorkGroupSize){
				emit sgnLogMessage(QString("OpenCL Warning: Out of Resources! Group Size specified: %1. Max Group Size supported on the kernel: %2. Changing the group size to %3.").arg(localThreads[0]).arg(kernelWorkGroupSize).arg(kernelWorkGroupSize));

				localThreads[0] = kernelWorkGroupSize;
			}
			kernelFirstRun= false;
		}

		if(!inputVectorsCopied){
			if(Options::keypointMatchingMode==Options::KEYPOINT_MATCHING_OPENCL_GPU){
				/* Enqueue clEnqueueWriteBuffer for input buffer */
				status = clEnqueueWriteBuffer(
							commandQueue,
							inputBuffer,
							CL_FALSE,
							0,
							sizeof(cl_float) * vectorSize * inputVectorsBufferSize,
							input.data(),
							0,
							NULL,
							&events[0]);
				if(status != CL_SUCCESS){
					emit sgnLogMessage(QString("OpenCL Error: clEnqueueWriteBuffer (inputBuffer) failed, code=%1").arg(status));
					return false;
				}

				/* wait for the write buffer call to finish execution */
				status = clWaitForEvents(1, &events[0]);
				if(status != CL_SUCCESS){
					emit sgnLogMessage(QString("OpenCL Error: clWaitForEvents (0) failed, code=%1").arg(status));
					return false;
				}
				status = clReleaseEvent(events[0]);
				if(status != CL_SUCCESS){
					emit sgnLogMessage(QString("OpenCL Error: clReleaseEvents (0) failed, code=%1").arg(status));
					return false;
				}
			}
			inputVectorsCopied= true;
		}

		if(Options::keypointMatchingMode==Options::KEYPOINT_MATCHING_OPENCL_GPU){
			/* Enqueue clEnqueueWriteBuffer for input vector buffer */
			status = clEnqueueWriteBuffer(
						commandQueue,
						vectorBuffer,
						CL_FALSE,
						0,
						sizeof(cl_float) * vectorSize,
						vector.data(),
						0,
						NULL,
						&events[1]);
			if(status != CL_SUCCESS){
				emit sgnLogMessage(QString("OpenCL Error: clEnqueueWriteBuffer (vectorBuffer) failed, code=%1").arg(status));
				return false;
			}

			/* wait for the write buffer call to finish execution */
			status = clWaitForEvents(1, &events[1]);
			if(status != CL_SUCCESS){
				emit sgnLogMessage(QString("OpenCL Error: clWaitForEvents (1) failed, code=%1").arg(status));
				return false;
			}

			status = clReleaseEvent(events[1]);
			if(status != CL_SUCCESS){
				emit sgnLogMessage(QString("OpenCL Error: clReleaseEvents (1) failed, code=%1").arg(status));
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
			emit sgnLogMessage(QString("OpenCL Error: clSetKernelArg failed. (outputBuffer), code=%1").arg(status));
			return false;
		}

		status = clSetKernelArg(
						kernel,
						1,
						sizeof(cl_mem),
						(void *)&inputBuffer);
		if(status != CL_SUCCESS){
			emit sgnLogMessage(QString("OpenCL Error: clSetKernelArg failed. (inputBuffer), code=%1").arg(status));
			return false;
		}

		status = clSetKernelArg(
						kernel,
						2,
						sizeof(cl_mem),
						(void *)&vectorBuffer);
		if(status != CL_SUCCESS){
			emit sgnLogMessage(QString("OpenCL Error: clSetKernelArg failed. (vectorBuffer), code=%1").arg(status));
			return false;
		}

		cl_uint clInputVectorsCount= inputVectorsCount;
		status = clSetKernelArg(
						kernel,
						3,
						sizeof(cl_uint),
						(void *)&clInputVectorsCount);
		if(status != CL_SUCCESS){
			emit sgnLogMessage(QString("OpenCL Error: clSetKernelArg failed. (inputVectorsCount), code=%1").arg(status));
			return false;
		}

		cl_float clMaxKeypointDistance= Options::MaxKeypointDistance*Options::MaxKeypointDistance;
		status = clSetKernelArg(
						kernel,
						4,
						sizeof(cl_float),
						(void *)&clMaxKeypointDistance);
		if(status != CL_SUCCESS){
			emit sgnLogMessage(QString("OpenCL Error: clSetKernelArg failed. (MaxKeypointDistance), code=%1").arg(status));
			return false;
		}

		/*
		 * Enqueue a kernel run call.
		 */
		status = clEnqueueNDRangeKernel(
				commandQueue,
				kernel,
				1,
				NULL,
				globalThreads,
				localThreads,
				0,
				NULL,
				&events[2]);

		if(status != CL_SUCCESS){
			emit sgnLogMessage(QString("OpenCL Error: clEnqueueNDRangeKernel failed, code=%1").arg(status));
			return false;
		}


		/* wait for the kernel call to finish execution */
		status = clWaitForEvents(1, &events[2]);
		if(status != CL_SUCCESS){
			emit sgnLogMessage(QString("OpenCL Error: clWaitForEvents (2) failed, code=%1").arg(status));
			return false;
		}

		/* Enqueue readBuffer*/
		status = clEnqueueReadBuffer(
					commandQueue,
					outputBuffer,
					CL_FALSE,
					0,
					inputVectorsBufferSize * sizeof(cl_float),
					output.data(),
					0,
					NULL,
					&events[3]);
		if(status != CL_SUCCESS){
			emit sgnLogMessage(QString("OpenCL Error: clEnqueueReadBuffer failed, code=%1").arg(status));
			return false;
		}

		/* Wait for the read buffer to finish execution */
		status = clWaitForEvents(1, &events[3]);
		if(status != CL_SUCCESS){
			emit sgnLogMessage(QString("OpenCL Error: clWaitForEvents (3) failed, code=%1").arg(status));
			return false;
		}

		status = clReleaseEvent(events[2]);
		if(status != CL_SUCCESS){
			emit sgnLogMessage(QString("OpenCL Error: clReleaseEvents (2) failed, code=%1").arg(status));
			return false;
		}

		status = clReleaseEvent(events[3]);
		if(status != CL_SUCCESS){
			emit sgnLogMessage(QString("OpenCL Error: clReleaseEvents (3) failed, code=%1").arg(status));
			return false;
		}

		return true;
	}

	SortedKeypointMatches PointMatcherOpenCL::CalcDistances(const QList<SIFT::Keypoint>& keypoints1,const QList<SIFT::Keypoint>& keypoints2){
		if(Options::keypointMatchingMode==Options::KEYPOINT_MATCHING_OPENCL_GPU)
			return CalcDistancesHybrid(keypoints1,keypoints2);
		else if(Options::keypointMatchingMode==Options::KEYPOINT_MATCHING_OPENCL_CPU)
			return CalcDistancesOpenCL(keypoints1,keypoints2);
		else
			return SortedKeypointMatches();
	}

	SortedKeypointMatches PointMatcherOpenCL::CalcDistancesHybrid(const QList<SIFT::Keypoint>& keypoints1,const QList<SIFT::Keypoint>& keypoints2){
		SortedKeypointMatches result;

		bool gpuBusy= false;
		bool openCLFailed= false;

		//Compute distances for all keypoints from keypoints1, then choose the minumum distance
		#pragma omp parallel
		for(QList<SIFT::Keypoint>::const_iterator it1=keypoints1.begin();it1!=keypoints1.end();it1++){
			#pragma omp single nowait
			if(!openCLFailed)
			{
				bool useGPU=false;
				#pragma omp critical
				{
					if(!gpuBusy && Options::keypointMatchingMode==Options::KEYPOINT_MATCHING_OPENCL_GPU){
						useGPU= true;
						gpuBusy= true;
					}
				}

				KeypointMatch match;
				float minDistance= std::numeric_limits<float>::infinity();

				if(useGPU && !kernelInitialized){
					if(!InitCL()){
						openCLFailed= true;
						useGPU= false;
					}
				}

				if(useGPU && kernelFirstRun){
					inputVectorsCount= keypoints2.size();
					//Copy data for keypoints2
					for(int i=0;i<keypoints2.size();i++){
						const SIFT::Keypoint& keypoint2= keypoints2[i];
						for(cl_uint j=0;j<vectorSize;j++)
							input[i*vectorSize+j]= keypoint2[j];
					}

					inputVectorsCopied= false;
				}

				if(useGPU){
					//Prepare current vector
					for(cl_uint i=0;i<vectorSize;i++)
						vector[i]= (*it1)[i];

					//Run OpenCL stuff
					if(!CalcDistances())
						openCLFailed= true;

					if(!openCLFailed){
						//Parse OpenCL result
						int min_i=0;
						for(int i=0;i<keypoints2.size();i++)
							if(output[i]<output[min_i])
								min_i=i;
						const SIFT::Keypoint& keypoint2= keypoints2[min_i];
						minDistance= it1->distance(keypoint2);
						if(minDistance<Options::MaxKeypointDistance){
							match.a= QPointF(it1->getX(),it1->getY());
							match.b= QPointF(keypoint2.getX(),keypoint2.getY());

							//Add point
							if(!result.contains(minDistance,match))
								result.insert(minDistance,match);
						}
					}
					gpuBusy= false;
				}else{
					minDistance= std::numeric_limits<float>::infinity();
					double MaxKeypointDistanceSquared= Options::MaxKeypointDistance*Options::MaxKeypointDistance;
					for(QList<SIFT::Keypoint>::const_iterator it2= keypoints2.begin();it2!=keypoints2.end();it2++){
						float distance= it1->distance(*it2,MaxKeypointDistanceSquared);
						if(distance<minDistance && distance<Options::MaxKeypointDistance){
							minDistance= distance;
							match.a= QPointF(it1->getX(),it1->getY());
							match.b= QPointF(it2->getX(),it2->getY());
						}
					}
				}

				if(minDistance!=std::numeric_limits<float>::infinity() && minDistance<Options::MaxKeypointDistance){
					#pragma omp critical
					{
						if(!result.contains(minDistance,match))
							result.insert(minDistance,match);
					}
				}
			}
		}

		if(openCLFailed)
			result= SortedKeypointMatches();

		return result;
	}


	SortedKeypointMatches PointMatcherOpenCL::CalcDistancesOpenCL(const QList<SIFT::Keypoint>& keypoints1,const QList<SIFT::Keypoint>& keypoints2){
		if(!kernelInitialized)
			if(!InitCL())
				return SortedKeypointMatches();

		SortedKeypointMatches result;

		inputVectorsCount= keypoints2.size();
		//Copy data for keypoints2
		for(int i=0;i<keypoints2.size();i++){
			const SIFT::Keypoint& keypoint2= keypoints2[i];
			for(cl_uint j=0;j<vectorSize;j++)
				input[i*vectorSize+j]= keypoint2[j];
		}

		inputVectorsCopied= false;

		//Compute distances for all keypoints from keypoints1, then choose the minumum distance
		for(QList<SIFT::Keypoint>::const_iterator it=keypoints1.begin();it!=keypoints1.end();it++){
			//Prepare current vector
			for(cl_uint i=0;i<vectorSize;i++)
				vector[i]= (*it)[i];

			//Run OpenCL stuff
			if(!CalcDistances())
				return SortedKeypointMatches();

			//Parse OpenCL result
			int min_i=0;
			for(int i=0;i<keypoints2.size();i++)
				if(output[i]<output[min_i])
					min_i=i;
			KeypointMatch match;
			float minDistance= output[min_i];
			if(minDistance<Options::MaxKeypointDistance){
				match.a= QPointF(it->getX(),it->getY());
				match.b= QPointF(keypoints2[min_i].getX(),keypoints2[min_i].getY());

				//Add point
				if(!result.contains(minDistance,match))
					result.insert(minDistance,match);
			}
		}

		return result;
	}
}

#endif // CYBERVISION_OPENCL
