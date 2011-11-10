//Kernel to compute the distances from a N-element vector to a set of M N-dimensional vectors

__kernel void computeDistances(__global   float  * outputDistances,
							   __global   float  * input1Vectors,
							   __global   float  * input2Vectors)
{
	uint gTid1 = get_global_id(0);
	uint gTid2 = get_global_id(1);
	uint lTid1 = get_local_id(0);
	uint lTid2 = get_local_id(1);
	//float input1Local[VECTOR_SIZE];
	__local float input2Local[MAX_THREADS_SECOND_DIMENSION*VECTOR_SIZE];

	//#pragma unroll
	//for(uint i=0;i<VECTOR_SIZE;i++)
	//	input1Local[i]= input1Vectors[gTid1*VECTOR_SIZE+i];

	//Copy to inputLocal
	#pragma unroll
	for(uint i=0;i<VECTOR_SIZE;i++)
		input2Local[lTid2*VECTOR_SIZE+i]= input2Vectors[gTid2*VECTOR_SIZE+i];
	barrier(CLK_LOCAL_MEM_FENCE);


	float sum = 0;
	//Compute Eucledian distance
	float sqr;

	#pragma unroll
	for(uint i=0;i<VECTOR_SIZE;i++){
		//sqr= input1Local[i]-input2Local[VECTOR_SIZE*lTid2+i];
		//sqr= input1Vectors[VECTOR_SIZE*gTid1+i]-input2Vectors[VECTOR_SIZE*gTid2+i];
		sqr= input1Vectors[VECTOR_SIZE*gTid1+i]-input2Local[VECTOR_SIZE*lTid2+i];
		sum+= sqr*sqr;
	}

	outputDistances[gTid1*get_global_size(1)+gTid2]= sqrt(sum);
}
