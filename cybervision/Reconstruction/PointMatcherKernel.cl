//Kernel to compute the distances from a N-element vector to a set of M N-dimensional vectors

__kernel void computeDistances(__global   float  * outputDistances,
							   __global   float  * inputVectors,
							   __constant float  * vector,
							   const      uint   inputVectorsCount,
							   const      float  MaxKeypointDistance)
{
	uint gTid = get_global_id(0);

	float sum = 0;
	//Compute Eucledian distance
	float sqr;
	#pragma unroll
	for(uint i=0;i<VECTOR_SIZE;i++){
		sqr= vector[i]-inputVectors[VECTOR_SIZE*gTid+i];
		sum+= sqr*sqr;
	}

	outputDistances[gTid]= sqrt(sum);
}
