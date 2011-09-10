//Kernel to compute the distances from a N-element vector to a set of M N-dimensional vectors

__kernel void computeDistances(__global  float  * outputDistances,
							   __global  float  * inputVectors,
							   __global  float  * vector,
							   const     uint  vectorSize,
							   const     uint  inputVectorsSize)
{
	uint tid = get_global_id(0);

	float sum = 0;
	//Compute Eucledian distance
	for(uint i = 0; i < vectorSize; ++i)
	{
		float vector_i= vector[i];
		float vectors_i= inputVectors[vectorSize*tid+i];
		float sqr= vector_i-vectors_i;
		sqr*= sqr;
		sum+= sqr;
	}

	//outputDistances[tid]= sum;
	outputDistances[tid]= sqrt(sum);
}
