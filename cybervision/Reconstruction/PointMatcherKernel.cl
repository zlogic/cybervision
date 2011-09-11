//Kernel to compute the distances from a N-element vector to a set of M N-dimensional vectors

__kernel void computeDistances(__global  float  * outputDistances,
							   __global  float  * inputVectors,
							   __global  float  * vector,
							   const     uint   * inputVectorsCount)
{
	uint gTid = get_global_id(0);
	uint lTid = get_local_id(0);

	if(gTid>=inputVectorsCount)
		return;

	float sum = 0;
	//Compute Eucledian distance
	for(uint i = 0; i < 128; ++i)
	{
		float vector_i= vector[i];
		float vectors_i= inputVectors[128*gTid+i];
		float sqr= vector_i-vectors_i;
		sqr*= sqr;
		sum+= sqr;
	}

	outputDistances[gTid]= sqrt(sum);
}
