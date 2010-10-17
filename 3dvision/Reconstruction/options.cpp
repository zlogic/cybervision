#include "options.h"


namespace cybervision{
	const double Options::MaxKeypointDistance= 0.5;
	const double Options::ReliableDistance= 0.5;
	const int Options::MinMatches= 16;
	const bool Options::UsePrecomputedKeypointData= true;
	const Options::KeypointMatchingMode Options::keypointMatchingMode= Options::KEYPOINT_MATCHING_KDTREE;
	const size_t Options::bbf_steps= 200;

	const int Options::RANSAC_k= 10000;
	const int Options::RANSAC_n= 8;
	const double Options::RANSAC_t= 1e-18;
	const int Options::RANSAC_d= 32;

	const float Options::surfaceSize= 15.0F;
	const float Options::surfaceDepth= 10.0F;
}
