#include "options.h"


namespace cybervision{
	const double Options::MaxKeypointDistance= 0.5;
	const double Options::ReliableDistance= 0.5;
	const int Options::MinMatches= 16;
	const bool Options::UsePrecomputedKeypointData= true;
	const bool Options::SaveFilteredMatches= false;
	const Options::KeypointMatchingMode Options::keypointMatchingMode= Options::KEYPOINT_MATCHING_SIMPLE;
	const size_t Options::bbf_steps= 1000;
	const Options::TriangulationMode Options::triangulationMode= Options::TRIANGULATION_PARALLEL;

	const int Options::RANSAC_k= 10000;
	const int Options::RANSAC_n= 8;
	const double Options::RANSAC_t= 1e-18;
	const int Options::RANSAC_d= 32;

	const double Options::scaleFocalDistance= 1e-2;

	const float Options::surfaceSize= 15.0F;
	const float Options::peakSize= 10.0F;
	const int Options::maxPeakFilterPasses= 10;
	const Options::ColladaFormat Options::colladaFormat= Options::COLLADA_INDEPENDENT_POLYGONS;
}
