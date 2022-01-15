#include "options.h"


namespace cybervision{

const double Options::SIFTContrastCorrection= 0.25;

const double Options::MaxKeypointDistance= 0.5;
const double Options::ReliableDistance= 0.5;
const int Options::MinMatches= 16;
const bool Options::UsePrecomputedKeypointData= false;
const bool Options::SaveFilteredMatches= false;
const Options::KeypointMatchingMode Options::keypointMatchingMode= Options::KEYPOINT_MATCHING_OPENCL_HYBRID;
const size_t Options::bbf_steps= 1000;
const Options::TriangulationMode Options::triangulationMode= Options::TRIANGULATION_PARALLEL;
const double Options::constraintsThreshold= 0.0000001;
const int Options::maxTriangulationPointSize= 8000;

const int Options::RANSAC_k= 10000;
const int Options::RANSAC_n= 8;
const double Options::RANSAC_t= 1e-18;
const int Options::RANSAC_d= 32;

const double Options::scaleFocalDistance= 1e-2;

const float Options::surfaceSize= 15.0F;
const float Options::peakSize= 10.0F;
const int Options::maxPeakFilterPasses= 25;
const Options::ColladaFormat Options::colladaFormat= Options::COLLADA_SHARED_POINTS;
const Options::AveragingNormalsMode Options::averageNormalsMode= Options::AVERAGE_NORMALS_TRIANGLE;
const bool Options::mapPointsToGrid= true;
const int Options::gridResolution= 6;
const bool Options::gridAddRealPoints= false;
const double Options::gridCellArea= 4.0;
const float Options::gridPeakFilterRadius= 5;
const float Options::gridPeakSize= 2.0;
const Options::StatsBaseLevelMethod Options::statsBaseLevelMethod= Options::STATS_BASELEVEL_HISTOGRAM;
const int Options::statsDepthHistogramSize= 50;

const bool Options::renderShiny= true;
const float Options::PointDiameter= 0.25;
const float Options::TextUnitScale= 1E6;

}
