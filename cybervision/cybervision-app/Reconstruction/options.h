#ifndef OPTIONS_H
#define OPTIONS_H
#include <cstddef>

namespace cybervision{
/*
 * Class for storing options, constants, etc. At the moment is read-only and can be changed before compilation.
 */
class Options{
public:
	//SIFT options
	static const double SIFTContrastCorrection;//SIFT constrast correction coefficient (amount to multiply default contrast threshold; can be used to lower or raise the threshold).
	//Reconstructor options
	static const double MaxKeypointDistance;//Maximum SIFT match vector distance, larger values get discarded
	static const double ReliableDistance;//Maximum keypoint distance on which camera pose (E, R&T) can be computed
	static const int MinMatches;//Minimum amount of matches needed for pose estimation
	static const bool UsePrecomputedKeypointData;//Use precomputed data when possible (to skip time-consuming steps)
	static const bool SaveFilteredMatches;//Save RANSAC-filtered matches to file
	enum KeypointMatchingMode {KEYPOINT_MATCHING_SIMPLE,KEYPOINT_MATCHING_KDTREE,KEYPOINT_MATCHING_OPENCL_CPU,KEYPOINT_MATCHING_OPENCL_GPU,KEYPOINT_MATCHING_OPENCL_HYBRID};//Mode for matching keypoints (simple comparison or BBF KT-tree or OpenCL)
	static const size_t bbf_steps;//Number of best-bin-first search iterations
	static const KeypointMatchingMode keypointMatchingMode;//Keypoint matching mode
	enum TriangulationMode {TRIANGULATION_PERSPECTIVE,TRIANGULATION_PARALLEL};//Mode for triangulation (perspective or parallel (V or SV) projection mode)
	static const TriangulationMode triangulationMode;//Selected point triangulation mode
	static const double constraintsThreshold;//Minimum non-zero value in Sigma matrix in SVD for linear constraints (only for parallel triangulation)
	static const int maxTriangulationPointSize;//Maximum point set size for parallel triangulation (to prevent SVD memory issues leading to a segfault)

	static const int RANSAC_k;//RANSAC k parameter
	static const int RANSAC_n;//RANSAC n parameter
	static const double RANSAC_t;//RANSAC t parameter
	static const int RANSAC_d;//RANSAC d parameter

	//Reconstructor data
	static const double scaleFocalDistance;//Focal distance for determining scale

	//Surface options
	static const float peakSize;//Minimum relative height of point (inside a triangle) for it to be considered to be a peak
	static const int maxPeakFilterPasses;//Maximum number of iterative applications of the peak filter
	static const float surfaceSize;//Target size (width~height) of surface, to fit into opengl viewport
	enum ColladaFormat {COLLADA_INDEPENDENT_POLYGONS,COLLADA_SHARED_POINTS}; //Shared points produces a more compact file but mangles normals
	static const ColladaFormat colladaFormat;//Selected Collada file format
	enum AveragingNormalsMode {AVERAGE_NORMALS_TRIANGLE,AVERAGE_NORMALS_LINE}; //Average normals for point: from neighboring lines or triangles
	static const AveragingNormalsMode averageNormalsMode;
	static const bool mapPointsToGrid;//Map points to grid (like Matlab)
	static const int gridResolution;//Grid resolution (number of iterations for splitting point set). Works only when mapPointsToGrid is true.
	static const bool gridAddRealPoints;//Add all points after grid interpolation (increases resolution for point-rich areas). Works only when mapPointsToGrid is true.
	static const double gridCellArea;//Maximum distance from point to cell center to use the point for averaging (may be more than sqrt(2) to overlap neighboring cells). Works only when mapPointsToGrid is true.
	static const float gridPeakFilterRadius;//Radius factor for checking the peak filter
	static const float gridPeakSize;//Minimum height ratio of two close point for the higher point to be considered to be a peak
	enum StatsBaseLevelMethod{STATS_BASELEVEL_MEDIAN,STATS_BASELEVEL_HISTOGRAM};//Methods for calculating the depth base level
	static const StatsBaseLevelMethod statsBaseLevelMethod;
	static const int statsDepthHistogramSize;//Number of histogram elements for computing base depth level

	//Rendering options
	static const bool renderShiny; //Render surface as a shiny texture
	enum PointDrawingMode {POINT_DRAW_AS_POINT,POINT_DRAW_AS_3DCIRCLE}; //Draw points as a simple point or as a small circle (with 3D scaling)
	static const float PointDiameter;//Point diameter for OpenGL
	static const float TextUnitScale;//Unit scale for outputting text (e.g. display all numbers in micrometers)
};

}
#endif // OPTIONS_H
