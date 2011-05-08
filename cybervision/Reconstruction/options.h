#ifndef OPTIONS_H
#define OPTIONS_H
#include <cstddef>

namespace cybervision{

class Options{
public:
	//Reconstructor options
	static const double MaxKeypointDistance;//Maximum SIFT match vector distance, larger values get discarded
	static const double ReliableDistance;//Maximum keypoint distance on which camera pose (E, R&T) can be computed
	static const int MinMatches;//Minimum amount of matches needed for pose estimation
	static const bool UsePrecomputedKeypointData;//Use precomputed data when possible (to skip time-consuming steps)
	static const bool SaveFilteredMatches;//Save RANSAC-filtered matches to file
	enum KeypointMatchingMode {KEYPOINT_MATCHING_SIMPLE,KEYPOINT_MATCHING_KDTREE};//Mode for matching keypoints (BBF KT-tree or simple comparison)
	static const size_t bbf_steps;//Number of best-bin-first search iterations
	static const KeypointMatchingMode keypointMatchingMode;//Keypoint matching mode
	enum TriangulationMode {TRIANGULATION_PERSPECTIVE,TRIANGULATION_PARALLEL_V,TRIANGULATION_PARALLEL_SV};//Mode for triangulation (perspective or parallel (V or SV) projection mode)
	static const TriangulationMode triangulationMode;

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

	//Rendering options
	static const bool renderShiny; //Render surface as a shiny texture
	enum RenderNormalsMode {RENDER_NORMALS_TRIANGLE,RENDER_NORMALS_POINT}; //Use normals from polygon or from point when rendering
	static const RenderNormalsMode renderNormalsMode; //Selected normals source for rendering
};

}
#endif // OPTIONS_H
