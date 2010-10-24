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
	enum KeypointMatchingMode {KEYPOINT_MATCHING_SIMPLE,KEYPOINT_MATCHING_KDTREE};//Mode for matching keypoints (BBF KT-tree or simple comparison)
	static const size_t bbf_steps;//Number of best-bin-first search iterations
	static const KeypointMatchingMode keypointMatchingMode;//Keypoint matching mode

	static const int RANSAC_k;//RANSAC k parameter
	static const int RANSAC_n;//RANSAC n parameter
	static const double RANSAC_t;//RANSAC t parameter
	static const int RANSAC_d;//RANSAC d parameter

	//Surface options
	static const float surfaceSize;//Target size (width~height) of surface, to fit into opengl viewport
	static const float surfaceDepth;//Target depth of surface, to fit into opengl viewport
	enum ColladaFormat {COLLADA_INDEPENDENT_POLYGONS,COLLADA_SHARED_POINTS}; //Shared points produces a more compact file but mangles normals
	static const ColladaFormat colladaFormat;//Selected Collada file format
};

}
#endif // OPTIONS_H
