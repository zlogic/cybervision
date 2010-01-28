#ifndef OPTIONS_H
#define OPTIONS_H

//Uses precumputed data when possible (to skip time-consuming steps)
#define USE_PRECOMPUTED_DATA

namespace cybervision{

class Options{
public:
	static const double MaxKeypointDistance;//Maximum SIFT match vector distance, larger values get discarded
	static const double ReliableDistance;//Maximum keypoint distance on which camera pose (E, R&T) can be computed
	static const int MinMatches;//Minimum amount of matches needed for pose estimation
	static const int RANSAC_k;//RANSAC k parameter
	static const int RANSAC_n;//RANSAC n parameter
	static const double RANSAC_t;//RANSAC t parameter
	static const int RANSAC_d;//RANSAC d parameter
};

}
#endif // OPTIONS_H
