#include "options.h"


namespace cybervision{
	const double Options::MaxKeypointDistance= 0.6;
	const double Options::ReliableDistance= 0.6;
	const int Options::MinMatches= 16;

	const int Options::RANSAC_k= 100000;
	const int Options::RANSAC_n= 8;
	const double Options::RANSAC_t= 1e-8;
	const int Options::RANSAC_d= 32;

	const float Options::surfaceSize= 15.0F;
	const float Options::surfaceDepth= 2.5F;
}
