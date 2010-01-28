#include "options.h"


namespace cybervision{
	const double Options::MaxKeypointDistance= 0.5;
	const double Options::ReliableDistance= 0.5;
	const int Options::MinMatches= 16;

	const int Options::RANSAC_k= 10000;
	const int Options::RANSAC_n= 8;
	const double Options::RANSAC_t= 1e-12;
	const int Options::RANSAC_d= 32;
}
