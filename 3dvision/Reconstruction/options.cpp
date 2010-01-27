#include "options.h"


namespace cybervision{
	const float Options::MaxKeypointDistance= 0.5;
	const float Options::ReliableDistance= 0.4;
	const int Options::MinMatches= 16;

	const int Options::RANSAC_k= 500;
	const int Options::RANSAC_n= 8;
	const float Options::RANSAC_t= 0.01F;
	const int Options::RANSAC_d= 32;
}
