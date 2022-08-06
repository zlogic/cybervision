#include "configuration.h"

const int cybervision_fast_threshold = 15;
const int cybervision_fast_mode = 12;
const int cybervision_fast_nonmax = 1;

const float cybervision_correlation_threshold = 0.9F;

// slower, but more effective
// const int cybervision_correlation_kernel_size = 10;
const int cybervision_correlation_kernel_size = 7;

const float cybervision_ransac_min_length = 3.0F;
const size_t cybervision_ransac_k = 10000;
const size_t cybervision_ransac_n = 10;
const float cybervision_ransac_t = 0.01F;
const size_t cybervision_ransac_d = 10;
