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

const float scales[] = {1.0F/8.0F, 1.0F/4.0F, 1.0F/2.0F, 1.0F};
const float *cybervision_crosscorrelation_scales = scales;
const int cybervision_crosscorrelation_scales_count = 4;
const int cybervision_crosscorrelation_kernel_size = 5;
const float cybervision_crosscorrelation_threshold = 0.8F;
const int cybervision_crosscorrelation_corridor_size = 5;
// const int cybervision_crosscorrelation_corridor = 7;
const int cybervision_crosscorrelation_neighbor_distance = 4;
const float cybervision_crosscorrelation_max_slope = 0.5F;
