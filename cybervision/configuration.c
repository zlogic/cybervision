#include "configuration.h"

const int cybervision_fast_threshold = 15;
const int cybervision_fast_mode = 12;
const int cybervision_fast_nonmax = 1;

const float cybervision_correlation_threshold = 0.9F;

// slower, but more effective
// const int cybervision_correlation_kernel_size = 10;
const int cybervision_correlation_kernel_size = 7;

const int cybervision_keypoint_scale_min_size = 512;

const float cybervision_ransac_min_length = 3.0F;
const size_t cybervision_ransac_k = 1E6;
const size_t cybervision_ransac_n_affine = 6;
const size_t cybervision_ransac_n_perspective = 10;
const float cybervision_ransac_collinear_epsilon = 1.0F;
const float cybervision_ransac_t = 20.0F;
const size_t cybervision_ransac_d = 10;
const size_t cybervision_ransac_check_interval = 100000;

const int cybervision_crosscorrelation_scale_min_size = 64;
const int cybervision_crosscorrelation_kernel_size = 5;
const float cybervision_crosscorrelation_threshold = 0.8F;
const int cybervision_crosscorrelation_corridor_size = 5;
// const int cybervision_crosscorrelation_corridor_size = 7;
// Decrease when using a low-powered GPU
const int cybervision_crosscorrelation_corridor_segment_length = 256;
const int cybervision_crosscorrelation_search_area_segment_length = 8;
const int cybervision_crosscorrelation_neighbor_distance = 6;
// const int cybervision_crosscorrelation_neighbor_distance = 8;
const float cybervision_crosscorrelation_max_slope = 0.5F;
const int cybervision_crosscorrelation_match_limit = 16;

#ifndef CYBERVISION_DISABLE_GPU
const correlation_mode cybervision_crosscorrelation_default_mode = CORRELATION_MODE_GPU;
#else
const correlation_mode cybervision_crosscorrelation_default_mode = CORRELATION_MODE_CPU;
#endif

const float cybervision_interpolation_epsilon = 1E-5;
