#include "configuration.h"

const int cybervision_fast_threshold = 15;
const int cybervision_fast_mode = 12;
const int cybervision_fast_nonmax = 1;

const float cybervision_correlation_threshold = 0.9F;
// slower, but more effective
// const int cybervision_correlation_kernel_size = 10;
const int cybervision_correlation_kernel_size = 7;

const int cybervision_keypoint_scale_min_size = 512;

const size_t cybervision_ransac_k = 1E7;
const size_t cybervision_ransac_n_affine = 4;
const size_t cybervision_ransac_n_perspective = 8;
const float cybervision_ransac_t_affine = 1.0F;
const float cybervision_ransac_t_perspective = 5.0F;
const size_t cybervision_ransac_d = 10;
const size_t cybervision_ransac_d_early_exit = 100;
const size_t cybervision_ransac_check_interval = 1E6;

const int cybervision_crosscorrelation_scale_min_size = 64;
const int cybervision_crosscorrelation_kernel_size = 5;
const float cybervision_crosscorrelation_threshold = 0.7F;
const int cybervision_crosscorrelation_corridor_size = 20;
// const int cybervision_crosscorrelation_corridor_size = 7;
// Decrease when using a low-powered GPU
const int cybervision_crosscorrelation_corridor_segment_length = 256;
const int cybervision_crosscorrelation_search_area_segment_length = 8;
const int cybervision_crosscorrelation_neighbor_distance = 10;
// const int cybervision_crosscorrelation_neighbor_distance = 8;
const float cybervision_crosscorrelation_corridor_extend_range = 1.0F;
const int cybervision_crosscorrelation_match_limit = 20;

#ifndef CYBERVISION_DISABLE_GPU
const correlation_mode cybervision_crosscorrelation_default_mode = CORRELATION_MODE_GPU;
#else
const correlation_mode cybervision_crosscorrelation_default_mode = CORRELATION_MODE_CPU;
#endif

const float cybervision_interpolation_epsilon = 1E-5;
