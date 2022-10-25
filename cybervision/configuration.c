#include "configuration.h"

const int cybervision_fast_threshold = 15;
const int cybervision_fast_mode = 12;
const int cybervision_fast_nonmax = 1;

const float cybervision_correlation_threshold = 0.95F;
const int cybervision_correlation_kernel_size = 7;

const int cybervision_keypoint_scale_min_size = 512;

const size_t cybervision_ransac_match_grid_size = 8;
const size_t cybervision_ransac_k_affine = 1E7;
const size_t cybervision_ransac_k_perspective = 1E7;
const size_t cybervision_ransac_n_affine = 4;
const size_t cybervision_ransac_n_perspective = 8;
const double cybervision_ransac_rank_epsilon = 0.0;
const double cybervision_ransac_t_affine = 0.1;
const double cybervision_ransac_t_perspective = 1.0;
const size_t cybervision_ransac_d = 10;
const size_t cybervision_ransac_d_early_exit = 1000;
const size_t cybervision_ransac_check_interval = 1E5;

const int cybervision_crosscorrelation_scale_min_size = 64;
const int cybervision_crosscorrelation_kernel_size = 5;
const float cybervision_crosscorrelation_threshold_parallel = 0.6F;
const float cybervision_crosscorrelation_threshold_perspective = 0.8F;
const int cybervision_crosscorrelation_corridor_size = 20;
// Decrease when using a low-powered GPU
const int cybervision_crosscorrelation_corridor_segment_length = 256;
const int cybervision_crosscorrelation_search_area_segment_length = 8;
const int cybervision_crosscorrelation_neighbor_distance = 10;
const float cybervision_crosscorrelation_corridor_extend_range = 1.0F;
const float cybervision_crosscorrelation_corridor_min_range = 2.5F;

const double cybervision_triangulation_min_scale = 1E-3;
const size_t cybervision_histogram_filter_bins = 100;
const float cybervision_histogram_filter_discard_percentile_parallel = 0.025F;
const float cybervision_histogram_filter_discard_percentile_perspective = 0.2F;
const float cybervision_histogram_filter_epsilon = 1E-3;

#ifndef CYBERVISION_DISABLE_GPU
const correlation_mode cybervision_crosscorrelation_default_mode = CORRELATION_MODE_GPU;
#else
const correlation_mode cybervision_crosscorrelation_default_mode = CORRELATION_MODE_CPU;
#endif

const float cybervision_interpolation_epsilon = 1E-5;
