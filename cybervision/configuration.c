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
const size_t cybervision_ransac_d_early_exit_parallel = 1000;
const size_t cybervision_ransac_d_early_exit_perspective = 20;
const size_t cybervision_ransac_check_interval = 5E5;

const size_t cybervision_lm_max_iterations = 30;
const double cybervision_lm_jacobian_h = 0.001;
const double cybervision_lm_lambda_start = 1E-2;
const double cybervision_lm_lambda_up = 11.0;
const double cybervision_lm_lambda_down = 9.0;
const double cybervision_lm_lambda_min = 1E-7;
const double cybervision_lm_lambda_max = 1E7;
const double cybervision_lm_rho_epsilon = 1E-1;
const double cybervision_lm_jt_residual_epsilon = 1E-3;
const double cybervision_lm_ratio_epsilon = 1E-3;
const double cybervision_lm_division_epsilon = 1E-12;

const int cybervision_crosscorrelation_scale_min_size = 64;
const int cybervision_crosscorrelation_kernel_size = 5;
const float cybervision_crosscorrelation_threshold_parallel = 0.6F;
const float cybervision_crosscorrelation_threshold_perspective = 0.7F;
const float cybervision_crosscorrelation_min_stdev_parallel = 1.0F;
const float cybervision_crosscorrelation_min_stdev_perspective = 25.0F;
const int cybervision_crosscorrelation_corridor_size = 20;
// Decrease when using a low-powered GPU
const int cybervision_crosscorrelation_corridor_segment_length = 256;
const int cybervision_crosscorrelation_search_area_segment_length = 8;
const int cybervision_crosscorrelation_neighbor_distance = 10;
const float cybervision_crosscorrelation_corridor_extend_range = 1.0F;
const float cybervision_crosscorrelation_corridor_min_range = 2.5F;

const double cybervision_triangulation_min_scale = 1E-3;
const size_t cybervision_triangulation_optimization_block_count = 100;
const size_t cybervision_histogram_filter_bins = 100;
const float cybervision_histogram_filter_discard_percentile = 0.025F;
const float cybervision_histogram_filter_epsilon = 1E-3;

#ifndef CYBERVISION_DISABLE_GPU
const correlation_mode cybervision_crosscorrelation_default_mode = CORRELATION_MODE_GPU;
#else
const correlation_mode cybervision_crosscorrelation_default_mode = CORRELATION_MODE_CPU;
#endif

const float cybervision_interpolation_epsilon = 1E-5;
