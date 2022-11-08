#ifndef CONFIGURATION_H
#define CONFIGURATION_H

#include <stddef.h>

extern const int cybervision_fast_threshold;
extern const int cybervision_fast_mode;
extern const int cybervision_fast_nonmax;

extern const float cybervision_correlation_threshold;
extern const int cybervision_correlation_kernel_size;

extern const int cybervision_keypoint_scale_min_size;

typedef enum
{
    PROJECTION_MODE_PARALLEL = 0,
    PROJECTION_MODE_PERSPECTIVE = 1
} projection_mode;

extern const size_t cybervision_ransac_match_grid_size;
extern const size_t cybervision_ransac_k_affine;
extern const size_t cybervision_ransac_k_perspective;
extern const size_t cybervision_ransac_n_affine;
extern const size_t cybervision_ransac_n_perspective;
extern const double cybervision_ransac_rank_epsilon;
extern const double cybervision_ransac_t_affine;
extern const double cybervision_ransac_t_perspective;
extern const size_t cybervision_ransac_d;
extern const size_t cybervision_ransac_d_early_exit_parallel;
extern const size_t cybervision_ransac_d_early_exit_perspective;
extern const size_t cybervision_ransac_check_interval;

extern const int cybervision_crosscorrelation_scale_min_size;
extern const int cybervision_crosscorrelation_kernel_size;
extern const float cybervision_crosscorrelation_threshold_parallel;
extern const float cybervision_crosscorrelation_threshold_perspective;
extern const float cybervision_crosscorrelation_min_stdev_parallel;
extern const float cybervision_crosscorrelation_min_stdev_perspective;
extern const int cybervision_crosscorrelation_corridor_size;
extern const int cybervision_crosscorrelation_corridor_segment_length;
extern const int cybervision_crosscorrelation_search_area_segment_length;
extern const int cybervision_crosscorrelation_neighbor_distance;
extern const float cybervision_crosscorrelation_corridor_extend_range;
extern const float cybervision_crosscorrelation_corridor_min_range;

extern const double cybervision_triangulation_min_scale;
extern const size_t cybervision_triangulation_optimization_block_count;
extern const size_t cybervision_histogram_filter_bins;
extern const float cybervision_histogram_filter_discard_percentile;
extern const float cybervision_histogram_filter_epsilon;

typedef enum 
{ 
    CORRELATION_MODE_CPU = 0,
    CORRELATION_MODE_GPU = 1
} correlation_mode;
extern const correlation_mode cybervision_crosscorrelation_default_mode;

extern const float cybervision_interpolation_epsilon;

#endif
