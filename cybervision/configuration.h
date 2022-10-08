#ifndef CONFIGURATION_H
#define CONFIGURATION_H

#include <stddef.h>

extern const int cybervision_fast_threshold;
extern const int cybervision_fast_mode;
extern const int cybervision_fast_nonmax;

extern const float cybervision_correlation_threshold;
extern const int cybervision_correlation_kernel_size;

extern const int cybervision_keypoint_scale_min_size;

extern const float cybervision_ransac_min_length;
extern const size_t cybervision_ransac_k;
extern const size_t cybervision_ransac_n_affine;
extern const size_t cybervision_ransac_n_perspective;
extern const float cybervision_ransac_collinear_epsilon;
extern const float cybervision_ransac_t;
extern const size_t cybervision_ransac_d;
extern const size_t cybervision_ransac_check_interval;

extern const int cybervision_crosscorrelation_scale_min_size;
extern const int cybervision_crosscorrelation_kernel_size;
extern const float cybervision_crosscorrelation_threshold;
extern const int cybervision_crosscorrelation_corridor_size;
extern const int cybervision_crosscorrelation_corridor_segment_length;
extern const int cybervision_crosscorrelation_search_area_segment_length;
extern const int cybervision_crosscorrelation_neighbor_distance;
extern const float cybervision_crosscorrelation_max_slope;
extern const int cybervision_crosscorrelation_match_limit;

typedef enum 
{ 
    CORRELATION_MODE_CPU = 0,
    CORRELATION_MODE_GPU = 1
} correlation_mode;
extern const correlation_mode cybervision_crosscorrelation_default_mode;

extern const float cybervision_interpolation_epsilon;

#endif
