#ifndef CONFIGURATION_H
#define CONFIGURATION_H

#include <stddef.h>

const int cybervision_fast_threshold;
const int cybervision_fast_mode;
const int cybervision_fast_nonmax;

const float cybervision_correlation_threshold;
const int cybervision_correlation_kernel_size;

const int cybervision_keypoint_scale_min_size;

const float cybervision_ransac_min_length;
const size_t cybervision_ransac_k;
const size_t cybervision_ransac_n;
const float cybervision_ransac_t;
const size_t cybervision_ransac_d;
const size_t cybervision_ransac_check_interval;

const int cybervision_crosscorrelation_scale_min_size;
const int cybervision_crosscorrelation_kernel_size;
const float cybervision_crosscorrelation_threshold;
const int cybervision_crosscorrelation_corridor_size;
const int cybervision_crosscorrelation_neighbor_distance;
const float cybervision_crosscorrelation_max_slope;
const int cybervision_crosscorrelation_match_limit;

const float cybervision_interpolation_epsilon;

#endif
