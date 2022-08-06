#ifndef CONFIGURATION_H
#define CONFIGURATION_H

#include <stddef.h>

const int cybervision_fast_threshold;
const int cybervision_fast_mode;
const int cybervision_fast_nonmax;

const float cybervision_correlation_threshold;
const int cybervision_correlation_kernel_size;

const float cybervision_ransac_min_length;
const size_t cybervision_ransac_k;
const size_t cybervision_ransac_n;
const float cybervision_ransac_t;
const size_t cybervision_ransac_d;

const float *cybervision_triangulation_scales;
const int cybervision_triangulation_scales_count;
const int cybervision_triangulation_kernel_size;
const float cybervision_triangulation_threshold;
const int cybervision_triangulation_corridor_size;
const int cybervision_triangulation_neighbor_distance;
const float cybervision_triangulation_max_slope;

#endif
