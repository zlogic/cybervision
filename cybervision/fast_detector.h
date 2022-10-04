#ifndef FAST_DETECTOR_H
#define FAST_DETECTOR_H

#include "correlation.h"

correlation_point* fast_detect(correlation_image *img, float scale, size_t  *count);

#endif
