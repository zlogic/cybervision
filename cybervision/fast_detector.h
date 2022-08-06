#ifndef FAST_DETECTOR_H
#define FAST_DETECTOR_H

#include "correlation.h"

correlation_point* fast_detect(correlation_image *img, size_t  *count);

#endif
