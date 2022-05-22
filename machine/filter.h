#ifndef FILTER_H
#define FILTER_H

#include "surface.h"

int filter_peaks(surface_data*, float sigma, int min_points, float threshold);

#endif
