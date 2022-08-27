#ifndef GPU_CORRELATION_H
#define GPU_CORRELATION_H

#include "correlation.h"

int gpu_correlation_cross_correlate_init(cross_correlate_task *t, size_t img1_pixels, size_t img2_pixels);
int gpu_correlation_cross_correlate_start(cross_correlate_task*);
void gpu_correlation_cross_correlate_cancel(cross_correlate_task*);
int gpu_correlation_cross_correlate_complete(cross_correlate_task*);
int gpu_correlation_cross_correlate_cleanup(cross_correlate_task*);

#endif
