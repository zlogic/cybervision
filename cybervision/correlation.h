#ifndef CORRELATION_H
#define CORRELATION_H

#include <stddef.h>
#include "fundamental_matrix.h"

typedef struct {
    int width, height;
    unsigned char* img;
} correlation_image;

typedef struct { int x,y; } correlation_point;
typedef struct {
    int point1, point2;
    float corr;
} correlation_match;
typedef void* match_task_internal;
typedef struct {
    correlation_image img1, img2;
    correlation_point *points1, *points2;
    size_t points1_size, points2_size;
    int num_threads;

    float percent_complete;
    int completed;

    correlation_match *matches;
    size_t matches_count;

    match_task_internal internal;
} match_task;
int correlation_match_points_start(match_task*);
void correlation_match_points_cancel(match_task*);
int correlation_match_points_complete(match_task*);

typedef void* cross_correlate_task_internal;
typedef struct {
    correlation_image img1, img2;
    matrix_3x3 fundamental_matrix;
    float scale;

    int num_threads;
    int iteration;

    float percent_complete;
    int completed;
    const char *error;

    int out_width, out_height;
    int *correlated_points;

    cross_correlate_task_internal internal;
} cross_correlate_task;
int cpu_correlation_cross_correlate_start(cross_correlate_task*);
void cpu_correlation_cross_correlate_cancel(cross_correlate_task*);
int cpu_correlation_cross_correlate_complete(cross_correlate_task*);

#endif
