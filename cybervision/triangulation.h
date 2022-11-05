#ifndef TRIANGULATION_H
#define TRIANGULATION_H

#include "configuration.h"
#include "linmath.h"

typedef void* triangulation_task_internal;
typedef struct {
    int width, height;
    int *correlated_points;
    projection_mode proj_mode;
    float scale_x, scale_y, scale_z;

    double projection_matrix_2[4*3];
    float tilt_angle;

    int num_threads;
    const char *error;

    float percent_complete;
    int completed;

    float* out_depth;

    triangulation_task_internal internal;
} triangulation_task;

int triangulation_triangulate_point(svd_internal svd_ctx, double projection2[4*3], double x1, double y1, double x2, double y2, double result[4]);
int triangulation_start(triangulation_task*);
void triangulation_cancel(triangulation_task*);
int triangulation_complete(triangulation_task*);

#endif
