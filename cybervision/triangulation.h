#ifndef TRIANGULATION_H
#define TRIANGULATION_H

#include "configuration.h"

typedef void* triangulation_task_internal;
typedef struct {
    int width, height;
    int *correlated_points;
    float scale_x, scale_y, scale_z;
    
    double fundamental_matrix[9];
    float tilt_angle;

    const char *error;

    float* out_depth;
} triangulation_task;

void triangulation_triangulate(triangulation_task*);

#endif
