#ifndef TRIANGULATION_H
#define TRIANGULATION_H

typedef enum
{ 
    TRIANGULATION_PROJECTION_MODE_PARALLEL = 0,
    TRIANGULATION_PROJECTION_MODE_PERSPECTIVE = 1
} triangulation_projection_mode;

typedef void* triangulation_task_internal;
typedef struct {
    int width, height;
    int *correlated_points;
    triangulation_projection_mode proj_mode;
    float depth_scale;
    
    double fundamental_matrix[9];
    float tilt_angle;

    int num_threads;
    const char *error;

    float percent_complete;
    int completed;

    float* out_depth;

    triangulation_task_internal internal;
} triangulation_task;

int triangulation_start(triangulation_task*);
void triangulation_cancel(triangulation_task*);
int triangulation_complete(triangulation_task*);

#endif
