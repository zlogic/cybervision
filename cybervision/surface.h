#ifndef SURFACE_H
#define SURFACE_H

typedef struct {
    int width, height;
    float* depth;
} surface_data;

typedef enum 
{ 
    INTERPOLATION_NONE = 0,
    INTERPOLATION_DELAUNAY = 1,
    INTERPOLATION_IDW = 2
} interpolation_mode;

typedef void* output_surface_task_internal;
typedef struct {
    surface_data surf;
    char* output_filename;
    interpolation_mode mode;

    int num_threads;

    float percent_complete;
    int completed;

    output_surface_task_internal internal;
} output_surface_task;
int surface_output_start(output_surface_task*);
void surface_output_cancel(output_surface_task*);
int surface_output_complete(output_surface_task*);

#endif
