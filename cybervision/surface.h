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

int surface_output(surface_data* surf, char* output_filename, interpolation_mode mode);

#endif
