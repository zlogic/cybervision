#ifndef TRIANGULATION_H
#define TRIANGULATION_H

typedef struct {
    int width, height;
    float* depth;
} surface_data;

typedef enum 
{ 
    OUTPUT_SURFACE_OBJ = 0,
    OUTPUT_SURFACE_PLY = 1
} output_surface_format;

int triangulation_triangulate(surface_data*, FILE*, output_surface_format);
int triangulation_interpolate(surface_data* data);

#endif
