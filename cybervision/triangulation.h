#ifndef TRIANGULATION_H
#define TRIANGULATION_H

typedef struct {
    int width, height;
    float* depth;
} surface_data;

int triangulation_triangulate(surface_data*, FILE*);
int triangulation_interpolate(surface_data* data);

#endif
