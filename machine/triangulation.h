#ifndef TRIANGULATION_H
#define TRIANGULATION_H

typedef struct {
    int width, height;
    float* depth;
} triangulation_data;

int triangulation_triangulate(triangulation_data*);

#endif
