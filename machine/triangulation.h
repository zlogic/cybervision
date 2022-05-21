#ifndef TRIANGULATION_H
#define TRIANGULATION_H

typedef struct {
    int x, y;
    float z;
} triangulation_point;

typedef struct {
    triangulation_point *points;
    size_t num_points;
} triangulation_data;

int triangulation_triangulate(triangulation_data*);

#endif
