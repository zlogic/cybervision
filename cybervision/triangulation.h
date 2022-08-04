#ifndef TRIANGULATION_H
#define TRIANGULATION_H

#include "surface.h"

int triangulation_triangulate(surface_data*, PyObject *out_points, PyObject *out_simplices);

#endif
