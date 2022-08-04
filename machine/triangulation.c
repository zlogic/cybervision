#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <stdlib.h>
#include <math.h>

#include "libqhull_r/libqhull_r.h"
#include "libqhull_r/poly_r.h"

#include "triangulation.h"

coordT* convert_points(surface_data* data, int *num_points)
{
    coordT *points;
    coordT *current_point = NULL;
    int np = 0;
    for (int y=0;y<data->height;y++)
    {
        for (int x=0;x<data->width;x++)
        {
            float depth = data->depth[y*data->width + x];
            if (!isfinite(depth))
                continue;
            np++;
        }
    }
    points = malloc(sizeof(coordT)*np*2);
    *num_points = np;
    current_point = points;
    for (int y=0;y<data->height;y++)
    {
        for (int x=0;x<data->width;x++)
        {
            float depth = data->depth[y*data->width + x];
            if (!isfinite(depth))
                continue;
            *(current_point++) = x;
            *(current_point++) = y;
        }
    }
    return points;
}

int output_points(qhT *qh, surface_data* data, PyObject *out)
{
    coordT *point, *pointtemp;
    FORALLpoints
    {
        int x = (int)point[0], y = (int)point[1];
        float z;
        if(qh_pointid(qh, point) == qh->num_points-1)
            break;
        if (x<0 || x>=data->width || y<0 || y>=data->height)
            return 0;
        z = data->depth[y*data->width+x];
        if (!isfinite(z))
            return 0;

        PyObject *new_point = Py_BuildValue("(iif)", x, y, z);
        PyList_Append(out, new_point);
        Py_DECREF(new_point);
    }
    return 1;
}

void output_simplices(qhT *qh, surface_data* data, PyObject *out)
{
    facetT *facet;
    vertexT *vertex, **vertexp;
    FORALLfacets
    {
        PyObject *simplex;
        if (facet->upperdelaunay)
            continue;
        simplex = PyList_New(0);
        PyList_Append(out, simplex);
        Py_DECREF(simplex);
        if ((facet->toporient ^ qh_ORIENTclock))
        {
            FOREACHvertexreverse12_(facet->vertices)
            {
                PyObject *value = Py_BuildValue("i", qh_pointid(qh, vertex->point));
                PyList_Append(simplex, value);
                Py_DECREF(value);
            }
        }
        else
        {
            FOREACHvertex_(facet->vertices)
            {
                PyObject *value = Py_BuildValue("i", qh_pointid(qh, vertex->point));
                PyList_Append(simplex, value);
                Py_DECREF(value);
            }
        }
    }
}

int triangulation_triangulate(surface_data* data, PyObject *out_points, PyObject *out_simplices)
{
    coordT *points;
    qhT qh = {0};
    int result = 0;
    int num_points = 0;
    int curlong, totlong;

    points = convert_points(data, &num_points);
    
    result = qh_new_qhull(&qh, 2, num_points, points, True, "qhull d Qt Qbb Qc Qz Q12", NULL, NULL) == 0;
    result = result && output_points(&qh, data, out_points);
    if (result)
        output_simplices(&qh, data, out_simplices);

    qh_freeqhull(&qh, qh_ALL);
    qh_memfreeshort(&qh, &curlong, &totlong);
    return result;
}
