#include <stdlib.h>
#include <math.h>

#include "libqhull_r/libqhull_r.h"

#include "triangulation.h"

coordT* convert_points(triangulation_data* data, size_t *num_points)
{
    coordT *points;
    coordT *current_point = NULL;
    size_t np = 0;
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

int output_obj(qhT *qh, triangulation_data* data)
{
    // TODO: rewrite this to interact better with Python
    FILE *outfile = fopen("out.obj", "w");
    facetT *facet;
    vertexT *vertex;
    FORALLvertices {
        int x = (int)vertex->point[0], y = (int)vertex->point[1];
        float z;
        if (x<0 || x>=data->width || y<0 || y>=data->height)
            return 0;
        z = data->depth[y*data->width+x];
        if (!isfinite(z))
            return 0;
        fprintf(outfile, "v %i %i %f\n", x, y, z);
    }
    FORALLfacets {
        vertexT *v1 = facet->vertices->e[0].p, *v2 = facet->vertices->e[1].p, *v3 = facet->vertices->e[2].p;
        fprintf(outfile, "f %i %i %i\n", qh_pointid(qh, v1->point)+1, qh_pointid(qh, v2->point)+1, qh_pointid(qh, v3->point)+1);
    }
    fclose(outfile);
    return 1;
}

int triangulation_triangulate(triangulation_data* data)
{
    coordT *points;
    qhT qh = {0};
    int result = 0;
    size_t num_points = 0;
    int curlong, totlong;

    points = convert_points(data, &num_points);
    
    result = qh_new_qhull(&qh, 2, num_points, points, True, "qhull d Qt", NULL, NULL) == 0;
    result = result && output_obj(&qh, data);

    qh_freeqhull(&qh, qh_ALL);
    qh_memfreeshort(&qh, &curlong, &totlong);
    return result;
}
