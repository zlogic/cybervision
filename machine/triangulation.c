#include <stdlib.h>
#include <math.h>

#include "libqhull_r/libqhull_r.h"
#include "libqhull_r/poly_r.h"

#include "triangulation.h"

coordT* convert_points(surface_data* data, size_t *num_points)
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

int output_obj(qhT *qh, surface_data* data)
{
    // TODO: rewrite this to interact better with Python
    FILE *outfile = fopen("out.obj", "w");
    facetT *facet;
    coordT *point, *pointtemp;
    vertexT *vertex, **vertexp;
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
        fprintf(outfile, "v %i %i %f\n", x, data->height-y, z);
    }
    FORALLfacets
    {
        fprintf(outfile, "f");
        if (facet->upperdelaunay)
            continue;
        if ((facet->toporient ^ qh_ORIENTclock))
        {
            FOREACHvertex_(facet->vertices)
                fprintf(outfile, " %d", qh_pointid(qh, vertex->point)+1);
        }
        else
        {
            FOREACHvertexreverse12_(facet->vertices)
                fprintf(outfile, " %d", qh_pointid(qh, vertex->point)+1);
        }
        fprintf(outfile, "\n");
    }
    fclose(outfile);
    return 1;
}

int triangulation_triangulate(surface_data* data)
{
    coordT *points;
    qhT qh = {0};
    int result = 0;
    size_t num_points = 0;
    int curlong, totlong;

    points = convert_points(data, &num_points);
    
    result = qh_new_qhull(&qh, 2, num_points, points, True, "qhull d Qt Qbb Qc Qz Q12", NULL, NULL) == 0;
    result = result && output_obj(&qh, data);

    qh_freeqhull(&qh, qh_ALL);
    qh_memfreeshort(&qh, &curlong, &totlong);
    return result;
}
