#include <stdlib.h>
#include <math.h>

#include <libqhull_r/libqhull_r.h>
#include <libqhull_r/poly_r.h>

#include "triangulation.h"
#include "configuration.h"

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

int output_points(qhT *qh, surface_data* data, FILE *output_file)
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
        
        fprintf(output_file, "v %i %i %f\n", x, data->height-y, z);
    }
    return 1;
}

void output_simplices(qhT *qh, surface_data* data, FILE *output_file)
{
    facetT *facet;
    vertexT *vertex, **vertexp;
    FORALLfacets
    {
        if (facet->upperdelaunay)
            continue;
        fprintf(output_file, "f");
        if ((facet->toporient ^ qh_ORIENTclock))
        {
            FOREACHvertexreverse12_(facet->vertices)
            {
                int point_index = qh_pointid(qh, vertex->point);
                fprintf(output_file, " %i", point_index+1);
            }
        }
        else
        {
            FOREACHvertex_(facet->vertices)
            {
                int point_index = qh_pointid(qh, vertex->point);
                fprintf(output_file, " %i", point_index+1);
            }
        }
        fprintf(output_file, "\n");
    }
    // TODO: also generate normals? (avg of all edges)
}

int triangulation_triangulate(surface_data* data, FILE *output_file)
{
    coordT *points;
    qhT qh = {0};
    int result = 0;
    int num_points = 0;
    int curlong, totlong;

    points = convert_points(data, &num_points);
    
    result = qh_new_qhull(&qh, 2, num_points, points, True, "qhull d Qt Qbb Qc Qz Q12", NULL, NULL) == 0;
    result = result && output_points(&qh, data, output_file);
    if (result)
        output_simplices(&qh, data, output_file);

    qh_freeqhull(&qh, qh_ALL);
    qh_memfreeshort(&qh, &curlong, &totlong);
    return result;
}

void interpolate_points(qhT *qh, surface_data* data)
{
    facetT *facet;
    vertexT *vertex, **vertexp;
    pointT p[2];
    int vx[3];
    int vy[3];
    float vz[3];
    const float epsilon = cybervision_interpolation_epsilon;
    FORALLfacets
    {
        if (facet->upperdelaunay)
            continue;
        int min_x=data->width, min_y=data->height, max_x=0, max_y=0;
        {
            int i = 0;
            FOREACHvertex_(facet->vertices)
            {
                if (i>2)
                {
                    i = 0;
                    break;
                }
                vx[i] = (int)vertex->point[0];
                vy[i] = (int)vertex->point[1];
                vz[i] = data->depth[data->width*vy[i]+vx[i]];
                if (!isfinite(vz[i]))
                    break;
                    
                min_x = min_x<vx[i]? min_x:vx[i];
                min_y = min_y<vy[i]? min_y:vy[i];
                max_x = max_x>vx[i]? max_x:vx[i];
                max_y = max_y>vy[i]? max_y:vy[i];
                i++;
            }
            if (i!=3)
            {
                continue;
            }
        }

        for (int y=min_y;y<=max_y;y++)
        {
            for (int x=min_x;x<=max_x;x++)
            {
                if (x<0 || y<0 || x>=data->width || y>=data->height)
                    continue;
                float *depth = &data->depth[y*data->width+x];
                if (isfinite(*depth))
                    continue;
                // Linear barycentric interpolation, see https://en.wikipedia.org/wiki/Barycentric_coordinate_system#Edge_approach
                float detT = (float)(vy[1]-vy[2])*(vx[0]-vx[2]) + (float)(vx[2]-vx[1])*(vy[0]-vy[2]);
                float lambda1 = ((float)(vy[1]-vy[2])*(x-vx[2]) + (float)(vx[2]-vx[1])*(y-vy[2])) / detT;
                float lambda2 = ((float)(vy[2]-vy[0])*(x-vx[2]) + (float)(vx[0]-vx[2])*(y-vy[2])) / detT;
                float lambda3 = 1.0F - lambda1 - lambda2;
                if (lambda1>(1.0F+epsilon) || lambda1<(0.0F-epsilon) || lambda2>(1.0F+epsilon) || lambda2<(0.0-epsilon) || lambda3>(1.0+epsilon) || lambda3<(0.0-epsilon))
                    continue;
                *depth = (float)(lambda1*vz[0] + lambda2*vz[1] + lambda3*vz[2]);
            }
        }
    }
}

int triangulation_interpolate(surface_data* data)
{
    coordT *points;
    qhT qh = {0};
    int result = 0;
    int num_points = 0;
    int curlong, totlong;

    points = convert_points(data, &num_points);
    
    result = qh_new_qhull(&qh, 2, num_points, points, True, "qhull d Qt Qbb Qc Qz Q12", NULL, NULL) == 0;
    if (result)
        interpolate_points(&qh, data);

    qh_freeqhull(&qh, qh_ALL);
    qh_memfreeshort(&qh, &curlong, &totlong);
    return result;
}
