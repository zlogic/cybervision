#include <stdlib.h>
#include <math.h>
#include <stdint.h>

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

int output_points_obj(qhT *qh, surface_data* data, FILE *output_file)
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

void output_simplices_obj(qhT *qh, surface_data* data, FILE *output_file)
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

void output_header_ply(qhT *qh, surface_data* data, FILE *output_file)
{
    facetT *facet;
    int num_facets = 0;
    char *endianess;
    unsigned int x = 1;
    if (*((char*)&x) == 1)
        endianess = "little";
    else
        endianess = "big";
    FORALLfacets
    {
        if (!facet->upperdelaunay)
            num_facets++;
    }
    fprintf(output_file, "ply\nformat binary_%s_endian 1.0\n", endianess);
    fprintf(output_file, "comment Cybervision 3D surface\n");
    fprintf(output_file, "element vertex %i\n", qh->num_points);
    fprintf(output_file, "property float x\nproperty float y\nproperty float z\n");
    fprintf(output_file, "element face %i\n", num_facets);
    fprintf(output_file, "property list uchar int vertex_indices\n");
    fprintf(output_file, "end_header\n");
}

int output_points_ply(qhT *qh, surface_data* data, FILE *output_file)
{
    coordT *point, *pointtemp;
    FORALLpoints
    {
        int x = (int)point[0], y = (int)point[1];
        float x_out = x, y_out = data->height-y, z;
        if(qh_pointid(qh, point) == qh->num_points-1)
            break;
        if (x<0 || x>=data->width || y<0 || y>=data->height)
            return 0;
        z = data->depth[y*data->width+x];
        if (!isfinite(z))
            return 0;
        fwrite(&x_out, sizeof(x_out), 1, output_file);
        fwrite(&y_out, sizeof(y_out), 1, output_file);
        fwrite(&z, sizeof(z), 1, output_file);
    }
    return 1;
}

void output_simplices_ply(qhT *qh, surface_data* data, FILE *output_file)
{
    facetT *facet;
    vertexT *vertex, **vertexp;
    const unsigned char point_count = 3;
    FORALLfacets
    {
        if (facet->upperdelaunay)
            continue;
        fwrite(&point_count, sizeof(point_count), 1, output_file);
        if ((facet->toporient ^ qh_ORIENTclock))
        {
            FOREACHvertexreverse12_(facet->vertices)
            {
                int32_t point_index = qh_pointid(qh, vertex->point);
                fwrite(&point_index, sizeof(point_index), 1, output_file);
            }
        }
        else
        {
            FOREACHvertex_(facet->vertices)
            {
                int32_t point_index = qh_pointid(qh, vertex->point);
                fwrite(&point_index, sizeof(point_index), 1, output_file);
            }
        }
    }
    // TODO: also generate normals? (avg of all edges)
}

int triangulation_triangulate(surface_data* data, FILE *output_file, output_surface_format format)
{
    coordT *points;
    qhT qh = {0};
    int result = 0;
    int num_points = 0;
    int curlong, totlong;

    points = convert_points(data, &num_points);
    
    result = qh_new_qhull(&qh, 2, num_points, points, True, "qhull d Qt Qbb Qc Qz Q12", NULL, NULL) == 0;
    if (format == OUTPUT_SURFACE_OBJ)
    {
        result = result && output_points_obj(&qh, data, output_file);
        if (result)
            output_simplices_obj(&qh, data, output_file);
    }
    else if (format == OUTPUT_SURFACE_PLY)
    {
        output_header_ply(&qh, data, output_file);
        result = result && output_points_ply(&qh, data, output_file);
        if (result)
            output_simplices_ply(&qh, data, output_file);
    }
    else
    {
        return 0;
    }

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
