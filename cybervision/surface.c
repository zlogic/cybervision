#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#include <libqhull_r/libqhull_r.h>
#include <libqhull_r/poly_r.h>

#include "system.h"
#include "surface.h"
#include "image.h"
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

void output_point_obj(int x, int y, float z, FILE* output_file)
{
    fprintf(output_file, "v %i %i %f\n", x, y, z);
}

void output_point_ply(int x, int y, float z, FILE* output_file)
{
    float x_out = x, y_out = y;
    fwrite(&x_out, sizeof(x), 1, output_file);
    fwrite(&y_out, sizeof(y), 1, output_file);
    fwrite(&z, sizeof(z), 1, output_file);
}

typedef void (*output_point_fn)(int x, int y, float z, FILE* output_file);
int output_points(qhT *qh, surface_data* data, FILE* output_file, output_point_fn output_fn)
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
        
        output_fn(x, data->height-y, z, output_file);
    }
    return 1;
}

void output_simplex_obj(int p1, int p2, int p3, FILE* output_file)
{
    fprintf(output_file, "f %i %i %i\n", p1+1, p2+1, p3+1);
}

void output_simplex_ply(int p1, int p2, int p3, FILE* output_file)
{
    const unsigned char point_count = 3;
    int32_t p1_out = p1, p2_out = p2, p3_out = p3;
    fwrite(&point_count, sizeof(point_count), 1, output_file);
    fwrite(&p1_out, sizeof(p1_out), 1, output_file);
    fwrite(&p2_out, sizeof(p2_out), 1, output_file);
    fwrite(&p3_out, sizeof(p3_out), 1, output_file);
}

typedef void (*output_simplex_fn)(int p1, int p2, int p3, FILE* output_file);
void output_simplices(qhT *qh, surface_data* data, FILE *output_file, output_simplex_fn output_fn)
{
    facetT *facet;
    vertexT *vertex, **vertexp;
    int points[3];
    // TODO: throw error if point count is out of range
    FORALLfacets
    {
        unsigned char i = 0;
        if (facet->upperdelaunay)
            continue;
        if ((facet->toporient ^ qh_ORIENTclock))
        {
            FOREACHvertexreverse12_(facet->vertices)
            {
                int point_index = qh_pointid(qh, vertex->point);
                if (i>=3) break;
                points[i++] = point_index;
            }
        }
        else
        {
            FOREACHvertex_(facet->vertices)
            {
                int point_index = qh_pointid(qh, vertex->point);
                if (i>=3) break;
                points[i++] = point_index;
            }
        }
        if (i==3)
        {
            output_fn(points[0], points[1], points[2], output_file);
        }
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


int triangulation_triangulate(surface_data* data, char* output_filename)
{
    coordT *points;
    qhT qh = {0};
    int result = 0;
    int num_points = 0;
    int curlong, totlong;
    char* output_fileextension = file_extension(output_filename);

    points = convert_points(data, &num_points);
    
    result = qh_new_qhull(&qh, 2, num_points, points, True, "qhull d Qt Qbb Qc Qz Q12", NULL, NULL) == 0;
    if (strcasecmp(output_fileextension, "obj") == 0)
    {
        FILE *output_file = fopen(output_filename, "w");
        result = result && output_points(&qh, data, output_file, output_point_obj);
        if (result)
            output_simplices(&qh, data, output_file, output_simplex_obj);
        fclose(output_file);
    }
    else if (strcasecmp(output_fileextension, "ply") == 0)
    {
        FILE *output_file = fopen(output_filename, "wb");
        output_header_ply(&qh, data, output_file);
        result = result && output_points(&qh, data, output_file, output_point_ply);
        if (result)
            output_simplices(&qh, data, output_file, output_simplex_ply);
        fclose(output_file);
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

int triangulation_interpolate_delaunay(surface_data* data)
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

int surface_output(surface_data* surf, char* output_filename, interpolation_mode mode)
{
    char* output_fileextension = file_extension(output_filename);
    if (mode == INTERPOLATION_DELAUNAY)
    {
        if (strcasecmp(output_fileextension, "obj") == 0 || strcasecmp(output_fileextension, "ply") == 0)
        {
            return triangulation_triangulate(surf, output_filename);
        }
        else if (strcasecmp(output_fileextension, "png") == 0)
        {
            return triangulation_interpolate_delaunay(surf) && save_surface(surf, output_filename);
        }
        else
        {
            return 0;
        }
    }
    else if (mode == INTERPOLATION_NONE)
    {
        if (strcasecmp(output_fileextension, "png") == 0)
        {
            return save_surface(surf, output_filename);
        }
        else
        {
            return 0;
        }
    }
    return 0;
}
