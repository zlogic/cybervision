#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
# include <pthread.h>
# define THREAD_FUNCTION void*
# define THREAD_RETURN_VALUE NULL
#elif defined(_WIN32)
# include "win32/pthread.h"
# define THREAD_FUNCTION DWORD WINAPI
# define THREAD_RETURN_VALUE 1
# include "win32/rand.h"
#else
# error "pthread is required"
#endif

#include <libqhull_r/libqhull_r.h>
#include <libqhull_r/poly_r.h>

#include "system.h"
#include "surface.h"
#include "image.h"
#include "configuration.h"

coordT* convert_points_qhull(surface_data data, int *num_points)
{
    coordT *points;
    coordT *current_point = NULL;
    int np = 0;
    for (int y=0;y<data.height;y++)
    {
        for (int x=0;x<data.width;x++)
        {
            float depth = data.depth[y*data.width + x];
            if (!isfinite(depth))
                continue;
            np++;
        }
    }
    points = malloc(sizeof(coordT)*np*2);
    *num_points = np;
    current_point = points;
    for (int y=0;y<data.height;y++)
    {
        for (int x=0;x<data.width;x++)
        {
            float depth = data.depth[y*data.width + x];
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
int output_points_qhull(qhT *qh, surface_data data, FILE* output_file, output_point_fn output_fn)
{
    coordT *point, *pointtemp;
    FORALLpoints
    {
        int x = (int)point[0], y = (int)point[1];
        float z;
        if(qh_pointid(qh, point) == qh->num_points-1)
            break;
        if (x<0 || x>=data.width || y<0 || y>=data.height)
            return 0;
        z = data.depth[y*data.width+x];
        if (!isfinite(z))
            return 0;
        
        output_fn(x, data.height-y, z, output_file);
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
void output_simplices_qhull(qhT *qh, FILE *output_file, output_simplex_fn output_fn)
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

void output_header_ply(int num_points, int num_simplices, FILE *output_file)
{
    char *endianess;
    unsigned int x = 1;
    if (*((char*)&x) == 1)
        endianess = "little";
    else
        endianess = "big";
    fprintf(output_file, "ply\nformat binary_%s_endian 1.0\n", endianess);
    fprintf(output_file, "comment Cybervision 3D surface\n");
    fprintf(output_file, "element vertex %i\n", num_points);
    fprintf(output_file, "property float x\nproperty float y\nproperty float z\n");
    fprintf(output_file, "element face %i\n", num_simplices);
    fprintf(output_file, "property list uchar int vertex_indices\n");
    fprintf(output_file, "end_header\n");
}

int triangulation_triangulate_delaunay(output_surface_task *task)
{
    coordT *points;
    qhT *qh = malloc(sizeof(qhT));
    int result = 0;
    int num_points = 0;
    int curlong, totlong;
    char* output_fileextension = file_extension(task->output_filename);

    points = convert_points_qhull(task->surf, &num_points);
    memset(qh, 0, sizeof(qhT));

    result = qh_new_qhull(qh, 2, num_points, points, True, "qhull d Qt Qbb Qc Qz Q12", NULL, NULL) == 0;
    if (strcasecmp(output_fileextension, "obj") == 0)
    {
        FILE *output_file = fopen(task->output_filename, "w");
        result = result && output_points_qhull(qh, task->surf, output_file, output_point_obj);
        if (result)
            output_simplices_qhull(qh, output_file, output_simplex_obj);
        fclose(output_file);
    }
    else if (strcasecmp(output_fileextension, "ply") == 0)
    {
        facetT *facet;
        int num_facets = 0;
        FILE *output_file = fopen(task->output_filename, "wb");
        FORALLfacets
        {
            if (!facet->upperdelaunay)
                num_facets++;
        }
        output_header_ply(num_points, num_facets, output_file);
        result = result && output_points_qhull(qh, task->surf, output_file, output_point_ply);
        if (result)
            output_simplices_qhull(qh, output_file, output_simplex_ply);
        fclose(output_file);
    }
    else
    {
        result = 0;
    }

    qh_freeqhull(qh, qh_ALL);
    qh_memfreeshort(qh, &curlong, &totlong);
    free(qh);
    return result;
}

static inline int has_forward_neighbor(surface_data* data, int x, int y)
{
    return (x<data->width-1 && y<data->height-1) && isfinite(data->depth[(y+1)*data->width+x]) && isfinite(data->depth[y*data->width+(x+1)]);
}

static inline int has_back_neighbor(surface_data* data, int x, int y)
{
    return (x>1 && y>1) && isfinite(data->depth[(y-1)*data->width+x]) && isfinite(data->depth[y*data->width+(x-1)]);
}

static inline void add_point_to_index(int *indices, int* point_count)
{
    if (*indices<0)
        *indices = (*point_count)++;
}

int triangulation_triangulate(surface_data data, char* output_filename)
{
    int *indices = malloc(sizeof(int)*data.width*data.height);
    int point_count = 0;
    int simplex_count = 0;
    int result = 0;
    FILE *output_file = NULL;
    char* output_fileextension = file_extension(output_filename);
    output_point_fn output_point;
    output_simplex_fn output_simplex;

    for(int i=0;i<data.width*data.height;i++)
        indices[i] = -1;
    
    for(int y=0;y<data.height;y++)
    {
        for(int x=0;x<data.width;x++)
        {
            if (!isfinite(data.depth[y*data.width+x]))
            {
                continue;
            }
            if (has_forward_neighbor(&data, x, y))
            {
                add_point_to_index(&indices[y*data.width+x], &point_count);
                add_point_to_index(&indices[y*data.width+(x+1)], &point_count);
                add_point_to_index(&indices[(y+1)*data.width+x], &point_count);
                simplex_count++;
            }
            if (has_back_neighbor(&data, x, y))
            {
                add_point_to_index(&indices[(y-1)*data.width+x], &point_count);
                add_point_to_index(&indices[y*data.width+(x-1)], &point_count);
                add_point_to_index(&indices[y*data.width+x], &point_count);
                simplex_count++;
            }
        }
    }

    if (strcasecmp(output_fileextension, "obj") == 0)
    {
        output_file = fopen(output_filename, "w");
        output_point = output_point_obj;
        output_simplex = output_simplex_obj;
    }
    else if (strcasecmp(output_fileextension, "ply") == 0)
    {
        output_file = fopen(output_filename, "wb");
        output_header_ply(point_count, simplex_count, output_file);
        output_point = output_point_ply;
        output_simplex = output_simplex_ply;
    }
    else
    {
        result = 0;
        goto cleanup;
    }

    typedef struct 
    {
        int x, y;
    } point;
    point *points = malloc(sizeof(point)*point_count);
    for(int y=0;y<data.height;y++)
    {
        for(int x=0;x<data.width;x++)
        {
            int pos = indices[y*data.width+x];
            if (pos<0)
            {
                continue;
            }
            points[pos].x = x;
            points[pos].y = y;
        }
    }
    for(int i=0;i<point_count;i++)
    {
        int x = points[i].x;
        int y = points[i].y;
        float z = data.depth[y*data.width+x];
        output_point(x, data.height-y, z, output_file);
    }
    free(points);

    for(int y=0;y<data.height;y++)
    {
        for(int x=0;x<data.width;x++)
        {
            if (indices[y*data.width+x]<0)
            {
                continue;
            }
            if (has_forward_neighbor(&data, x, y))
            {
                output_simplex(indices[y*data.width+x], indices[(y+1)*data.width+x], indices[y*data.width+(x+1)], output_file);
            }
            if (has_back_neighbor(&data, x, y))
            {
                output_simplex(indices[y*data.width+x], indices[(y-1)*data.width+x], indices[y*data.width+(x-1)], output_file);
            }
        }
    }

    fclose(output_file);
    result = 1;

cleanup:
    free(indices);
    return result;
}

void interpolate_points_delaunay(qhT *qh, surface_data data)
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
        int min_x=data.width, min_y=data.height, max_x=0, max_y=0;
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
                vz[i] = data.depth[data.width*vy[i]+vx[i]];
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
                if (x<0 || y<0 || x>=data.width || y>=data.height)
                    continue;
                float *depth = &data.depth[y*data.width+x];
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

typedef struct {
    int y;
    int threads_completed;
    float *output;
    pthread_mutex_t lock;
    pthread_t *threads;
} output_surface_task_ctx;

THREAD_FUNCTION interpolate_points_idw(void *args)
{
    output_surface_task *task = args;
    output_surface_task_ctx *ctx = task->internal;
    const int idw_search_radius = cybervision_interpolation_idw_radius;
    surface_data data = task->surf;
    
    while (!task->completed)
    {
        int y;
        if (pthread_mutex_lock(&ctx->lock) != 0)
            goto cleanup;
        task->percent_complete = 80.0F*(float)ctx->y/(float)data.width;
        y = ctx->y++;
        if (pthread_mutex_unlock(&ctx->lock) != 0)
            break;
        if (y>=data.height)
            break;
        for(int x=0;x<data.width;x++)
        {
            ctx->output[y*data.width+x] = NAN;
            float sum_values = 0.0F;
            float divider = 0.0F;
            for(int j=-idw_search_radius;j<=idw_search_radius;j++)
            {
                if (y+j<0 || y+j>=data.height)
                    continue;
                for(int i=-idw_search_radius;i<=idw_search_radius;i++)
                {
                    if (x+i<0 || x+i>=data.width)
                        continue;
                    float depth = data.depth[(y+j)*data.width+(x+i)];
                    if (!isfinite(depth))
                        continue;
                    float distance = powf(sqrtf(i*i+j*j), cybervision_interpolation_idw_power)+cybervision_interpolation_idw_offset;
                    sum_values += depth/distance;
                    divider += 1.0F/distance;
                }
            }
            if (divider != 0.0F)
                ctx->output[y*data.width+x] = sum_values/divider;
        }
    }

cleanup:
    pthread_mutex_lock(&ctx->lock);
    ctx->threads_completed++;
    pthread_mutex_unlock(&ctx->lock);
    if (ctx->threads_completed >= task->num_threads)
        task->completed = 1;
    return THREAD_RETURN_VALUE;
}

int triangulation_interpolate_delaunay(surface_data data)
{
    coordT *points;
    qhT qh = {0};
    int result = 0;
    int num_points = 0;
    int curlong, totlong;

    points = convert_points_qhull(data, &num_points);
    
    result = qh_new_qhull(&qh, 2, num_points, points, True, "qhull d Qt Qbb Qc Qz Q12", NULL, NULL) == 0;
    if (result)
        interpolate_points_delaunay(&qh, data);

    qh_freeqhull(&qh, qh_ALL);
    qh_memfreeshort(&qh, &curlong, &totlong);
    return result;
}

int surface_output_start(output_surface_task *task)
{
    char* output_fileextension = file_extension(task->output_filename);
    task->internal = NULL;
    task->completed = 1;

    if (task->mode == INTERPOLATION_DELAUNAY)
    {
        task->num_threads = 0;
        if (strcasecmp(output_fileextension, "obj") == 0 || strcasecmp(output_fileextension, "ply") == 0)
        {
            return triangulation_triangulate_delaunay(task);
        }
        else if (strcasecmp(output_fileextension, "png") == 0)
        {
            return triangulation_interpolate_delaunay(task->surf) && save_surface_image(task->surf, task->output_filename);
        }
        else
        {
            return 0;
        }
    }
    else if (task->mode == INTERPOLATION_IDW)
    {
        output_surface_task_ctx *ctx = malloc(sizeof(output_surface_task_ctx));
        task->internal = ctx;
        ctx->threads = malloc(sizeof(pthread_t)*task->num_threads);
        task->completed = 0;
        task->percent_complete = 0.0F;
        ctx->output = malloc(sizeof(float)*task->surf.width*task->surf.height);
        ctx->y = 0;
        ctx->threads_completed = 0;

        if (pthread_mutex_init(&ctx->lock, NULL) != 0)
            return 0;
        for (int i = 0; i < task->num_threads; i++)
            pthread_create(&ctx->threads[i], NULL, interpolate_points_idw, task);
        return 1;
    }
    else if (task->mode == INTERPOLATION_NONE)
    {
        task->num_threads = 0;
        return 1;
    }
    return 0;
}

int surface_output_complete(output_surface_task* task)
{
    output_surface_task_ctx *ctx;
    if (task == NULL)
        return 1;
    ctx = task->internal;
    task->internal = NULL;
    if (ctx != NULL)
    {
        for (int i = 0; i < task->num_threads; i++)
            pthread_join(ctx->threads[i], NULL);

        if (task->mode == INTERPOLATION_IDW)
        {
            for (int i=0;i<task->surf.width*task->surf.height;i++)
                if (isfinite(ctx->output[i]))
                    task->surf.depth[i] = ctx->output[i];
        }

        if (ctx->output != NULL)
            free(ctx->output);
        
        pthread_mutex_destroy(&ctx->lock);
        free(ctx->threads);
        free(ctx);
    }
    
    if (task->mode == INTERPOLATION_NONE || task->mode == INTERPOLATION_IDW)
    {
        char* output_fileextension = file_extension(task->output_filename);
        if (strcasecmp(output_fileextension, "obj") == 0 || strcasecmp(output_fileextension, "ply") == 0)
        {
            return triangulation_triangulate(task->surf, task->output_filename);
        }
        else if (strcasecmp(output_fileextension, "png") == 0)
        {
            return save_surface_image(task->surf, task->output_filename);
        }
        else
        {
            return 0;
        }
    }

    return 1;
}