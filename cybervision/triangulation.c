#include <stdlib.h>
#include <pthread.h>

#define _USE_MATH_DEFINES
#include <math.h>

#include "configuration.h"
#include "triangulation.h"
#include "linmath.h"

typedef struct {
    int y;

    double camera_2[4*3];
    float *out_coords;
    float *out_depth;

    int threads_completed;
    pthread_mutex_t lock;
    pthread_t *threads;
} triangulation_task_ctx;

void find_min_max_depth(triangulation_task *t, float* out_depth, float *min_depth, float *max_depth)
{
    float min = NAN, max = NAN;
    for(size_t i=0;i<t->width*t->height;i++)
    {
        float depth = out_depth[i];
        if (!isfinite(depth))
            continue;
        min = (depth<min || !isfinite(min))? depth:min;
        max = (depth>max || !isfinite(max))? depth:max;
    }
    *min_depth = min;
    *max_depth = max;
}

void filter_depth_histogram(triangulation_task *t, float* out_depth, const float histogram_discard_percentile, float *min_depth, float *max_depth)
{
    const size_t histogram_bins = cybervision_histogram_filter_bins;
    const float histogram_depth_epsilon = cybervision_histogram_filter_epsilon;
    float min, max;
    size_t *histogram = malloc(sizeof(size_t)*histogram_bins);
    size_t histogram_sum = 0;
    size_t current_histogram_sum;
    find_min_max_depth(t, out_depth, &min, &max);
    *min_depth = min;
    *max_depth = max;
    for(int i=0;i<histogram_bins;i++)
        histogram[i] = 0;
    for(size_t i=0;i<t->width*t->height;i++)
    {
        float depth = out_depth[i];
        if (!isfinite(depth))
            continue;
        int pos = (int)roundf((depth-min)*histogram_bins/(max-min));
        pos = pos>=0? pos:0;
        pos = pos<histogram_bins? pos:histogram_bins-1;
        histogram[pos]++;
        histogram_sum++;
    }
    current_histogram_sum = 0;
    for(size_t i=0;i<histogram_bins;i++)
    {
        current_histogram_sum += histogram[i];
        if (((float)current_histogram_sum/(float)histogram_sum)>histogram_discard_percentile)
            break;
        *min_depth = min + ((float)i/(float)histogram_bins)*(max-min) - histogram_depth_epsilon*(max-min);
    }
    current_histogram_sum = 0;
    for(size_t i=histogram_bins-1;i>=0;i--)
    {
        current_histogram_sum += histogram[i];
        if (((float)current_histogram_sum/(float)histogram_sum)>histogram_discard_percentile)
            break;
        *max_depth = min + ((float)i/(float)histogram_bins)*(max-min) + histogram_depth_epsilon*(max-min);
    }
    for(size_t i=0;i<t->width*t->height;i++)
    {
        float depth = out_depth[i];
        if (!isfinite(depth))
            continue;
        if (depth<*min_depth || depth>*max_depth)
            out_depth[i] = NAN;
    }
}

void triangulation_parallel(triangulation_task *t)
{
    // TODO: use tilt angle factorization instead of this simple 
    for (int y1=0;y1<t->height;y1++)
    {
        for (int x1=0;x1<t->width;x1++)
        {
            size_t pos = y1*t->width+x1;
            int x2 = t->correlated_points[pos*2];
            int y2 = t->correlated_points[pos*2+1];
            if (x2<0 || y2<0)
            {
                t->out_depth[y1*t->width+x1] = NAN;
                continue;
            }
            float dx = (float)x1-(float)x2, dy = (float)y1-(float)y2;
            t->out_depth[y1*t->width+x1] = sqrtf(dx*dx+dy*dy)*t->depth_scale;
        }
    }
    float min_depth, max_depth;
    filter_depth_histogram(t, t->out_depth, cybervision_histogram_filter_discard_percentile_parallel, &min_depth, &max_depth);
    t->completed = 1;
}

int triangulation_perspective_cameras(triangulation_task *t)
{
    triangulation_task_ctx *ctx = t->internal;
    svd_internal svd_ctx = init_svd();
    double f[9];
    int result = 0;
    for (size_t i=0;i<3;i++)
        for (size_t j=0;j<3;j++)
            f[j*3+i] = t->fundamental_matrix[i*3+j];

    double u[9], s[3], vt[9];
    if (!svdd(svd_ctx, f, 3, 3, u, s, vt))
        goto cleanup;

    // Using e' (epipole in second image) to calculate projection matrix for second image
    double e2[3];
    for(size_t i=0;i<3;i++)
        e2[i] = u[3*2+i];
    double e2_skewsymmetric[9] = {0.0, -e2[2], e2[1], e2[2], 0.0, -e2[0], -e2[1], e2[0], 0.0};
    double e2sf[9];
    multiply_matrix_3x3(e2_skewsymmetric, t->fundamental_matrix, e2sf);
    for (size_t i=0;i<3;i++)
        for (size_t j=0;j<3;j++)
            ctx->camera_2[4*i+j] = e2sf[3*i+j]/e2[2];
    for (size_t i=0;i<3;i++)
        ctx->camera_2[4*i+3] = e2[i]/e2[2];
    result = 1;
cleanup:
    free_svd(svd_ctx);
    return result;
}

void* triangulation_perspective_task(void *args)
{
    triangulation_task *t = args;
    triangulation_task_ctx *ctx = t->internal;
    double a[4*4];
    double u[4*4], s[4], vt[4*4];
    double *p2 = ctx->camera_2; 
    svd_internal svd_ctx = init_svd();

    while (!t->completed)
    {
        int y1;
        if (pthread_mutex_lock(&ctx->lock) != 0)
            goto cleanup;
        y1 = ctx->y++;
        if (pthread_mutex_unlock(&ctx->lock) != 0)
            goto cleanup;

        if (y1>=t->height)
            break;

        t->percent_complete = 100.0F*(float)ctx->y/(t->height);
   
        for (int x1=0;x1<t->width;x1++)
        {
            size_t pos = y1*t->width+x1;
            int x2 = t->correlated_points[pos*2];
            int y2 = t->correlated_points[pos*2+1];
            ctx->out_depth[y1*t->width+x1] = NAN;
            if (x2<0 || y2<0)
                continue;

            // Linear triangulation method
            // First row of A: x1*[0 0 1 0]-[1 0 0 0]
            a[0+4*0]= -1.0;
            a[0+4*1]= 0.0;
            a[0+4*2]= x1;
            a[0+4*3]= 0.0;
            // Second row of A: y1*[0 0 1 0]-[0 1 0 0]
            a[1+4*0]= 0.0;
            a[1+4*1]= -1.0;
            a[1+4*2]= y1;
            a[1+4*3]= 0.0;
            // Third row of A: x2*camera_2[2]-camera_2[0]
            a[2+4*0]= (double)x2*p2[2*4+0]-p2[0+0];
            a[2+4*1]= (double)x2*p2[2*4+1]-p2[0+1];
            a[2+4*2]= (double)x2*p2[2*4+2]-p2[0+2];
            a[2+4*3]= (double)x2*p2[2*4+3]-p2[0+3];
            // Fourch row of A: y2*camera_2[2]-camera_2[1]
            a[3+4*0]= (double)y2*p2[2*4+0]-p2[4+0];
            a[3+4*1]= (double)y2*p2[2*4+1]-p2[4+1];
            a[3+4*2]= (double)y2*p2[2*4+2]-p2[4+2];
            a[3+4*3]= (double)y2*p2[2*4+3]-p2[4+3];

            if (!svdd(svd_ctx, a, 4, 4, u, s, vt))
                continue;
            
            double point[4];
            for(size_t i=0;i<4;i++)
                point[i] = vt[i*4+3];

            if (fabs(point[3])<1E-3)
                continue;
            const double point_x = point[0]/point[3];
            const double point_y = point[1]/point[3];
            const double point_z = point[2]/point[3];
            const size_t out_pos = y1*t->width+x1;
            ctx->out_coords[out_pos*2] = point_x;
            ctx->out_coords[out_pos*2+1] = point_y;
            //const int target_x1 = x1;//(int)round(point_x/point_z);
            //const int target_y1 = y1;//(int)round(point_y/point_z);
            //if (target_x1<0 || target_x1>=t->width || target_y1<0 || target_y1>=t->height)
            //    continue;
            ctx->out_depth[out_pos] = point_z;
        }
    }
cleanup:
    free_svd(svd_ctx);
    pthread_mutex_lock(&ctx->lock);
    ctx->threads_completed++;
    pthread_mutex_unlock(&ctx->lock);
    if (ctx->threads_completed >= t->num_threads)
        t->completed = 1;
}

int triangulation_start(triangulation_task *task)
{
    task->percent_complete = 0.0F;
    task->completed = 0;
    task->error = NULL;
    if (task->proj_mode == PROJECTION_MODE_PARALLEL)
    {
        task->internal = NULL;
        triangulation_parallel(task);
        return 1;
    }
    if(task->proj_mode == PROJECTION_MODE_PERSPECTIVE)
    {
        triangulation_task_ctx *ctx = malloc(sizeof(triangulation_task_ctx));
        task->internal = ctx;
        ctx->y = 0;
        ctx->threads= malloc(sizeof(pthread_t)*task->num_threads);
        ctx->threads_completed = 0;
        ctx->out_coords = malloc(sizeof(float)*2*task->width*task->height);
        ctx->out_depth = malloc(sizeof(float)*task->width*task->height);

        for(size_t i=0;i<2*task->width*task->height;i++)
            ctx->out_coords[i] = NAN;
        for(size_t i=0;i<task->width*task->height;i++)
            ctx->out_depth[i] = NAN;

        if (!triangulation_perspective_cameras(task))
        {
            task->error = "Failed to compute projection matrices";
            return 0;
        }
            
        if (pthread_mutex_init(&ctx->lock, NULL) != 0)
            return 0;

        for (int i = 0; i < task->num_threads; i++)
            pthread_create(&ctx->threads[i], NULL, triangulation_perspective_task, task);
        return 1;
    }
    task->error = "Unsupported projection mode";
    task->completed = 1;
    return 0;
}

void triangulation_cancel(triangulation_task *t)
{
    if (t == NULL)
        return;
    t->completed = 1;
}

int triangulation_complete(triangulation_task *t)
{
    triangulation_task_ctx *ctx;
    if (t == NULL || t->internal == NULL)
        return 1;
    ctx = t->internal;
    for (int i = 0; i < t->num_threads; i++)
        pthread_join(ctx->threads[i], NULL);

    pthread_mutex_destroy(&ctx->lock);
    free(ctx->threads);

    if (t->proj_mode == PROJECTION_MODE_PERSPECTIVE)
    {
        float min_depth, max_depth;
        filter_depth_histogram(t, ctx->out_depth, cybervision_histogram_filter_discard_percentile_perspective, &min_depth, &max_depth);
        float min_size = (float)(t->width<t->height? t->width:t->height);
        float depth_scale = t->depth_scale*min_size/(max_depth-min_depth);
        float min_x = NAN, max_x = NAN, min_y = NAN, max_y = NAN;
        for(size_t i=0;i<t->width*t->height;i++)
        {
            float x = ctx->out_coords[i*2];
            float y = ctx->out_coords[i*2+1];
            if (!isfinite(x) || !isfinite(y))
                continue;
            min_x = (x<min_x || !isfinite(min_x))? x:min_x;
            max_x = (x>max_x || !isfinite(max_x))? x:max_x;
            min_y = (y<min_y || !isfinite(min_y))? y:min_y;
            max_y = (y>max_y || !isfinite(max_y))? y:max_y;
        }
        for(size_t i=0;i<t->width*t->height;i++)
            t->out_depth[i] = NAN;
        float x_scale = t->width/(max_x-min_x);
        float y_scale = t->height/(max_y-min_y);
        for(size_t i=0;i<t->width*t->height;i++)
        {
            float depth = ctx->out_depth[i];
            const int x = (int)roundf((ctx->out_coords[i*2]-min_x)*x_scale);
            const int y = (int)roundf((ctx->out_coords[i*2+1]-min_y)*y_scale);
            if (!isfinite(depth) || x<0 || x>=t->width || y<0 || y>=t->height)
                continue;
            depth = (depth-min_depth)*depth_scale;

            t->out_depth[y*t->width+x] = depth;
        }
        free(ctx->out_coords);
        free(ctx->out_depth);
    }

    free(t->internal);
    t->internal = NULL;
    return t->error == NULL;
}
