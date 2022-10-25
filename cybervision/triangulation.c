#include <stdlib.h>
#include <pthread.h>

#define _USE_MATH_DEFINES
#include <math.h>

#include "configuration.h"
#include "triangulation.h"
#include "linmath.h"

typedef struct {
    int y;

    double epipole_2[3];
    double camera_1[4*3];
    double camera_2[4*3];

    int threads_completed;
    pthread_mutex_t lock;
    pthread_t *threads;
} triangulation_task_ctx;

void find_min_max_depth(triangulation_task *t, float *min_depth, float *max_depth)
{
    float min = NAN, max = NAN;
    for(size_t i=0;i<t->width*t->height;i++)
    {
        float depth = t->out_depth[i];
        if (!isfinite(depth))
            continue;
        min = (depth<min || !isfinite(min))? depth:min;
        max = (depth>max || !isfinite(max))? depth:max;
    }
    *min_depth = min;
    *max_depth = max;
}

void filter_depth_histogram(triangulation_task *t, const float histogram_discard_percentile, float *min_depth, float *max_depth)
{
    const size_t histogram_bins = cybervision_histogram_filter_bins;
    const float histogram_depth_epsilon = cybervision_histogram_filter_epsilon;
    float min, max;
    size_t *histogram = malloc(sizeof(size_t)*histogram_bins);
    size_t histogram_sum = 0;
    size_t current_histogram_sum;
    find_min_max_depth(t, &min, &max);
    *min_depth = min;
    *max_depth = max;
    for(int i=0;i<histogram_bins;i++)
        histogram[i] = 0;
    for(size_t i=0;i<t->width*t->height;i++)
    {
        float depth = t->out_depth[i];
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
        *min_depth = min + ((float)i/(float)histogram_bins-histogram_depth_epsilon)*(max-min);
    }
    current_histogram_sum = 0;
    for(size_t i=histogram_bins-1;i>=0;i--)
    {
        current_histogram_sum += histogram[i];
        if (((float)current_histogram_sum/(float)histogram_sum)>histogram_discard_percentile)
            break;
        *max_depth = min + ((float)i/(float)histogram_bins+histogram_depth_epsilon)*(max-min);
    }
    for(size_t i=0;i<t->width*t->height;i++)
    {
        float depth = t->out_depth[i];
        if (!isfinite(depth))
            continue;
        if (depth<*min_depth || depth>*max_depth)
            t->out_depth[i] = NAN;
    }
}

void triangulation_parallel(triangulation_task *t)
{
    const float depth_scale = t->scale_z*((t->scale_x+t->scale_y)/2.0F);
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
            t->out_depth[y1*t->width+x1] = sqrtf(dx*dx+dy*dy)*depth_scale;
        }
    }
    float min_depth, max_depth;
    filter_depth_histogram(t, cybervision_histogram_filter_discard_percentile_parallel, &min_depth, &max_depth);
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

    for (size_t i=0;i<4*3;i++)
        ctx->camera_1[i] = 0;
    ctx->camera_1[0] = 1;
    ctx->camera_1[4*1+1] = 1;
    ctx->camera_1[4*2+2] = 1;
    // Using e' (epipole in second image) to calculate projection matrix for second image
    double *e2 = ctx->epipole_2;
    for(size_t i=0;i<3;i++)
        e2[i] = u[3*2+i];
    double e2_skewsymmetric[9] = {0.0, -e2[2], e2[1], e2[2], 0.0, -e2[0], -e2[1], e2[0], 0.0};
    double e2sf[9];
    multiply_matrix_3x3(e2_skewsymmetric, t->fundamental_matrix, e2sf);
    for (size_t i=0;i<3;i++)
        for (size_t j=0;j<3;j++)
            ctx->camera_2[4*i+j] = e2sf[3*i+j];
    for (size_t i=0;i<3;i++)
        ctx->camera_2[4*i+3] = e2[i];
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
    double *p1 = ctx->camera_1; 
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
            t->out_depth[pos] = NAN;
            if (x2<0 || y2<0)
                continue;
            // Linear triangulation method
            float x1n = x1;
            float y1n = y1;
            float x2n = x2;
            float y2n = y2;
            // First row of A: x1*camera_1[2]-camera_1[0]
            a[0+4*0]= x1n*p1[2*4+0]-p1[0+0];
            a[0+4*1]= x1n*p1[2*4+1]-p1[0+1];
            a[0+4*2]= x1n*p1[2*4+2]-p1[0+2];
            a[0+4*3]= x1n*p1[2*4+3]-p1[0+3];
            // Second row of A: y1*camera_1[2]-camera_1[0]
            a[1+4*0]= y1n*p1[2*4+0]-p1[4+0];
            a[1+4*1]= y1n*p1[2*4+1]-p1[4+1];
            a[1+4*2]= y1n*p1[2*4+2]-p1[4+2];
            a[1+4*3]= y1n*p1[2*4+3]-p1[4+3];
            // Third row of A: x2*camera_2[2]-camera_2[0]
            a[2+4*0]= x2n*p2[2*4+0]-p2[0+0];
            a[2+4*1]= x2n*p2[2*4+1]-p2[0+1];
            a[2+4*2]= x2n*p2[2*4+2]-p2[0+2];
            a[2+4*3]= x2n*p2[2*4+3]-p2[0+3];
            // Fourth row of A: y2*camera_2[2]-camera_2[1]
            a[3+4*0]= y2n*p2[2*4+0]-p2[4+0];
            a[3+4*1]= y2n*p2[2*4+1]-p2[4+1];
            a[3+4*2]= y2n*p2[2*4+2]-p2[4+2];
            a[3+4*3]= y2n*p2[2*4+3]-p2[4+3];

            if (!svdd(svd_ctx, a, 4, 4, u, s, vt))
                continue;
            
            double point[4];
            for(size_t i=0;i<4;i++)
                point[i] = vt[i*4+3];

            if (fabs(point[3])<cybervision_triangulation_min_scale)
                continue;

            t->out_depth[pos] = point[2]/point[3];
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
        filter_depth_histogram(t, cybervision_histogram_filter_discard_percentile_perspective, &min_depth, &max_depth);
        float min_size = (float)(t->width<t->height? t->width:t->height);
        float depth_scale = t->scale_z*min_size/(max_depth-min_depth);
        for(size_t i=0;i<t->width*t->height;i++)
        {
            float depth = t->out_depth[i];
            if (!isfinite(depth))
                continue;
            depth = (depth-min_depth)*depth_scale;
            t->out_depth[i] = depth;
        }
    }

    free(t->internal);
    t->internal = NULL;
    return t->error == NULL;
}
