#include <stdlib.h>
#include <pthread.h>

#define _USE_MATH_DEFINES
#include <math.h>

#include "configuration.h"
#include "triangulation.h"
#include "linmath.h"

typedef struct {
    int y;

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

int triangulation_triangulate_point(svd_internal svd_ctx, double projection2[4*3], double x1, double y1, double x2, double y2, double result[4])
{
    // Linear triangulation method
    double a[4*4];
    double u[4*4], s[4], vt[4*4];
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
    a[2+4*0]= x2*projection2[2*4+0]-projection2[0+0];
    a[2+4*1]= x2*projection2[2*4+1]-projection2[0+1];
    a[2+4*2]= x2*projection2[2*4+2]-projection2[0+2];
    a[2+4*3]= x2*projection2[2*4+3]-projection2[0+3];
    // Fourth row of A: y2*camera_2[2]-camera_2[1]
    a[3+4*0]= y2*projection2[2*4+0]-projection2[4+0];
    a[3+4*1]= y2*projection2[2*4+1]-projection2[4+1];
    a[3+4*2]= y2*projection2[2*4+2]-projection2[4+2];
    a[3+4*3]= y2*projection2[2*4+3]-projection2[4+3];

    if (!svdd(svd_ctx, a, 4, 4, u, s, vt))
        return 0;

    for(size_t i=0;i<4;i++)
        result[i] = vt[i*4+3];
    return 1;
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
    free(histogram);
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
    filter_depth_histogram(t, cybervision_histogram_filter_discard_percentile, &min_depth, &max_depth);
    t->completed = 1;
}

void* triangulation_perspective_task(void *args)
{
    triangulation_task *t = args;
    triangulation_task_ctx *ctx = t->internal;
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

            double point[4];
            if(!triangulation_triangulate_point(svd_ctx, t->projection_matrix_2, x1, y1, x2, y2, point))
                continue;

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
    return NULL;
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
        filter_depth_histogram(t, cybervision_histogram_filter_discard_percentile, &min_depth, &max_depth);
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
