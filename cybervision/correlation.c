#include <stdlib.h>
#include <pthread.h>

#define _USE_MATH_DEFINES
#include <math.h>

#include "correlation.h"
#include "configuration.h"
#include "linmath.h"
#include "system.h"

#define MATCH_RESULT_GROW_SIZE 1000

void compute_correlation_data(correlation_image *image, int kernel_size, int x, int y, float *stdev, float *delta)
{
    int kernel_width = 2*kernel_size + 1;
    int kernel_point_count = kernel_width*kernel_width;
    int width = image->width, height = image->height;

    float avg = 0;

    if (x-kernel_size<0 || x+kernel_size>=width || y-kernel_size<0 || y+kernel_size>=height)
    {
        for (int i=0;i<kernel_point_count;i++)
            delta[i] = 0;
        *stdev = INFINITY;
        return;
    }

    *stdev = 0;
    for(int j=-kernel_size;j<=kernel_size;j++)
    {
        for (int i=-kernel_size;i<=kernel_size;i++)
        {
            float value;
            value = (float)(unsigned char)image->img[(y+j)*width + (x+i)];
            avg += value;
            delta[(j+kernel_size)*kernel_width + (i+kernel_size)] = value;
        }
    }
    avg /= (float)kernel_point_count;
    for (int i=0;i<kernel_point_count;i++)
    {
        delta[i] -= avg;
        *stdev += delta[i] * delta[i];
    }
    *stdev = sqrtf(*stdev/(float)kernel_point_count);
}

void compute_compact_correlation_data(correlation_image *img, int kernel_size, float *stdev, float *avg)
{
    float kernel_point_count = (2*kernel_size+1)*(2*kernel_size+1);
    for(int y=kernel_size;y<img->height-kernel_size;y++)
    {
        for(int x=kernel_size;x<img->width-kernel_size;x++)
        {
            float point_avg = 0.0F;
            float point_stdev = 0.0F;

            for (int j=-kernel_size;j<=kernel_size;j++)
            {
                for (int i=-kernel_size;i<=kernel_size;i++)
                {
                    float value = img->img[(y+j)*img->width+(x+i)];
                    point_avg += value;
                }
            }
            point_avg /= kernel_point_count;

            for (int j=-kernel_size;j<=kernel_size;j++)
            {
                for (int i=-kernel_size;i<=kernel_size;i++)
                {
                    float value = img->img[(y+j)*img->width+(x+i)];
                    float delta = value-point_avg;
                    point_stdev += delta*delta;
                }
            }
            point_stdev = sqrtf(point_stdev/kernel_point_count);
            avg[y*img->width+x] = point_avg;
            stdev[y*img->width+x] = point_stdev;
        }
    }
}

typedef struct {
    int kernel_point_count;
    size_t matches_limit;
    size_t p1;
    float *delta2;
    float *stdev2;
    int threads_completed;
    pthread_mutex_t lock;
    pthread_t *threads;
} match_task_ctx;

void* correlate_points_task(void *args)
{
    match_task *t = args;
    match_task_ctx *ctx = t->internal;

    int kernel_size = cybervision_correlation_kernel_size;
    int kernel_point_count = ctx->kernel_point_count;
    size_t points1_size = t->points1_size;
    size_t points2_size = t->points2_size;
    correlation_point *points1 = t->points1;
    correlation_point *points2 = t->points2;
    int w1 = t->img1.width, h1 = t->img1.height;
    int w2 = t->img2.width, h2 = t->img2.height;

    float *delta1 = malloc(sizeof(float)*kernel_point_count);
    float stdev1;
    const size_t max_matches = cybervision_correlation_max_matches_per_point;
    correlation_match *p1_matches = malloc(sizeof(correlation_match)*max_matches);
    while (!t->completed)
    {
        size_t p1;
        int x1, y1;
        float min_corr = 0.0F;
        size_t min_match_pos = 0;
        if (pthread_mutex_lock(&ctx->lock) != 0)
            goto cleanup;
        p1 = ctx->p1++;
        if (pthread_mutex_unlock(&ctx->lock) != 0)
            break;

        if (p1 >= points1_size)
            goto cleanup;

        t->percent_complete = 100.0F*(float)p1/(float)points1_size;

        x1 = points1[p1].x, y1 = points1[p1].y;
        if (x1-kernel_size<0 || x1+kernel_size>=w1 || y1-kernel_size<0 || y1+kernel_size>=h1)
            continue;

        for (size_t i=0;i<max_matches;i++)
            p1_matches[i].corr = 0.0F;

        compute_correlation_data(&t->img1, kernel_size, x1, y1, &stdev1, delta1);
        for (size_t p2=0;p2<points2_size;p2++)
        {
            int x2 = points2[p2].x, y2 = points2[p2].y;
            float stdev2 = ctx->stdev2[p2];
            float *delta2 = &ctx->delta2[p2*kernel_point_count];
            float corr = 0;

            if (x2-kernel_size<0 || x2+kernel_size>=w2 || y2-kernel_size<0 || y2+kernel_size>=h2)
                continue;
            for (int i=0;i<kernel_point_count;i++)
                corr += delta1[i] * delta2[i];
            corr = corr/(stdev1*stdev2*(float)kernel_point_count);
            
            if (corr >= cybervision_correlation_threshold && corr > min_corr)
            {
                correlation_match *m;
                m = &p1_matches[min_match_pos];
                m->point1 = (int)p1;
                m->point2 = (int)p2;
                m->corr = corr;

                min_corr = p1_matches[0].corr;
                min_match_pos = 0;
                for (size_t i=1;i<max_matches;i++)
                {
                    if (p1_matches[i].corr<min_corr)
                    {
                        min_corr = p1_matches[i].corr;
                        min_match_pos = i;
                    }
                }
            }
        }

        if (pthread_mutex_lock(&ctx->lock) != 0)
            goto cleanup;
        for (size_t i=0;i<max_matches;i++)
        {
            correlation_match *m = &p1_matches[i];
            if (m->corr<cybervision_correlation_threshold)
                continue;
            if (t->matches_count >= ctx->matches_limit)
            {
                ctx->matches_limit += MATCH_RESULT_GROW_SIZE;
                t->matches = realloc(t->matches, sizeof(correlation_match)*ctx->matches_limit);
            }
            t->matches[t->matches_count] = *m;
            t->matches_count++;
        }
        if (pthread_mutex_unlock(&ctx->lock) != 0)
            goto cleanup;
    }

cleanup:
    free(delta1);
    free(p1_matches);
    pthread_mutex_lock(&ctx->lock);
    ctx->threads_completed++;
    pthread_mutex_unlock(&ctx->lock);
    if (ctx->threads_completed >= t->num_threads)
        t->completed = 1;
    return NULL;
}

int correlation_match_points_start(match_task *task)
{
    int kernel_size = cybervision_correlation_kernel_size;
    int kernel_point_count = (2*kernel_size+1)*(2*kernel_size+1);
    match_task_ctx *ctx = malloc(sizeof(match_task_ctx));
    
    task->internal = ctx;
    ctx->kernel_point_count = kernel_point_count;
    ctx->threads= malloc(sizeof(pthread_t)*task->num_threads);

    ctx->p1 = 0;
    task->percent_complete = 0.0;
    ctx->threads_completed = 0;
    task->completed = 0;

    ctx->delta2 = malloc(sizeof(float)*kernel_point_count*task->points2_size);
    ctx->stdev2 = malloc(sizeof(float)*task->points2_size);
    ctx->matches_limit = MATCH_RESULT_GROW_SIZE;
    task->matches = malloc(sizeof(correlation_match)*ctx->matches_limit);
    task->matches_count = 0;
    for(size_t p=0;p<task->points2_size;p++){
        int x = task->points2[p].x, y = task->points2[p].y;
        float *delta2 = &ctx->delta2[p*kernel_point_count];
        float *stdev2 = &ctx->stdev2[p];
        compute_correlation_data(&task->img2, kernel_size, x, y, stdev2, delta2);
    }

    if (pthread_mutex_init(&ctx->lock, NULL) != 0)
        return 0;

    for (int i = 0; i < task->num_threads; i++)
        pthread_create(&ctx->threads[i], NULL, correlate_points_task, task);

    return 1;
}

void correlation_match_points_cancel(match_task *t)
{
    if (t == NULL)
        return;
    t->completed = 1;
}

int correlation_match_points_complete(match_task *t)
{
    match_task_ctx *ctx;
    if (t == NULL || t->internal == NULL)
        return 1;
    ctx = t->internal;
    for (int i = 0; i < t->num_threads; i++)
        pthread_join(ctx->threads[i], NULL);

    pthread_mutex_destroy(&ctx->lock);
    free(ctx->threads);
    free(ctx->delta2);
    free(ctx->stdev2);
    free(t->internal);
    t->internal = NULL;
    return 1;
}

static inline int fit_range(int val, int min, int max)
{
    if (val<min)
        return min;
    if (val>max)
        return max;
    return val;
}

typedef struct {
    float coeff_x, coeff_y;
    float add_x, add_y;
    int corridor_offset_x, corridor_offset_y;

    int kernel_size;
    int kernel_point_count;
    int x1, y1;

    float stdev1;
    float *delta1;
    float *avg2;
    float *stdev2;
    float *stdev_range;

    int corridor_offset;
    float best_corr;
    int best_match_x, best_match_y;
    int match_count;
} corridor_area_ctx;
int estimate_search_range(cross_correlate_task *t, corridor_area_ctx *ctx, int x1, int y1, int *corridor_start, int *corridor_end)
{
    float mid_corridor = 0.0F;
    int neighbor_count = 0;
    float range_stdev = 0.0F;
    float inv_scale = 1.0F/t->scale;
    int x_min = (int)floorf((x1-cybervision_crosscorrelation_neighbor_distance)*inv_scale);
    int x_max = (int)ceilf((x1+cybervision_crosscorrelation_neighbor_distance)*inv_scale);
    int y_min = (int)floorf((y1-cybervision_crosscorrelation_neighbor_distance)*inv_scale);
    int y_max = (int)ceilf((y1+cybervision_crosscorrelation_neighbor_distance)*inv_scale);
    int corridor_vertical = fabsf(ctx->coeff_y) > fabsf(ctx->coeff_x);

    x_min = fit_range(x_min, 0, t->out_width);
    x_max = fit_range(x_max, 0, t->out_width);
    y_min = fit_range(y_min, 0, t->out_height);
    y_max = fit_range(y_max, 0, t->out_height);
    for (int j=y_min;j<y_max;j++)
    {
        for (int i=x_min;i<x_max;i++)
        {
            int out_pos = j*t->out_width + i;
            float x2 = t->scale*(float)t->correlated_points[out_pos*2];
            float y2 = t->scale*(float)t->correlated_points[out_pos*2+1];
            if (x2<0 || y2<0)
                continue;

            int corridor_pos = (int)(corridor_vertical? roundf((y2-ctx->add_y)/ctx->coeff_y):roundf((x2-ctx->add_x)/ctx->coeff_x));
            ctx->stdev_range[neighbor_count++] = corridor_pos;
            mid_corridor += corridor_pos;
        }
    }
    if (neighbor_count==0)
        return 0;

    mid_corridor /= (float)neighbor_count;
    for (int i=0;i<neighbor_count;i++)
    {
        float delta = ctx->stdev_range[i]-mid_corridor;
        range_stdev += delta*delta;
    }
    range_stdev = sqrtf(range_stdev/(float)neighbor_count);
    
    int corridor_center = (int)roundf(mid_corridor);
    int corridor_length = (int)roundf(range_stdev*cybervision_crosscorrelation_corridor_extend_range);
    *corridor_start = fit_range(corridor_center - corridor_length, *corridor_start, *corridor_end);
    *corridor_end = fit_range(corridor_center + corridor_length, *corridor_start, *corridor_end);
    return 1;
}

static inline void calculate_epipolar_line(cross_correlate_task *t, corridor_area_ctx* c)
{
    float scale = t->scale;
    float p1[3] = {(float)c->x1/scale, (float)c->y1/scale, 1.0F};
    float Fp1[3];
    multiply_f_vector(t->fundamental_matrix, p1, Fp1);
    if (fabs(Fp1[0])>fabs(Fp1[1])) 
    {
        c->coeff_x = (float)(-Fp1[1]/Fp1[0]);
        c->add_x = (float)(-scale*Fp1[2]/Fp1[0]);
        c->corridor_offset_x = 1;
        c->coeff_y = 1.0F;
        c->add_y = 0.0F;
        c->corridor_offset_y = 0;
    }
    else
    {
        c->coeff_x = 1.0F;
        c->add_x = 0.0F;
        c->corridor_offset_x = 0;
        c->coeff_y = (float)(-Fp1[0]/Fp1[1]);
        c->add_y = (float)(-scale*Fp1[2]/Fp1[1]);
        c->corridor_offset_y = 1;
    }
}

static inline void correlate_corridor_area(cross_correlate_task *t, corridor_area_ctx* c, int corridor_start, int corridor_end)
{
    int kernel_size = c->kernel_size;
    int kernel_width = 2*kernel_size + 1;
    int kernel_point_count = c->kernel_point_count;
    int x1 = c->x1, y1 = c->y1;
    int w1 = t->img1.width;
    int w2 = t->img2.width, h2 = t->img2.height;
    float inv_scale = 1.0F/t->scale;
    for (int i=corridor_start;i<corridor_end;i++)
    {
        int x2 = (int)(c->coeff_x*i + c->add_x) + c->corridor_offset*c->corridor_offset_x;
        int y2 = (int)(c->coeff_y*i + c->add_y) + c->corridor_offset*c->corridor_offset_y;
        float corr = 0;
        if (x2 < kernel_size || x2 >= w2-kernel_size || y2 < kernel_size || y2 >= h2-kernel_size)
            continue;

        float avg2 = c->avg2[y2*w2+x2];
        float stdev2 = c->stdev2[y2*w2+x2];
        for (int j=-kernel_size;j<=kernel_size;j++)
        {
            for (int k=-kernel_size;k<=kernel_size;k++)
            {
                float delta1 = c->delta1[(j+kernel_size)*kernel_width + (k+kernel_size)];
                float delta2 = t->img2.img[(y2+j)*w2+(x2+k)] - avg2;
                corr += delta1 * delta2;
            }
        }
        corr = corr/(c->stdev1*stdev2*(float)kernel_point_count);
        
        if (corr >= cybervision_crosscorrelation_threshold && corr > c->best_corr)
        {
            c->best_match_x = (int)roundf(inv_scale*x2);
            c->best_match_y = (int)roundf(inv_scale*y2);
            c->best_corr = corr;
            c->match_count++;
        }
    }
}

typedef struct {
    int y;
    int threads_completed;
    size_t processed_points;
    float *avg2;
    float *stdev2;
    int *correlated_points;
    pthread_mutex_t lock;
    pthread_t *threads;
} cross_correlation_task_ctx;

void* cross_correlation_task(void *args)
{
    cross_correlate_task *t = args;
    cross_correlation_task_ctx *ctx = t->internal;

    corridor_area_ctx corr_ctx;

    int kernel_size = cybervision_crosscorrelation_kernel_size;
    int corridor_size = cybervision_crosscorrelation_corridor_size;
    int kernel_point_count = (2*kernel_size+1)*(2*kernel_size+1);
    int w1 = t->img1.width, h1 = t->img1.height;
    int w2 = t->img2.width, h2 = t->img2.height;
    int processed_points = 0;
    int points_to_process = (w1-2*kernel_size)*(h1-2*kernel_size);

    float inv_scale = 1.0F/t->scale;
    int neighbor_size = 2*(int)(ceilf(cybervision_crosscorrelation_neighbor_distance*inv_scale))+1;
    corr_ctx.kernel_size = kernel_size;
    corr_ctx.kernel_point_count = kernel_point_count;

    corr_ctx.delta1 = malloc(sizeof(float)*kernel_point_count);
    corr_ctx.avg2 = ctx->avg2;
    corr_ctx.stdev2 = ctx->stdev2;
    corr_ctx.stdev_range = malloc(sizeof(float)*neighbor_size*neighbor_size);
    
    while (!t->completed)
    {
        int y1;
        if (pthread_mutex_lock(&ctx->lock) != 0)
            goto cleanup;
        y1 = ctx->y++;
        if (pthread_mutex_unlock(&ctx->lock) != 0)
            goto cleanup;

        if (y1>=h1-kernel_size)
            break;

        for (int x1=kernel_size;x1<w1-kernel_size;x1++)
        {
            int corridor_start = kernel_size;
            int corridor_end = 0;

            corr_ctx.best_match_x = -1;
            corr_ctx.best_match_y = -1;
            corr_ctx.best_corr = 0;
            corr_ctx.x1 = x1;
            corr_ctx.y1 = y1;
            corr_ctx.match_count = 0;

            compute_correlation_data(&t->img1, kernel_size, x1, y1, &corr_ctx.stdev1, corr_ctx.delta1);
            if (!isfinite(corr_ctx.stdev1))
                continue;

            calculate_epipolar_line(t, &corr_ctx);
            if (!isfinite(corr_ctx.coeff_x) || !isfinite(corr_ctx.coeff_y) || !isfinite(corr_ctx.add_x) || !isfinite(corr_ctx.add_y))
                continue;

            int corridor_vertical = fabsf(corr_ctx.coeff_y) > fabsf(corr_ctx.coeff_x);
            corridor_end = corridor_vertical? h2-kernel_size : w2-kernel_size;
            processed_points++;
            if (t->iteration > 0)
                if (!estimate_search_range(t, &corr_ctx, x1, y1, &corridor_start, &corridor_end))
                    continue;

            for (int corridor_offset=-corridor_size;corridor_offset<=corridor_size;corridor_offset++)
            {
                corr_ctx.corridor_offset = corridor_offset;
                correlate_corridor_area(t, &corr_ctx, corridor_start, corridor_end);
                if (corr_ctx.match_count>cybervision_crosscorrelation_match_limit)
                {
                    corr_ctx.best_match_x = -1;
                    corr_ctx.best_match_y = -1;
                    corr_ctx.best_corr = 0;
                    break;
                }
            }
            if (corr_ctx.best_match_x>=0 && corr_ctx.best_match_y>=0)
            {
                size_t out_pos = y1*t->img1.width + x1;
                ctx->correlated_points[out_pos*2] = corr_ctx.best_match_x;
                ctx->correlated_points[out_pos*2+1] = corr_ctx.best_match_y;
            }
        }
        if (pthread_mutex_lock(&ctx->lock) != 0)
            goto cleanup;
        ctx->processed_points += processed_points;
        if (points_to_process != 0)
            t->percent_complete = 100.0F*(float)ctx->processed_points/(points_to_process);
        if (pthread_mutex_unlock(&ctx->lock) != 0)
            goto cleanup;
        processed_points = 0;
    }

cleanup:
    free(corr_ctx.delta1);
    free(corr_ctx.stdev_range);
    pthread_mutex_lock(&ctx->lock);
    ctx->threads_completed++;
    pthread_mutex_unlock(&ctx->lock);
    if (ctx->threads_completed >= t->num_threads)
        t->completed = 1;
    return NULL;
}

int cpu_correlation_cross_correlate_start(cross_correlate_task* task)
{
    cross_correlation_task_ctx *ctx = malloc(sizeof(cross_correlation_task_ctx));
    
    task->internal = ctx;
    ctx->threads= malloc(sizeof(pthread_t)*task->num_threads);

    ctx->correlated_points = malloc(sizeof(int)*task->img1.width*task->img1.height*2);
    for(size_t i = 0; i<task->img1.width*task->img1.height*2; i++)
        ctx->correlated_points[i] = -1;
    ctx->processed_points = 0;
    task->percent_complete = 0.0;
    ctx->threads_completed = 0;
    task->completed = 0;
    task->error = NULL;

    ctx->y = cybervision_crosscorrelation_kernel_size;

    ctx->avg2 = malloc(sizeof(float)*task->img2.width*task->img2.height);
    ctx->stdev2 = malloc(sizeof(float)*task->img2.width*task->img2.height);
    compute_compact_correlation_data(&task->img2, cybervision_crosscorrelation_kernel_size, ctx->stdev2, ctx->avg2);

    if (pthread_mutex_init(&ctx->lock, NULL) != 0)
        return 0;

    for (int i = 0; i < task->num_threads; i++)
        pthread_create(&ctx->threads[i], NULL, cross_correlation_task, task);

    return 1;
}

void cpu_correlation_cross_correlate_cancel(cross_correlate_task *t)
{
    if (t == NULL)
        return;
    t->completed = 1;
}

int cpu_correlation_cross_correlate_complete(cross_correlate_task *t)
{
    cross_correlation_task_ctx *ctx;
    if (t == NULL || t->internal == NULL)
        return 1;
    ctx = t->internal;
    for (int i = 0; i < t->num_threads; i++)
        pthread_join(ctx->threads[i], NULL);

    float inv_scale = 1/t->scale;
    for (int y=0;y<t->img1.height;y++)
    {
        for (int x=0;x<t->img1.width;x++)
        {
            size_t match_pos = y*t->img1.width + x;
            int x2 = ctx->correlated_points[match_pos*2];
            int y2 = ctx->correlated_points[match_pos*2+1];
            if(x2<0 || y2<0)
                continue;
            int out_point_pos = ((int)roundf(inv_scale*y))*t->out_width + (int)roundf(inv_scale*x);
            t->correlated_points[out_point_pos*2] = x2;
            t->correlated_points[out_point_pos*2+1] = y2;
        }
    }

    pthread_mutex_destroy(&ctx->lock);
    free(ctx->threads);
    free(ctx->correlated_points);
    free(ctx->avg2);
    free(ctx->stdev2);
    free(t->internal);
    t->internal = NULL;
    return t->error == NULL;
}

#ifdef CYBERVISION_DISABLE_GPU

int gpu_correlation_cross_correlate_init(cross_correlate_task *t, size_t img1_pixels, size_t img2_pixels)
{
    t->error = "Compiled without GPU support";
    return 0;
}
int gpu_correlation_cross_correlate_start(cross_correlate_task *t)
{
    t->error = "Compiled without GPU support";
    return 0;
}
void gpu_correlation_cross_correlate_cancel(cross_correlate_task *t)
{
    t->error = "Compiled without GPU support";
}
int gpu_correlation_cross_correlate_complete(cross_correlate_task *t)
{
    t->error = "Compiled without GPU support";
    return 0;
}
int gpu_correlation_cross_correlate_cleanup(cross_correlate_task *t)
{
    t->error = "Compiled without GPU support";
    return 0;
}

#endif
