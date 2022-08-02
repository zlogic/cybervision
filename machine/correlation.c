#include <stdlib.h>
#include <string.h>

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
# include <pthread.h>
# define THREAD_FUNCTION void*
# define THREAD_RETURN_VALUE NULL
#elif defined(_WIN32)
# include "win32/pthread.h"
# define THREAD_FUNCTION DWORD WINAPI
# define THREAD_RETURN_VALUE 1
#else
# error "pthread is required"
#endif

#define _USE_MATH_DEFINES
#include <math.h>

#include "correlation.h"

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

THREAD_FUNCTION correlate_points_task(void *args)
{
    match_task *t = args;
    match_task_ctx *ctx = t->internal;

    int kernel_size = t->kernel_size;
    int kernel_point_count = ctx->kernel_point_count;
    size_t points1_size = t->points1_size;
    size_t points2_size = t->points2_size;
    correlation_point *points1 = t->points1;
    correlation_point *points2 = t->points2;
    int w1 = t->img1.width, h1 = t->img1.height;
    int w2 = t->img2.width, h2 = t->img2.height;

    float *delta1 = malloc(sizeof(float)*kernel_point_count);
    float stdev1;
    while (!t->completed)
    {
        size_t p1;
        int x1, y1;
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
            
            if (corr >= t->threshold)
            {
                correlation_match m;
                m.point1 = (int)p1;
                m.point2 = (int)p2;
                m.corr = corr;
                if (pthread_mutex_lock(&ctx->lock) != 0)
                    goto cleanup;
                if (t->matches_count >= ctx->matches_limit)
                {
                    ctx->matches_limit += MATCH_RESULT_GROW_SIZE;
                    t->matches = realloc(t->matches, sizeof(correlation_match)*ctx->matches_limit);
                }
                t->matches[t->matches_count] = m;
                t->matches_count++;
                if (pthread_mutex_unlock(&ctx->lock) != 0)
                    goto cleanup;
            }
        }
    }

cleanup:
    free(delta1);
    pthread_mutex_lock(&ctx->lock);
    ctx->threads_completed++;
    pthread_mutex_unlock(&ctx->lock);
    if (ctx->threads_completed >= t->num_threads)
        t->completed = 1;
    return THREAD_RETURN_VALUE;
}

int correlation_match_points_start(match_task *task)
{
    int kernel_size = task->kernel_size;
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

inline int fit_range(int val, int min, int max)
{
    if (val<min)
        return min;
    if (val>max)
        return max;
    return val;
}

int estimate_search_range(cross_correlate_task *t, int x1, int y1, float *min_distance, float *max_distance)
{
    float min_depth, max_depth;
    int found = 0;
    float inv_scale = 1.0f/t->scale;
    int x_min = (int)floorf((x1-t->neighbor_distance)*inv_scale);
    int x_max = (int)ceilf((x1+t->neighbor_distance)*inv_scale);
    int y_min = (int)floorf((y1-t->neighbor_distance)*inv_scale);
    int y_max = (int)ceilf((y1+t->neighbor_distance)*inv_scale);

    x_min = fit_range(x_min, 0, t->out_width-1);
    x_max = fit_range(x_max, 0, t->out_width-1);
    y_min = fit_range(y_min, 0, t->out_height-1);
    y_max = fit_range(y_max, 0, t->out_height-1);
    for (int j=y_min;j<y_max;j++)
    {
        for (int i=x_min;i<x_max;i++)
        {
            int out_pos = j*t->out_width + i;
            float current_depth, distance;
            float dx, dy;
            float min, max;
            current_depth = t->out_points[out_pos];

            if (!isfinite(current_depth))
                continue;

            dx = (float)i-(float)x1*inv_scale;
            dy = (float)j-(float)y1*inv_scale;
            distance = sqrtf(dx*dx + dy*dy);
            min = current_depth - distance*t->max_slope;
            max = current_depth + distance*t->max_slope;

            if (!found)
            {
                min_depth = min;
                max_depth = max;
                found = 1;
            }
            else
            {
                min_depth = min<min_depth? min:min_depth;
                max_depth = max>max_depth? max:max_depth;
            }
        }
    }
    if (!found)
        return 0;
    *min_distance = min_depth;
    *max_distance = max_depth;
    return 1;
}

typedef struct {
    int corridor_vertical;
    float corridor_coeff;
    float kernel_point_count;
    int x1, y1;
    int w2, h2;

    float stdev1;
    float *delta1;
    float *delta2;

    int corridor_offset;
    float best_corr;
    float best_distance;
} corridor_area_ctx;

inline void correlate_corridor_area(cross_correlate_task *t, corridor_area_ctx* c, int corridor_start, int corridor_end)
{
    const int kernel_size = t->kernel_size;
    const int kernel_point_count = c->kernel_point_count;
    const int x1 = c->x1, y1 = c->y1;
    const int w2 = c->w2, h2 = c->h2;
    const float stdev1 = c->stdev1;
    for (int i=corridor_start;i<corridor_end;i++)
    {
        int x2, y2;
        float stdev2;
        float corr = 0;
        float dx, dy;
        float distance;
        if (c->corridor_vertical)
        {
            y2 = i;
            x2 = x1+c->corridor_offset + (int)((y2-y1)*c->corridor_coeff);
            if (x2 < kernel_size || x2 >= w2-kernel_size)
                continue;
        }
        else
        {
            x2 = i;
            y2 = y1+c->corridor_offset + (int)((x2-x1)*c->corridor_coeff);
            if (y2 < kernel_size || y2 >= h2-kernel_size)
                continue;
        }

        compute_correlation_data(&t->img2, kernel_size, x2, y2, &stdev2, c->delta2);

        if (!isfinite(stdev2))
            continue;

        dx = (float)(x2-x1)/t->scale;
        dy = (float)(y2-y1)/t->scale;
        distance = sqrtf(dx*dx+dy*dy);

        for (int l=0;l<kernel_point_count;l++)
            corr += c->delta1[l] * c->delta2[l];
        corr = corr/(c->stdev1*stdev2*(float)kernel_point_count);
        
        if (corr >= t->threshold && corr > c->best_corr)
        {
            c->best_distance = distance;
            c->best_corr = corr;
        }
    }
}

typedef struct {
    int y;
    int threads_completed;
    size_t processed_points;
    pthread_mutex_t lock;
    pthread_t *threads;
} cross_correlation_task_ctx;

THREAD_FUNCTION correlate_cross_correlation_task(void *args)
{
    cross_correlate_task *t = args;
    cross_correlation_task_ctx *ctx = t->internal;

    corridor_area_ctx corr_ctx;

    int kernel_size = t->kernel_size;
    int kernel_point_count = (2*kernel_size+1)*(2*kernel_size+1);
    int w1 = t->img1.width, h1 = t->img1.height;
    int w2 = t->img2.width, h2 = t->img2.height;
    int processed_points = 0;
    int points_to_process = (w1-2*kernel_size)*(h1-2*kernel_size);

    int corridor_vertical = fabsf(t->dir_y)>fabsf(t->dir_x);
    int corridor_start = kernel_size;
    int corridor_end = corridor_vertical? h2-kernel_size : w2-kernel_size;
    float inv_scale = 1.0f/t->scale;

    float dir_length = sqrtf(t->dir_x*t->dir_x + t->dir_y*t->dir_y);
    float distance_coeff = fabsf(corridor_vertical? t->dir_y/dir_length : t->dir_x/dir_length)*t->scale;
    float corridor_cot = fabsf(t->dir_x/dir_length);

    corr_ctx.corridor_vertical = corridor_vertical;
    corr_ctx.corridor_coeff = corridor_vertical? t->dir_x/t->dir_y : t->dir_y/t->dir_x;
    corr_ctx.kernel_point_count = kernel_point_count;
    corr_ctx.w2 = w2;
    corr_ctx.h2 = h2;

    corr_ctx.delta1 = malloc(sizeof(float)*kernel_point_count);
    corr_ctx.delta2 = malloc(sizeof(float)*kernel_point_count);
    
    for(;;){
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
            int out_pos = ((int)roundf(inv_scale*y1))*t->out_width + (int)roundf(inv_scale*x1);
            float min_distance, max_distance;

            corr_ctx.best_distance = NAN;
            corr_ctx.best_corr = 0;
            corr_ctx.x1 = x1;
            corr_ctx.y1 = y1;

            compute_correlation_data(&t->img1, kernel_size, x1, y1, &corr_ctx.stdev1, corr_ctx.delta1);

            if (!isfinite(corr_ctx.stdev1))
                continue;

            if (t->iteration > 1)
                if (!estimate_search_range(t, x1, y1, &min_distance, &max_distance))
                    continue;

            for (int corridor_offset=-t->corridor_size;corridor_offset<=t->corridor_size;corridor_offset++)
            {
                corr_ctx.corridor_offset = corridor_offset;
                if (t->iteration > 1)
                {
                    int current_pos, min_pos, max_pos;
                    int start, end;
                    if (corridor_vertical)
                        current_pos = y1;
                    else
                        current_pos = x1;
                    min_pos = (int)floorf(min_distance*distance_coeff);
                    max_pos = (int)ceilf(max_distance*distance_coeff);
                    start = fit_range(current_pos+min_pos, corridor_start, corridor_end);
                    end = fit_range(current_pos+max_pos, corridor_start, corridor_end);
                    correlate_corridor_area(t, &corr_ctx, start, end);
                    start = fit_range(current_pos-max_pos, corridor_start, corridor_end);
                    end = fit_range(current_pos-min_pos, corridor_start, corridor_end);
                    correlate_corridor_area(t, &corr_ctx, start, end);
                }
                else
                {
                    correlate_corridor_area(t, &corr_ctx, corridor_start, corridor_end);
                }
                if (isfinite(corr_ctx.best_distance))
                {
                    t->out_points[out_pos] = corr_ctx.best_distance;
                }
            }
            processed_points++;
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
    free(corr_ctx.delta2);
    pthread_mutex_lock(&ctx->lock);
    ctx->threads_completed++;
    pthread_mutex_unlock(&ctx->lock);
    if (ctx->threads_completed >= t->num_threads)
        t->completed = 1;
    return THREAD_RETURN_VALUE;
}

int correlation_cross_correlate_start(cross_correlate_task* task)
{
    cross_correlation_task_ctx *ctx = malloc(sizeof(cross_correlation_task_ctx));
    
    task->internal = ctx;
    ctx->threads= malloc(sizeof(pthread_t)*task->num_threads);

    ctx->processed_points = 0;
    task->percent_complete = 0.0;
    ctx->threads_completed = 0;
    task->completed = 0;
    task->error = NULL;

    ctx->y = task->kernel_size;

    if (pthread_mutex_init(&ctx->lock, NULL) != 0)
        return 0;

    for (int i = 0; i < task->num_threads; i++)
        pthread_create(&ctx->threads[i], NULL, correlate_cross_correlation_task, task);

    return 1;
}

void correlation_cross_correlate_cancel(cross_correlate_task *t)
{
    if (t == NULL)
        return;
    t->completed = 1;
}

int correlation_cross_correlate_complete(cross_correlate_task *t)
{
    cross_correlation_task_ctx *ctx;
    if (t == NULL || t->internal == NULL)
        return 1;
    ctx = t->internal;
    for (int i = 0; i < t->num_threads; i++)
        pthread_join(ctx->threads[i], NULL);

    pthread_mutex_destroy(&ctx->lock);
    free(ctx->threads);
    free(t->internal);
    t->internal = NULL;
    return 1;
}
