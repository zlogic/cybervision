#include <stdlib.h>
#include <string.h>

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
# include <pthread.h>
# define THREAD_FUNCTION void*
# define THREAD_RETURN_VALUE NULL
#elif defined(_WIN32)
# include "win32/pthread.h"
#define THREAD_FUNCTION DWORD WINAPI
# define THREAD_RETURN_VALUE 1
#else
# error "pthread is required"
#endif

#define _USE_MATH_DEFINES
#include <math.h>

#include "correlation.h"

#define MATCH_RESULT_GROW_SIZE 1000
#define CORRELATION_STRIPE_WIDTH 64
#define CORRELATION_STRIPE_VERTICAL 0
#define CORRELATION_STRIPE_HORIZONTAL 1

void compute_correlation_data(correlation_image *image, int kernel_size, int x, int y, float *stddev, float *delta)
{
    int kernel_width = 2*kernel_size + 1;
    int kernel_point_count = kernel_width*kernel_width;
    int width = image->width, height = image->height;

    float avg = 0;

    if (x-kernel_size<0 || x+kernel_size>=width || y-kernel_size<0 || y+kernel_size>=height)
    {
        for (int i=0;i<kernel_point_count;i++)
            delta[i] = 0;
        *stddev = INFINITY;
        return;
    }

    *stddev = 0;
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
        *stddev += delta[i] * delta[i];
    }
    *stddev = sqrtf(*stddev/(float)kernel_point_count);
}

typedef struct {
    int kernel_point_count;
    size_t matches_limit;
    size_t p1;
    float *delta2;
    float *stddev2;
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
    float stddev1;
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
        
        compute_correlation_data(&t->img1, kernel_size, x1, y1, &stddev1, delta1);
        for (size_t p2=0;p2<points2_size;p2++)
        {
            int x2 = points2[p2].x, y2 = points2[p2].y;
            float stddev2 = ctx->stddev2[p2];
            float *delta2 = &ctx->delta2[p2*kernel_point_count];
            float corr = 0;

            if (x2-kernel_size<0 || x2+kernel_size>=w2 || y2-kernel_size<0 || y2+kernel_size>=h2)
                continue;
            for (int i=0;i<kernel_point_count;i++)
                corr += delta1[i] * delta2[i];
            corr = corr/(stddev1*stddev2*(float)kernel_point_count);
            
            if (corr >= t->threshold)
            {
                correlation_match m;
                m.point1 = p1;
                m.point2 = p2;
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
    ctx->stddev2 = malloc(sizeof(float)*task->points2_size);
    ctx->matches_limit = MATCH_RESULT_GROW_SIZE;
    task->matches = malloc(sizeof(correlation_match)*ctx->matches_limit);
    task->matches_count = 0;
    for(size_t p=0;p<task->points2_size;p++){
        int x = task->points2[p].x, y = task->points2[p].y;
        float *delta2 = &ctx->delta2[p*kernel_point_count];
        float *stddev2 = &ctx->stddev2[p];
        compute_correlation_data(&task->img2, kernel_size, x, y, stddev2, delta2);
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
    free(ctx->stddev2);
    free(t->internal);
    t->internal = NULL;
    return 1;
}

typedef struct {
    correlation_image *img;
    int *stripe_front;
    char stripe_direction;
    float *delta;
    float *stddev;
    int kernel_size, kernel_point_count;
    int corridor_size;
    int last_pos;
} correlation_stripe_cache;

inline void corridor_add_stripe(correlation_stripe_cache *cache, int stripe)
{
    int corridor_width = (2*cache->corridor_size)+1;
    int next_pos = (cache->last_pos+1) % corridor_width;
    int stripe_length = cache->stripe_direction == CORRELATION_STRIPE_VERTICAL ? cache->img->height : cache->img->width;
    int c_offset = next_pos*stripe_length;
    for (int i=0;i<stripe_length;i++)
    {
        int stripe_pos = stripe + cache->stripe_front[i];
        float *stddev = &cache->stddev[c_offset+i];
        float *delta = &cache->delta[(c_offset+i)*cache->kernel_point_count];
        int x = cache->stripe_direction == CORRELATION_STRIPE_VERTICAL ? stripe_pos : i;
        int y = cache->stripe_direction == CORRELATION_STRIPE_VERTICAL ? i : stripe_pos;
        compute_correlation_data(cache->img, cache->kernel_size, x, y, stddev, delta);
    }
    cache->last_pos = next_pos;
}

inline int cache_get_offset(correlation_stripe_cache *cache, int c)
{
    int corridor_width = (2*cache->corridor_size)+1;
    return (c + cache->corridor_size + cache->last_pos + corridor_width) % corridor_width;
}

typedef struct {
    int kernel_point_count;
    int stripe, stripe_max;
    int *stripe_front;
    char stripe_direction;
    int threads_completed;
    size_t processed_points;
    pthread_mutex_t lock;
    pthread_t *threads;
} cross_correlation_task_ctx;

THREAD_FUNCTION correlate_cross_correlation_task(void *args)
{
    cross_correlate_task *t = args;
    cross_correlation_task_ctx *ctx = t->internal;

    int kernel_size = t->kernel_size;
    int kernel_point_count = ctx->kernel_point_count;
    int corridor_size = t->corridor_size;
    int corridor_stripes = 2*corridor_size + 1;
    int stripe1_length = ctx->stripe_direction == CORRELATION_STRIPE_VERTICAL ? t->img1.height : t->img1.width;
    int stripe2_length = ctx->stripe_direction == CORRELATION_STRIPE_VERTICAL ? t->img2.height : t->img2.width;
    int w1 = t->img1.width, h1 = t->img1.height;
    int processed_points = 0;
    int points_to_process = (w1-2*kernel_size)*(h1-2*kernel_size);

    float *delta1 = malloc(sizeof(float)*kernel_point_count);

    correlation_stripe_cache img2_cache;
    img2_cache.img = &t->img2;
    img2_cache.stripe_front = ctx->stripe_front;
    img2_cache.stripe_direction = ctx->stripe_direction;
    img2_cache.delta = malloc(sizeof(float)*kernel_point_count*stripe2_length*corridor_stripes);
    img2_cache.stddev = malloc(sizeof(float)*stripe2_length*corridor_stripes);
    img2_cache.kernel_size = kernel_size;
    img2_cache.kernel_point_count = kernel_point_count;
    img2_cache.corridor_size = corridor_size;
    img2_cache.last_pos = -1;

    for(;;){
        int stripe;
        int stripe_max;
        if (pthread_mutex_lock(&ctx->lock) != 0)
            goto cleanup;
        stripe = ctx->stripe;
        ctx->stripe += CORRELATION_STRIPE_WIDTH;
        stripe_max = (stripe+CORRELATION_STRIPE_WIDTH) < ctx->stripe_max ? stripe+CORRELATION_STRIPE_WIDTH : ctx->stripe_max;
        if (pthread_mutex_unlock(&ctx->lock) != 0)
            goto cleanup;

        if (stripe >= stripe_max)
            break;
        
        img2_cache.last_pos = -1;
        for (int c=-corridor_size;c<corridor_size;c++)
        {
            int i = stripe + c;
            corridor_add_stripe(&img2_cache, i);
        }

        for (int i=stripe;i<stripe_max;i++)
        {
            corridor_add_stripe(&img2_cache, i+corridor_size);
            for (int j1=0;j1<stripe1_length;j1++)
            {
                float stddev1;
                int x1 = ctx->stripe_direction == CORRELATION_STRIPE_VERTICAL ? i + ctx->stripe_front[j1] : j1;
                int y1 = ctx->stripe_direction == CORRELATION_STRIPE_VERTICAL ? j1 : i + ctx->stripe_front[j1];
                float best_distance = NAN;
                float best_corr = 0;

                compute_correlation_data(&t->img1, kernel_size, x1, y1, &stddev1, delta1);

                if (!isfinite(stddev1))
                    continue;

                for (int c=-corridor_size;c<=corridor_size;c++)
                {
                    int c_offset = cache_get_offset(&img2_cache, c)*stripe2_length;
                    for (int j2=0;j2<stripe2_length;j2++)
                    {
                        int x2 = ctx->stripe_direction == CORRELATION_STRIPE_VERTICAL ? i + c + ctx->stripe_front[j2] : j2;
                        int y2 = ctx->stripe_direction == CORRELATION_STRIPE_VERTICAL ? j2 : i + c + ctx->stripe_front[j2];
                        float stddev2 = img2_cache.stddev[c_offset+j2];
                        float *delta2 = &img2_cache.delta[(c_offset+j2)*kernel_point_count];
                        float corr = 0;

                        if (!isfinite(stddev2))
                            continue;

                        for (int l=0;l<kernel_point_count;l++)
                            corr += delta1[l] * delta2[l];
                        corr = corr/(stddev1*stddev2*(float)kernel_point_count);
                        
                        if (corr >= t->threshold && corr > best_corr)
                        {
                            float dx = (float)(x2-x1);
                            float dy = (float)(y2-y1);
                            best_distance = -sqrtf(dx*dx+dy*dy);
                            best_corr = corr;
                        }
                    }
                    if(isfinite(best_distance))
                    {
                        t->out_points[y1*w1 + x1] = best_distance;
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
    }

cleanup:
    free(delta1);
    free(img2_cache.delta);
    free(img2_cache.stddev);
    pthread_mutex_lock(&ctx->lock);
    ctx->threads_completed++;
    pthread_mutex_unlock(&ctx->lock);
    if (ctx->threads_completed >= t->num_threads)
        t->completed = 1;
    return THREAD_RETURN_VALUE;
}

int correlation_cross_correlate_start(cross_correlate_task* task)
{
    int w1 = task->img1.width, h1 = task->img1.height;
    int w2 = task->img2.width, h2 = task->img2.height;
    int kernel_point_count = (2*task->kernel_size+1)*(2*task->kernel_size+1);
    cross_correlation_task_ctx *ctx = malloc(sizeof(cross_correlation_task_ctx));
    
    task->internal = ctx;
    ctx->threads= malloc(sizeof(pthread_t)*task->num_threads);

    ctx->processed_points = 0;
    task->percent_complete = 0.0;
    ctx->threads_completed = 0;
    task->completed = 0;
    task->error = NULL;

    int max_height = h1>h2 ? h1 : h2;
    int max_width = w1>w2 ? w1 : w2;

    ctx->kernel_point_count = kernel_point_count;

    for (int i=0;i<w1*h1;i++)
        task->out_points[i] = NAN;

    if (fabs(task->dir_y)>=fabs(task->dir_x))
    {
        int offset = (int)((float)h1*fabs(task->dir_y));
        ctx->stripe_direction = CORRELATION_STRIPE_VERTICAL;
        ctx->stripe_front = malloc(sizeof(int)*max_height);
        for (int y=0;y<max_height;y++)
            ctx->stripe_front[y] = (int)(task->dir_x/task->dir_y*y);
        ctx->stripe = -offset;
        ctx->stripe_max = w1 + offset;
    }
    else
    {
        int offset = (int)((float)w1*fabs(task->dir_x));
        ctx->stripe_direction = CORRELATION_STRIPE_HORIZONTAL;
        ctx->stripe_front = malloc(sizeof(int)*max_width);
        for (int x=0;x<max_width;x++)
            ctx->stripe_front[x] = (int)(task->dir_y/task->dir_x*x);
        ctx->stripe = -offset;
        ctx->stripe_max = h1 + offset;
    }

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
    free(ctx->stripe_front);
    free(t->internal);
    t->internal = NULL;
    return 1;
}
