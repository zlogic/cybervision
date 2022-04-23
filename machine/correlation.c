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
#define ANGLE_EPSILON 1e-5

void compute_correlation_data(correlation_image *image, int kernel_size, int x, int y, float *sigma, float *delta)
{
    int kernel_width = 2*kernel_size + 1;
    int kernel_point_count = kernel_width*kernel_width;
    int width = image->width, height = image->height;

    float avg = 0;

    if (x-kernel_size<0 || x+kernel_size>=width || y-kernel_size<0 || y+kernel_size>=height)
    {
        for (int i=0;i<kernel_point_count;i++)
            delta[i] = 0;
        *sigma = INFINITY;
        return;
    }

    *sigma = 0;
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
        *sigma += delta[i] * delta[i];
    }
    *sigma = sqrtf(*sigma/(float)kernel_point_count);
}

typedef struct {
    int threads_completed;
    size_t matches_limit;
    size_t p1;
    float *delta2;
    float *sigma2;
    pthread_mutex_t lock;
    pthread_t *threads;
} match_task_ctx;

THREAD_FUNCTION correlate_points_task(void *args)
{
    match_task *t = args;
    match_task_ctx *ctx = t->internal;

    int kernel_size = t->kernel_size;
    int kernel_point_count = (2*kernel_size+1)*(2*kernel_size+1);
    size_t points1_size = t->points1_size;
    size_t points2_size = t->points2_size;
    correlation_point *points1 = t->points1;
    correlation_point *points2 = t->points2;
    int w1 = t->img1.width, h1 = t->img1.height;
    int w2 = t->img2.width, h2 = t->img2.height;

    float *delta1 = malloc(sizeof(float)*kernel_point_count);
    float sigma1;
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

        t->percent_complete = 100.0*(float)p1/(float)points1_size;

        x1 = points1[p1].x, y1 = points1[p1].y;
        if (x1-kernel_size<0 || x1+kernel_size>=w1 || y1-kernel_size<0 || y1+kernel_size>=h1)
            continue;
        
        compute_correlation_data(&t->img1, kernel_size, x1, y1, &sigma1, delta1);
        for (size_t p2=0;p2<points2_size;p2++)
        {
            int x2 = points2[p2].x, y2 = points2[p2].y;
            float sigma2 = ctx->sigma2[p2];
            float *delta2 = &ctx->delta2[p2*kernel_point_count];
            float corr = 0;

            if (x2-kernel_size<0 || x2+kernel_size>=w2 || y2-kernel_size<0 || y2+kernel_size>=h2)
                continue;
            for (int i=0;i<kernel_point_count;i++)
                corr += delta1[i] * delta2[i];
            corr = corr/(sigma1*sigma2*(float)kernel_point_count);
            
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
    ctx->threads= malloc(sizeof(pthread_t)*task->num_threads);

    ctx->p1 = 0;
    task->percent_complete = 0.0;
    ctx->threads_completed = 0;
    task->completed = 0;

    ctx->delta2 = malloc(sizeof(float)*kernel_point_count*task->points2_size);
    ctx->sigma2 = malloc(sizeof(float)*task->points2_size);
    ctx->matches_limit = MATCH_RESULT_GROW_SIZE;
    task->matches = malloc(sizeof(correlation_match)*ctx->matches_limit);
    task->matches_count = 0;
    for(size_t p=0;p<task->points2_size;p++){
        int x = task->points2[p].x, y = task->points2[p].y;
        float *delta2 = &ctx->delta2[p*kernel_point_count];
        float *sigma2 = &ctx->sigma2[p];
        compute_correlation_data(&task->img2, kernel_size, x, y, sigma2, delta2);
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
    free(ctx->sigma2);
    free(t->internal);
    t->internal = NULL;
    return 1;
}

typedef struct {
    correlation_image *img;
    int *x_front;
    float *delta;
    float *sigma;
    int kernel_size, kernel_point_count;
    int corridor_size;
    int last_pos;
} correlation_stripe_cache;

void correlate_corridor_add_stripe(correlation_stripe_cache *cache, int x)
{
    int corridor_width = (2*cache->corridor_size)+1;
    int next_pos = (cache->last_pos+1) % corridor_width;
    int c_offset = next_pos*cache->img->height;
    for (int y=0;y<cache->img->height;y++)
    {
        int x_pos = x + cache->x_front[y];
        float *sigma = &cache->sigma[c_offset+y];
        float *delta = &cache->delta[(c_offset+y)*cache->kernel_point_count];
        compute_correlation_data(cache->img, cache->kernel_size, x_pos, y, sigma, delta);
    }
    cache->last_pos = next_pos;
}

int cache_get_offset(correlation_stripe_cache *cache, int c)
{
    int corridor_width = (2*cache->corridor_size)+1;
    return (c + cache->corridor_size + cache->last_pos + corridor_width) % corridor_width;
}

typedef struct {
    correlation_image *img1, *img2;
    int corridor_size;
    int kernel_size, kernel_point_count;
    float threshold;
    int x_stripe, x_max;
    int *x_front;
    float *out_points;
    pthread_mutex_t lock;
} correlate_image_args;

THREAD_FUNCTION correlate_images_task(void *args)
{
    correlate_image_args *c_args = args;

    int kernel_size = c_args->kernel_size;
    int kernel_point_count = c_args->kernel_point_count;
    int corridor_size = c_args->corridor_size;
    int corridor_stripes = 2*corridor_size + 1;
    int w1 = c_args->img1->width, h1 = c_args->img1->height;
    int h2 = c_args->img2->height;

    float *delta1 = malloc(sizeof(float)*kernel_point_count);

    correlation_stripe_cache img2_cache;
    img2_cache.img = c_args->img2;
    img2_cache.x_front = c_args->x_front;
    img2_cache.delta = malloc(sizeof(float)*kernel_point_count*h2*corridor_stripes);
    img2_cache.sigma = malloc(sizeof(float)*h2*corridor_stripes);
    img2_cache.kernel_size = kernel_size;
    img2_cache.kernel_point_count = kernel_point_count;
    img2_cache.corridor_size = corridor_size;
    img2_cache.last_pos = -1;

    for(;;){
        int x_stripe;
        int x_max;
        if (pthread_mutex_lock(&c_args->lock) != 0)
            return THREAD_RETURN_VALUE;
        x_stripe = c_args->x_stripe;
        c_args->x_stripe += CORRELATION_STRIPE_WIDTH;
        x_max = (x_stripe+CORRELATION_STRIPE_WIDTH) < c_args->x_max ? x_stripe+CORRELATION_STRIPE_WIDTH : c_args->x_max;
        if (pthread_mutex_unlock(&c_args->lock) != 0)
            return THREAD_RETURN_VALUE;

        if (x_stripe >= x_max)
            break;
        
        img2_cache.last_pos = -1;
        for (int c=-corridor_size;c<corridor_size;c++)
        {
            int x = x_stripe + c;
            correlate_corridor_add_stripe(&img2_cache, x);
        }

        for (int i=x_stripe;i<x_max;i++)
        {
            correlate_corridor_add_stripe(&img2_cache, i+corridor_size);
            for (int y1=0;y1<h1;y1++)
            {
                float sigma1;
                int x1 = i + c_args->x_front[y1];
                float best_distance = NAN;
                float best_corr = 0;

                compute_correlation_data(c_args->img1, kernel_size, x1, y1, &sigma1, delta1);

                if (!isfinite(sigma1))
                    continue;

                for (int c=-corridor_size;c<=corridor_size;c++)
                {
                    int c_offset = cache_get_offset(&img2_cache, c)*h2;
                    for (int y2=0;y2<h2;y2++)
                    {
                        int x2 = i + c + c_args->x_front[y2];
                        float sigma2 = img2_cache.sigma[c_offset+y2];
                        float *delta2 = &img2_cache.delta[(c_offset+y2)*kernel_point_count];
                        float corr = 0;

                        if (!isfinite(sigma2))
                            continue;

                        for (int l=0;l<kernel_point_count;l++)
                            corr += delta1[l] * delta2[l];
                        corr = corr/(sigma1*sigma2*(float)kernel_point_count);
                        
                        if (corr >= c_args->threshold && corr > best_corr)
                        {
                            float dx = (float)(x2-x1);
                            float dy = (float)(y2-y1);
                            best_distance = -sqrtf(dx*dx+dy*dy);
                            best_corr = corr;
                        }
                    }
                    if(isfinite(best_distance))
                    {
                        c_args->out_points[y1*w1 + x1] = best_distance;
                    }
                }
            }
        }
    }

    free(delta1);
    free(img2_cache.delta);
    free(img2_cache.sigma);

    return THREAD_RETURN_VALUE;
}

int correlation_correlate_images(correlation_image *img1, correlation_image *img2,
    float angle, int corridor_size,
    int kernel_size, float threshold, int num_threads,
    float *out_points)
{
    correlate_image_args args;
    pthread_t *threads = malloc(sizeof(pthread_t)*num_threads);

    int kernel_point_count = (2*kernel_size+1)*(2*kernel_size+1);
    int max_height = img1->height>img2->height ? img1->height : img2->height;

    args.img1 = img1;
    args.img2 = img2;
    args.corridor_size = corridor_size;
    args.kernel_size = kernel_size;
    args.kernel_point_count = kernel_point_count;
    args.threshold = threshold;
    args.x_front = malloc(sizeof(int)*max_height);
    args.out_points = out_points;

    for (int i=0;i<img1->width*img1->height;i++)
        out_points[i] = NAN;

    {
        float a = 0;
        int l_offset = 0, r_offset = 0;
        if (fabs(angle-M_PI/2)>ANGLE_EPSILON)
        {
            a = 1.0f/tanf(angle);
            if (a>0)
            {
                l_offset = (int)(-(float)img1->height*a);
                r_offset = (int)((float)img1->height*a);
            }
            else
            {
                r_offset = (int)(-(float)img1->height*a);
            }
        }
        for (int y=0;y<max_height;y++)
            args.x_front[y] = (int)(a*y)+l_offset;
        args.x_stripe = l_offset;
        args.x_max = img1->width + r_offset;
    }

    if (pthread_mutex_init(&args.lock, NULL) != 0)
        return 0;

    for (int i = 0; i < num_threads; i++)
        pthread_create(&threads[i], NULL, correlate_images_task, &args);

    for (int i = 0; i < num_threads; i++)
        pthread_join(threads[i], NULL);

    pthread_mutex_destroy(&args.lock);

    free(args.x_front);
    free(threads);
    return 1;
}
