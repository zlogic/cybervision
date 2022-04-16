#include <stdlib.h>
#include <string.h>

#if defined(_POSIX_THREADS) || defined(__APPLE__)
# include <pthread.h>
#else
# error "pthread is required, Windows builds are not supported yet"
#endif

#include <math.h>

#include "correlation.h"

typedef struct {
    context *ctx;
    const char *img;
    int y;
    pthread_mutex_t lock;
} prepare_line_args; 

void *ctx_prepare_task(void *args)
{
    prepare_line_args *prepare_line_args = args;
    context *ctx = prepare_line_args->ctx;
    int kernel_point_count = ctx->kernel_point_count;
    int kernel_size = ctx->kernel_size;
    int kernel_width = 2*kernel_size + 1;
    int width = ctx->width, height = ctx->height;

    for(;;){
        int y;
        if (pthread_mutex_lock(&prepare_line_args->lock) != 0)
            return NULL;
        y = prepare_line_args->y++;
        if (pthread_mutex_unlock(&prepare_line_args->lock) != 0)
            return NULL;
        if (y >= height)
            break;
        
        for (int x=0;x<width;x++)
        {
            float avg = 0, sigma = 0;
            int pos = y*width + x;
            float *delta = &ctx->delta[pos*kernel_point_count];

            if (x-kernel_size<0 || x+kernel_size>=width || y-kernel_size<0 || y+kernel_size>=height)
            {
                for (int i=0;i<kernel_point_count;i++)
                    delta[i] = 0;
                ctx->sigma[y*width + x] = INFINITY;
                continue;
            }

            for(int j=-kernel_size;j<=kernel_size;j++)
            {
                for (int i=-kernel_size;i<=kernel_size;i++)
                {
                    float value;
                    value = (float)(unsigned char)prepare_line_args->img[(y+j)*width + (x+i)];
                    avg += value;
                    delta[(j+kernel_size)*kernel_width + (i+kernel_size)] = value;
                }
            }
            avg /= (float)kernel_point_count;
            for (int i=0;i<kernel_point_count;i++)
            {
                delta[i] -= avg;
                sigma += delta[i] * delta[i];
            }
            ctx->sigma[y*width + x] = sqrt(sigma/(float)kernel_point_count);
        }
    }
    return NULL;
}

int ctx_init(context *ctx, const char* img, int width, int height, int kernel_size, int num_threads)
{
    pthread_t *threads;
    prepare_line_args thread_args;

    ctx->width = width;
    ctx->height = height;
    ctx->kernel_size = kernel_size;
    ctx->kernel_point_count = (2*kernel_size+1)*(2*kernel_size+1);
    ctx->delta = malloc(sizeof(float)*ctx->kernel_point_count*width*height);
    ctx->sigma = malloc(sizeof(float)*width*height);

    threads = malloc(sizeof(pthread_t)*num_threads);
    thread_args.ctx = ctx;
    thread_args.y = 0;
    thread_args.img = img;
    if (pthread_mutex_init(&thread_args.lock, NULL) != 0)
        return 0;

    for (int i = 0; i < num_threads; i++)
        pthread_create(&threads[i], NULL, ctx_prepare_task, &thread_args);

    for (int i = 0; i < num_threads; i++)
        pthread_join(threads[i], NULL);

    pthread_mutex_destroy(&thread_args.lock);
    return 1;
}

void ctx_free(context *ctx)
{
    if (ctx == NULL)
        return;
    if (ctx->delta != NULL)
        free(ctx->delta);
    ctx->delta = NULL;
    if (ctx->sigma != NULL)
        free(ctx->sigma);
    ctx->sigma = NULL;
}

typedef struct {
    context *ctx1, *ctx2;
    size_t p1;
    correlation_point *points1, *points2;
    size_t points1_size, points2_size;
    float threshold;
    correlation_matched cb;
    void *cb_args;
    pthread_mutex_t lock;
} correlate_args; 

void *ctx_correlate_task(void *args)
{
    correlate_args *correlate_args = args;

    context *ctx1 = correlate_args->ctx1;
    context *ctx2 = correlate_args->ctx2;
    int kernel_size = ctx1->kernel_size;
    int kernel_point_count = ctx1->kernel_point_count;
    size_t points1_size= correlate_args->points1_size;
    size_t points2_size= correlate_args->points2_size;
    correlation_point *points1 = correlate_args->points1;
    correlation_point *points2 = correlate_args->points2;
    int w1 = ctx1->width, h1 = ctx1->height;
    int w2 = ctx2->width, h2 = ctx2->height;

    for(;;){
        size_t p1;
        int x1, y1;
        if (pthread_mutex_lock(&correlate_args->lock) != 0)
            return NULL;
        p1 = correlate_args->p1++;
        if (pthread_mutex_unlock(&correlate_args->lock) != 0)
            return NULL;

        if (p1 >= points1_size)
            break;

        x1 = points1[p1].x, y1 = points1[p1].y;
        if (x1-kernel_size<0 || x1+kernel_size>=w1 || y1-kernel_size<0 || y1+kernel_size>=h1)
            continue;
        for (size_t p2=0;p2<points2_size;p2++)
        {
            int x2 = points2[p2].x, y2 = points2[p2].y;
            float *delta1 =  &ctx1->delta[(y1*w1 + x1) * kernel_point_count];
            float *delta2 = &ctx2->delta[(y2*w2 + x2) * kernel_point_count];
            float sigma1 = ctx1->sigma[y1*w1 + x1];
            float sigma2 = ctx2->sigma[y2*w2 + x2];
            float corr = 0;

            if (x2-kernel_size<0 || x2+kernel_size>=w2 || y2-kernel_size<0 || y2+kernel_size>=h2)
                continue;
            for (int i=0;i<kernel_point_count;i++)
                corr += delta1[i] * delta2[i];
            corr = corr/(sigma1*sigma2*(float)kernel_point_count);
            
            if (corr >= correlate_args->threshold)
            {
                if (pthread_mutex_lock(&correlate_args->lock) != 0)
                    return NULL;
                correlate_args->cb(p1, p2, corr, correlate_args->cb_args);
                if (pthread_mutex_unlock(&correlate_args->lock) != 0)
                    return NULL;
            }
        }
    }
    return NULL;
}

int ctx_correlate(context *ctx1, context *ctx2,
    correlation_point *points1, correlation_point *points2, size_t points1_size, size_t points2_size, 
    float threshold, int num_threads,
    correlation_matched cb, void *cb_args)
{
    pthread_t *threads;
    correlate_args thread_args;

    threads = malloc(sizeof(pthread_t)*num_threads);

    thread_args.ctx1 = ctx1;
    thread_args.ctx2 = ctx2;
    thread_args.p1 = 0;
    thread_args.points1 = points1;
    thread_args.points2 = points2;
    thread_args.points1_size = points1_size;
    thread_args.points2_size = points2_size;
    thread_args.threshold = threshold;
    thread_args.cb = cb;
    thread_args.cb_args = cb_args;

    if (pthread_mutex_init(&thread_args.lock, NULL) != 0)
        return 0;

    for (int i = 0; i < num_threads; i++)
        pthread_create(&threads[i], NULL, ctx_correlate_task, &thread_args);

    for (int i = 0; i < num_threads; i++)
        pthread_join(threads[i], NULL);

    pthread_mutex_destroy(&thread_args.lock);
    return 1;
}
