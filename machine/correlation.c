#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#if defined(_POSIX_THREADS) || defined(__APPLE__)
# include <pthread.h>
#else
# error "pthread is required, Windows builds are not supported yet"
#endif

#include <math.h>

#include "correlation.h"

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
    *sigma = sqrt(*sigma/(float)kernel_point_count);
}

typedef struct {
    correlation_image *img1, *img2;
    size_t p1;
    correlation_point *points1, *points2;
    size_t points1_size, points2_size;
    float *delta2;
    float *sigma2;
    int kernel_size, kernel_point_count;
    float threshold;
    correlation_matched cb;
    void *cb_args;
    pthread_mutex_t lock;
} correlate_points_args; 

void *correlate_points_task(void *args)
{
    correlate_points_args *c_args = args;

    int kernel_size = c_args->kernel_size;
    int kernel_point_count = c_args->kernel_point_count;
    size_t points1_size= c_args->points1_size;
    size_t points2_size= c_args->points2_size;
    correlation_point *points1 = c_args->points1;
    correlation_point *points2 = c_args->points2;
    int w1 = c_args->img1->width, h1 = c_args->img1->height;
    int w2 = c_args->img2->width, h2 = c_args->img2->height;

    float *delta1 = malloc(sizeof(float)*kernel_point_count);
    float sigma1;
    for(;;){
        size_t p1;
        int x1, y1;
        if (pthread_mutex_lock(&c_args->lock) != 0)
            return NULL;
        p1 = c_args->p1++;
        if (pthread_mutex_unlock(&c_args->lock) != 0)
            return NULL;

        if (p1 >= points1_size)
            break;

        x1 = points1[p1].x, y1 = points1[p1].y;
        if (x1-kernel_size<0 || x1+kernel_size>=w1 || y1-kernel_size<0 || y1+kernel_size>=h1)
            continue;
        
        compute_correlation_data(c_args->img1, kernel_size, x1, y1, &sigma1, delta1);
        for (size_t p2=0;p2<points2_size;p2++)
        {
            int x2 = points2[p2].x, y2 = points2[p2].y;
            float sigma2 = c_args->sigma2[p2];
            float *delta2 = &c_args->delta2[p2*kernel_point_count];
            float corr = 0;

            if (x2-kernel_size<0 || x2+kernel_size>=w2 || y2-kernel_size<0 || y2+kernel_size>=h2)
                continue;
            for (int i=0;i<kernel_point_count;i++)
                corr += delta1[i] * delta2[i];
            corr = corr/(sigma1*sigma2*(float)kernel_point_count);
            
            if (corr >= c_args->threshold)
            {
                if (pthread_mutex_lock(&c_args->lock) != 0)
                    return NULL;
                c_args->cb(p1, p2, corr, c_args->cb_args);
                if (pthread_mutex_unlock(&c_args->lock) != 0)
                    return NULL;
            }
        }
    }
    free(delta1);
    return NULL;
}

int correlation_correlate_points(correlation_image *img1, correlation_image *img2,
    correlation_point *points1, correlation_point *points2, size_t points1_size, size_t points2_size, 
    int kernel_size, float threshold, int num_threads,
    correlation_matched cb, void *cb_args)
{
    pthread_t *threads;
    correlate_points_args args;

    int kernel_point_count = (2*kernel_size+1)*(2*kernel_size+1);
    
    threads = malloc(sizeof(pthread_t)*num_threads);

    args.img1 = img1;
    args.img2 = img2;
    args.p1 = 0;
    args.points1 = points1;
    args.points2 = points2;
    args.points1_size = points1_size;
    args.points2_size = points2_size;
    args.kernel_size = kernel_size;
    args.kernel_point_count = kernel_point_count;
    args.threshold = threshold;
    args.cb = cb;
    args.cb_args = cb_args;

    args.delta2 = malloc(sizeof(float)*kernel_point_count*points2_size);
    args.sigma2 = malloc(sizeof(float)*points2_size);
    for(size_t p=0;p<points2_size;p++){
        int x = points2[p].x, y = points2[p].y;
        float *delta2 = &args.delta2[p*kernel_point_count];
        float *sigma2 = &args.sigma2[p];
        compute_correlation_data(img2, kernel_size, x, y, sigma2, delta2);
    }

    if (pthread_mutex_init(&args.lock, NULL) != 0)
        return 0;

    for (int i = 0; i < num_threads; i++)
        pthread_create(&threads[i], NULL, correlate_points_task, &args);

    for (int i = 0; i < num_threads; i++)
        pthread_join(threads[i], NULL);

    pthread_mutex_destroy(&args.lock);
    free(threads);
    free(args.delta2);
    free(args.sigma2);
    return 1;
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

#define STRIPE_WIDTH 64

void *correlate_images_task(void *args)
{
    correlate_image_args *c_args = args;

    int kernel_size = c_args->kernel_size;
    int kernel_point_count = c_args->kernel_point_count;
    int corridor_size = c_args->corridor_size;
    int corridor_stripes = 2*corridor_size + 1;
    int w1 = c_args->img1->width, h1 = c_args->img1->height;
    int h2 = c_args->img2->height;

    float *delta1 = malloc(sizeof(float)*kernel_point_count);
    float *sigma2_values = malloc(sizeof(float)*h2*corridor_stripes);
    float *delta2_values = malloc(sizeof(float)*kernel_point_count*h2*corridor_stripes);

    for(;;){
        int x_stripe;
        int x_max;
        if (pthread_mutex_lock(&c_args->lock) != 0)
            return NULL;
        x_stripe = c_args->x_stripe;
        c_args->x_stripe += STRIPE_WIDTH;
        x_max = (x_stripe+STRIPE_WIDTH) < c_args->x_max ? x_stripe+STRIPE_WIDTH : c_args->x_max;
        if (pthread_mutex_unlock(&c_args->lock) != 0)
            return NULL;

        if (x_stripe >= x_max)
            break;

        for (int i=x_stripe;i<x_max;i++)
        {
            for (int c=-corridor_size;c<=corridor_size;c++)
            {
                int c_offset = (c+corridor_size)*h2;
                for (int y2=0;y2<h2;y2++)
                {
                    int x2 = i + c + c_args->x_front[y2];
                    float *sigma2 = &sigma2_values[c_offset+y2];
                    float *delta2 = &delta2_values[(c_offset+y2)*kernel_point_count];
                    compute_correlation_data(c_args->img2, kernel_size, x2, y2, sigma2, delta2);
                }
            }
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
                    int c_offset = (c+corridor_size)*h2;
                    for (int y2=0;y2<h2;y2++)
                    {
                        int x2 = i + c + c_args->x_front[y2];
                        float sigma2 = sigma2_values[c_offset+y2];
                        float *delta2 = &delta2_values[(c_offset+y2)*kernel_point_count];
                        float corr = 0;

                        if (!isfinite(sigma2))
                            continue;

                        for (int l=0;l<kernel_point_count;l++)
                            corr += delta1[l] * delta2[l];
                        corr = corr/(sigma1*sigma2*(float)kernel_point_count);
                        
                        if (corr >= c_args->threshold && corr > best_corr)
                        {
                            float dx = x2-x1;
                            float dy = y2-y1;
                            // TODO: use distance from tilt center
                            float sgn = y2>y1 ? -1 : 1;
                            best_distance = sgn*sqrt(dx*dx+dy*dy);
                            best_corr = corr;
                        }
                    }
                    if(isfinite(best_distance))
                    {
                        c_args->out_points[y1*w1 + x1] = best_distance;
                        //printf("x=%i y=%i distance=%f corr=%f\n", x1, y1, best_distance, best_corr);
                    }
                }
            }
            //printf("i=%i\n", i);
        }
    }

    free(delta1);
    free(delta2_values);
    free(sigma2_values);

    return NULL;
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

    for (int y=0;y<img1->height;y++)
        for (int x=0;x<img1->width;x++)
            out_points[y*img1->width + x] = NAN;

    {
        float a = 0;
        int l_offset = 0, r_offset = 0;
        if (fabs(angle-M_PI/2)>1e-5)
        {
            a = 1.0/tan(angle);
            if (a>0)
            {
                l_offset = -(float)img1->height*a;
                r_offset = (float)img1->height*a;
            }
            else
            {
                r_offset = -(float)img1->height*a;
            }
        }
        for (int y=0;y<max_height;y++)
            args.x_front[y] = a*y+l_offset;
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

    free(threads);
    return 1;
}
