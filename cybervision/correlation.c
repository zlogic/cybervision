#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <pthread.h>

#define _USE_MATH_DEFINES
#include <math.h>

#include "correlation.h"
#include "configuration.h"
#include "linmath.h"

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
            
            if (corr >= cybervision_correlation_threshold)
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

typedef struct {
    size_t iteration;

    int threads_completed;
    pthread_mutex_t lock;
    pthread_t *threads;
} ransac_task_ctx;

static inline float ransac_calculate_error(ransac_task *t, size_t selected_match, float* f)
{
    // Calculate Sampson distance
    ransac_match *match = &t->matches[selected_match];
    float p1[3] = {match->x1, match->y1, 1.0F};
    float p2[3] = {match->x2, match->y2, 1.0F};
    float nominator = 0.0F;
    float p2tFp1[3];
    multiplyf(p2, f, p2tFp1, 1, 3, 3, 0, 0);
    multiplyf(p2tFp1, p1, &nominator, 1, 1, 3, 0, 0);
    float Fp1[3];
    float Ftp2[3];
    multiplyf(f, p1, Fp1, 3, 1, 3, 0, 0);
    multiplyf(f, p2, Ftp2, 3, 1, 3, 1, 0);
    float denominator = Fp1[0]*Fp1[0]+Fp1[1]*Fp1[1]+Ftp2[0]*Ftp2[0]+Ftp2[1]*Fp1[1];
    return nominator*nominator/denominator;
}

static inline int ransac_calculate_model(svd_internal svd_ctx, ransac_task *t, size_t *selected_matches, size_t selected_matches_count, float* f)
{
    // TODO: preallocate memory
    // 8-point algorithm
    // Recenter & rescale points
    float centerX1 = 0.0F, centerY1 = 0.0F;
    float centerX2 = 0.0F, centerY2 = 0.0F;
    for(size_t i=0;i<selected_matches_count;i++)
    {
        size_t selected_match = selected_matches[i];
        ransac_match *match = &t->matches[selected_match];
        centerX1 += match->x1;
        centerY1 += match->y1;
        centerX2 += match->x2;
        centerY2 += match->y2;
    }
    centerX1 /= (float)selected_matches_count;
    centerY1 /= (float)selected_matches_count;
    centerX2 /= (float)selected_matches_count;
    centerY2 /= (float)selected_matches_count;
    float scale1 = 0.0F, scale2 = 0.0F;
    for(size_t i=0;i<selected_matches_count;i++)
    {
        size_t selected_match = selected_matches[i];
        ransac_match *match = &t->matches[selected_match];
        float dx1 = (float)match->x1-centerX1;
        float dy1 = (float)match->y1-centerY1;
        float dx2 = (float)match->x2-centerX2;
        float dy2 = (float)match->y2-centerY2;
        scale1 += sqrtf(dx1*dx1 + dy1*dy1);
        scale2 += sqrtf(dx2*dx2 + dy2*dy2);
    }
    scale1 = sqrtf(2)*selected_matches_count/scale1;
    scale2 = sqrtf(2)*selected_matches_count/scale2;
    // Calculate fundamental matrix using the 8-point algorithm
    float *a = malloc(sizeof(float)*selected_matches_count*9);
    for(size_t i=0;i<selected_matches_count;i++)
    {
        size_t selected_match = selected_matches[i];
        ransac_match *match = &t->matches[selected_match];
        float x1 = ((float)match->x1-centerX1)*scale1;
        float y1 = ((float)match->y1-centerY1)*scale1;
        float x2 = ((float)match->x2-centerX2)*scale2;
        float y2 = ((float)match->y2-centerY2)*scale2;
        a[i*9  ] = x1*x2;
        a[i*9+1] = x1*y2;
        a[i*9+2] = x1;
        a[i*9+3] = y1*x2;
        a[i*9+4] = y1*y2;
        a[i*9+5] = y1;
        a[i*9+6] = x2;
        a[i*9+7] = y2;
        a[i*9+8] = 1.0F;
    }
    float *s = malloc(sizeof(float)*selected_matches_count);
    float v[9*9];
    int result = svdf(svd_ctx, a, selected_matches_count, 9, NULL, s, v);
    if (!result)
        return result;

    // TODO: figure out how to correctly read matrix V
    float f_temp[9];
    for(size_t i=0;i<9;i++)
        f_temp[i] = v[9*8+i];
    //    f_temp[i] = v[9*i+8];

    float u[9];
    float s_new[3];
    result = svdf(svd_ctx, f_temp, 3, 3, u, s_new, v);
    if (!result)
        return result;

    float s_matrix[9] = {s_new[0], 0.0F, 0.0F, 0.0F, s_new[1], 0.0F, 0.0F, 0.0F};
    multiplyf(u, s_matrix, f_temp, 3, 3, 3, 0, 0);
    multiplyf(f_temp, v, f, 3, 3, 3, 0, 0);

    // Scale back to image coordinates
    float m1[9] = {scale1, 0.0F, -centerX1*scale1, 0.0F, scale1, -centerY1*scale1, 0.0F, 0.0F, 1.0F};
    float m2[9] = {scale2, 0.0F, -centerX2*scale2, 0.0F, scale2, -centerY2*scale2, 0.0F, 0.0F, 1.0F};
    multiplyf(m2, f, f_temp, 3, 3, 3, 1, 0);
    multiplyf(f_temp, m1, f, 3, 3, 3, 0, 0);
    free(a);
    free(s);
    return 1;
}

void* correlate_ransac_task(void *args)
{
    ransac_task *t = args;
    ransac_task_ctx *ctx = t->internal;
    svd_internal svd_ctx = init_svd();
    size_t ransac_n = cybervision_ransac_n;
    size_t *inliers = malloc(sizeof(size_t)*ransac_n);
    size_t *extended_inliers = malloc(sizeof(size_t)*t->matches_count);
    float fundamental_matrix[9];
    size_t extended_inliers_count = 0;
    unsigned int rand_seed;
    
    if (pthread_mutex_lock(&ctx->lock) != 0)
        goto cleanup;
    rand_seed = rand();
    if (pthread_mutex_unlock(&ctx->lock) != 0)
        goto cleanup;

    while (!t->completed)
    {
        size_t iteration;
        if (pthread_mutex_lock(&ctx->lock) != 0)
            goto cleanup;
        iteration = ctx->iteration++;
        t->percent_complete = 100.0F*(float)iteration/cybervision_ransac_k;
        if (pthread_mutex_unlock(&ctx->lock) != 0)
            goto cleanup;

        if (iteration > cybervision_ransac_k)
            break;

        if (iteration % cybervision_ransac_check_interval == 0 && t->result_matches_count > 0)
            t->completed = 1;

        extended_inliers_count = 0;
        
        for(size_t i=0;i<ransac_n;i++)
        {
            size_t m;
            int unique = 0;
            while(!unique)
            {
                m = rand_r(&rand_seed)%t->matches_count;
                unique = 1;
                for (size_t j=0;j<i;j++)
                {
                    if (inliers[j] == m)
                    {
                        unique = 0;
                        break;
                    }
                }
            }
            inliers[i] = m;
        }

        if (!ransac_calculate_model(svd_ctx, t, inliers, ransac_n, fundamental_matrix))
        {
            t->error = "Failed to calculate fundamental matrix";
            t->completed = 1;
            break;
        }

        float fundamental_matrix_sum = 0.0F;
        for(size_t i=0;i<9;i++)
            fundamental_matrix_sum += fabsf(fundamental_matrix[i]);
        if (fundamental_matrix_sum == 0.0F)
            continue;

        for (size_t i=0;i<t->matches_count;i++)
        {
            int already_exists = 0;
            for (size_t j=0;j<ransac_n;j++)
            {
                if (inliers[j] == i)
                {
                    already_exists = 1;
                    break;
                }
            }
            if (already_exists)
                continue;
            
            if (ransac_calculate_error(t, i, fundamental_matrix) > cybervision_ransac_t)
                continue;

            extended_inliers[extended_inliers_count++] = i;
        }

        if (extended_inliers_count < cybervision_ransac_d)
            continue;
        
        for (size_t i=0;i<ransac_n;i++)
            extended_inliers[extended_inliers_count++] = inliers[i];

        /*
        if (!ransac_calculate_model(svd_ctx, t, extended_inliers, extended_inliers_count, fundamental_matrix))
        {
            t->error = "Failed to calculate extended fundamental matrix";
            t->completed = 1;
            break;
        }
        */

        float error = 0.0F;
        for (size_t i=0;i<extended_inliers_count;i++)
            error += ransac_calculate_error(t, extended_inliers[i], fundamental_matrix);
        error /= (float)extended_inliers_count;

        if (pthread_mutex_lock(&ctx->lock) != 0)
            goto cleanup;
        if (extended_inliers_count > t->result_matches_count)
        {
            for (size_t i=0;i<9;i++)
                t->fundamental_matrix[i] = fundamental_matrix[i];
            t->result_matches_count = extended_inliers_count;
        }
        if (pthread_mutex_unlock(&ctx->lock) != 0)
            goto cleanup;
    }
cleanup:
    free(inliers);
    free(extended_inliers);
    free_svd(svd_ctx);
    pthread_mutex_lock(&ctx->lock);
    ctx->threads_completed++;
    pthread_mutex_unlock(&ctx->lock);
    if (ctx->threads_completed >= t->num_threads)
        t->completed = 1;
    return NULL;
}

int correlation_ransac_start(ransac_task *task)
{
    ransac_task_ctx *ctx = malloc(sizeof(ransac_task_ctx));

    srand((unsigned int)time(NULL));
    
    task->internal = ctx;
    ctx->threads= malloc(sizeof(pthread_t)*task->num_threads);

    ctx->iteration = 0;
    task->percent_complete = 0.0;
    ctx->threads_completed = 0;
    task->completed = 0;
    task->error = NULL;

    task->dir_x = NAN;
    task->dir_y = NAN;
    task->result_matches_count = 0;

    if (pthread_mutex_init(&ctx->lock, NULL) != 0)
        return 0;

    for (int i = 0; i < task->num_threads; i++)
        pthread_create(&ctx->threads[i], NULL, correlate_ransac_task, task);

    return 1;
}

void correlation_ransac_cancel(ransac_task *t)
{
    if (t == NULL)
        return;
    t->completed = 1;
}

int correlation_ransac_complete(ransac_task *t)
{
    ransac_task_ctx *ctx;
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

static inline int fit_range(int val, int min, int max)
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
    float inv_scale = 1.0F/t->scale;
    int x_min = (int)floorf((x1-cybervision_crosscorrelation_neighbor_distance)*inv_scale);
    int x_max = (int)ceilf((x1+cybervision_crosscorrelation_neighbor_distance)*inv_scale);
    int y_min = (int)floorf((y1-cybervision_crosscorrelation_neighbor_distance)*inv_scale);
    int y_max = (int)ceilf((y1+cybervision_crosscorrelation_neighbor_distance)*inv_scale);

    x_min = fit_range(x_min, 0, t->out_width);
    x_max = fit_range(x_max, 0, t->out_width);
    y_min = fit_range(y_min, 0, t->out_height);
    y_max = fit_range(y_max, 0, t->out_height);
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
            min = current_depth - distance*cybervision_crosscorrelation_max_slope;
            max = current_depth + distance*cybervision_crosscorrelation_max_slope;

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
    int kernel_size;
    int kernel_point_count;
    int x1, y1;

    float stdev1;
    float *delta1;
    float *avg2;
    float *stdev2;

    int corridor_offset;
    float best_corr;
    float best_distance;
    int match_count;
} corridor_area_ctx;

static inline void correlate_corridor_area(cross_correlate_task *t, corridor_area_ctx* c, int corridor_start, int corridor_end)
{
    int kernel_size = c->kernel_size;
    int kernel_width = 2*kernel_size + 1;
    int kernel_point_count = c->kernel_point_count;
    int x1 = c->x1, y1 = c->y1;
    int w1 = t->img1.width;
    int w2 = t->img2.width, h2 = t->img2.height;
    for (int i=corridor_start;i<corridor_end;i++)
    {
        int x2, y2;
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

        dx = (float)(x2-x1)/t->scale;
        dy = (float)(y2-y1)/t->scale;
        distance = sqrtf(dx*dx+dy*dy);

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
            c->best_distance = distance;
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
    float *out_points;
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

    int corridor_vertical = fabsf(t->dir_y)>fabsf(t->dir_x);
    int corridor_start = kernel_size;
    int corridor_end = corridor_vertical? h2-kernel_size : w2-kernel_size;
    float inv_scale = 1.0F/t->scale;

    float dir_length = sqrtf(t->dir_x*t->dir_x + t->dir_y*t->dir_y);
    float distance_coeff = fabsf(corridor_vertical? t->dir_y/dir_length : t->dir_x/dir_length)*t->scale;

    corr_ctx.corridor_vertical = corridor_vertical;
    corr_ctx.corridor_coeff = corridor_vertical? t->dir_x/t->dir_y : t->dir_y/t->dir_x;
    corr_ctx.kernel_size = kernel_size;
    corr_ctx.kernel_point_count = kernel_point_count;

    corr_ctx.delta1 = malloc(sizeof(float)*kernel_point_count);
    corr_ctx.avg2 = ctx->avg2;
    corr_ctx.stdev2 = ctx->stdev2;
    
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
            float min_distance, max_distance;

            corr_ctx.best_distance = NAN;
            corr_ctx.best_corr = 0;
            corr_ctx.x1 = x1;
            corr_ctx.y1 = y1;
            corr_ctx.match_count = 0;

            compute_correlation_data(&t->img1, kernel_size, x1, y1, &corr_ctx.stdev1, corr_ctx.delta1);

            if (!isfinite(corr_ctx.stdev1))
                continue;

            processed_points++;
            if (t->iteration > 0)
                if (!estimate_search_range(t, x1, y1, &min_distance, &max_distance))
                    continue;

            for (int corridor_offset=-corridor_size;corridor_offset<=corridor_size;corridor_offset++)
            {
                corr_ctx.corridor_offset = corridor_offset;
                if (t->iteration > 0)
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
                if (corr_ctx.match_count>cybervision_crosscorrelation_match_limit)
                {
                    corr_ctx.best_distance = NAN;
                    corr_ctx.best_corr = 0;
                    break;
                }
            }
            if (isfinite(corr_ctx.best_distance))
            {
                ctx->out_points[y1*t->img1.width + x1] = corr_ctx.best_distance;
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

    ctx->out_points = malloc(sizeof(float)*task->img1.width*task->img1.height);
    for(int i = 0; i < task->img1.width*task->img1.height; i++)
        ctx->out_points[i] = NAN;
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
            float value = ctx->out_points[y*t->img1.width + x];
            int out_point_pos = ((int)roundf(inv_scale*y))*t->out_width + (int)roundf(inv_scale*x);
            if(isfinite(value))
            {
                t->out_points[out_point_pos] = value;
            }
        }
    }

    pthread_mutex_destroy(&ctx->lock);
    free(ctx->threads);
    free(ctx->out_points);
    free(ctx->avg2);
    free(ctx->stdev2);
    free(t->internal);
    t->internal = NULL;
    return 1;
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
