#include <stdlib.h>
#include <string.h>
#include <time.h>
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

    double best_error;

    int threads_completed;
    pthread_mutex_t lock;
    pthread_t *threads;
} ransac_task_ctx;

static inline double ransac_calculate_error(ransac_task *t, size_t selected_match, matrix_3x3 f)
{
    // Calculate Sampson distance
    ransac_match *match = &t->matches[selected_match];
    double p1[3] = {match->x1, match->y1, 1.0};
    double p2[3] = {match->x2, match->y2, 1.0};
    double nominator = 0.0;
    double p2tFp1[3];
    p2tFp1[0] = p2[0]*f[0]+p2[1]*f[3]+p2[2]*f[6];
    p2tFp1[1] = p2[0]*f[1]+p2[1]*f[4]+p2[2]*f[7];
    p2tFp1[2] = p2[0]*f[2]+p2[1]*f[5]+p2[2]*f[8];
    nominator = p2tFp1[0]*p1[0]+p2tFp1[1]*p1[1]+p2tFp1[2]*p1[2];
    double Fp1[3];
    double Ftp2[3];
    multiply_f_vector(f, p1, Fp1);
    multiply_ft_vector(f, p2, Ftp2);
    double denominator = Fp1[0]*Fp1[0]+Fp1[1]*Fp1[1]+Ftp2[0]*Ftp2[0]+Ftp2[1]*Ftp2[1];
    return nominator*nominator/denominator;
}

typedef struct {
    svd_internal svd;
    double *a;
    double *s;
    double *v;
} ransac_memory;

static inline void normalize_points(ransac_task *t, size_t *selected_matches, size_t selected_matches_count, matrix_3x3 m1, matrix_3x3 m2)
{
    // Recenter & rescale points
    double centerX1 = 0.0, centerY1 = 0.0;
    double centerX2 = 0.0, centerY2 = 0.0;
    for(size_t i=0;i<selected_matches_count;i++)
    {
        size_t selected_match = selected_matches[i];
        ransac_match *match = &t->matches[selected_match];
        centerX1 += (double)match->x1;
        centerY1 += (double)match->y1;
        centerX2 += (double)match->x2;
        centerY2 += (double)match->y2;
    }
    centerX1 /= (double)selected_matches_count;
    centerY1 /= (double)selected_matches_count;
    centerX2 /= (double)selected_matches_count;
    centerY2 /= (double)selected_matches_count;
    double scale1 = 0.0, scale2 = 0.0;
    for(size_t i=0;i<selected_matches_count;i++)
    {
        size_t selected_match = selected_matches[i];
        ransac_match *match = &t->matches[selected_match];
        double dx1 = (double)match->x1-centerX1;
        double dy1 = (double)match->y1-centerY1;
        double dx2 = (double)match->x2-centerX2;
        double dy2 = (double)match->y2-centerY2;
        scale1 += sqrt(dx1*dx1 + dy1*dy1);
        scale2 += sqrt(dx2*dx2 + dy2*dy2);
    }
    scale1 = sqrt(2.0)/(scale1/(double)selected_matches_count);
    scale2 = sqrt(2.0)/(scale2/(double)selected_matches_count);

    m1[0] = scale1; m1[1] = 0.0; m1[2] = -centerX1*scale1;
    m1[3] = 0.0; m1[3+1] = scale1; m1[3+2] = -centerY1*scale1;
    m1[6] = 0.0; m1[6+1] = 0.0; m1[6+2] = 1.0;

    m2[0] = scale2; m2[1] = 0.0; m2[2] = -centerX2*scale2;
    m2[3] = 0.0; m2[3+1] = scale2; m2[3+2] = -centerY2*scale2;
    m2[6] = 0.0; m2[6+1] = 0.0; m2[6+2] = 1.0;
}

static inline int ransac_calculate_model_perspective(ransac_memory *ctx, ransac_task *t, size_t *selected_matches, size_t selected_matches_count, matrix_3x3 f)
{
    matrix_3x3 m1;
    matrix_3x3 m2;
    normalize_points(t, selected_matches, selected_matches_count, m1, m2);
    // Calculate fundamental matrix using the 8-point algorithm
    double *a = ctx->a;
    double *s = ctx->s, *v = ctx->v;
    for(size_t i=0;i<selected_matches_count;i++)
    {
        size_t selected_match = selected_matches[i];
        ransac_match *match = &t->matches[selected_match];
        double x1 = (double)match->x1*m1[0] + (double)match->y1*m1[1] + m1[2];
        double y1 = (double)match->x1*m1[3] + (double)match->y1*m1[4] + m1[5];
        double x2 = (double)match->x2*m2[0] + (double)match->y2*m2[1] + m2[2];
        double y2 = (double)match->x2*m2[3] + (double)match->y2*m2[4] + m2[5];
        a[i*9  ] = x2*x1;
        a[i*9+1] = x2*y1;
        a[i*9+2] = x2;
        a[i*9+3] = y2*x1;
        a[i*9+4] = y2*y1;
        a[i*9+5] = y2;
        a[i*9+6] = x1;
        a[i*9+7] = y1;
        a[i*9+8] = 1.0;
    }
    int result = svdd(ctx->svd, a, selected_matches_count, 9, s, v);
    if (!result)
        return result;

   for(size_t i=0;i<9;i++)
        a[i] = v[9*8+i];

    result = svdd(ctx->svd, a, 3, 3, s, v);
    if (!result)
        return result;

    matrix_3x3 s_matrix = {s[0], 0.0, 0.0, 0.0, s[1], 0.0, 0.0, 0.0, 0.0};
    matrix_3x3 f_temp;
    multiply_matrix_3x3(a, s_matrix, f_temp);
    multiply_matrix_3x3(f_temp, v, f);

    // Scale back to image coordinates
    multiply_matrix_3tx3(m2, f, f_temp);
    multiply_matrix_3x3(f_temp, m1, f);
    return 1;
}

static inline int ransac_calculate_model_affine(ransac_memory *ctx, ransac_task *t, size_t *selected_matches, size_t selected_matches_count, matrix_3x3 f)
{
    matrix_3x3 m1;
    matrix_3x3 m2;
    // Calculate fundamental matrix using the 4-point algorithm
    double *a = ctx->a, *s = ctx->s, *v = ctx->v;
    double mean_x1 = 0.0, mean_y1 = 0.0;
    double mean_x2 = 0.0, mean_y2 = 0.0;
    for(size_t i=0;i<selected_matches_count;i++)
    {
        size_t selected_match = selected_matches[i];
        ransac_match *match = &t->matches[selected_match];
        double x1 = match->x1;
        double y1 = match->y1;
        double x2 = match->x2;
        double y2 = match->y2;
        a[i*4  ] = x2;
        a[i*4+1] = y2;
        a[i*4+2] = x1;
        a[i*4+3] = y1;
        mean_x2 += x2;
        mean_y2 += y2;
        mean_x1 += x1;
        mean_y1 += y1;
    }
    mean_x1 /= (double)selected_matches_count;
    mean_y1 /= (double)selected_matches_count;
    mean_x2 /= (double)selected_matches_count;
    mean_y2 /= (double)selected_matches_count;
    for(size_t i=0;i<selected_matches_count;i++)
    {
        size_t selected_match = selected_matches[i];
        ransac_match *match = &t->matches[selected_match];
        a[i*4  ] -= mean_x2;
        a[i*4+1] -= mean_y2;
        a[i*4+2] -= mean_x1;
        a[i*4+3] -= mean_y1;
    }
    int result = svdd(ctx->svd, a, selected_matches_count, 4, s, v);
    if (!result)
        return result;
    v = &v[4*3];

    // Check if matrix rank is too low
    if (s[3]<cybervision_ransac_rank_epsilon)
        return 0;

    f[0] = 0.0; f[1] = 0.0; f[2] = v[0];
    f[3] = 0.0; f[4] = 0.0; f[5] = v[1];
    f[6] = v[2]; f[7] = v[3]; f[8] = -(v[0]*mean_x2+v[1]*mean_y2+v[2]*mean_x1+v[3]*mean_y1);

    return 1;
}

void* correlate_ransac_task(void *args)
{
    ransac_task *t = args;
    ransac_task_ctx *ctx = t->internal;
    size_t ransac_n;
    size_t *inliers;
    double ransac_t;
    matrix_3x3 fundamental_matrix;
    size_t extended_inliers_count = 0;
    ransac_memory ctx_memory = {0};
    int (*ransac_calculate_model)(ransac_memory *ctx, ransac_task *t, size_t *selected_matches, size_t selected_matches_count, matrix_3x3 f);
    unsigned int rand_seed = thread_id() ^ (unsigned int)time(NULL);
    
    if (t->proj_mode == PROJECTION_MODE_PARALLEL)
    {
        ransac_calculate_model = ransac_calculate_model_affine;
        ransac_n = cybervision_ransac_n_affine;
        ransac_t = cybervision_ransac_t_affine;
    }
    else if (t->proj_mode == PROJECTION_MODE_PERSPECTIVE)
    {
        ransac_calculate_model = ransac_calculate_model_perspective;
        ransac_n = cybervision_ransac_n_perspective;
        ransac_t = cybervision_ransac_t_perspective/(t->keypoint_scale*t->keypoint_scale);
    }
    inliers = malloc(sizeof(size_t)*ransac_n);
    ctx_memory.svd = init_svd();
    ctx_memory.a = malloc(sizeof(double)*(ransac_n>9?ransac_n*ransac_n:9*9));
    ctx_memory.s = malloc(sizeof(double)*ransac_n);
    ctx_memory.v = malloc(sizeof(double)*9*9);
    
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
            ransac_match match;
            while(!unique)
            {
                m = rand_r(&rand_seed)%t->matches_count;
                match = t->matches[m];
                unique = 1;
                for (size_t j=0;j<i;j++)
                {
                    ransac_match check_match = t->matches[inliers[j]];
                    if ((match.x1 == check_match.x1 && match.y1 == check_match.y1) ||
                        (match.x2 == check_match.x2 && match.y2 == check_match.y2))
                    {
                        unique = 0;
                        break;
                    }
                }
            }
            inliers[i] = m;
        }

        if (!ransac_calculate_model(&ctx_memory, t, inliers, ransac_n, fundamental_matrix))
            continue;

        double fundamental_matrix_sum = 0.0;
        for(size_t i=0;i<9;i++)
        {
            if (!isfinite(fundamental_matrix[i]))
            {
                fundamental_matrix_sum = NAN;
                break;
            }
            fundamental_matrix_sum += fabs(fundamental_matrix[i]);
        }
        if (fundamental_matrix_sum == 0.0 || !isfinite(fundamental_matrix_sum))
            continue;

        double inliers_error = 0.0F;
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
            
            double inlier_error = fabs(ransac_calculate_error(t, i, fundamental_matrix));
            if (inlier_error > ransac_t)
                continue;

            extended_inliers_count++;
            inliers_error += inlier_error;
        }

        if (extended_inliers_count < cybervision_ransac_d)
            continue;

        for (size_t i=0;i<ransac_n;i++)
        {
            double inlier_error = fabs(ransac_calculate_error(t, inliers[i], fundamental_matrix));
            if (inlier_error > ransac_t)
            {
                inliers_error = NAN;
                break;
            }
            extended_inliers_count++;
            inliers_error += inlier_error;
        }
        if (!isfinite(inliers_error))
            continue;
        inliers_error = fabs(inliers_error/(double)extended_inliers_count);

        if (pthread_mutex_lock(&ctx->lock) != 0)
            goto cleanup;
        if (extended_inliers_count >= t->result_matches_count && inliers_error <= ctx->best_error)
        {
            for (size_t i=0;i<9;i++)
                t->fundamental_matrix[i] = fundamental_matrix[i];
            t->result_matches_count = extended_inliers_count;
            ctx->best_error = inliers_error;
        }
        if (pthread_mutex_unlock(&ctx->lock) != 0)
            goto cleanup;
    }
cleanup:
    free(inliers);
    free(ctx_memory.a);
    free(ctx_memory.s);
    free(ctx_memory.v);
    free_svd(ctx_memory.svd);
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

    ctx->best_error = INFINITY;
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
    return t->error == NULL;
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
    double scale = t->scale;
    double p1[3] = {(double)c->x1/scale, (double)c->y1/scale, 1.0};
    double Fp1[3];
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
