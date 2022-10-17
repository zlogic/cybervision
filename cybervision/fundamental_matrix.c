#include <stdlib.h>
#include <pthread.h>

#define _USE_MATH_DEFINES
#include <math.h>

#include "fundamental_matrix.h"
#include "linmath.h"
#include "configuration.h"

typedef struct {
    size_t iteration;
    unsigned int thread_id;

    float best_error;

    int threads_completed;
    pthread_mutex_t lock;
    pthread_t *threads;
} ransac_task_ctx;

static inline float ransac_calculate_error(ransac_task *t, size_t selected_match, matrix_3x3 f)
{
    // Calculate Sampson distance
    ransac_match *match = &t->matches[selected_match];
    float p1[3] = {match->x1, match->y1, 1.0F};
    float p2[3] = {match->x2, match->y2, 1.0F};
    float nominator = 0.0F;
    float p2tFp1[3];
    p2tFp1[0] = p2[0]*f[0]+p2[1]*f[3]+p2[2]*f[6];
    p2tFp1[1] = p2[0]*f[1]+p2[1]*f[4]+p2[2]*f[7];
    p2tFp1[2] = p2[0]*f[2]+p2[1]*f[5]+p2[2]*f[8];
    nominator = p2tFp1[0]*p1[0]+p2tFp1[1]*p1[1]+p2tFp1[2]*p1[2];
    float Fp1[3];
    float Ftp2[3];
    multiply_f_vector(f, p1, Fp1);
    multiply_ft_vector(f, p2, Ftp2);
    float denominator = Fp1[0]*Fp1[0]+Fp1[1]*Fp1[1]+Ftp2[0]*Ftp2[0]+Ftp2[1]*Ftp2[1];
    return nominator*nominator/denominator;
}

typedef struct {
    svd_internal svd;
    double *a;
    double *u, *s, *vt;
} ransac_memory;

static inline void normalize_points(ransac_task *t, size_t *selected_matches, size_t selected_matches_count, matrix_3x3 m1, matrix_3x3 m2)
{
    // Recenter & rescale points
    float centerX1 = 0.0F, centerY1 = 0.0F;
    float centerX2 = 0.0F, centerY2 = 0.0F;
    for(size_t i=0;i<selected_matches_count;i++)
    {
        size_t selected_match = selected_matches[i];
        ransac_match *match = &t->matches[selected_match];
        centerX1 += (float)match->x1;
        centerY1 += (float)match->y1;
        centerX2 += (float)match->x2;
        centerY2 += (float)match->y2;
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
    scale1 = sqrtf(2.0F)/(scale1/(float)selected_matches_count);
    scale2 = sqrtf(2.0F)/(scale2/(float)selected_matches_count);

    m1[0] = scale1; m1[1] = 0.0F; m1[2] = -centerX1*scale1;
    m1[3] = 0.0F; m1[3+1] = scale1; m1[3+2] = -centerY1*scale1;
    m1[6] = 0.0F; m1[6+1] = 0.0F; m1[6+2] = 1.0F;

    m2[0] = scale2; m2[1] = 0.0F; m2[2] = -centerX2*scale2;
    m2[3] = 0.0F; m2[3+1] = scale2; m2[3+2] = -centerY2*scale2;
    m2[6] = 0.0F; m2[6+1] = 0.0F; m2[6+2] = 1.0F;
}

static inline int ransac_calculate_model_perspective(ransac_memory *ctx, ransac_task *t, size_t *selected_matches, size_t selected_matches_count, matrix_3x3 f)
{
    matrix_3x3 m1;
    matrix_3x3 m2;
    normalize_points(t, selected_matches, selected_matches_count, m1, m2);
    // Calculate fundamental matrix using the 8-point algorithm
    double *a = ctx->a;
    double *u = ctx->u, *s = ctx->s, *vt = ctx->vt;
    for(size_t i=0;i<selected_matches_count;i++)
    {
        size_t selected_match = selected_matches[i];
        ransac_match *match = &t->matches[selected_match];
        float x1 = (float)match->x1*m1[0] + (float)match->y1*m1[1] + m1[2];
        float y1 = (float)match->x1*m1[3] + (float)match->y1*m1[4] + m1[5];
        float x2 = (float)match->x2*m2[0] + (float)match->y2*m2[1] + m2[2];
        float y2 = (float)match->x2*m2[3] + (float)match->y2*m2[4] + m2[5];
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
    if (!svd(ctx->svd, a, selected_matches_count, 9, u, s, vt))
        return 0;

   for(size_t i=0;i<9;i++)
        a[i] = vt[9*8+i];

    if (!svd(ctx->svd, a, 3, 3, u, s, vt))
        return 0;

    // Check if matrix rank is too low
    if (fabs(s[8])<cybervision_ransac_rank_epsilon)
        return 0;

    matrix_3x3 u_float, v_float;
    for(size_t i=0;i<9;i++)
    {
        u_float[i] = u[i];
        v_float[i] = vt[i];
    }
    matrix_3x3 s_matrix = {s[0], 0.0F, 0.0F, 0.0F, s[1], 0.0F, 0.0F, 0.0F, 0.0F};
    matrix_3x3 f_temp;
    multiply_matrix_3x3(u_float, s_matrix, f_temp);
    multiply_matrix_3x3(f_temp, v_float, f);

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
    double *a = ctx->a, *u = ctx->u, *s = ctx->s, *vt = ctx->vt;
    float mean_x1 = 0.0F, mean_y1 = 0.0F;
    float mean_x2 = 0.0F, mean_y2 = 0.0F;
    for(size_t i=0;i<selected_matches_count;i++)
    {
        size_t selected_match = selected_matches[i];
        ransac_match *match = &t->matches[selected_match];
        float x1 = match->x1;
        float y1 = match->y1;
        float x2 = match->x2;
        float y2 = match->y2;
        a[i*4  ] = x2;
        a[i*4+1] = y2;
        a[i*4+2] = x1;
        a[i*4+3] = y1;
        mean_x2 += x2;
        mean_y2 += y2;
        mean_x1 += x1;
        mean_y1 += y1;
    }
    mean_x1 /= (float)selected_matches_count;
    mean_y1 /= (float)selected_matches_count;
    mean_x2 /= (float)selected_matches_count;
    mean_y2 /= (float)selected_matches_count;
    for(size_t i=0;i<selected_matches_count;i++)
    {
        a[i*4  ] -= mean_x2;
        a[i*4+1] -= mean_y2;
        a[i*4+2] -= mean_x1;
        a[i*4+3] -= mean_y1;
    }
    if (!svd(ctx->svd, a, selected_matches_count, 4, u, s, vt))
        return 0;
    vt = &vt[4*3];

    // Check if matrix rank is too low
    if (fabs(s[3])<cybervision_ransac_rank_epsilon)
        return 0;

    f[0] = 0.0F; f[1] = 0.0F; f[2] = vt[0];
    f[3] = 0.0F; f[4] = 0.0F; f[5] = vt[1];
    f[6] = vt[2]; f[7] = vt[3]; f[8] = -(vt[0]*mean_x2+vt[1]*mean_y2+vt[2]*mean_x1+vt[3]*mean_y1);

    return 1;
}

void* correlate_ransac_task(void *args)
{
    ransac_task *t = args;
    ransac_task_ctx *ctx = t->internal;
    size_t ransac_n;
    size_t *inliers;
    float ransac_t;
    matrix_3x3 fundamental_matrix;
    size_t extended_inliers_count = 0;
    ransac_memory ctx_memory = {0};
    int (*ransac_calculate_model)(ransac_memory *ctx, ransac_task *t, size_t *selected_matches, size_t selected_matches_count, matrix_3x3 f);
    unsigned int rand_seed;

    if (pthread_mutex_lock(&ctx->lock) != 0)
        goto cleanup;
    rand_seed = (ctx->thread_id++) ^ (unsigned int)time(NULL);
    if (pthread_mutex_unlock(&ctx->lock) != 0)
        goto cleanup;

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
    ctx_memory.svd = init_svd(&rand_seed);
    ctx_memory.a = malloc(sizeof(double)*(ransac_n*9));
    ctx_memory.u = malloc(sizeof(double)*(ransac_n*ransac_n));
    ctx_memory.s = malloc(sizeof(double)*(ransac_n>9?ransac_n:9));
    ctx_memory.vt = malloc(sizeof(double)*9*9);
    
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

        if (iteration % cybervision_ransac_check_interval == 0 && t->result_matches_count > cybervision_ransac_d_early_exit)
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

        float fundamental_matrix_sum = 0.0F;
        for(size_t i=0;i<9;i++)
        {
            if (!isfinite(fundamental_matrix[i]))
            {
                fundamental_matrix_sum = NAN;
                break;
            }
            fundamental_matrix_sum += fabsf(fundamental_matrix[i]);
        }
        if (fundamental_matrix_sum == 0.0F || !isfinite(fundamental_matrix_sum))
            continue;

        float inliers_error = 0.0F;
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
            
            float inlier_error = fabsf(ransac_calculate_error(t, i, fundamental_matrix));
            if (inlier_error > ransac_t)
                continue;

            extended_inliers_count++;
            inliers_error += inlier_error;
        }

        if (extended_inliers_count < cybervision_ransac_d)
            continue;

        for (size_t i=0;i<ransac_n;i++)
        {
            float inlier_error = fabsf(ransac_calculate_error(t, inliers[i], fundamental_matrix));
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
        inliers_error = fabsf(inliers_error/(float)extended_inliers_count);

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
    free(ctx_memory.u);
    free(ctx_memory.s);
    free(ctx_memory.vt);
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
    ctx->thread_id = 0;
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
