#include <stdlib.h>
#include <pthread.h>

#define _USE_MATH_DEFINES
#include <math.h>

#include "triangulation.h"
#include "linmath.h"

typedef struct {
    int y;

    double camera_2[4*3];

    int threads_completed;
    pthread_mutex_t lock;
    pthread_t *threads;
} triangulation_task_ctx;

void triangulation_parallel(triangulation_task *t)
{
    // TODO: use tilt angle factorization instead of this simple 
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
            t->out_depth[y1*t->width+x1] = sqrtf(dx*dx+dy*dy)*t->depth_scale;
        }
    }
    t->completed = 1;
}

int triangulation_perspective_cameras(triangulation_task *t)
{
    triangulation_task_ctx *ctx = t->internal;
    svd_internal svd_ctx = init_svd();
    double f[9];
    int result = 0;
    for (size_t i=0;i<3;i++)
        for (size_t j=0;j<3;j++)
            f[j*3+i] = t->fundamental_matrix[i*3+j];

    double u[9], s[3], vt[9];
    if (!svdd(svd_ctx, f, 3, 3, u, s, vt))
        goto cleanup;

    // Using e' (epipole in second image) to calculate projection matrix for second image
    double e2[3];
    for(size_t i=0;i<3;i++)
        e2[i] = u[3*2+i];
    float e2_skewsymmetric[9] = {0.0, -e2[2], e2[1], e2[2], 0.0, -e2[0], -e2[1], e2[0], 0.0};
    float e2sf[9];
    multiply_matrix_3x3(e2_skewsymmetric, t->fundamental_matrix, e2sf);
    for (size_t i=0;i<3;i++)
        for (size_t j=0;j<3;j++)
            ctx->camera_2[4*i+j] = e2sf[3*i+j]/e2[2];
    for (size_t i=0;i<3;i++)
        ctx->camera_2[4*i+3] = e2[i]/e2[2];
    result = 1;
cleanup:
    free_svd(svd_ctx);
    return result;
}

void* triangulation_perspective_task(void *args)
{
    triangulation_task *t = args;
    triangulation_task_ctx *ctx = t->internal;
    double a[4*4];
    double u[4*4], s[4], vt[4*4];
    double *p2 = ctx->camera_2; 
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
            t->out_depth[y1*t->width+x1] = NAN;
            if (x2<0 || y2<0)
                continue;

            // Linear triangulation method
            // First row of A: x1*[0 0 1 0]-[1 0 0 0]
            a[0+3*0]= -1.0;
            a[0+3*1]= 0.0;
            a[0+3*2]= x1;
            a[0+3*3]= 0.0;
            // Second row of A: y1*[0 0 1 0]-[0 1 0 0]
            a[1+3*0]= 0.0;
            a[1+3*1]= -1.0;
            a[1+3*2]= y1;
            a[1+3*3]= 0.0;
            // Third row of A: x2*camera_2[2]-camera_2[0]
            a[2+3*0]= (float)x2*p2[2*4+0]-p2[0+0];
            a[2+3*1]= (float)x2*p2[2*4+1]-p2[0+1];
            a[2+3*2]= (float)x2*p2[2*4+2]-p2[0+2];
            a[2+3*3]= (float)x2*p2[2*4+3]-p2[0+3];
            // Fourch row of A: y2*camera_2[2]-camera_2[1]
            a[3+3*0]= (float)y2*p2[2*4+0]-p2[4+0];
            a[3+3*1]= (float)y2*p2[2*4+1]-p2[4+1];
            a[3+3*2]= (float)y2*p2[2*4+2]-p2[4+2];
            a[3+3*3]= (float)y2*p2[2*4+3]-p2[4+3];

            if (!svdd(svd_ctx, a, 4, 4, u, s, vt))
                continue;
            
            double point[4];
            for(size_t i=0;i<4;i++)
                point[i] = vt[i*4+3];
            if (fabs(point[3])>1.0E-3)
                t->out_depth[y1*t->width+x1] = point[2]/point[3];
        }
    }
cleanup:
    free_svd(svd_ctx);
    pthread_mutex_lock(&ctx->lock);
    ctx->threads_completed++;
    pthread_mutex_unlock(&ctx->lock);
    if (ctx->threads_completed >= t->num_threads)
        t->completed = 1;
}

int triangulation_start(triangulation_task *task)
{
    task->percent_complete = 0.0F;
    task->completed = 0;
    task->error = NULL;
    if (task->proj_mode == TRIANGULATION_PROJECTION_MODE_PARALLEL)
    {
        task->internal = NULL;
        triangulation_parallel(task);
        return 1;
    }
    if(task->proj_mode == TRIANGULATION_PROJECTION_MODE_PERSPECTIVE)
    {
        triangulation_task_ctx *ctx = malloc(sizeof(triangulation_task_ctx));
        task->internal = ctx;
        ctx->y = 0;
        ctx->threads= malloc(sizeof(pthread_t)*task->num_threads);
        ctx->threads_completed = 0;

        if (!triangulation_perspective_cameras(task))
        {
            task->error = "Failed to compute projection matrices";
            return 0;
        }
            
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
    free(t->internal);
    t->internal = NULL;
    return t->error == NULL;
}
