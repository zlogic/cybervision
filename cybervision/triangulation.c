#include <stdlib.h>
#include <pthread.h>

#define _USE_MATH_DEFINES
#include <math.h>

#include "triangulation.h"

typedef struct {
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
        task->internal = NULL;
        triangulation_parallel(task);
        return 1;
    }
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
