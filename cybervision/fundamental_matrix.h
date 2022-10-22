#ifndef FUNDAMENTAL_MATRIX_H
#define FUNDAMENTAL_MATRIX_H

#include <stddef.h>

typedef enum
{ 
    PROJECTION_MODE_PARALLEL = 0,
    PROJECTION_MODE_PERSPECTIVE = 1
} projection_mode;
typedef struct { 
    int x1,y1;
    int x2,y2;
} ransac_match;
typedef struct {
    ransac_match *matches;
    size_t matches_count;
} ransac_match_bucket;
typedef void* ransac_task_internal;
typedef double matrix_3x3[3*3];
typedef struct {
    ransac_match_bucket *match_buckets;
    size_t match_buckets_count;

    projection_mode proj_mode;
    float keypoint_scale;

    int num_threads;
    const char *error;

    float percent_complete;
    int completed;

    matrix_3x3 fundamental_matrix;
    size_t result_matches_count;

    ransac_task_internal internal;
} ransac_task;
int correlation_ransac_start(ransac_task*);
void correlation_ransac_cancel(ransac_task*);
int correlation_ransac_complete(ransac_task*);

#endif