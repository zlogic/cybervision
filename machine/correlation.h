#ifndef CORRELATION_H
#define CORRELATION_H

typedef struct {
    int width, height;
    char* img;
} correlation_image;

typedef struct { int x,y; } correlation_point;
typedef struct {
    int point1, point2;
    float corr;
} correlation_match;
typedef void* match_task_internal;
typedef struct {
    correlation_image img1, img2;
    correlation_point *points1, *points2;
    size_t points1_size, points2_size;
    int kernel_size;
    float threshold;
    int num_threads;

    float percent_complete;
    int completed;

    correlation_match *matches;
    size_t matches_count;

    match_task_internal internal;
} match_task;
int correlation_match_points_start(match_task *match_task);
void correlation_match_points_cancel(match_task *match_task);
int correlation_match_points_complete(match_task *match_task);

int correlation_correlate_images(correlation_image *img1, correlation_image *img2,
    float angle, int corridor_size,
    int kernel_size, float threshold, int num_threads,
    float *out_points);

#endif
