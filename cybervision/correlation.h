#ifndef CORRELATION_H
#define CORRELATION_H

typedef struct {
    int width, height;
    unsigned char* img;
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
    int num_threads;

    float percent_complete;
    int completed;

    correlation_match *matches;
    size_t matches_count;

    match_task_internal internal;
} match_task;
int correlation_match_points_start(match_task*);
void correlation_match_points_cancel(match_task*);
int correlation_match_points_complete(match_task*);

typedef struct { 
    int x1,y1;
    int x2,y2;
} ransac_match;
typedef void* ransac_task_internal;
typedef struct {
    ransac_match *matches;
    size_t matches_count;

    int num_threads;
    const char *error;

    float percent_complete;
    int completed;

    float dir_x, dir_y;
    size_t result_matches_count;

    ransac_task_internal internal;
} ransac_task;
int correlation_ransac_start(ransac_task*);
void correlation_ransac_cancel(ransac_task*);
int correlation_ransac_complete(ransac_task*);

typedef void* cross_correlate_task_internal;
typedef enum 
{ 
    CORRELATION_MODE_CPU = 0,
    CORRELATION_MODE_GPU = 1
} correlation_mode;
typedef struct {
    correlation_image img1, img2;
    float dir_x, dir_y;
    float scale;
    int neighbor_distance;
    float max_slope;
    int corridor_size;
    int kernel_size;
    float threshold;

    int num_threads;
    int iteration;

    float percent_complete;
    int completed;
    const char *error;

    int out_width, out_height;
    float *out_points;

    cross_correlate_task_internal internal;
} cross_correlate_task;
int correlation_cross_correlate_start(cross_correlate_task*);
void correlation_cross_correlate_cancel(cross_correlate_task*);
int correlation_cross_correlate_complete(cross_correlate_task*);

#endif
