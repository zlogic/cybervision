#ifndef CORRELATION_H
#define CORRELATION_H

typedef struct {
    int width, height;
    const char* img;
} correlation_image;

typedef void (*correlation_matched)(size_t p1, size_t p2, float correlation, void *cb_args);

typedef struct { int x,y; } correlation_point;
int correlation_correlate_points(correlation_image *img1, correlation_image *img2,
    correlation_point *points1, correlation_point *points2, size_t points1_size, size_t points2_size,
    int kernel_size, float threshold, int num_threads,
    correlation_matched cb, void *cb_args);

int correlation_correlate_images(correlation_image *img1, correlation_image *img2,
    float angle, int corridor_size,
    int kernel_size, float threshold, int num_threads,
    float *out_points);

#endif
