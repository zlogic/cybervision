#ifndef CORRELATION_H
#define CORRELATION_H

typedef struct {
    int width, height;
    int kernel_size, kernel_point_count;
    float *delta, *sigma;
} context;

int ctx_init(context *ctx, const char* img, int width, int height, int kernel_size, int num_threads);
void ctx_free(context *ctx);

typedef void (*correlation_matched)(size_t p1, size_t p2, float correlation, void *cb_args);

typedef struct { int x,y; } correlation_point;
int ctx_correlate(context *ctx1, context *ctx2,
    correlation_point *points1, correlation_point *points2, size_t points1_size, size_t points2_size,
    float threshold, int num_threads,
    correlation_matched cb, void *cb_args);

#endif
