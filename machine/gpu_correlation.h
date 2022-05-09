#ifndef GPU_CORRELATION_H
#define GPU_CORRELATION_H

typedef void* gpu_context_internal;
typedef struct {
    int width, height;
    gpu_context_internal internal;
} gpu_context;
int correlation_gpu_init(gpu_context*);
int correlation_gpu_free(gpu_context*);

#endif
