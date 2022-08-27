#include <objc/objc.h>
#include <objc/objc-runtime.h>

#include <stdlib.h>
#include <math.h>
#include <string.h>

// TODO: remove this
#include <stdio.h>

#include <pthread.h>
#define THREAD_FUNCTION void*
#define THREAD_RETURN_VALUE NULL

#include "configuration.h"
#include "gpu_correlation.h"
#include "shaders_metal.h"

/*
 * Objective C interop helpers
 */
id MTLCreateSystemDefaultDevice();
// From NSObjCRuntime.h - use the default for 64-bit OS editions
typedef unsigned long NSUInteger;
typedef struct
{
    NSUInteger width, height, depth;
} __attribute__((packed)) MTLSize;

typedef struct {
    id pool;
    id device;
    id library;
    id pso_prepare_initialdata, pso_prepare_searchdata, pso_cross_correlate;
    id command_queue;
    id buffer_img, buffer_internal, buffer_internal_int, buffer_out;
} metal_device;

typedef struct {
    metal_device dev;
    pthread_t thread;
    int thread_started;
} metal_context;

typedef struct {
    int32_t img1_width;
    int32_t img1_height;
    int32_t img2_width;
    int32_t img2_height;
    int32_t output_width;
    int32_t output_height;
    float dir_x, dir_y;
    float scale;
    int32_t iteration;
    int32_t corridor_offset;
    int32_t corridor_start;
    int32_t corridor_end;
    int32_t kernel_size;
    float threshold;
    int32_t neighbor_distance;
    float max_slope;
    int32_t match_limit;
} shader_params;

const char* nserror_localized_description(id error)
{
    id errorNSString = ((id (*)(id, SEL))objc_msgSend)(error, sel_registerName("localizedDescription"));
    const char* errorCStr = ((const char* (*)(id, SEL, NSUInteger))objc_msgSend)(errorNSString, sel_registerName("cStringUsingEncoding:"), 5);
    return errorCStr;
}

id init_autoreleasepool()
{
    SEL allocSel = sel_registerName("alloc");
    SEL initSel = sel_registerName("init");
    Class NSAutoreleasePoolClass = objc_getClass("NSAutoreleasePool");
    id poolAlloc = ((id (*)(Class, SEL))objc_msgSend)(NSAutoreleasePoolClass, allocSel);
    return ((id (*)(id, SEL))objc_msgSend)(poolAlloc, initSel);
}

void drain_autoreleasepool(id pool)
{
    ((void (*)(id, SEL))objc_msgSend)(pool, sel_registerName("release"));
}

int gpu_init_device(metal_device *dev)
{
    SEL autoreleaseSel = sel_registerName("autorelease");
    dev->device = MTLCreateSystemDefaultDevice();
    if (dev->device == NULL)
        return 0;
    ((void (*)(id, SEL))objc_msgSend)(dev->device, autoreleaseSel);
    return 1;
}

int gpu_init_functions(metal_device *dev)
{
    SEL allocSel = sel_registerName("alloc");
    SEL autoreleaseSel = sel_registerName("autorelease");
    // Begin: delete from here
    FILE *fd = fopen("correlation.metal", "rb");
    void *correlation_metallib = malloc(1000000);
    int correlation_metallib_len = fread(correlation_metallib, 1, 1000000, fd);
    fclose(fd);
    // end: delete to here
    char *shaders_str = malloc(sizeof(char)*correlation_metallib_len+1);
    memcpy(shaders_str, correlation_metallib, correlation_metallib_len);
    shaders_str[correlation_metallib_len] = 0;

    Class NSStringClass = objc_getClass("NSString");
    
    id sourceString = ((id (*)(Class, SEL, char*, NSUInteger))objc_msgSend)(NSStringClass, sel_registerName("stringWithCString:encoding:"), shaders_str, 5);
    free(shaders_str);
    if (sourceString == NULL)
        return 0;
    
    const NSUInteger METAL_2_2 = 131074;
    Class MTLCompileOptions = objc_getClass("MTLCompileOptions");
    id compileOptions = ((id (*)(Class, SEL))objc_msgSend)(MTLCompileOptions, allocSel);
    ((void (*)(id, SEL, NSUInteger))objc_msgSend)(compileOptions, sel_registerName("setLanguageVersion:"), METAL_2_2);
    ((void (*)(id, SEL))objc_msgSend)(compileOptions, autoreleaseSel);

    id error = NULL;
    dev->library = ((id (*)(id, SEL, id, id, id*))objc_msgSend)(dev->device, sel_registerName("newLibraryWithSource:options:error:"), sourceString, compileOptions, &error);
    if (error != NULL)
        return 0;
    if (dev->library == NULL)
        return 0;

    id prepare_initialdataFunctionName = ((id (*)(Class, SEL, char*, NSUInteger))objc_msgSend)(NSStringClass, sel_registerName("stringWithCString:encoding:"), "prepare_initialdata", 5);
    if (prepare_initialdataFunctionName == NULL)
        return 0;
    id prepare_searchdataFunctionName = ((id (*)(Class, SEL, char*, NSUInteger))objc_msgSend)(NSStringClass, sel_registerName("stringWithCString:encoding:"), "prepare_searchdata", 5);
    if (prepare_searchdataFunctionName == NULL)
        return 0;
    id cross_correlateFunctionName = ((id (*)(Class, SEL, char*, NSUInteger))objc_msgSend)(NSStringClass, sel_registerName("stringWithCString:encoding:"), "cross_correlate", 5);
    if (cross_correlateFunctionName == NULL)
        return 0;

    id function_prepare_initialdata = ((id (*)(id, SEL, id))objc_msgSend)(dev->library, sel_registerName("newFunctionWithName:"), prepare_initialdataFunctionName);
    if (function_prepare_initialdata == NULL)
        return 0;
    ((void (*)(id, SEL))objc_msgSend)(function_prepare_initialdata, autoreleaseSel);
    id function_prepare_searchdata = ((id (*)(id, SEL, id))objc_msgSend)(dev->library, sel_registerName("newFunctionWithName:"), prepare_searchdataFunctionName);
    if (function_prepare_searchdata == NULL)
        return 0;
    ((void (*)(id, SEL))objc_msgSend)(function_prepare_searchdata, autoreleaseSel);
    id function_cross_correlate = ((id (*)(id, SEL, id))objc_msgSend)(dev->library, sel_registerName("newFunctionWithName:"), cross_correlateFunctionName);
    if (function_cross_correlate == NULL)
        return 0;
    ((void (*)(id, SEL))objc_msgSend)(function_cross_correlate, autoreleaseSel);

    dev->pso_prepare_initialdata = ((id (*)(id, SEL, id, id*))objc_msgSend)(dev->device, sel_registerName("newComputePipelineStateWithFunction:error:"), function_prepare_initialdata, &error);
    if (dev->pso_prepare_initialdata == NULL)
        return 0;
    ((void (*)(id, SEL))objc_msgSend)(dev->pso_prepare_initialdata, autoreleaseSel);
    if (error != NULL)
        return 0;

    dev->pso_prepare_searchdata = ((id (*)(id, SEL, id, id*))objc_msgSend)(dev->device, sel_registerName("newComputePipelineStateWithFunction:error:"), function_prepare_searchdata, &error);
    if (dev->pso_prepare_searchdata == NULL)
        return 0;
    ((void (*)(id, SEL))objc_msgSend)(dev->pso_prepare_searchdata, autoreleaseSel);
    if (error != NULL)
        return 0;

    dev->pso_cross_correlate = ((id (*)(id, SEL, id, id*))objc_msgSend)(dev->device, sel_registerName("newComputePipelineStateWithFunction:error:"), function_cross_correlate, &error);
    if (dev->pso_cross_correlate == NULL)
        return 0;
    ((void (*)(id, SEL))objc_msgSend)(dev->pso_cross_correlate, autoreleaseSel);
    if (error != NULL)
        return 0;

    return 1;
}

int gpu_init_queue(metal_device *dev)
{
    SEL autoreleaseSel = sel_registerName("autorelease");
    dev->command_queue = ((id (*)(id, SEL))objc_msgSend)(dev->device, sel_registerName("newCommandQueue"));
    if (dev->command_queue == NULL)
        return 0;
    ((void (*)(id, SEL))objc_msgSend)(dev->command_queue, autoreleaseSel);
    return 1;
}

id gpu_create_buffer(metal_device *dev, NSUInteger size, int gpuonly)
{
    SEL autoreleaseSel = sel_registerName("autorelease");
    const NSUInteger STORAGE_MODE_SHARED = 0 << 4;
    const NSUInteger STORAGE_MODE_PRIVATE = 2 << 4;
    
    NSUInteger mode = gpuonly? STORAGE_MODE_PRIVATE : STORAGE_MODE_SHARED;
    id buffer = ((id (*)(id, SEL, NSUInteger, NSUInteger))objc_msgSend)(dev->device, sel_registerName("newBufferWithLength:options:"), size, mode);
    ((void (*)(id, SEL))objc_msgSend)(buffer, autoreleaseSel);
    return buffer;
}

int gpu_transfer_in_images(cross_correlate_task *t, id buffer)
{
    float *payload = ((float* (*)(id, SEL))objc_msgSend)(buffer, sel_registerName("contents"));
    if (payload == NULL)
        return 0;

    for (int i=0;i<t->img1.width*t->img1.height;i++)
        payload[i] = (unsigned char)t->img1.img[i];
    payload += t->img1.width*t->img1.height;
    for (int i=0;i<t->img2.width*t->img2.height;i++)
        payload[i] = (unsigned char)t->img2.img[i];
    payload += t->img2.width*t->img2.height;
    for (int i=0;i<t->out_width*t->out_height;i++)
        payload[i] = t->out_points[i];

    free(t->img1.img);
    t->img1.img = NULL;
    free(t->img2.img);
    t->img2.img = NULL;

    return 1;
}

int gpu_run_command(metal_device *dev, int max_width, int max_height, shader_params *params, id pipeline)
{
    id commandBuffer = ((id (*)(id, SEL))objc_msgSend)(dev->command_queue, sel_registerName("commandBuffer"));
    if (commandBuffer == NULL)
        return 0;

    id computeEncoder = ((id (*)(id, SEL))objc_msgSend)(commandBuffer, sel_registerName("computeCommandEncoder"));
    if (computeEncoder == NULL)
        return 0;

    ((void (*)(id, SEL, id))objc_msgSend)(computeEncoder, sel_registerName("setComputePipelineState:"), pipeline);
    ((void (*)(id, SEL, void*, NSUInteger, NSUInteger))objc_msgSend)(computeEncoder, sel_registerName("setBytes:length:atIndex:"), params, sizeof(shader_params), 0);
    ((void (*)(id, SEL, id, NSUInteger, NSUInteger))objc_msgSend)(computeEncoder, sel_registerName("setBuffer:offset:atIndex:"), dev->buffer_img, 0, 1);
    ((void (*)(id, SEL, id, NSUInteger, NSUInteger))objc_msgSend)(computeEncoder, sel_registerName("setBuffer:offset:atIndex:"), dev->buffer_internal, 0, 2);
    ((void (*)(id, SEL, id, NSUInteger, NSUInteger))objc_msgSend)(computeEncoder, sel_registerName("setBuffer:offset:atIndex:"), dev->buffer_internal_int, 0, 3);
    ((void (*)(id, SEL, id, NSUInteger, NSUInteger))objc_msgSend)(computeEncoder, sel_registerName("setBuffer:offset:atIndex:"), dev->buffer_out, 0, 4);

    MTLSize threadgroupSize = {16, 16, 1};
    MTLSize threadgroupCount;
    threadgroupCount.width = (max_width + threadgroupSize.width-1)/threadgroupSize.width;
    threadgroupCount.height = (max_height + threadgroupSize.height-1)/threadgroupSize.height;
    threadgroupCount.depth = 1;

    ((void (*)(id, SEL, MTLSize, MTLSize))objc_msgSend)(computeEncoder, sel_registerName("dispatchThreadgroups:threadsPerThreadgroup:"), threadgroupCount, threadgroupSize);

    ((void (*)(id, SEL))objc_msgSend)(computeEncoder, sel_registerName("endEncoding"));
    ((void (*)(id, SEL))objc_msgSend)(commandBuffer, sel_registerName("commit"));
    ((void (*)(id, SEL))objc_msgSend)(commandBuffer, sel_registerName("waitUntilCompleted"));

    return 1;
}

int gpu_transfer_out_image(cross_correlate_task *t, metal_device *dev)
{
    float *output_points = ((float* (*)(id, SEL))objc_msgSend)(dev->buffer_out, sel_registerName("contents"));
    float inv_scale = 1.0F/t->scale;
    if (output_points == NULL)
        return 0;
    for (int y=0;y<t->img1.height;y++)
    {
        for (int x=0;x<t->img1.width;x++)
        {
            int out_point_pos = ((int)roundf(inv_scale*y))*t->out_width + (int)roundf(inv_scale*x);
            float value = output_points[y*t->img1.width + x];
            if (isfinite(value)){
                t->out_points[out_point_pos] = value;
            }
        }
    }

    return 1;
}

THREAD_FUNCTION gpu_correlate_cross_correlation_task(void *args)
{
    cross_correlate_task *t = args;
    metal_context* ctx = t->internal;
    int kernel_size = cybervision_crosscorrelation_kernel_size;
    int corridor_size = cybervision_crosscorrelation_corridor_size;
    int corridor_stripes = 2*corridor_size+1;
    int corridor_segment_length = cybervision_crosscorrelation_corridor_segment_length;
    int max_width = t->img1.width > t->img2.width ? t->img1.width:t->img2.width;
    int max_height = t->img1.height > t->img2.height ? t->img1.height:t->img2.height;
    int corridor_length = (fabs(t->dir_y)>fabs(t->dir_x)? t->img2.height:t->img2.width);
    int corridor_segments = corridor_length/cybervision_crosscorrelation_corridor_segment_length + 1;

    shader_params params = {0};

    params.img1_width = t->img1.width;
    params.img1_height = t->img1.height;
    params.img2_width = t->img2.width;
    params.img2_height = t->img2.height;
    params.output_width = t->out_width;
    params.output_height = t->out_height;
    params.dir_x = t->dir_x;
    params.dir_y = t->dir_y;
    params.scale = t->scale;
    params.iteration = t->iteration;
    params.kernel_size = cybervision_crosscorrelation_kernel_size;
    params.threshold = cybervision_crosscorrelation_threshold;
    params.neighbor_distance = cybervision_crosscorrelation_neighbor_distance;
    params.max_slope = cybervision_crosscorrelation_max_slope;
    params.match_limit = cybervision_crosscorrelation_match_limit;

    id autoreleasepool = init_autoreleasepool();
    if (!gpu_transfer_in_images(t, ctx->dev.buffer_img))
    {
        t->error = "Failed to transfer images into device buffer";
        goto cleanup;
    }

    if (!gpu_run_command(&ctx->dev, max_width, max_height, &params, ctx->dev.pso_prepare_initialdata))
    {
        t->error = "Failed to run initialization kernel";
        goto cleanup;
    }

    if (t->iteration > 0)
    {
        int y_limit = (int)ceilf((cybervision_crosscorrelation_neighbor_distance)/t->scale);
        int batch_size = cybervision_crosscorrelation_search_area_segment_length;
        for(int y=-y_limit;y<=y_limit;y+=batch_size)
        {
            params.corridor_start = y;
            params.corridor_end = y+batch_size;
            if (!gpu_run_command(&ctx->dev, t->img1.width, t->img1.height, &params, ctx->dev.pso_prepare_searchdata))
            {
                t->error = "Failed to run search area estimation kernel";
                goto cleanup;
            }
        }
    }

    t->percent_complete = 2.0F;

    for (int c=-corridor_size;c<=corridor_size;c++)
    {
        for (int l=0;l<corridor_segments;l++)
        {
            params.corridor_offset = c;
            params.corridor_start = kernel_size + l*corridor_segment_length;
            params.corridor_end = kernel_size + (l+1)*corridor_segment_length;
            if (params.corridor_end > corridor_length-kernel_size)
                params.corridor_end = corridor_length-kernel_size;

            if (!gpu_run_command(&ctx->dev, t->img1.width, t->img1.height, &params, ctx->dev.pso_cross_correlate))
            {
                t->error = "Failed to run correlation kernel";
                goto cleanup;
            }

            float corridor_complete = (float)(params.corridor_end - kernel_size) / (corridor_length-2*kernel_size);
            t->percent_complete = 2.0F + 98.0F*(c+corridor_size + corridor_complete)/corridor_stripes;
        }
    }

    if (!gpu_transfer_out_image(t, &ctx->dev))
        t-> error = "Failed to read output image";

cleanup:
    drain_autoreleasepool(autoreleasepool);
    t->completed = 1;
    return THREAD_RETURN_VALUE;
}


int gpu_correlation_cross_correlate_init(cross_correlate_task *t, size_t img1_pixels, size_t img2_pixels)
{
    metal_context* ctx = malloc(sizeof(metal_context));
    metal_device *dev;
    t->internal = ctx;
    ctx->thread_started = 0;

    memset(ctx, 0, sizeof(metal_context));
    ctx->dev.pool = init_autoreleasepool();

    if (!gpu_init_device(&ctx->dev))
    {
        t->error = "Failed to initialize Metal device";
        return 0;
    }
    dev = &ctx->dev; 
    if (!gpu_init_functions(dev))
    {
        t->error = "Failed to init function";
        return 0;
    }
    if (!gpu_init_queue(dev))
    {
        t->error = "Failed to init queue";
        return 0;
    }

    dev->buffer_img = gpu_create_buffer(dev, sizeof(float)*(img1_pixels+img2_pixels+t->out_width*t->out_height), 0);
    dev->buffer_internal = gpu_create_buffer(dev, sizeof(float)*(img1_pixels*2+img2_pixels*2+t->out_width*t->out_height*3), 1);
    dev->buffer_internal_int = gpu_create_buffer(dev, sizeof(int32_t)*(t->out_width*t->out_height), 1);
    dev->buffer_out = gpu_create_buffer(dev, sizeof(float)*t->out_width*t->out_height, 0);
    if (dev->buffer_img == NULL || dev->buffer_internal == NULL || dev->buffer_internal_int == NULL || dev->buffer_out == NULL)
    {
        t->error = "Failed to create buffers";
        return 0;
    }

    return 1;
}

int gpu_correlation_cross_correlate_start(cross_correlate_task *t)
{
    metal_context* ctx = t->internal;
    t->completed = 0;
    t->percent_complete = 0.0;
    t->error = NULL;

    pthread_create(&ctx->thread, NULL, gpu_correlate_cross_correlation_task, t);

    return 1;
}

int gpu_correlation_cross_correlate_complete(cross_correlate_task *t)
{
    metal_context* ctx;
    int result = 1;
    if (t == NULL || t->internal == NULL)
        return 1;
    ctx = t->internal;
    if (ctx->thread_started)
        pthread_join(ctx->thread, NULL);
    ctx->thread_started = 0;
    return 1;
}

int gpu_correlation_cross_correlate_cleanup(cross_correlate_task *t)
{
    metal_context* ctx;
    if (t == NULL || t->internal == NULL)
        return 1;
    ctx = t->internal;
    
    drain_autoreleasepool(ctx->dev.pool);
    free(ctx);
    t->internal = NULL;
    return 1;
}
