#include <objc/objc.h>
#include <objc/objc-runtime.h>
#include <dispatch/dispatch.h>

#include <stdlib.h>
#include <math.h>
#include <string.h>

#include <pthread.h>

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
    id pso_prepare_initialdata_searchdata, pso_prepare_initialdata_correlation, pso_prepare_searchdata, pso_cross_correlate;
    id command_queue;
    id buffer_img, buffer_previous_result, buffer_internal, buffer_internal_int, buffer_out;
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
    int32_t pad[2]; // metal requires matrices to be aligned to 16 bytes
    float fundamental_matrix[3*4]; // matrices are row-major and each row is aligned to 4-component vectors
    float scale;
    int32_t iteration;
    int32_t corridor_offset;
    int32_t corridor_start;
    int32_t corridor_end;
    int32_t phase;
    int32_t kernel_size;
    float threshold;
    int32_t neighbor_distance;
    float extend_range;
    float min_range;
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
    char *shaders_str = malloc(sizeof(char)*correlation_metallib_len+1);
    memcpy(shaders_str, correlation_metallib, correlation_metallib_len);
    shaders_str[correlation_metallib_len] = 0;

    Class NSStringClass = objc_getClass("NSString");

    dispatch_data_t dispatch_library = dispatch_data_create(correlation_metallib, correlation_metallib_len, NULL, DISPATCH_DATA_DESTRUCTOR_FREE);
    id error = NULL;
    dev->library = ((id (*)(id, SEL, dispatch_data_t, id*))objc_msgSend)(dev->device, sel_registerName("newLibraryWithData:error:"), dispatch_library, &error);
    dispatch_release(dispatch_library);
    if (error != NULL)
        return 0;
    if (dev->library == NULL)
        return 0;

    id prepare_initialdata_searchdataFunctionName = ((id (*)(Class, SEL, char*, NSUInteger))objc_msgSend)(NSStringClass, sel_registerName("stringWithCString:encoding:"), "prepare_initialdata_searchdata", 5);
    if (prepare_initialdata_searchdataFunctionName == NULL)
        return 0;
    id prepare_initialdata_correlationFunctionName = ((id (*)(Class, SEL, char*, NSUInteger))objc_msgSend)(NSStringClass, sel_registerName("stringWithCString:encoding:"), "prepare_initialdata_correlation", 5);
    if (prepare_initialdata_correlationFunctionName == NULL)
        return 0;
    id prepare_searchdataFunctionName = ((id (*)(Class, SEL, char*, NSUInteger))objc_msgSend)(NSStringClass, sel_registerName("stringWithCString:encoding:"), "prepare_searchdata", 5);
    if (prepare_searchdataFunctionName == NULL)
        return 0;
    id cross_correlateFunctionName = ((id (*)(Class, SEL, char*, NSUInteger))objc_msgSend)(NSStringClass, sel_registerName("stringWithCString:encoding:"), "cross_correlate", 5);
    if (cross_correlateFunctionName == NULL)
        return 0;

    id function_prepare_initialdata_searchdata = ((id (*)(id, SEL, id))objc_msgSend)(dev->library, sel_registerName("newFunctionWithName:"), prepare_initialdata_searchdataFunctionName);
    if (function_prepare_initialdata_searchdata == NULL)
        return 0;
    ((void (*)(id, SEL))objc_msgSend)(function_prepare_initialdata_searchdata, autoreleaseSel);
    id function_prepare_initialdata_correlation = ((id (*)(id, SEL, id))objc_msgSend)(dev->library, sel_registerName("newFunctionWithName:"), prepare_initialdata_correlationFunctionName);
    if (function_prepare_initialdata_correlation == NULL)
        return 0;
    ((void (*)(id, SEL))objc_msgSend)(function_prepare_initialdata_correlation, autoreleaseSel);
    id function_prepare_searchdata = ((id (*)(id, SEL, id))objc_msgSend)(dev->library, sel_registerName("newFunctionWithName:"), prepare_searchdataFunctionName);
    if (function_prepare_searchdata == NULL)
        return 0;
    ((void (*)(id, SEL))objc_msgSend)(function_prepare_searchdata, autoreleaseSel);
    id function_cross_correlate = ((id (*)(id, SEL, id))objc_msgSend)(dev->library, sel_registerName("newFunctionWithName:"), cross_correlateFunctionName);
    if (function_cross_correlate == NULL)
        return 0;
    ((void (*)(id, SEL))objc_msgSend)(function_cross_correlate, autoreleaseSel);

    dev->pso_prepare_initialdata_searchdata = ((id (*)(id, SEL, id, id*))objc_msgSend)(dev->device, sel_registerName("newComputePipelineStateWithFunction:error:"), function_prepare_initialdata_searchdata, &error);
    if (dev->pso_prepare_initialdata_searchdata == NULL)
        return 0;
    ((void (*)(id, SEL))objc_msgSend)(dev->pso_prepare_initialdata_searchdata, autoreleaseSel);
    if (error != NULL)
        return 0;

    dev->pso_prepare_initialdata_correlation = ((id (*)(id, SEL, id, id*))objc_msgSend)(dev->device, sel_registerName("newComputePipelineStateWithFunction:error:"), function_prepare_initialdata_correlation, &error);
    if (dev->pso_prepare_initialdata_correlation == NULL)
        return 0;
    ((void (*)(id, SEL))objc_msgSend)(dev->pso_prepare_initialdata_correlation, autoreleaseSel);
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

    free(t->img1.img);
    t->img1.img = NULL;
    free(t->img2.img);
    t->img2.img = NULL;

    return 1;
}

int gpu_transfer_in_previous_results(cross_correlate_task *t, id buffer)
{
    int32_t *payload  = ((int32_t* (*)(id, SEL))objc_msgSend)(buffer, sel_registerName("contents"));
    if (payload == NULL)
        return 0;

    for (int i=0;i<t->out_width*t->out_height*2;i++)
        payload[i] = t->correlated_points[i];

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
    ((void (*)(id, SEL, id, NSUInteger, NSUInteger))objc_msgSend)(computeEncoder, sel_registerName("setBuffer:offset:atIndex:"), dev->buffer_previous_result, 0, 2);
    ((void (*)(id, SEL, id, NSUInteger, NSUInteger))objc_msgSend)(computeEncoder, sel_registerName("setBuffer:offset:atIndex:"), dev->buffer_internal, 0, 3);
    ((void (*)(id, SEL, id, NSUInteger, NSUInteger))objc_msgSend)(computeEncoder, sel_registerName("setBuffer:offset:atIndex:"), dev->buffer_internal_int, 0, 4);
    ((void (*)(id, SEL, id, NSUInteger, NSUInteger))objc_msgSend)(computeEncoder, sel_registerName("setBuffer:offset:atIndex:"), dev->buffer_out, 0, 5);

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
    int32_t *correlated_points = ((int32_t* (*)(id, SEL))objc_msgSend)(dev->buffer_out, sel_registerName("contents"));
    float inv_scale = 1.0F/t->scale;
    if (correlated_points == NULL)
        return 0;
    for (int y=0;y<t->img1.height;y++)
    {
        for (int x=0;x<t->img1.width;x++)
        {
            size_t match_pos = y*t->img1.width + x;
            int x2 = correlated_points[match_pos*2];
            int y2 = correlated_points[match_pos*2+1];
            if(x2<0 || y2<0)
                continue;
            int out_point_pos = ((int)roundf(inv_scale*y))*t->out_width + (int)roundf(inv_scale*x);
            t->correlated_points[out_point_pos*2] = x2;
            t->correlated_points[out_point_pos*2+1] = y2;
        }
    }

    return 1;
}

void* gpu_correlate_cross_correlation_task(void *args)
{
    cross_correlate_task *t = args;
    metal_context* ctx = t->internal;
    int kernel_size = cybervision_crosscorrelation_kernel_size;
    int corridor_size = cybervision_crosscorrelation_corridor_size;
    int corridor_stripes = 2*corridor_size+1;
    int corridor_segment_length = cybervision_crosscorrelation_corridor_segment_length;
    int max_width = t->img1.width > t->img2.width ? t->img1.width:t->img2.width;
    int max_height = t->img1.height > t->img2.height ? t->img1.height:t->img2.height;
    int corridor_length = (t->img2.width>t->img2.height? t->img2.width:t->img2.height)-(kernel_size*2);
    int corridor_segments = corridor_length/cybervision_crosscorrelation_corridor_segment_length + 1;
    float progressbar_completed_percentage = 2.0F;

    shader_params params = {0};

    params.img1_width = t->img1.width;
    params.img1_height = t->img1.height;
    params.img2_width = t->img2.width;
    params.img2_height = t->img2.height;
    params.output_width = t->out_width;
    params.output_height = t->out_height;
    for(size_t i=0;i<3;i++)
        for(size_t j=0;j<3;j++)
            params.fundamental_matrix[i*4+j] = t->fundamental_matrix[i*3+j];
    params.scale = t->scale;
    params.iteration = t->iteration;
    params.kernel_size = cybervision_crosscorrelation_kernel_size;
    params.threshold = cybervision_crosscorrelation_threshold;
    params.neighbor_distance = cybervision_crosscorrelation_neighbor_distance;
    params.extend_range = cybervision_crosscorrelation_corridor_extend_range;
    params.min_range = cybervision_crosscorrelation_corridor_min_range;

    id autoreleasepool = init_autoreleasepool();
    if (!gpu_transfer_in_images(t, ctx->dev.buffer_img))
    {
        t->error = "Failed to transfer input images";
        goto cleanup;
    }

    if (!gpu_transfer_in_previous_results(t, ctx->dev.buffer_previous_result))
    {
        t->error = "Failed to transfer previous results";
        goto cleanup;
    }

    if (t->iteration > 0)
    {
        int y_limit = (int)ceilf((cybervision_crosscorrelation_neighbor_distance)/t->scale);
        int batch_size = cybervision_crosscorrelation_search_area_segment_length;
        if (!gpu_run_command(&ctx->dev, t->img1.width, t->img1.height, &params, ctx->dev.pso_prepare_initialdata_searchdata))
        {
            t->error = "Failed to run kernel (search area estimation initialization stage)";
            goto cleanup;
        }
        t->percent_complete = 2.0F;
        params.phase = 0;
        for(int y=-y_limit;y<=y_limit;y+=batch_size)
        {
            params.corridor_start = y;
            params.corridor_end = y+batch_size;
            params.corridor_end = params.corridor_end<y_limit?params.corridor_end:y_limit;
            if (!gpu_run_command(&ctx->dev, t->img1.width, t->img1.height, &params, ctx->dev.pso_prepare_searchdata))
            {
                t->error = "Failed to run kernel (search area estimation stage phase 0)";
                goto cleanup;
            }
            t->percent_complete = 2.0F+29.0F*(y+y_limit)/(2.0F*y_limit+1.0F);
        }
        t->percent_complete = 31.0F;
        params.phase = 1;
        for(int y=-y_limit;y<=y_limit;y+=batch_size)
        {
            params.corridor_start = y;
            params.corridor_end = y+batch_size;
            params.corridor_end = params.corridor_end<y_limit?params.corridor_end:y_limit;
            if (!gpu_run_command(&ctx->dev, t->img1.width, t->img1.height, &params, ctx->dev.pso_prepare_searchdata))
            {
                t->error = "Failed to run kernel (search area estimation stage phase 1)";
                goto cleanup;
            }
            t->percent_complete = 31.0F+29.0F*(y+y_limit)/(2.0F*y_limit+1.0F);
        }
        progressbar_completed_percentage = 60.0F;
    }
    params.phase = -1;

    if (!gpu_run_command(&ctx->dev, max_width, max_height, &params, ctx->dev.pso_prepare_initialdata_correlation))
    {
        t->error = "Failed to run kernel (initialization stage)";
        goto cleanup;
    }

    t->percent_complete = progressbar_completed_percentage;

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

            float corridor_complete = (float)(params.corridor_end - kernel_size) / (corridor_length);
            t->percent_complete = progressbar_completed_percentage + (100.0F-progressbar_completed_percentage)*(c+corridor_size + corridor_complete)/corridor_stripes;
        }
    }

    if (!gpu_transfer_out_image(t, &ctx->dev))
        t-> error = "Failed to read output image";

cleanup:
    drain_autoreleasepool(autoreleasepool);
    t->completed = 1;
    return NULL;
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

    dev->buffer_img = gpu_create_buffer(dev, sizeof(float)*(img1_pixels+img2_pixels), 0);
    dev->buffer_previous_result = gpu_create_buffer(dev, sizeof(int32_t)*(t->out_width*t->out_height*2), 0);
    dev->buffer_internal = gpu_create_buffer(dev, sizeof(float)*(img1_pixels*3+img2_pixels*2), 1);
    dev->buffer_internal_int = gpu_create_buffer(dev, sizeof(int32_t)*img1_pixels*3, 1);
    dev->buffer_out = gpu_create_buffer(dev, sizeof(int32_t)*img1_pixels*2, 0);
    if (dev->buffer_img == NULL || dev->buffer_previous_result == NULL || dev->buffer_internal == NULL || dev->buffer_internal_int == NULL || dev->buffer_out == NULL)
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
    return t->error == NULL;
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
