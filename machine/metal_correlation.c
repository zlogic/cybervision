#include <objc/objc.h>
#include <objc/objc-runtime.h>

#include <stdlib.h>
#include <math.h>
#include <string.h>

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
# include <pthread.h>
# define THREAD_FUNCTION void*
# define THREAD_RETURN_VALUE NULL
#elif defined(_WIN32)
# include "win32/pthread.h"
#define THREAD_FUNCTION DWORD WINAPI
# define THREAD_RETURN_VALUE 1
#else
# error "pthread is required"
#endif

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
    id pso_prepare_initialdata, pso_cross_correlate;
    id command_queue;
    id buffer_img, buffer_internal, buffer_out;
} metal_device;

typedef struct {
    pthread_t thread;
} metal_context;

#define CORRIDOR_SEGMENT_LENGTH 1024

typedef struct {
    int32_t img1_width;
    int32_t img1_height;
    int32_t img2_width;
    int32_t img2_height;
    float dir_x, dir_y;
    int32_t corridor_offset;
    int32_t corridor_segment;
    int32_t kernel_size;
    float threshold;
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
    char *shaders_str = malloc(sizeof(char)*shaders_correlation_metal_len+1);
    memcpy(shaders_str, shaders_correlation_metal, shaders_correlation_metal_len);
    shaders_str[shaders_correlation_metal_len] = 0;

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
    if (dev->library == NULL)
        return 0;
    if (error != NULL)
        return 0;

    id prepare_initialdataFunctionName = ((id (*)(Class, SEL, char*, NSUInteger))objc_msgSend)(NSStringClass, sel_registerName("stringWithCString:encoding:"), "prepare_initialdata", 5);
    if (prepare_initialdataFunctionName == NULL)
        return 0;
    id cross_correlateFunctionName = ((id (*)(Class, SEL, char*, NSUInteger))objc_msgSend)(NSStringClass, sel_registerName("stringWithCString:encoding:"), "cross_correlate", 5);
    if (cross_correlateFunctionName == NULL)
        return 0;

    id function_prepare_initialdata = ((id (*)(id, SEL, id))objc_msgSend)(dev->library, sel_registerName("newFunctionWithName:"), prepare_initialdataFunctionName);
    if (function_prepare_initialdata == NULL)
        return 0;
    ((void (*)(id, SEL))objc_msgSend)(function_prepare_initialdata, autoreleaseSel);
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

int gpu_prepare_device(cross_correlate_task *t, metal_device *dev, const char** error)
{
    *error = NULL;
    if (!gpu_init_functions(dev))
    {
        *error = "Failed to init function";
        return 0;
    }

    if (!gpu_init_queue(dev))
    {
        *error = "Failed to init queue";
        return 0;
    }

    dev->buffer_img = gpu_create_buffer(dev, sizeof(float)*(t->img1.width*t->img1.height + t->img2.width*t->img2.height), 0);
    dev->buffer_internal = gpu_create_buffer(dev, sizeof(float)*(t->img1.width*t->img1.height*3 + t->img2.width*t->img2.height*2), 1);
    dev->buffer_out = gpu_create_buffer(dev, sizeof(float)*t->img1.width*t->img1.height, 0);
    if (dev->buffer_img == NULL || dev->buffer_internal == NULL || dev->buffer_out == NULL)
    {
        *error = "Failed to create buffers";
        return 0;
    }

    return 1;
}

int gpu_transfer_in_images(correlation_image img1, correlation_image img2, id buffer)
{
    float *payload = ((float* (*)(id, SEL))objc_msgSend)(buffer, sel_registerName("contents"));
    if (payload == NULL)
        return 0;

    for (int i=0;i<img1.width*img1.height;i++)
        payload[i] = (unsigned char)img1.img[i];
    payload += img1.width*img1.height;
    for (int i=0;i<img2.width*img2.height;i++)
        payload[i] = (unsigned char)img2.img[i];

    return 1;
}

int gpu_run_command(metal_device *dev, int max_width, int max_height, shader_params *params, int initial_run)
{
    id commandBuffer = ((id (*)(id, SEL))objc_msgSend)(dev->command_queue, sel_registerName("commandBuffer"));
    if (commandBuffer == NULL)
        return 0;

    id computeEncoder = ((id (*)(id, SEL))objc_msgSend)(commandBuffer, sel_registerName("computeCommandEncoder"));
    if (computeEncoder == NULL)
        return 0;

    ((void (*)(id, SEL, id))objc_msgSend)(computeEncoder, sel_registerName("setComputePipelineState:"), initial_run? dev->pso_prepare_initialdata : dev->pso_cross_correlate);
    ((void (*)(id, SEL, void*, NSUInteger, NSUInteger))objc_msgSend)(computeEncoder, sel_registerName("setBytes:length:atIndex:"), params, sizeof(shader_params), 0);
    ((void (*)(id, SEL, id, NSUInteger, NSUInteger))objc_msgSend)(computeEncoder, sel_registerName("setBuffer:offset:atIndex:"), dev->buffer_img, 0, 1);
    ((void (*)(id, SEL, id, NSUInteger, NSUInteger))objc_msgSend)(computeEncoder, sel_registerName("setBuffer:offset:atIndex:"), dev->buffer_internal, 0, 2);
    ((void (*)(id, SEL, id, NSUInteger, NSUInteger))objc_msgSend)(computeEncoder, sel_registerName("setBuffer:offset:atIndex:"), dev->buffer_out, 0, 3);

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
    if (output_points == NULL)
        return 0;
    for (int i=0;i<t->img1.width*t->img1.height;i++)
        t->out_points[i] = output_points[i];

    return 1;
}

THREAD_FUNCTION gpu_correlate_cross_correlation_task(void *args)
{
    cross_correlate_task *t = args;
    int kernel_size = t->kernel_size;
    int corridor_size = t->corridor_size;
    int corridor_stripes = 2*t->corridor_size+1;
    int max_width = t->img1.width > t->img2.width ? t->img1.width:t->img2.width;
    int max_height = t->img1.height > t->img2.height ? t->img1.height:t->img2.height;
    int corridor_length = (fabs(t->dir_y)>fabs(t->dir_x)? t->img2.height:t->img2.width) - 2*kernel_size;
    int corridor_segments = corridor_length/CORRIDOR_SEGMENT_LENGTH + 1;

    metal_device dev = {0};
    shader_params params = {0};

    params.img1_width = t->img1.width;
    params.img1_height = t->img1.height;
    params.img2_width = t->img2.width;
    params.img2_height = t->img2.height;
    params.dir_x = t->dir_x;
    params.dir_y = t->dir_y;
    params.corridor_offset = 0;
    params.corridor_segment = 0;
    params.kernel_size = t->kernel_size;
    params.threshold = t->threshold;

    id autoreleasepool = init_autoreleasepool();
    if (!gpu_init_device(&dev))
    {
        t->error = "Failed to initialize Metal device";
        goto cleanup;
    }
    if (!gpu_prepare_device(t, &dev, &t->error))
    {
        goto cleanup;
    }
    if (!gpu_transfer_in_images(t->img1, t->img2, dev.buffer_img))
    {
        t->error = "Failed to transfer images into device buffer";
        goto cleanup;
    }

    if (!gpu_run_command(&dev, max_width, max_height, &params, 1))
    {
        t->error = "Failed to run initialization kernel";
        goto cleanup;
    }

    t->percent_complete = 2.0F;

    for (int c=-corridor_size;c<=corridor_size;c++)
    {
        for (int l=0;l<corridor_segments;l++)
        {
            float corridor_complete = (float)(l)*CORRIDOR_SEGMENT_LENGTH/corridor_length;
            params.corridor_offset = c;
            params.corridor_segment = l;

            if (!gpu_run_command(&dev, t->img1.width, t->img1.height, &params, 0))
            {
                t->error = "Failed to run correlation kernel";
                goto cleanup;
            }

            t->percent_complete = 2.0F + 98.0F*(c+corridor_size + corridor_complete)/corridor_stripes;
        }
    }

    if (!gpu_transfer_out_image(t, &dev))
        t-> error = "Failed to read output image";

cleanup:
    drain_autoreleasepool(autoreleasepool);
    t->completed = 1;
    return THREAD_RETURN_VALUE;
}

int gpu_correlation_cross_correlate_start(cross_correlate_task *t)
{
    metal_context* ctx = malloc(sizeof(metal_context));
    t->internal = ctx;
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
    pthread_join(ctx->thread, NULL);
    
    free(ctx);
    t->internal = NULL;
    return result;
}
