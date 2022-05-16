#include <objc/objc.h>
#include <objc/objc-runtime.h>

#include <stdlib.h>
#include <math.h>
#include <string.h>

#include <stdio.h>

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

typedef struct {
    id pool;
    id device;
    id library;
    id function_prepare_initialdata;
    id function_cross_correlate;
} metal_device;

typedef struct {
    metal_device dev;
    pthread_t thread;
} metal_context;

const char* nserror_localized_description(id error)
{
    id errorNSString = ((id (*)(id, SEL))objc_msgSend)(error, sel_registerName("localizedDescription"));
    const char* errorCStr = ((const char* (*)(id, SEL, NSUInteger))objc_msgSend)(errorNSString, sel_registerName("cStringUsingEncoding:"), 5);
    return errorCStr;
}

int gpu_init_device(metal_device *dev)
{
	SEL allocSel = sel_registerName("alloc");
    SEL initSel = sel_registerName("init");
	SEL autoreleaseSel = sel_registerName("autorelease");

    Class NSAutoreleasePoolClass = objc_getClass("NSAutoreleasePool");
	id poolAlloc = ((id (*)(Class, SEL))objc_msgSend)(NSAutoreleasePoolClass, allocSel);
	dev->pool = ((id (*)(id, SEL))objc_msgSend)(poolAlloc, initSel);

    // TODO: choose the best device?
    dev->device = MTLCreateSystemDefaultDevice();
    if (dev->device == NULL)
        return 0;
    ((id (*)(id, SEL))objc_msgSend)(dev->device, autoreleaseSel);

    return 1;
}

int gpu_init_library(metal_device *dev)
{
	SEL allocSel = sel_registerName("alloc");
    SEL releaseSel = sel_registerName("release");
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
    ((id (*)(id, SEL, NSUInteger))objc_msgSend)(compileOptions, sel_registerName("setLanguageVersion:"), METAL_2_2);

    id error = NULL;
    dev->library = ((id (*)(id, SEL, id, id, id*))objc_msgSend)(dev->device, sel_registerName("newLibraryWithSource:options:error:"), sourceString, compileOptions, &error);
    ((id (*)(id, SEL))objc_msgSend)(compileOptions, releaseSel);
    if (error != NULL)
        return 0;
    if (dev->library == NULL)
        return 0;

    id prepare_initialdataFunctionName = ((id (*)(Class, SEL, char*, NSUInteger))objc_msgSend)(NSStringClass, sel_registerName("stringWithCString:encoding:"), "prepare_initialdata", 5);
    id cross_correlateFunctionName = ((id (*)(Class, SEL, char*, NSUInteger))objc_msgSend)(NSStringClass, sel_registerName("stringWithCString:encoding:"), "cross_correlate", 5);
    if (prepare_initialdataFunctionName == NULL || cross_correlateFunctionName == NULL)
        return 0;

    dev->function_prepare_initialdata = ((id (*)(id, SEL, id))objc_msgSend)(dev->library, sel_registerName("newFunctionWithName:"), prepare_initialdataFunctionName);
    dev->function_cross_correlate = ((id (*)(id, SEL, id))objc_msgSend)(dev->library, sel_registerName("newFunctionWithName:"), cross_correlateFunctionName);
    if (dev->function_prepare_initialdata == NULL || dev->function_cross_correlate == NULL)
        return 0;

    return 1;
}

int gpu_prepare_device(metal_device *dev, const char** error)
{
    *error = NULL;
    if (!gpu_init_library(dev))
    {
        *error = "Failed to init library";
        return 0;
    }

    return 1;
}

THREAD_FUNCTION gpu_correlate_cross_correlation_task(void *args)
{
    cross_correlate_task *t = args;
    metal_context *ctx = t->internal;

    if (!gpu_init_device(&ctx->dev))
    {
        t->error = "Failed to initialize Metal device";
        goto cleanup;
    }
    if (!gpu_prepare_device(&ctx->dev, &t->error))
    {
        goto cleanup;
    }

cleanup:
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

int gpu_free_device(metal_device *dev)
{
    printf("started cleanup\n");
    if (dev->device != NULL)
        ((void (*)(id, SEL))objc_msgSend)(dev->device, sel_registerName("release"));
    if (dev->library != NULL)
        ((void (*)(id, SEL))objc_msgSend)(dev->function_cross_correlate, sel_registerName("release"));
    if (dev->function_prepare_initialdata != NULL)
        ((void (*)(id, SEL))objc_msgSend)(dev->function_prepare_initialdata, sel_registerName("release"));
    if (dev->function_cross_correlate != NULL)
        ((void (*)(id, SEL))objc_msgSend)(dev->function_cross_correlate, sel_registerName("release"));
    ((void (*)(id, SEL))objc_msgSend)(dev->pool, sel_registerName("drain"));
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

    if (!gpu_free_device(&ctx->dev))
        result = 0;
    
    free(ctx);
    t->internal = NULL;
    return result;
}
