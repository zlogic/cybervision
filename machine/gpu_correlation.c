#include <vulkan/vulkan.h>
#include <stdlib.h>
#include "gpu_correlation.h"

typedef struct {
    VkInstance instance;
    VkDebugReportCallbackEXT debugReportCallback;
    VkPhysicalDevice physicalDevice;
    VkDevice device;
} vulkan_context;

VkResult gpu_vk_create(vulkan_context* ctx)
{
    VkApplicationInfo applicationInfo;
    VkInstanceCreateInfo createInfo;
    
    applicationInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    applicationInfo.pApplicationName = "Cybervision";
    applicationInfo.applicationVersion = 0;
    applicationInfo.pEngineName = "cybervision";
    applicationInfo.engineVersion = 0;
    applicationInfo.apiVersion = VK_API_VERSION_1_0;
    
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.flags = 0;
    createInfo.pApplicationInfo = &applicationInfo;
    
    createInfo.enabledLayerCount = 0;
    createInfo.ppEnabledLayerNames = NULL;
    createInfo.enabledExtensionCount = 0;
    createInfo.ppEnabledExtensionNames = NULL;

    return vkCreateInstance(&createInfo, NULL, &ctx->instance);
}

int gpu_vk_find_device(vulkan_context* ctx)
{
    uint32_t deviceCount;
    VkPhysicalDevice *devices = NULL;
    int result = 0;
    
    if(vkEnumeratePhysicalDevices(ctx->instance, &deviceCount, NULL) != VK_SUCCESS)
        goto cleanup;
    if (deviceCount == 0)
        goto cleanup;

    devices = malloc(sizeof(VkPhysicalDevice)*deviceCount);
    if (vkEnumeratePhysicalDevices(ctx->instance, &deviceCount, devices) != VK_SUCCESS)
        goto cleanup;

    for(uint32_t i=0;i<deviceCount;i++)
    {
        // TODO: choose the best device (or devices)
        VkPhysicalDeviceProperties deviceProperties;
        VkPhysicalDeviceFeatures deviceFeatures;
        vkGetPhysicalDeviceProperties(devices[i], &deviceProperties);
        vkGetPhysicalDeviceFeatures(devices[i], &deviceFeatures);
        // For now, choose the first device
        ctx->physicalDevice = devices[i];
        break;
    }

    result = 1;
cleanup:
    if (devices != NULL)
        free(devices);
    return result;
}


int correlation_gpu_init(gpu_context *gc)
{
    vulkan_context* ctx = malloc(sizeof(vulkan_context));
    gc->internal = ctx;
    if (gpu_vk_create(ctx) != VK_SUCCESS)
        return 0;

    if (!gpu_vk_find_device(ctx))
        return 0;
    return 1;
}

int correlation_gpu_free(gpu_context *gc)
{
    vulkan_context* ctx;
    if (gc == NULL || gc->internal == NULL)
        return 1;
    ctx = gc->internal;
    vkDestroyDevice(ctx->device, NULL);
    vkDestroyInstance(ctx->instance, NULL);	
    free(ctx);
    gc->internal = NULL;
    return 1;
}
