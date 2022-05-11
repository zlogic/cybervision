#include <vulkan/vulkan.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "gpu_correlation.h"
#include "shaders_spv.h"

typedef struct {
    VkPhysicalDevice physicalDevice;
    VkDevice device;
    VkQueue queue;
    uint32_t queueFamilyIndex;

    VkBuffer params_buffer, img1_buffer, img2_buffer, out_buffer;
    VkDeviceMemory params_bufferMemory, img1_bufferMemory, img2_bufferMemory, out_bufferMemory;

    VkDescriptorSetLayout descriptorSetLayout;
    VkDescriptorPool descriptorPool;
    VkDescriptorSet descriptorSet;

    VkShaderModule computeShaderModule;
    VkPipeline pipeline;
    VkPipelineLayout pipelineLayout;
    VkCommandPool commandPool;
    VkCommandBuffer commandBuffer;
} vulkan_device;

typedef struct {
    VkInstance instance;

    vulkan_device dev;
} vulkan_context;

typedef struct {
    int32_t img1_width;
    int32_t img1_height;
    int32_t img2_width;
    int32_t img2_height;
    float dir_x, dir_y;
    int32_t corridor_size;
    int32_t kernel_size;
    float threshold;
} shader_params;

VkResult gpu_vk_create(vulkan_context* ctx)
{
    VkApplicationInfo applicationInfo = {0};
    VkInstanceCreateInfo createInfo = {0};
    
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

int gpu_find_compute_queue_index(vulkan_device *device, uint32_t *index)
{
    uint32_t queueFamilyCount;
    VkQueueFamilyProperties *queueFamilies = NULL;
    int result = 0;

    vkGetPhysicalDeviceQueueFamilyProperties(device->physicalDevice, &queueFamilyCount, NULL);

    queueFamilies = malloc(sizeof(VkQueueFamilyProperties)*queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(device->physicalDevice, &queueFamilyCount, queueFamilies);
    
    for(uint32_t i=0;i<queueFamilyCount;i++)
    {
        VkQueueFamilyProperties props = queueFamilies[i];
        if (props.queueCount > 0 && (props.queueFlags & VK_QUEUE_COMPUTE_BIT))
        {
            *index = i;
            result = 1;
            break;
        }
    }

    free(queueFamilies);
    return result;
}

int gpu_create_device(vulkan_device *dev)
{
    VkDeviceQueueCreateInfo queueCreateInfo = {};
    VkDeviceCreateInfo deviceCreateInfo = {};
    VkPhysicalDeviceFeatures deviceFeatures = {};

    float queuePriorities[] = {1.0f};

    queueCreateInfo.pNext = NULL;
    queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    if (!gpu_find_compute_queue_index(dev, &dev->queueFamilyIndex))
        return 0;
    queueCreateInfo.queueFamilyIndex = dev->queueFamilyIndex;
    queueCreateInfo.queueCount = 1;
    queueCreateInfo.pQueuePriorities = queuePriorities;

    deviceCreateInfo.pNext = NULL;
    deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;
    deviceCreateInfo.queueCreateInfoCount = 1;
    deviceCreateInfo.pEnabledFeatures = &deviceFeatures;

    if (vkCreateDevice(dev->physicalDevice, &deviceCreateInfo, NULL, &dev->device) != VK_SUCCESS)
        return 0;

    vkGetDeviceQueue(dev->device, queueCreateInfo.queueFamilyIndex, 0, &dev->queue);
    return 1;
}

int gpu_find_device(vulkan_context* ctx)
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
        VkPhysicalDevice physicalDevice = devices[i];
        vkGetPhysicalDeviceProperties(physicalDevice, &deviceProperties);
        vkGetPhysicalDeviceFeatures(physicalDevice, &deviceFeatures);
        if (deviceProperties.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU)
            continue;
        // For now, choose the first device
        ctx->dev.physicalDevice = physicalDevice;
        if (!gpu_create_device(&ctx->dev))
            goto cleanup;
        break;
    }

    result = 1;
cleanup:
    if (devices != NULL)
        free(devices);
    return result;
}

int gpu_create_buffer(vulkan_device *dev, VkDeviceSize bufferSize, VkBuffer *buffer, VkDeviceMemory *bufferMemory)
{
    VkBufferCreateInfo bufferCreateInfo = {0};
    VkMemoryRequirements memoryRequirements;
    VkMemoryAllocateInfo allocateInfo = {0};
    VkPhysicalDeviceMemoryProperties memoryProperties;

    bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferCreateInfo.size = bufferSize;
    bufferCreateInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(dev->device, &bufferCreateInfo, NULL, buffer) != VK_SUCCESS)
        return 0;
    
    vkGetBufferMemoryRequirements(dev->device, *buffer, &memoryRequirements);

    allocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocateInfo.allocationSize = memoryRequirements.size;
    allocateInfo.memoryTypeIndex = VK_MAX_MEMORY_TYPES;

    vkGetPhysicalDeviceMemoryProperties(dev->physicalDevice, &memoryProperties);
    for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; ++i)
    {
        if ((memoryProperties.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) == 0 ||
            (memoryProperties.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) == 0 ||
            memoryProperties.memoryHeaps[memoryProperties.memoryTypes[i].heapIndex].size < bufferSize)
            continue;
        allocateInfo.memoryTypeIndex = i;
        break;
    }

    if (allocateInfo.memoryTypeIndex == VK_MAX_MEMORY_TYPES)
        return 0;

    if (vkAllocateMemory(dev->device, &allocateInfo, NULL, bufferMemory) != VK_SUCCESS)
        return 0;
    if (vkBindBufferMemory(dev->device, *buffer, *bufferMemory, 0) != VK_SUCCESS)
        return 0;
    
    return 1;
}

int gpu_create_descriptor_set_layout(vulkan_device *dev)
{
    VkDescriptorSetLayoutBinding descriptorSetLayoutBindings[4];
    VkDescriptorSetLayoutBinding* descriptorSetLayoutBinding = descriptorSetLayoutBinding;

    for(uint32_t i=0;i<4;i++)
    {
        VkDescriptorSetLayoutBinding descriptorSetLayoutBinding = {0};
        descriptorSetLayoutBinding.binding = i;
        descriptorSetLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descriptorSetLayoutBinding.descriptorCount = 1;
        descriptorSetLayoutBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        descriptorSetLayoutBindings[i] = descriptorSetLayoutBinding;
    }

    VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = {};
    descriptorSetLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    descriptorSetLayoutCreateInfo.bindingCount = 4;
    descriptorSetLayoutCreateInfo.pBindings = descriptorSetLayoutBindings; 

    if (vkCreateDescriptorSetLayout(dev->device, &descriptorSetLayoutCreateInfo, NULL, &dev->descriptorSetLayout) != VK_SUCCESS)
        return 0;
    return 1;
}

int gpu_create_descriptor_set(vulkan_device *dev)
{
    VkDescriptorPoolSize descriptorPoolSize = {0};
    VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = {0};
    VkDescriptorSetAllocateInfo descriptorSetAllocateInfo = {0};
    VkWriteDescriptorSet writeDescriptorSets[4];

    VkDescriptorBufferInfo descriptorBufferInfo[4] = {0};

    descriptorPoolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorPoolSize.descriptorCount = 4;

    descriptorPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    descriptorPoolCreateInfo.maxSets = 1;
    descriptorPoolCreateInfo.poolSizeCount = 1;
    descriptorPoolCreateInfo.pPoolSizes = &descriptorPoolSize;

    if (vkCreateDescriptorPool(dev->device, &descriptorPoolCreateInfo, NULL, &dev->descriptorPool) != VK_SUCCESS)
        return 0;

    descriptorSetAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO; 
    descriptorSetAllocateInfo.descriptorPool = dev->descriptorPool;
    descriptorSetAllocateInfo.descriptorSetCount = 1;
    descriptorSetAllocateInfo.pSetLayouts = &dev->descriptorSetLayout;

    if (vkAllocateDescriptorSets(dev->device, &descriptorSetAllocateInfo, &dev->descriptorSet) != VK_SUCCESS)
        return 0;

    descriptorBufferInfo[0].buffer = dev->params_buffer;
    descriptorBufferInfo[0].offset = 0;
    descriptorBufferInfo[0].range = VK_WHOLE_SIZE;

    descriptorBufferInfo[1].buffer = dev->img1_buffer;
    descriptorBufferInfo[1].offset = 0;
    descriptorBufferInfo[1].range = VK_WHOLE_SIZE;

    descriptorBufferInfo[2].buffer = dev->img2_buffer;
    descriptorBufferInfo[2].offset = 0;
    descriptorBufferInfo[2].range = VK_WHOLE_SIZE;

    descriptorBufferInfo[3].buffer = dev->out_buffer;
    descriptorBufferInfo[3].offset = 0;
    descriptorBufferInfo[3].range = VK_WHOLE_SIZE;

    for(uint32_t i=0;i<4;i++)
    {
        VkWriteDescriptorSet writeDescriptorSet = {0};
        writeDescriptorSet.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writeDescriptorSet.dstSet = dev->descriptorSet;
        writeDescriptorSet.dstBinding = i;
        writeDescriptorSet.descriptorCount = 1;
        writeDescriptorSet.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writeDescriptorSet.pBufferInfo = &descriptorBufferInfo[i];
        writeDescriptorSets[i] = writeDescriptorSet;
    }

    vkUpdateDescriptorSets(dev->device, 4, writeDescriptorSets, 0, NULL);
    return 1;
}

int gpu_create_compute_pipeline(vulkan_device *dev)
{
    VkShaderModuleCreateInfo createInfo = {0};
    VkPipelineShaderStageCreateInfo shaderStageCreateInfo = {0};
    VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = {0};
    VkComputePipelineCreateInfo pipelineCreateInfo = {0};

    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.pCode = shaders_spv;
    createInfo.codeSize = sizeof(shaders_spv);

    if (vkCreateShaderModule(dev->device, &createInfo, NULL, &dev->computeShaderModule) != VK_SUCCESS)
        return 0;

    shaderStageCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shaderStageCreateInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    shaderStageCreateInfo.module = dev->computeShaderModule;
    shaderStageCreateInfo.pName = "main";

    pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutCreateInfo.setLayoutCount = 1;
    pipelineLayoutCreateInfo.pSetLayouts = &dev->descriptorSetLayout; 
    if (vkCreatePipelineLayout(dev->device, &pipelineLayoutCreateInfo, NULL, &dev->pipelineLayout) != VK_SUCCESS)
        return 0;
    
    pipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineCreateInfo.stage = shaderStageCreateInfo;
    pipelineCreateInfo.layout = dev->pipelineLayout;

    if (vkCreateComputePipelines(dev->device, VK_NULL_HANDLE, 1, &pipelineCreateInfo, NULL, &dev->pipeline) != VK_SUCCESS)
        return 0;
    return 1;
}

int gpu_create_command_buffer(vulkan_device *dev, int w1, int h1)
{
    VkCommandPoolCreateInfo commandPoolCreateInfo = {0};
    VkCommandBufferAllocateInfo commandBufferAllocateInfo = {0};

    commandPoolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    commandPoolCreateInfo.flags = 0;
    commandPoolCreateInfo.queueFamilyIndex = dev->queueFamilyIndex;
    if (vkCreateCommandPool(dev->device, &commandPoolCreateInfo, NULL, &dev->commandPool) != VK_SUCCESS)
        return 0;

    commandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    commandBufferAllocateInfo.commandPool = dev->commandPool;
    commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    commandBufferAllocateInfo.commandBufferCount = 1;
    if (vkAllocateCommandBuffers(dev->device, &commandBufferAllocateInfo, &dev->commandBuffer) != VK_SUCCESS)
        return 0;

    VkCommandBufferBeginInfo beginInfo = {};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(dev->commandBuffer, &beginInfo) != VK_SUCCESS)
        return 0;

    vkCmdBindPipeline(dev->commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, dev->pipeline);
    vkCmdBindDescriptorSets(dev->commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, dev->pipelineLayout, 0, 1, &dev->descriptorSet, 0, NULL);

    vkCmdDispatch(dev->commandBuffer, (uint32_t)ceilf(w1/16.0f), (uint32_t)ceilf(h1/16.0f), 1);

    if (vkEndCommandBuffer(dev->commandBuffer) != VK_SUCCESS)
        return 0;
    return 1;
}

int gpu_run_command_buffer(vulkan_device *dev)
{
    VkSubmitInfo submitInfo = {0};

    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &dev->commandBuffer;

    if (vkQueueSubmit(dev->queue, 1, &submitInfo, VK_NULL_HANDLE) != VK_SUCCESS)
        return 0;

    if (vkQueueWaitIdle(dev->queue) != VK_SUCCESS)
        return 0;
    return 1;
}

int gpu_transfer_in_image(correlation_image img, VkDevice device, VkDeviceMemory bufferMemory)
{
    float *payload;

    if (vkMapMemory(device, bufferMemory, 0, VK_WHOLE_SIZE, 0, (void*)&payload) != VK_SUCCESS)
        return 0;
    for (int i=0;i<img.width*img.height;i++)
        payload[i] = (unsigned char)img.img[i];
    
    vkUnmapMemory(device, bufferMemory);
    return 1;
}

int gpu_transfer_in_params(cross_correlate_task *t, vulkan_device *dev)
{
    shader_params *payload;

    if (vkMapMemory(dev->device, dev->params_bufferMemory, 0, VK_WHOLE_SIZE, 0, (void*)&payload) != VK_SUCCESS)
        return 0;

    payload->img1_width = t->img1.width;
    payload->img1_height = t->img1.height;
    payload->img2_width = t->img2.width;
    payload->img2_height = t->img2.height;
    payload->dir_x = t->dir_x;
    payload->dir_y = t->dir_y;
    payload->corridor_size = t->corridor_size;
    payload->kernel_size = t->kernel_size;
    payload->threshold = t->threshold;

    vkUnmapMemory(dev->device, dev->params_bufferMemory);
    return 1;
}

int gpu_transfer_out_image(cross_correlate_task *t, vulkan_device *dev)
{
    float *payload;
    if (vkMapMemory(dev->device, dev->out_bufferMemory, 0, VK_WHOLE_SIZE, 0, (void*)&payload) != VK_SUCCESS)
        return 0;
    for (int i=0;i<t->img1.width*t->img1.height;i++)
        t->out_points[i] = payload[i];
    
    vkUnmapMemory(dev->device, dev->out_bufferMemory);
    return 1;
}

int gpu_correlation_cross_correlate_start(cross_correlate_task *t)
{
    vulkan_context* ctx = malloc(sizeof(vulkan_context));
    t->internal = ctx;
    t->completed = 1;

    if (gpu_vk_create(ctx) != VK_SUCCESS)
        return 0;
    if (!gpu_find_device(ctx))
        return 0;

    if (!gpu_create_buffer(&ctx->dev, sizeof(shader_params), &ctx->dev.params_buffer, &ctx->dev.params_bufferMemory))
        return 0;
    if (!gpu_create_buffer(&ctx->dev, sizeof(float)*t->img1.width*t->img1.height, &ctx->dev.img1_buffer, &ctx->dev.img1_bufferMemory))
        return 0;
    if (!gpu_create_buffer(&ctx->dev, sizeof(float)*t->img2.width*t->img2.height, &ctx->dev.img2_buffer, &ctx->dev.img2_bufferMemory))
        return 0;
    if (!gpu_create_buffer(&ctx->dev, sizeof(float)*t->img1.width*t->img1.height, &ctx->dev.out_buffer, &ctx->dev.out_bufferMemory))
        return 0;

    if (!gpu_transfer_in_image(t->img1, ctx->dev.device, ctx->dev.img1_bufferMemory))
        return 0;
    if (!gpu_transfer_in_image(t->img2, ctx->dev.device, ctx->dev.img2_bufferMemory))
        return 0;
    if (!gpu_transfer_in_params(t, &ctx->dev))
        return 0;

    if (!gpu_create_descriptor_set_layout(&ctx->dev))
        return 0;
    if (!gpu_create_descriptor_set(&ctx->dev))
        return 0;

    if (!gpu_create_compute_pipeline(&ctx->dev))
        return 0;

    if (!gpu_create_command_buffer(&ctx->dev, t->img1.width, t->img1.height))
        return 0;
    if (!gpu_run_command_buffer(&ctx->dev))
        return 0;

    if (!gpu_transfer_out_image(t, &ctx->dev))
        return 0;

    return 1;
}

int gpu_correlation_cross_correlate_complete(cross_correlate_task *t)
{
    vulkan_context* ctx;
    VkDevice device;
    if (t == NULL || t->internal == NULL)
        return 1;
    ctx = t->internal;
    device = ctx->dev.device;
    vkFreeMemory(device, ctx->dev.params_bufferMemory, NULL);
    vkFreeMemory(device, ctx->dev.img1_bufferMemory, NULL);
    vkFreeMemory(device, ctx->dev.img2_bufferMemory, NULL);
    vkFreeMemory(device, ctx->dev.out_bufferMemory, NULL);
    vkDestroyBuffer(device, ctx->dev.params_buffer, NULL);
    vkDestroyBuffer(device, ctx->dev.img1_buffer, NULL);
    vkDestroyBuffer(device, ctx->dev.img2_buffer, NULL);
    vkDestroyBuffer(device, ctx->dev.out_buffer, NULL);
    vkDestroyShaderModule(device, ctx->dev.computeShaderModule, NULL);
    vkDestroyDescriptorPool(device, ctx->dev.descriptorPool, NULL);
    vkDestroyDescriptorSetLayout(device, ctx->dev.descriptorSetLayout, NULL);
    vkDestroyPipelineLayout(device, ctx->dev.pipelineLayout, NULL);
    vkDestroyPipeline(device, ctx->dev.pipeline, NULL);
    vkDestroyCommandPool(device, ctx->dev.commandPool, NULL);
    vkDestroyDevice(device, NULL);
    vkDestroyInstance(ctx->instance, NULL);	
    free(ctx);
    t->internal = NULL;
    return 1;
}
