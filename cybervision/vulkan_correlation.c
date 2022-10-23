#include <vulkan/vulkan.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <pthread.h>

#include "configuration.h"
#include "gpu_correlation.h"
#include "shaders_spv.h"

typedef struct {
    VkPhysicalDevice physicalDevice;
    VkDevice device;
    VkQueue queue;
    uint32_t queueFamilyIndex;

    VkBuffer params_buffer, img_buffer, previous_result_buffer, internal_buffer, internal_int_buffer, out_buffer;
    VkDeviceMemory params_bufferMemory, img_bufferMemory, previous_result_bufferMemory, internal_bufferMemory, internal_int_bufferMemory, out_bufferMemory;

    VkDescriptorSetLayout descriptorSetLayout;
    VkDescriptorPool descriptorPool;
    VkDescriptorSet descriptorSet;

    VkShaderModule computeShaderModule;
    VkPipeline pipeline;
    VkPipelineLayout pipelineLayout;
    VkFence fence;
    VkCommandPool commandPool;
} vulkan_device;

typedef struct {
    VkInstance instance;

    vulkan_device dev;

    pthread_t thread;
    int thread_started;
} vulkan_context;

typedef struct {
    int32_t img1_width;
    int32_t img1_height;
    int32_t img2_width;
    int32_t img2_height;
    int32_t output_width;
    int32_t output_height;
    int32_t pad[2]; // std430 layout requires matrices to be aligned to 16 bytes
    float fundamental_matrix[4*3]; // in std430 matrices are column-major and each column is aligned to 4-component vectors
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
    VkDeviceQueueCreateInfo queueCreateInfo = {0};
    VkDeviceCreateInfo deviceCreateInfo = {0};
    VkPhysicalDeviceFeatures deviceFeatures = {0};

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
    VkPhysicalDevice best_device;
    float best_device_score = -1;
    
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
        float device_score = 0.0F;
        vkGetPhysicalDeviceProperties(physicalDevice, &deviceProperties);
        vkGetPhysicalDeviceFeatures(physicalDevice, &deviceFeatures);
        switch(deviceProperties.deviceType)
        {
            case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
                device_score = 3.0F;
                break;
            case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:
                device_score = 2.0F;
                break;
            case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
                device_score = 1.0F;
                break;
            default:
                device_score = 0.0F;
                break;
        }
        if (device_score > best_device_score)
        {
            best_device_score = device_score;
            best_device = physicalDevice;
        }
    }

    if (best_device_score>=0.0F)
    {
        ctx->dev.physicalDevice = best_device;
        if (!gpu_create_device(&ctx->dev))
            goto cleanup;
    }
    else
    {
        goto cleanup;
    }

    result = 1;
cleanup:
    if (devices != NULL)
        free(devices);
    return result;
}

int gpu_create_buffer(vulkan_device *dev, VkDeviceSize bufferSize, VkBuffer *buffer, VkDeviceMemory *bufferMemory, int gpuonly)
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
        if (memoryProperties.memoryHeaps[memoryProperties.memoryTypes[i].heapIndex].size < bufferSize)
            continue;
        if (gpuonly)
        {
            if ((memoryProperties.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) == 0)
                continue;
        }
        else
        {
            if ((memoryProperties.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) == 0 ||
                (memoryProperties.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) == 0)
                continue;
        }

        allocateInfo.memoryTypeIndex = i;
        if (vkAllocateMemory(dev->device, &allocateInfo, NULL, bufferMemory) != VK_SUCCESS)
            continue;
        if (vkBindBufferMemory(dev->device, *buffer, *bufferMemory, 0) != VK_SUCCESS)
            return 0;
        return 1;
    }

    return 0;
}

int gpu_create_descriptor_set_layout(vulkan_device *dev)
{
    VkDescriptorSetLayoutBinding descriptorSetLayoutBindings[6];
    VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = {0};

    for(uint32_t i=0;i<6;i++)
    {
        VkDescriptorSetLayoutBinding descriptorSetLayoutBinding = {0};
        descriptorSetLayoutBinding.binding = i;
        descriptorSetLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descriptorSetLayoutBinding.descriptorCount = 1;
        descriptorSetLayoutBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        descriptorSetLayoutBindings[i] = descriptorSetLayoutBinding;
    }

    descriptorSetLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    descriptorSetLayoutCreateInfo.bindingCount = 6;
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
    VkWriteDescriptorSet writeDescriptorSets[6];

    VkDescriptorBufferInfo descriptorBufferInfo[6] = {0};

    descriptorPoolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorPoolSize.descriptorCount = 6;

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

    descriptorBufferInfo[1].buffer = dev->img_buffer;
    descriptorBufferInfo[1].offset = 0;
    descriptorBufferInfo[1].range = VK_WHOLE_SIZE;

    descriptorBufferInfo[2].buffer = dev->previous_result_buffer;
    descriptorBufferInfo[2].offset = 0;
    descriptorBufferInfo[2].range = VK_WHOLE_SIZE;

    descriptorBufferInfo[3].buffer = dev->internal_buffer;
    descriptorBufferInfo[3].offset = 0;
    descriptorBufferInfo[3].range = VK_WHOLE_SIZE;

    descriptorBufferInfo[4].buffer = dev->internal_int_buffer;
    descriptorBufferInfo[4].offset = 0;
    descriptorBufferInfo[4].range = VK_WHOLE_SIZE;

    descriptorBufferInfo[5].buffer = dev->out_buffer;
    descriptorBufferInfo[5].offset = 0;
    descriptorBufferInfo[5].range = VK_WHOLE_SIZE;

    for(uint32_t i=0;i<6;i++)
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

    vkUpdateDescriptorSets(dev->device, 6, writeDescriptorSets, 0, NULL);
    return 1;
}

int gpu_create_compute_pipeline(vulkan_device *dev)
{
    VkShaderModuleCreateInfo createInfo = {0};
    VkPipelineShaderStageCreateInfo shaderStageCreateInfo = {0};
    VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = {0};
    VkComputePipelineCreateInfo pipelineCreateInfo = {0};

    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.pCode = (const uint32_t*)correlation_spv;
    createInfo.codeSize = correlation_spv_len;

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

int gpu_create_command_pool(vulkan_device *dev)
{
    VkCommandPoolCreateInfo commandPoolCreateInfo = {0};
    VkFenceCreateInfo fenceCreateInfo = {0};

    commandPoolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    commandPoolCreateInfo.flags = 0;
    commandPoolCreateInfo.queueFamilyIndex = dev->queueFamilyIndex;
    if (vkCreateCommandPool(dev->device, &commandPoolCreateInfo, NULL, &dev->commandPool) != VK_SUCCESS)
        return 0;

    fenceCreateInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    if (vkCreateFence(dev->device, &fenceCreateInfo, NULL, &dev->fence) != VK_SUCCESS)
        return 0;

    return 1;
}

int gpu_run_command_buffer(vulkan_device *dev, int w1, int h1)
{
    VkSubmitInfo submitInfo = {0};
    VkCommandBufferBeginInfo beginInfo = {0};
    VkCommandBufferAllocateInfo commandBufferAllocateInfo = {0};
    VkCommandBuffer commandBuffer = {0};

    commandBufferAllocateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    commandBufferAllocateInfo.commandPool = dev->commandPool;
    commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    commandBufferAllocateInfo.commandBufferCount = 1;
    if (vkAllocateCommandBuffers(dev->device, &commandBufferAllocateInfo, &commandBuffer) != VK_SUCCESS)
        return 0;

    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS)
        return 0;

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, dev->pipeline);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, dev->pipelineLayout, 0, 1, &dev->descriptorSet, 0, NULL);

    vkCmdDispatch(commandBuffer, (uint32_t)ceilf(w1/16.0f), (uint32_t)ceilf(h1/16.0f), 1);

    if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS)
        return 0;

    if (vkQueueSubmit(dev->queue, 1, &submitInfo, dev->fence) != VK_SUCCESS)
        return 0;

    if (vkWaitForFences(dev->device, 1, &dev->fence, VK_TRUE, UINT64_MAX) != VK_SUCCESS)
        return 0;

    if (vkResetFences(dev->device, 1,  &dev->fence) != VK_SUCCESS)
        return 0;

    vkFreeCommandBuffers(dev->device, dev->commandPool, 1, &commandBuffer);
    return 1;
}

int gpu_transfer_in_images(cross_correlate_task *t, VkDevice device, VkDeviceMemory bufferMemory)
{
    float *payload;
    if (vkMapMemory(device, bufferMemory, 0, VK_WHOLE_SIZE, 0, (void*)&payload) != VK_SUCCESS)
        return 0;

    for (int i=0;i<t->img1.width*t->img1.height;i++)
        payload[i] = (unsigned char)t->img1.img[i];
    payload += t->img1.width*t->img1.height;
    for (int i=0;i<t->img2.width*t->img2.height;i++)
        payload[i] = (unsigned char)t->img2.img[i];

    vkUnmapMemory(device, bufferMemory);

    free(t->img1.img);
    t->img1.img = NULL;
    free(t->img2.img);
    t->img2.img = NULL;

    return 1;
}

int gpu_transfer_in_previous_results(cross_correlate_task *t, VkDevice device, VkDeviceMemory bufferMemory)
{
    int32_t *payload;
    if (vkMapMemory(device, bufferMemory, 0, VK_WHOLE_SIZE, 0, (void*)&payload) != VK_SUCCESS)
        return 0;

    for (int i=0;i<t->out_width*t->out_height*2;i++)
        payload[i] = t->correlated_points[i];

    vkUnmapMemory(device, bufferMemory);
    return 1;
}

int gpu_transfer_in_params(cross_correlate_task *t, vulkan_device *dev, int corridor_offset, int corridor_start, int corridor_end, int phase)
{
    shader_params *payload;

    if (vkMapMemory(dev->device, dev->params_bufferMemory, 0, VK_WHOLE_SIZE, 0, (void*)&payload) != VK_SUCCESS)
        return 0;

    payload->img1_width = t->img1.width;
    payload->img1_height = t->img1.height;
    payload->img2_width = t->img2.width;
    payload->img2_height = t->img2.height;
    payload->output_width = t->out_width;
    payload->output_height = t->out_height;
    for(size_t i=0;i<3;i++)
        for(size_t j=0;j<3;j++)
            payload->fundamental_matrix[j*4+i] = t->fundamental_matrix[i*3+j];
    payload->scale = t->scale;
    payload->iteration = t->iteration;
    payload->corridor_offset = corridor_offset;
    payload->corridor_start = corridor_start;
    payload->corridor_end = corridor_end;
    payload->phase = phase;
    payload->kernel_size = cybervision_crosscorrelation_kernel_size;
    payload->threshold = t->proj_mode==PROJECTION_MODE_PARALLEL? cybervision_crosscorrelation_threshold_parallel : cybervision_crosscorrelation_threshold_perspective;
    payload->neighbor_distance = cybervision_crosscorrelation_neighbor_distance;
    payload->extend_range = cybervision_crosscorrelation_corridor_extend_range;
    payload->min_range = cybervision_crosscorrelation_corridor_min_range;

    vkUnmapMemory(dev->device, dev->params_bufferMemory);
    return 1;
}

int gpu_transfer_out_image(cross_correlate_task *t, vulkan_device *dev)
{
    int32_t *correlated_points;
    float inv_scale = 1.0F/t->scale;
    if (vkMapMemory(dev->device, dev->out_bufferMemory, 0, VK_WHOLE_SIZE, 0, (void*)&correlated_points) != VK_SUCCESS)
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

    vkUnmapMemory(dev->device, dev->out_bufferMemory);
    return 1;
}

void* gpu_correlate_cross_correlation_task(void *args)
{
    cross_correlate_task *t = args;
    vulkan_context *ctx = t->internal;
    int kernel_size = cybervision_crosscorrelation_kernel_size;
    int corridor_size = cybervision_crosscorrelation_corridor_size;
    int corridor_segment_length = cybervision_crosscorrelation_corridor_segment_length;
    int corridor_stripes = 2*corridor_size+1;
    int max_width = t->img1.width > t->img2.width ? t->img1.width:t->img2.width;
    int max_height = t->img1.height > t->img2.height ? t->img1.height:t->img2.height;
    int corridor_length = (t->img2.width>t->img2.height? t->img2.width:t->img2.height)-(kernel_size*2);
    int corridor_segments = corridor_length/cybervision_crosscorrelation_corridor_segment_length + 1;
    float progressbar_completed_percentage = 2.0F;

    if (!gpu_transfer_in_images(t, ctx->dev.device, ctx->dev.img_bufferMemory))
    {
        t->error = "Failed to transfer input images";
        t->completed = 1;
        return 0;
    }
    if (!gpu_transfer_in_previous_results(t, ctx->dev.device, ctx->dev.previous_result_bufferMemory))
    {
        t->error = "Failed to transfer previous results";
        t->completed = 1;
        return 0;
    }

    if (t->iteration > 0)
    {
        int y_limit = (int)ceilf((cybervision_crosscorrelation_neighbor_distance)/t->scale);
        int batch_size = cybervision_crosscorrelation_search_area_segment_length;
        if (!gpu_transfer_in_params(t, &ctx->dev, 0, 0, 0, 1))
        {
            t->error = "Failed to transfer input parameters (search area estimation initialization stage)";
            t->completed = 1;
            return NULL;
        }
        if (!gpu_run_command_buffer(&ctx->dev, max_width, max_height))
        {
            t->error = "Failed to run command buffer (search area estimation initialization stage)";
            t->completed = 1;
            return NULL;
        }
        t->percent_complete = 2.0F;
        for(int y=-y_limit;y<=y_limit;y+=batch_size)
        {
            int corridor_end = y+batch_size;
            corridor_end = corridor_end<y_limit?corridor_end:y_limit;
            if (!gpu_transfer_in_params(t, &ctx->dev, 0, y, corridor_end, 2))
            {
                t->error = "Failed to transfer input parameters (search area estimation stage phase 0)";
                t->completed = 1;
                return NULL;
            }
            if (!gpu_run_command_buffer(&ctx->dev, t->img1.width, t->img1.height))
            {
                t->error = "Failed to run command buffer (search area estimation stage phase 0)";
                t->completed = 1;
                return NULL;
            }
            t->percent_complete = 2.0F+29.0F*(y+y_limit)/(2.0F*y_limit+1.0F);
        }
        t->percent_complete = 31.0F;
        for(int y=-y_limit;y<=y_limit;y+=batch_size)
        {
            int corridor_end = y+batch_size;
            corridor_end = corridor_end<y_limit?corridor_end:y_limit;
            if (!gpu_transfer_in_params(t, &ctx->dev, 0, y, corridor_end, 3))
            {
                t->error = "Failed to transfer input parameters (search area estimation stage phase 1)";
                t->completed = 1;
                return NULL;
            }
            if (!gpu_run_command_buffer(&ctx->dev, t->img1.width, t->img1.height))
            {
                t->error = "Failed to run command buffer (search area estimation stage phase 1)";
                t->completed = 1;
                return NULL;
            }
            t->percent_complete = 31.0F+29.0F*(y+y_limit)/(2.0F*y_limit+1.0F);
        }
        progressbar_completed_percentage = 60.0F;
    }

    if (!gpu_transfer_in_params(t, &ctx->dev, 0, 0, 0, 4))
    {
        t->error = "Failed to transfer input parameters (initialization stage)";
        t->completed = 1;
        return NULL;
    }
    if (!gpu_run_command_buffer(&ctx->dev, max_width, max_height))
    {
        t->error = "Failed to run command buffer (initialization stage)";
        t->completed = 1;
        return NULL;
    }

    t->percent_complete = progressbar_completed_percentage;

    for (int c=-corridor_size;c<=corridor_size;c++)
    {
        for (int l=0;l<corridor_segments;l++)
        {
            int corridor_start = kernel_size + l*corridor_segment_length;
            int corridor_end = kernel_size + (l+1)*corridor_segment_length;
            if (corridor_end> corridor_length-kernel_size)
                corridor_end = corridor_length-kernel_size;
            if (!gpu_transfer_in_params(t, &ctx->dev, c, corridor_start, corridor_end, 5))
            {
                t->error = "Failed to transfer input parameters";
                break;
            }
            if (!gpu_run_command_buffer(&ctx->dev, t->img1.width, t->img1.height))
            {
                t->error = "Failed to run command buffer";
                break;
            }

            float corridor_complete = (float)(corridor_end - kernel_size) / (corridor_length);
            t->percent_complete = progressbar_completed_percentage + (100.0F-progressbar_completed_percentage)*(c+corridor_size + corridor_complete)/corridor_stripes;
        }
    }

    if (!gpu_transfer_out_image(t, &ctx->dev))
        t->error = "Failed to read output image";
    t->completed = 1;
    return NULL;
}

int gpu_correlation_cross_correlate_init(cross_correlate_task *t, size_t img1_pixels, size_t img2_pixels)
{
    vulkan_context* ctx = malloc(sizeof(vulkan_context));
    t->internal = ctx;
    ctx->thread_started = 0;

    if (gpu_vk_create(ctx) != VK_SUCCESS)
    {
        t->error = "Failed to create device";
        return 0;
    }
    if (!gpu_find_device(ctx))
    {
        t->error = "Failed to find suitable device";
        return 0;
    }

    if (!gpu_create_buffer(&ctx->dev, sizeof(shader_params), &ctx->dev.params_buffer, &ctx->dev.params_bufferMemory, 0))
    {
        t->error = "Failed to create buffer (parameters)";
        return 0;
    }
    if (!gpu_create_buffer(&ctx->dev, sizeof(float)*(img1_pixels+img2_pixels), &ctx->dev.img_buffer, &ctx->dev.img_bufferMemory, 0))
    {
        t->error = "Failed to create buffer (images)";
        return 0;
    }
    if (!gpu_create_buffer(&ctx->dev, sizeof(int32_t)*(t->out_width*t->out_height*2), &ctx->dev.previous_result_buffer, &ctx->dev.previous_result_bufferMemory, 0))
    {
        t->error = "Failed to create buffer (previous results)";
        return 0;
    }
    if (!gpu_create_buffer(&ctx->dev, sizeof(float)*(img1_pixels*3+img2_pixels*2), &ctx->dev.internal_buffer, &ctx->dev.internal_bufferMemory, 0))
    {
        t->error = "Failed to create buffer (internal float)";
        return 0;
    }
    if (!gpu_create_buffer(&ctx->dev, sizeof(int32_t)*img1_pixels*3, &ctx->dev.internal_int_buffer, &ctx->dev.internal_int_bufferMemory, 0))
    {
        t->error = "Failed to create buffer (internal int)";
        return 0;
    }
    if (!gpu_create_buffer(&ctx->dev, sizeof(int32_t)*img1_pixels*2, &ctx->dev.out_buffer, &ctx->dev.out_bufferMemory, 0))
    {
        t->error = "Failed to create buffer (output)";
        return 0;
    }

    if (!gpu_create_descriptor_set_layout(&ctx->dev))
    {
        t->error = "Failed to create descriptor set layout";
        return 0;
    }
    if (!gpu_create_descriptor_set(&ctx->dev))
    {
        t->error = "Failed to create descriptor set";
        return 0;
    }

    if (!gpu_create_compute_pipeline(&ctx->dev))
    {
        t->error = "Failed to create compute pipeline";
        return 0;
    }
    if (!gpu_create_command_pool(&ctx->dev))
    {
        t->error = "Failed to create command pool";
        return 0;
    }

    return 1;
}

int gpu_correlation_cross_correlate_start(cross_correlate_task *t)
{
    vulkan_context* ctx = t->internal;
    t->completed = 0;
    t->percent_complete = 0.0;
    t->error = NULL;

    ctx->thread_started = 1;
    pthread_create(&ctx->thread, NULL, gpu_correlate_cross_correlation_task, t);

    return 1;
}

int gpu_correlation_cross_correlate_complete(cross_correlate_task *t)
{
    vulkan_context* ctx;
    VkDevice device;
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
    vulkan_context* ctx;
    VkDevice device;
    if (t == NULL || t->internal == NULL)
        return 1;
    ctx = t->internal;
    device = ctx->dev.device;
    
    vkDeviceWaitIdle(device);
    vkFreeMemory(device, ctx->dev.params_bufferMemory, NULL);
    vkFreeMemory(device, ctx->dev.img_bufferMemory, NULL);
    vkFreeMemory(device, ctx->dev.previous_result_bufferMemory, NULL);
    vkFreeMemory(device, ctx->dev.internal_bufferMemory, NULL);
    vkFreeMemory(device, ctx->dev.internal_int_bufferMemory, NULL);
    vkFreeMemory(device, ctx->dev.out_bufferMemory, NULL);
    vkDestroyBuffer(device, ctx->dev.params_buffer, NULL);
    vkDestroyBuffer(device, ctx->dev.img_buffer, NULL);
    vkDestroyBuffer(device, ctx->dev.previous_result_buffer, NULL);
    vkDestroyBuffer(device, ctx->dev.internal_buffer, NULL);
    vkDestroyBuffer(device, ctx->dev.internal_int_buffer, NULL);
    vkDestroyBuffer(device, ctx->dev.out_buffer, NULL);
    vkDestroyShaderModule(device, ctx->dev.computeShaderModule, NULL);
    vkDestroyDescriptorPool(device, ctx->dev.descriptorPool, NULL);
    vkDestroyDescriptorSetLayout(device, ctx->dev.descriptorSetLayout, NULL);
    vkDestroyPipelineLayout(device, ctx->dev.pipelineLayout, NULL);
    vkDestroyPipeline(device, ctx->dev.pipeline, NULL);
    vkDestroyFence(device, ctx->dev.fence, NULL);
    vkDestroyCommandPool(device, ctx->dev.commandPool, NULL);
    vkDestroyDevice(device, NULL);
    vkDestroyInstance(ctx->instance, NULL);	
    free(ctx);
    t->internal = NULL;
    return 1;
}
