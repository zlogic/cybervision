use std::{cmp::Ordering, collections::HashMap, error, ffi::CStr, slice};

use ash::{prelude::VkResult, vk};
use rayon::iter::ParallelIterator;

use crate::{
    correlation::{gpu::ShaderParams, Match},
    data::{Grid, Point2D},
};

use super::{CorrelationDirection, GpuError, HardwareMode};

// This should be supported by most modern GPUs, even old/budget ones like Celeron N3350.
// Based on https://www.vulkan.gpuinfo.org, only old integrated GPUs like 4th gen Core i5 have a
// limit of 4.
// But storing everything in the same buffer will likely have other issues like memory limits.
const MAX_BINDINGS: u32 = 6;

const _: () = {
    // Validate that ShaderParams fits into the minimum guaranteed push constants size.
    // Exceeding this size would require refactoring code (e.g. switch to uniform buffers), or risk
    // dropping support for some basic devices (like integrated Intel GPUs).
    assert!(std::mem::size_of::<ShaderParams>() < 128);
};

pub struct Device {
    instance: ash::Instance,
    name: String,
    device: ash::Device,
    memory_properties: vk::PhysicalDeviceMemoryProperties,
    buffers: Option<DeviceBuffers>,
    max_buffer_size: usize,
    descriptor_sets: DescriptorSets,
    pipelines: HashMap<ShaderModuleType, ShaderPipeline>,
    control: Control,
}

#[derive(Copy, Clone)]
struct Buffer {
    buffer: vk::Buffer,
    buffer_memory: vk::DeviceMemory,
    host_visible: bool,
    host_coherent: bool,
}

enum BufferType {
    GpuOnly,
    GpuDestination,
    GpuSource,
    HostSource,
    HostDestination,
}

struct DeviceBuffers {
    buffer_img: Buffer,
    buffer_internal_img1: Buffer,
    buffer_internal_img2: Buffer,
    buffer_internal_int: Buffer,
    buffer_out: Buffer,
    buffer_out_reverse: Buffer,
    buffer_out_corr: Buffer,
}

struct DescriptorSets {
    descriptor_pool: vk::DescriptorPool,
    regular_layout: vk::DescriptorSetLayout,
    cross_check_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    descriptor_sets: Vec<vk::DescriptorSet>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ShaderModuleType {
    InitOutData,
    PrepareInitialdataSearchdata,
    PrepareInitialdataCorrelation,
    PrepareSearchdata,
    CrossCorrelate,
    CrossCheckFilter,
}

struct ShaderPipeline {
    shader_module: vk::ShaderModule,
    pipeline: vk::Pipeline,
}

struct Control {
    queue: vk::Queue,
    command_pool: vk::CommandPool,
    fence: vk::Fence,
    command_buffer: vk::CommandBuffer,
}

pub struct DeviceContext {
    entry: ash::Entry,
    low_power: bool,
    device: Option<Device>,
}

impl DeviceContext {
    pub fn new(hardware_mode: HardwareMode) -> Result<DeviceContext, Box<dyn error::Error>> {
        if !matches!(hardware_mode, HardwareMode::Gpu | HardwareMode::GpuLowPower) {
            return Err(GpuError::new("GPU mode is not enabled").into());
        };
        let low_power = matches!(hardware_mode, HardwareMode::GpuLowPower);
        let entry = unsafe { ash::Entry::load()? };
        Ok(DeviceContext {
            entry,
            low_power,
            device: None,
        })
    }
}

impl super::DeviceContext<Device> for DeviceContext {
    fn is_low_power(&self) -> bool {
        self.low_power
    }

    fn get_device_name(&self) -> Option<String> {
        self.device().ok().map(|device| device.name.to_owned())
    }

    fn prepare_device(
        &mut self,
        img1_dimensions: (usize, usize),
        img2_dimensions: (usize, usize),
    ) -> Result<(), Box<dyn error::Error>> {
        let img1_pixels = img1_dimensions.0 * img1_dimensions.1;
        let img2_pixels = img2_dimensions.0 * img2_dimensions.1;

        // Ensure there's enough memory for the largest buffer.
        let max_pixels = img1_pixels.max(img2_pixels);
        let max_buffer_size = max_pixels * 4 * std::mem::size_of::<i32>();

        let current_buffer_size = self
            .device
            .as_ref()
            .map_or(0, |device| device.max_buffer_size);
        if current_buffer_size < max_buffer_size || self.device.is_none() {
            self.device = None;
            self.device = Some(Device::new(&self.entry, max_buffer_size)?);
        }
        let device = self.device_mut()?;
        unsafe {
            if let Some(buffers) = device.buffers.as_ref() {
                buffers.destroy(&device.device);
                device.buffers = None;
            }
        }
        let buffers = unsafe {
            Device::create_buffers(
                &device.device,
                &device.memory_properties,
                img1_pixels,
                img2_pixels,
            )?
        };
        device.buffers = Some(buffers);
        device.set_buffer_direction(&CorrelationDirection::Forward)?;
        Ok(())
    }

    fn device(&self) -> Result<&Device, GpuError> {
        match self.device.as_ref() {
            Some(device) => Ok(device),
            None => Err(GpuError::new("Device not initialized")),
        }
    }

    fn device_mut(&mut self) -> Result<&mut Device, GpuError> {
        match self.device.as_mut() {
            Some(device) => Ok(device),
            None => Err(GpuError::new("Device not initialized")),
        }
    }
}

impl Drop for DeviceContext {
    fn drop(&mut self) {
        self.device = None;
    }
}

impl Device {
    fn new(entry: &ash::Entry, max_buffer_size: usize) -> Result<Device, Box<dyn error::Error>> {
        // Init adapter.
        let instance = unsafe { Device::init_vk(entry)? };
        let cleanup_err = |err| unsafe {
            // TODO: look at refactoring this to follow RAII patterns (like ashpan/scopeguard).
            instance.destroy_instance(None);
            err
        };
        let (physical_device, name, compute_queue_index) =
            unsafe { Device::find_device(&instance, max_buffer_size).map_err(cleanup_err)? };
        let name = name.to_string();
        let device = unsafe {
            match Device::create_device(&instance, physical_device, compute_queue_index) {
                Ok(dev) => dev,
                Err(err) => {
                    instance.destroy_instance(None);
                    return Err(err);
                }
            }
        };
        let memory_properties =
            unsafe { instance.get_physical_device_memory_properties(physical_device) };
        let cleanup_err = |err| unsafe {
            device.destroy_device(None);
            instance.destroy_instance(None);
            err
        };
        // Init pipelines and shaders.
        let descriptor_sets =
            unsafe { Device::create_descriptor_sets(&device).map_err(cleanup_err)? };
        let cleanup_err = |err| unsafe {
            descriptor_sets.destroy(&device);
            device.destroy_device(None);
            instance.destroy_instance(None);
            err
        };
        let pipelines =
            unsafe { Device::create_pipelines(&device, &descriptor_sets).map_err(cleanup_err)? };
        let cleanup_err = |err| unsafe {
            destroy_pipelines(&device, &pipelines);
            descriptor_sets.destroy(&device);
            device.destroy_device(None);
            instance.destroy_instance(None);
            err
        };
        // Init control struct - queues, fences, command buffer.
        let control =
            unsafe { Device::create_control(&device, compute_queue_index).map_err(cleanup_err)? };
        let result = Device {
            instance,
            name,
            device,
            memory_properties,
            buffers: None,
            max_buffer_size,
            descriptor_sets,
            pipelines,
            control,
        };
        Ok(result)
    }

    unsafe fn map_buffer_write<F, T>(
        &self,
        dst_buffer: &Buffer,
        size: usize,
        f: F,
    ) -> Result<(), Box<dyn error::Error>>
    where
        F: FnOnce(&mut [T]),
    {
        // Not all code paths here are fully tested - some actions like flushing memory if memory
        // is not host_coherent might not work as expected.
        let size_bytes = size * std::mem::size_of::<T>();
        let handle_buffer = |buffer: &Buffer| -> VkResult<()> {
            {
                let memory = self.device.map_memory(
                    buffer.buffer_memory,
                    0,
                    size_bytes as u64,
                    vk::MemoryMapFlags::empty(),
                )?;
                let mapped_slice = slice::from_raw_parts_mut(memory as *mut T, size);
                f(mapped_slice);
            }

            if !buffer.host_coherent {
                let flush_memory_ranges = vk::MappedMemoryRange::builder()
                    .memory(buffer.buffer_memory)
                    .offset(0)
                    .size(size_bytes as u64);
                self.device
                    .flush_mapped_memory_ranges(&[flush_memory_ranges.build()])?;
            }
            self.device.unmap_memory(buffer.buffer_memory);
            Ok(())
        };

        if dst_buffer.host_visible {
            // If memory is available to the host, copy data directly to the buffer.
            handle_buffer(dst_buffer)?;
            return Ok(());
        }

        let temp_buffer = Device::create_buffer(
            &self.device,
            &self.memory_properties,
            size_bytes,
            BufferType::HostSource,
        )?;
        let cleanup_err = |err| {
            temp_buffer.destroy(&self.device);
            err
        };

        handle_buffer(&temp_buffer).map_err(cleanup_err)?;

        self.copy_buffer_to_buffer(&temp_buffer, dst_buffer, size_bytes)
            .map_err(cleanup_err)?;

        temp_buffer.destroy(&self.device);

        Ok(())
    }

    unsafe fn copy_buffer_to_buffer(
        &self,
        src: &Buffer,
        dst: &Buffer,
        size_bytes: usize,
    ) -> VkResult<()> {
        let command_buffer = self.control.command_buffer;
        self.device.reset_fences(&[self.control.fence])?;
        self.device
            .reset_command_buffer(command_buffer, vk::CommandBufferResetFlags::empty())?;

        let info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        self.device.begin_command_buffer(command_buffer, &info)?;
        let regions = vk::BufferCopy::builder().size(size_bytes as u64);
        self.device
            .cmd_copy_buffer(command_buffer, src.buffer, dst.buffer, &[regions.build()]);
        self.device.end_command_buffer(command_buffer)?;

        let command_buffers = [command_buffer];
        let submit_info = vk::SubmitInfo::builder().command_buffers(&command_buffers);
        self.device.queue_submit(
            self.control.queue,
            &[submit_info.build()],
            self.control.fence,
        )?;
        self.device
            .wait_for_fences(&[self.control.fence], true, u64::MAX)
    }

    unsafe fn map_buffer_read<F, T>(
        &self,
        buffer: &Buffer,
        size: usize,
        f: F,
    ) -> Result<(), Box<dyn error::Error>>
    where
        F: FnOnce(&[T]),
    {
        // Not all code paths here are fully tested - some actions like flushing memory if memory
        // is not host_coherent might not work as expected.
        let size_bytes = size * std::mem::size_of::<T>();
        let handle_buffer = |buffer: &Buffer| -> VkResult<()> {
            let memory = self.device.map_memory(
                buffer.buffer_memory,
                0,
                size_bytes as u64,
                vk::MemoryMapFlags::empty(),
            )?;
            if !buffer.host_coherent {
                let invalidate_memory_ranges = vk::MappedMemoryRange::builder()
                    .memory(buffer.buffer_memory)
                    .offset(0)
                    .size(size_bytes as u64);
                self.device
                    .invalidate_mapped_memory_ranges(&[invalidate_memory_ranges.build()])?;
            }
            {
                let mapped_slice = slice::from_raw_parts(memory as *const T, size);
                f(mapped_slice);
            }

            self.device.unmap_memory(buffer.buffer_memory);
            Ok(())
        };

        if buffer.host_visible {
            // If memory is available to the host, copy data directly to the buffer.
            handle_buffer(buffer)?;
            return Ok(());
        }

        let temp_buffer = Device::create_buffer(
            &self.device,
            &self.memory_properties,
            size_bytes,
            BufferType::HostDestination,
        )?;
        let cleanup_err = |err| {
            temp_buffer.destroy(&self.device);
            err
        };

        self.copy_buffer_to_buffer(buffer, &temp_buffer, size_bytes)
            .map_err(cleanup_err)?;

        handle_buffer(&temp_buffer).map_err(cleanup_err)?;
        temp_buffer.destroy(&self.device);

        Ok(())
    }

    unsafe fn init_vk(entry: &ash::Entry) -> VkResult<ash::Instance> {
        let app_name = CStr::from_bytes_with_nul_unchecked(b"Cybervision\0");
        let engine_name = CStr::from_bytes_with_nul_unchecked(b"cybervision\0");
        let appinfo = vk::ApplicationInfo::builder()
            .application_name(app_name)
            .application_version(0)
            .engine_name(engine_name)
            .engine_version(0)
            .api_version(vk::make_api_version(0, 1, 0, 0));

        let create_flags = vk::InstanceCreateFlags::default();
        let create_info = vk::InstanceCreateInfo::builder()
            .application_info(&appinfo)
            .flags(create_flags);
        entry.create_instance(&create_info, None)
    }

    unsafe fn find_device(
        instance: &ash::Instance,
        max_buffer_size: usize,
    ) -> Result<(vk::PhysicalDevice, String, u32), Box<dyn error::Error>> {
        let devices = instance.enumerate_physical_devices()?;
        let device = devices
            .iter()
            .filter_map(|device| {
                let device = *device;
                let props = instance.get_physical_device_properties(device);
                if props.limits.max_push_constants_size < std::mem::size_of::<ShaderParams>() as u32
                    || props.limits.max_bound_descriptor_sets < 2
                    || props.limits.max_per_stage_descriptor_storage_buffers < MAX_BINDINGS
                    || props.limits.max_storage_buffer_range < max_buffer_size as u32
                {
                    return None;
                }
                let queue_index = Device::find_compute_queue(instance, device)?;

                let device_name = CStr::from_ptr(props.device_name.as_ptr());
                let device_name = String::from_utf8_lossy(device_name.to_bytes()).to_string();
                // TODO: allow to specify a device name filter/regex?
                let score = match props.device_type {
                    vk::PhysicalDeviceType::DISCRETE_GPU => 3,
                    vk::PhysicalDeviceType::VIRTUAL_GPU => 2,
                    vk::PhysicalDeviceType::INTEGRATED_GPU => 1,
                    _ => 0,
                };
                // Prefer real devices instead of dzn emulation.
                let is_dzn = device_name
                    .to_lowercase()
                    .starts_with("microsoft direct3d12");
                let score = (score, is_dzn);
                Some((device, device_name, queue_index, score))
            })
            .max_by(|(_, _, _, a), (_, _, _, b)| {
                if a.1 && !b.1 {
                    return Ordering::Less;
                } else if !a.1 && b.1 {
                    return Ordering::Greater;
                }
                a.0.cmp(&b.0)
            });
        let (device, name, queue_index) = if let Some((device, name, queue_index, _score)) = device
        {
            (device, name, queue_index)
        } else {
            return Err(GpuError::new("Device not found").into());
        };
        Ok((device, name, queue_index))
    }

    unsafe fn find_compute_queue(
        instance: &ash::Instance,
        device: vk::PhysicalDevice,
    ) -> Option<u32> {
        instance
            .get_physical_device_queue_family_properties(device)
            .iter()
            .enumerate()
            .flat_map(|(index, queue)| {
                if queue
                    .queue_flags
                    .contains(vk::QueueFlags::COMPUTE | vk::QueueFlags::TRANSFER)
                    && queue.queue_count > 0
                {
                    Some(index as u32)
                } else {
                    None
                }
            })
            .next()
    }

    unsafe fn create_device(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        compute_queue_index: u32,
    ) -> Result<ash::Device, Box<dyn error::Error>> {
        let queue_info = vk::DeviceQueueCreateInfo::builder()
            .queue_family_index(compute_queue_index)
            .queue_priorities(&[1.0f32]);
        let device_create_info =
            vk::DeviceCreateInfo::builder().queue_create_infos(std::slice::from_ref(&queue_info));
        match instance.create_device(physical_device, &device_create_info, None) {
            Ok(device) => Ok(device),
            Err(err) => Err(err.into()),
        }
    }

    unsafe fn create_buffers(
        device: &ash::Device,
        memory_properties: &vk::PhysicalDeviceMemoryProperties,
        img1_pixels: usize,
        img2_pixels: usize,
    ) -> Result<DeviceBuffers, Box<dyn error::Error>> {
        let max_pixels = img1_pixels.max(img2_pixels);
        let mut buffers: Vec<Buffer> = vec![];
        let cleanup_err = |buffers: &[Buffer], err| {
            buffers.iter().for_each(|buffer| {
                device.free_memory(buffer.buffer_memory, None);
                device.destroy_buffer(buffer.buffer, None)
            });
            err
        };
        let buffer_img = Device::create_buffer(
            device,
            memory_properties,
            (img1_pixels + img2_pixels) * std::mem::size_of::<f32>(),
            BufferType::GpuDestination,
        )
        .map_err(|err| cleanup_err(buffers.as_slice(), err))?;
        buffers.push(buffer_img);

        let buffer_internal_img1 = Device::create_buffer(
            device,
            memory_properties,
            (img1_pixels * 2) * std::mem::size_of::<f32>(),
            BufferType::GpuOnly,
        )
        .map_err(|err| cleanup_err(buffers.as_slice(), err))?;
        buffers.push(buffer_internal_img1);

        let buffer_internal_img2 = Device::create_buffer(
            device,
            memory_properties,
            (img2_pixels * 2) * std::mem::size_of::<f32>(),
            BufferType::GpuOnly,
        )
        .map_err(|err| cleanup_err(buffers.as_slice(), err))?;
        buffers.push(buffer_internal_img2);

        let buffer_internal_int = Device::create_buffer(
            device,
            memory_properties,
            max_pixels * 4 * std::mem::size_of::<i32>(),
            BufferType::GpuOnly,
        )
        .map_err(|err| cleanup_err(buffers.as_slice(), err))?;
        buffers.push(buffer_internal_int);

        let buffer_out = Device::create_buffer(
            device,
            memory_properties,
            img1_pixels * 2 * std::mem::size_of::<i32>(),
            BufferType::GpuSource,
        )
        .map_err(|err| cleanup_err(buffers.as_slice(), err))?;
        buffers.push(buffer_out);

        let buffer_out_reverse = Device::create_buffer(
            device,
            memory_properties,
            img2_pixels * 2 * std::mem::size_of::<i32>(),
            BufferType::GpuOnly,
        )
        .map_err(|err| cleanup_err(buffers.as_slice(), err))?;
        buffers.push(buffer_out_reverse);

        let buffer_out_corr = Device::create_buffer(
            device,
            memory_properties,
            max_pixels * std::mem::size_of::<f32>(),
            BufferType::GpuSource,
        )
        .map_err(|err| cleanup_err(buffers.as_slice(), err))?;

        Ok(DeviceBuffers {
            buffer_img,
            buffer_internal_img1,
            buffer_internal_img2,
            buffer_internal_int,
            buffer_out,
            buffer_out_reverse,
            buffer_out_corr,
        })
    }

    fn buffers(&self) -> Result<&DeviceBuffers, GpuError> {
        match self.buffers.as_ref() {
            Some(buffers) => Ok(buffers),
            None => Err(GpuError::new("Buffers not initialized")),
        }
    }

    unsafe fn create_buffer(
        device: &ash::Device,
        memory_properties: &vk::PhysicalDeviceMemoryProperties,
        size: usize,
        buffer_type: BufferType,
    ) -> Result<Buffer, Box<dyn error::Error>> {
        let size = size as u64;
        let required_memory_properties = match buffer_type {
            BufferType::GpuOnly | BufferType::GpuDestination | BufferType::GpuSource => {
                vk::MemoryPropertyFlags::DEVICE_LOCAL
            }
            BufferType::HostSource | BufferType::HostDestination => {
                vk::MemoryPropertyFlags::HOST_VISIBLE
            }
        };
        let extra_usage_flags = match buffer_type {
            BufferType::HostSource => vk::BufferUsageFlags::TRANSFER_SRC,
            BufferType::HostDestination => vk::BufferUsageFlags::TRANSFER_DST,
            BufferType::GpuOnly => vk::BufferUsageFlags::STORAGE_BUFFER,
            BufferType::GpuSource => {
                vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::STORAGE_BUFFER
            }
            BufferType::GpuDestination => {
                vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::STORAGE_BUFFER
            }
        };
        let buffer_create_info = vk::BufferCreateInfo {
            size,
            usage: extra_usage_flags,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            ..Default::default()
        };
        let buffer = device.create_buffer(&buffer_create_info, None)?;
        let memory_requirements = device.get_buffer_memory_requirements(buffer);
        // Most vendors provide a sorted list - with less features going first.
        // As soon as the right flag is found, this search will stop, so it should pick a memory
        // type with the closest match.
        let buffer_memory = (0..memory_properties.memory_type_count as usize)
            .flat_map(|i| {
                let memory_type = memory_properties.memory_types[i];
                if memory_properties.memory_heaps[memory_type.heap_index as usize].size
                    < memory_requirements.size
                {
                    return None;
                }
                if ((1 << i) & memory_requirements.memory_type_bits) == 0 {
                    return None;
                }
                let property_flags = memory_type.property_flags;
                if !property_flags.contains(required_memory_properties) {
                    return None;
                }
                let host_visible = property_flags.contains(vk::MemoryPropertyFlags::HOST_VISIBLE);
                let host_coherent = property_flags.contains(vk::MemoryPropertyFlags::HOST_COHERENT);
                let allocate_info = vk::MemoryAllocateInfo {
                    allocation_size: memory_requirements.size,
                    memory_type_index: i as u32,
                    ..Default::default()
                };
                // Some buffers may fill up, in this case allocating memory can fail.
                let mem = device.allocate_memory(&allocate_info, None).ok()?;

                Some((mem, host_visible, host_coherent))
            })
            .next();

        let (buffer_memory, host_visible, host_coherent) = if let Some(mem) = buffer_memory {
            mem
        } else {
            device.destroy_buffer(buffer, None);
            return Err(GpuError::new("Cannot find suitable memory").into());
        };
        let result = Buffer {
            buffer,
            buffer_memory,
            host_visible,
            host_coherent,
        };
        device.bind_buffer_memory(buffer, buffer_memory, 0)?;
        Ok(result)
    }

    unsafe fn create_descriptor_sets(
        device: &ash::Device,
    ) -> Result<DescriptorSets, Box<dyn error::Error>> {
        let create_layout_bindings = |count| {
            let bindings = (0..count)
                .map(|i| {
                    vk::DescriptorSetLayoutBinding::builder()
                        .binding(i as u32)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .descriptor_count(1)
                        .stage_flags(vk::ShaderStageFlags::COMPUTE)
                        .build()
                })
                .collect::<Vec<_>>();
            let layout_info =
                vk::DescriptorSetLayoutCreateInfo::builder().bindings(bindings.as_slice());
            device.create_descriptor_set_layout(&layout_info, None)
        };
        let descriptor_pool_size = [vk::DescriptorPoolSize::builder()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(6)
            .build()];
        let descriptor_pool_info = vk::DescriptorPoolCreateInfo::builder()
            .max_sets(2)
            .pool_sizes(&descriptor_pool_size);
        let descriptor_pool = device.create_descriptor_pool(&descriptor_pool_info, None)?;
        let cleanup_err = |err| {
            device.destroy_descriptor_pool(descriptor_pool, None);
            err
        };
        let regular_layout = create_layout_bindings(6).map_err(cleanup_err)?;
        let cleanup_err = |err| {
            device.destroy_descriptor_set_layout(regular_layout, None);
            device.destroy_descriptor_pool(descriptor_pool, None);
            err
        };
        let cross_check_layout = create_layout_bindings(2).map_err(cleanup_err)?;
        let cleanup_err = |err| {
            device.destroy_descriptor_set_layout(cross_check_layout, None);
            device.destroy_descriptor_set_layout(regular_layout, None);
            device.destroy_descriptor_pool(descriptor_pool, None);
            err
        };
        let layouts = [regular_layout, cross_check_layout];
        let push_constant_ranges = vk::PushConstantRange::builder()
            .offset(0)
            .size(std::mem::size_of::<ShaderParams>() as u32)
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .build();
        let pipeline_layout = device
            .create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::builder()
                    .set_layouts(&layouts)
                    .push_constant_ranges(&[push_constant_ranges]),
                None,
            )
            .map_err(cleanup_err)?;
        let cleanup_err = |err| {
            device.destroy_pipeline_layout(pipeline_layout, None);
            device.destroy_descriptor_set_layout(cross_check_layout, None);
            device.destroy_descriptor_set_layout(regular_layout, None);
            device.destroy_descriptor_pool(descriptor_pool, None);
            err
        };
        let descriptor_set_allocate_info = vk::DescriptorSetAllocateInfo::builder()
            .descriptor_pool(descriptor_pool)
            .set_layouts(&layouts);
        let descriptor_sets = device
            .allocate_descriptor_sets(&descriptor_set_allocate_info)
            .map_err(cleanup_err)?;

        Ok(DescriptorSets {
            descriptor_pool,
            regular_layout,
            cross_check_layout,
            pipeline_layout,
            descriptor_sets,
        })
    }

    pub fn set_buffer_direction(&self, direction: &CorrelationDirection) -> Result<(), GpuError> {
        let descriptor_sets = &self.descriptor_sets;
        let buffers = &self.buffers()?;
        let create_buffer_infos = |buffers: &[Buffer]| {
            buffers
                .iter()
                .map(|buf| {
                    vk::DescriptorBufferInfo::builder()
                        .buffer(buf.buffer)
                        .offset(0)
                        .range(vk::WHOLE_SIZE)
                        .build()
                })
                .collect::<Vec<_>>()
        };
        let create_write_descriptor = |i: usize, buffer_infos: &[vk::DescriptorBufferInfo]| {
            vk::WriteDescriptorSet::builder()
                .dst_set(descriptor_sets.descriptor_sets[i])
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(buffer_infos)
                .build()
        };
        let (buffer_internal_img1, buffer_internal_img2, buffer_out, buffer_out_reverse) =
            match direction {
                CorrelationDirection::Forward => (
                    buffers.buffer_internal_img1,
                    buffers.buffer_internal_img2,
                    buffers.buffer_out,
                    buffers.buffer_out_reverse,
                ),
                CorrelationDirection::Reverse => (
                    buffers.buffer_internal_img2,
                    buffers.buffer_internal_img1,
                    buffers.buffer_out_reverse,
                    buffers.buffer_out,
                ),
            };
        let regular_buffer_infos = create_buffer_infos(&[
            buffers.buffer_img,
            buffer_internal_img1,
            buffer_internal_img2,
            buffers.buffer_internal_int,
            buffer_out,
            buffers.buffer_out_corr,
        ]);
        let cross_check_buffer_infos = create_buffer_infos(&[buffer_out, buffer_out_reverse]);
        let write_descriptors = [
            create_write_descriptor(0, regular_buffer_infos.as_slice()),
            create_write_descriptor(1, cross_check_buffer_infos.as_slice()),
        ];
        unsafe {
            self.device.update_descriptor_sets(&write_descriptors, &[]);
        }
        Ok(())
    }

    unsafe fn load_shaders(
        device: &ash::Device,
    ) -> Result<Vec<(ShaderModuleType, vk::ShaderModule)>, Box<dyn error::Error>> {
        let mut result = vec![];
        let cleanup_err = |result: &mut Vec<(ShaderModuleType, vk::ShaderModule)>, err| {
            result
                .iter()
                .for_each(|(_type, shader)| device.destroy_shader_module(*shader, None));
            err
        };
        for module_type in ShaderModuleType::VALUES {
            let shader = module_type
                .load(device)
                .map_err(|err| cleanup_err(&mut result, err))?;
            result.push((module_type, shader));
        }
        Ok(result)
    }

    unsafe fn create_pipelines(
        device: &ash::Device,
        descriptor_sets: &DescriptorSets,
    ) -> Result<HashMap<ShaderModuleType, ShaderPipeline>, Box<dyn error::Error>> {
        let shader_modules = Device::load_shaders(device)?;

        let main_module_name = CStr::from_bytes_with_nul_unchecked(b"main\0");

        let pipeline_create_info = shader_modules
            .iter()
            .map(|(_shader_type, module)| {
                let stage_create_info = vk::PipelineShaderStageCreateInfo::builder()
                    .module(*module)
                    .name(main_module_name)
                    .stage(vk::ShaderStageFlags::COMPUTE);
                vk::ComputePipelineCreateInfo::builder()
                    .stage(stage_create_info.build())
                    .layout(descriptor_sets.pipeline_layout)
                    .build()
            })
            .collect::<Vec<_>>();
        let pipelines = match device.create_compute_pipelines(
            vk::PipelineCache::null(),
            pipeline_create_info.as_slice(),
            None,
        ) {
            Ok(pipelines) => pipelines,
            Err(err) => {
                err.0
                    .iter()
                    .for_each(|pipeline| device.destroy_pipeline(*pipeline, None));
                shader_modules
                    .iter()
                    .for_each(|(_shader_type, module)| device.destroy_shader_module(*module, None));
                return Err(err.1.into());
            }
        };
        let mut result = HashMap::new();
        shader_modules
            .iter()
            .zip(pipelines.iter())
            .for_each(|(shader_module, pipeline)| {
                result.insert(
                    shader_module.0,
                    ShaderPipeline {
                        shader_module: shader_module.1,
                        pipeline: *pipeline,
                    },
                );
            });
        Ok(result)
    }

    unsafe fn create_control(
        device: &ash::Device,
        queue_family_index: u32,
    ) -> Result<Control, Box<dyn error::Error>> {
        let queue = device.get_device_queue(queue_family_index, 0);
        let command_pool_info = vk::CommandPoolCreateInfo::builder()
            .queue_family_index(queue_family_index)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let command_pool = device.create_command_pool(&command_pool_info, None)?;
        let cleanup_err = |err| {
            device.destroy_command_pool(command_pool, None);
            err
        };
        let fence_create_info = vk::FenceCreateInfo::builder();
        let fence = device
            .create_fence(&fence_create_info, None)
            .map_err(cleanup_err)?;
        let cleanup_err = |err| {
            device.destroy_command_pool(command_pool, None);
            device.destroy_fence(fence, None);
            err
        };
        let command_buffers_info = vk::CommandBufferAllocateInfo::builder()
            .command_buffer_count(1)
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY);
        let command_buffer = device
            .allocate_command_buffers(&command_buffers_info)
            .map_err(cleanup_err)?[0];
        Ok(Control {
            queue,
            command_pool,
            fence,
            command_buffer,
        })
    }
}

impl super::Device for Device {
    unsafe fn run_shader(
        &mut self,
        dimensions: (usize, usize),
        shader_type: ShaderModuleType,
        shader_params: ShaderParams,
    ) -> Result<(), Box<dyn error::Error>> {
        let pipeline_config = self.pipelines.get(&shader_type).unwrap();
        let command_buffer = self.control.command_buffer;

        let workgroup_size = ((dimensions.0 + 15) / 16, ((dimensions.1 + 15) / 16));
        self.device.reset_fences(&[self.control.fence])?;
        self.device
            .reset_command_buffer(command_buffer, vk::CommandBufferResetFlags::empty())?;
        let info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        self.device.begin_command_buffer(command_buffer, &info)?;

        self.device.cmd_bind_pipeline(
            self.control.command_buffer,
            vk::PipelineBindPoint::COMPUTE,
            pipeline_config.pipeline,
        );
        // It's way easier to map all descriptor sets identically, instead of ensuring that every
        // kernel gets to use set = 0.
        // The cross correlation kernel will need to switch to descriptor set = 1.
        self.device.cmd_bind_descriptor_sets(
            command_buffer,
            vk::PipelineBindPoint::COMPUTE,
            self.descriptor_sets.pipeline_layout,
            0,
            &self.descriptor_sets.descriptor_sets,
            &[],
        );

        let push_constants_data = slice::from_raw_parts(
            &shader_params as *const ShaderParams as *const u8,
            std::mem::size_of::<ShaderParams>(),
        );
        self.device.cmd_push_constants(
            command_buffer,
            self.descriptor_sets.pipeline_layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            push_constants_data,
        );
        self.device.cmd_dispatch(
            command_buffer,
            workgroup_size.0 as u32,
            workgroup_size.1 as u32,
            1,
        );

        self.device.end_command_buffer(command_buffer)?;

        let command_buffers = [command_buffer];
        let submit_info = vk::SubmitInfo::builder().command_buffers(&command_buffers);
        self.device.queue_submit(
            self.control.queue,
            &[submit_info.build()],
            self.control.fence,
        )?;

        self.device
            .wait_for_fences(&[self.control.fence], true, u64::MAX)?;

        Ok(())
    }

    unsafe fn transfer_in_images(
        &self,
        img1: &Grid<u8>,
        img2: &Grid<u8>,
    ) -> Result<(), Box<dyn error::Error>> {
        let buffers = self.buffers()?;
        let img2_offset = img1.width() * img1.height();
        let size = img1.width() * img1.height() + img2.width() * img2.height();
        let copy_images = |img_slice: &mut [f32]| {
            img1.iter()
                .for_each(|(x, y, val)| img_slice[y * img1.width() + x] = *val as f32);
            img2.iter().for_each(|(x, y, val)| {
                img_slice[img2_offset + y * img2.width() + x] = *val as f32
            });
        };

        self.map_buffer_write(&buffers.buffer_img, size, copy_images)?;

        Ok(())
    }

    unsafe fn save_corr(
        &self,
        correlation_values: &mut Grid<Option<f32>>,
        correlation_threshold: f32,
    ) -> Result<(), Box<dyn error::Error>> {
        let buffers = self.buffers()?;
        let size = correlation_values.width() * correlation_values.height();
        let width = correlation_values.width();
        let copy_corr_data = |out_corr: &[f32]| {
            correlation_values
                .par_iter_mut()
                .for_each(|(x, y, out_point)| {
                    let corr = out_corr[y * width + x];
                    if corr > correlation_threshold {
                        *out_point = Some(corr);
                    }
                });
        };

        self.map_buffer_read(&buffers.buffer_out_corr, size, copy_corr_data)?;

        Ok(())
    }

    unsafe fn save_result(
        &self,
        out_image: &mut Grid<Option<Match>>,
        correlation_values: &Grid<Option<f32>>,
    ) -> Result<(), Box<dyn error::Error>> {
        let buffers = self.buffers()?;
        let size = out_image.width() * out_image.height() * 2;
        let width = out_image.width();
        let copy_out_image = |out_data: &[i32]| {
            out_image.par_iter_mut().for_each(|(x, y, out_point)| {
                let pos = 2 * (y * width + x);
                let (match_x, match_y) = (out_data[pos], out_data[pos + 1]);
                if let Some(corr) = correlation_values.val(x, y) {
                    *out_point = if match_x > 0 && match_y > 0 {
                        let point_match = Point2D::new(match_x as u32, match_y as u32);
                        Some((point_match, *corr))
                    } else {
                        None
                    };
                } else {
                    *out_point = None;
                };
            });
        };

        self.map_buffer_read(&buffers.buffer_out, size, copy_out_image)?;

        Ok(())
    }

    unsafe fn destroy_buffers(&mut self) {
        if let Some(buffers) = self.buffers.as_ref() {
            buffers.destroy(&self.device)
        }
        self.buffers = None;
    }
}

impl Buffer {
    unsafe fn destroy(&self, device: &ash::Device) {
        device.free_memory(self.buffer_memory, None);
        device.destroy_buffer(self.buffer, None);
    }
}

impl DeviceBuffers {
    unsafe fn destroy(&self, device: &ash::Device) {
        [
            self.buffer_img,
            self.buffer_internal_img1,
            self.buffer_internal_img2,
            self.buffer_internal_int,
            self.buffer_out,
            self.buffer_out_reverse,
            self.buffer_out_corr,
        ]
        .iter()
        .for_each(|buffer| {
            buffer.destroy(device);
        });
    }
}

impl DescriptorSets {
    unsafe fn destroy(&self, device: &ash::Device) {
        let _ = device.free_descriptor_sets(self.descriptor_pool, self.descriptor_sets.as_slice());
        device.destroy_pipeline_layout(self.pipeline_layout, None);
        device.destroy_descriptor_set_layout(self.cross_check_layout, None);
        device.destroy_descriptor_set_layout(self.regular_layout, None);
        device.destroy_descriptor_pool(self.descriptor_pool, None);
    }
}

unsafe fn destroy_pipelines(
    device: &ash::Device,
    pipelines: &HashMap<ShaderModuleType, ShaderPipeline>,
) {
    pipelines.values().for_each(|shader_pipeline| {
        device.destroy_shader_module(shader_pipeline.shader_module, None);
        device.destroy_pipeline(shader_pipeline.pipeline, None);
    });
}

impl ShaderModuleType {
    const VALUES: [Self; 6] = [
        Self::InitOutData,
        Self::PrepareInitialdataSearchdata,
        Self::PrepareInitialdataCorrelation,
        Self::PrepareSearchdata,
        Self::CrossCorrelate,
        Self::CrossCheckFilter,
    ];

    unsafe fn load(&self, device: &ash::Device) -> Result<vk::ShaderModule, Box<dyn error::Error>> {
        const SHADER_INIT_OUT_DATA: &[u8] = include_bytes!("shaders/init_out_data.spv");
        const SHADER_PREPARE_INITIALDATA_SEARCHDATA: &[u8] =
            include_bytes!("shaders/prepare_initialdata_searchdata.spv");
        const SHADER_PREPARE_INITIALDATA_CORRELATION: &[u8] =
            include_bytes!("shaders/prepare_initialdata_correlation.spv");
        const SHADER_PREPARE_SEARCHDATA: &[u8] = include_bytes!("shaders/prepare_searchdata.spv");
        const SHADER_CROSS_CORRELATE: &[u8] = include_bytes!("shaders/cross_correlate.spv");
        const SHADER_CROSS_CHECK_FILTER: &[u8] = include_bytes!("shaders/cross_check_filter.spv");

        let shader_module_spv = match self {
            ShaderModuleType::InitOutData => SHADER_INIT_OUT_DATA,
            ShaderModuleType::PrepareInitialdataSearchdata => SHADER_PREPARE_INITIALDATA_SEARCHDATA,
            ShaderModuleType::PrepareInitialdataCorrelation => {
                SHADER_PREPARE_INITIALDATA_CORRELATION
            }
            ShaderModuleType::PrepareSearchdata => SHADER_PREPARE_SEARCHDATA,
            ShaderModuleType::CrossCorrelate => SHADER_CROSS_CORRELATE,
            ShaderModuleType::CrossCheckFilter => SHADER_CROSS_CHECK_FILTER,
        };
        let shader_code = ash::util::read_spv(&mut std::io::Cursor::new(shader_module_spv))?;
        let shader_module = device.create_shader_module(
            &vk::ShaderModuleCreateInfo::builder()
                .flags(vk::ShaderModuleCreateFlags::empty())
                .code(shader_code.as_slice()),
            None,
        )?;

        Ok(shader_module)
    }
}

impl Control {
    unsafe fn destroy(&self, device: &ash::Device) {
        device.free_command_buffers(self.command_pool, &[self.command_buffer]);
        device.destroy_command_pool(self.command_pool, None);
        device.destroy_fence(self.fence, None);
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        unsafe {
            self.control.destroy(&self.device);
            destroy_pipelines(&self.device, &self.pipelines);
            self.descriptor_sets.destroy(&self.device);
            if let Some(buffers) = self.buffers.as_ref() {
                buffers.destroy(&self.device)
            }
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}
