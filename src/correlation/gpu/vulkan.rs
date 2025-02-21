use std::{
    cmp::Ordering,
    collections::HashMap,
    error, fmt, io,
    ops::{Deref, DerefMut},
    slice,
};

use super::Device as GpuDevice;
use ash::{prelude::VkResult, vk};
use rayon::iter::ParallelIterator;

use crate::{
    correlation::{Match, gpu::ShaderParams},
    data::{Grid, Point2D},
};

use super::{CorrelationDirection, HardwareMode};

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
    direction: CorrelationDirection,
    max_buffer_size: usize,
    descriptor_sets: DescriptorSets,
    pipelines: HashMap<ShaderModuleType, ShaderPipeline>,
    control: Control,
}

#[derive(Copy, Clone)]
struct Buffer {
    buffer: vk::Buffer,
    buffer_memory: vk::DeviceMemory,
    size: usize,
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
    layout: vk::DescriptorSetLayout,
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
    pub fn new(hardware_mode: HardwareMode) -> Result<DeviceContext, GpuError> {
        if !matches!(hardware_mode, HardwareMode::Gpu | HardwareMode::GpuLowPower) {
            return Err("GPU mode is not enabled".into());
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
    ) -> Result<(), GpuError> {
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
        if let Some(buffers) = device.buffers.as_ref() {
            buffers.destroy(&device.device);
            device.buffers = None;
        }
        let buffers = Device::create_buffers(
            &device.device,
            &device.memory_properties,
            img1_pixels,
            img2_pixels,
        )?;
        device.buffers = Some(buffers);
        device.set_buffer_direction(&CorrelationDirection::Forward)?;
        Ok(())
    }

    fn device(&self) -> Result<&Device, GpuError> {
        match self.device.as_ref() {
            Some(device) => Ok(device),
            None => Err("Device not initialized".into()),
        }
    }

    fn device_mut(&mut self) -> Result<&mut Device, GpuError> {
        match self.device.as_mut() {
            Some(device) => Ok(device),
            None => Err("Device not initialized".into()),
        }
    }
}

impl Drop for DeviceContext {
    fn drop(&mut self) {
        self.device = None;
    }
}

impl Device {
    fn new(entry: &ash::Entry, max_buffer_size: usize) -> Result<Device, GpuError> {
        // Init adapter.
        let instance = Self::init_vk(entry)?;
        let instance = ScopeRollback::new(instance, |instance| unsafe {
            instance.destroy_instance(None)
        });
        let (physical_device, name, compute_queue_index) =
            Self::find_device(&instance, max_buffer_size)?;
        let name = name.to_string();
        let device = Self::create_device(&instance, physical_device, compute_queue_index)?;
        let device = ScopeRollback::new(device, |device| unsafe { device.destroy_device(None) });
        let memory_properties =
            unsafe { instance.get_physical_device_memory_properties(physical_device) };
        // Init pipelines and shaders.
        let descriptor_sets = Self::create_descriptor_sets(&device)?;
        let descriptor_sets = ScopeRollback::new(descriptor_sets, |descriptor_sets| {
            descriptor_sets.destroy(&device)
        });
        let pipelines = Self::create_pipelines(&device, &descriptor_sets)?;
        let pipelines = ScopeRollback::new(pipelines, |pipelines| {
            destroy_pipelines(&device, &pipelines)
        });
        let direction = CorrelationDirection::Forward;
        // Init control struct - queues, fences, command buffer.
        let control = Self::create_control(&device, compute_queue_index)?;

        // All is good - consume instances and defuse the scope rollback.
        let pipelines = pipelines.consume();
        let descriptor_sets = descriptor_sets.consume();
        let result = Device {
            instance: instance.consume(),
            name,
            device: device.consume(),
            memory_properties,
            buffers: None,
            direction,
            max_buffer_size,
            descriptor_sets,
            pipelines,
            control,
        };
        Ok(result)
    }

    fn map_buffer_write<F, T>(&self, dst_buffer: &Buffer, size: usize, f: F) -> Result<(), GpuError>
    where
        F: FnOnce(&mut [T]),
    {
        // Not all code paths here are fully tested - some actions like flushing memory if memory
        // is not host_coherent might not work as expected.
        let size_bytes = size * std::mem::size_of::<T>();
        if dst_buffer.size < size_bytes {
            return Err("Mapped write buffer is smaller than expected".into());
        }
        let handle_buffer = |buffer: &Buffer| -> VkResult<()> {
            {
                let mapped_slice = unsafe {
                    let memory = self.device.map_memory(
                        buffer.buffer_memory,
                        0,
                        size_bytes as u64,
                        vk::MemoryMapFlags::empty(),
                    )?;
                    slice::from_raw_parts_mut(memory as *mut T, size)
                };
                f(mapped_slice);
            }

            if !buffer.host_coherent {
                let flush_memory_ranges = vk::MappedMemoryRange::default()
                    .memory(buffer.buffer_memory)
                    .offset(0)
                    .size(size_bytes as u64);
                unsafe {
                    self.device
                        .flush_mapped_memory_ranges(&[flush_memory_ranges])?
                };
            }
            unsafe { self.device.unmap_memory(buffer.buffer_memory) };
            Ok(())
        };

        if dst_buffer.host_visible {
            // If memory is available to the host, copy data directly to the buffer.
            handle_buffer(dst_buffer)?;
            return Ok(());
        }

        let temp_buffer = Self::create_buffer(
            &self.device,
            &self.memory_properties,
            size_bytes,
            BufferType::HostSource,
        )?;
        let temp_buffer =
            ScopeRollback::new(temp_buffer, |temp_buffer| temp_buffer.destroy(&self.device));

        handle_buffer(&temp_buffer)?;
        self.copy_buffer_to_buffer(&temp_buffer, dst_buffer, size_bytes)?;

        Ok(())
    }

    fn copy_buffer_to_buffer(
        &self,
        src: &Buffer,
        dst: &Buffer,
        size_bytes: usize,
    ) -> Result<(), GpuError> {
        if src.size < size_bytes {
            return Err("Copy source buffer is smaller than expected".into());
        }
        if dst.size < size_bytes {
            return Err("Copy destination buffer is smaller than expected".into());
        }
        unsafe {
            let command_buffer = self.control.command_buffer;
            self.device.reset_fences(&[self.control.fence])?;
            self.device
                .reset_command_buffer(command_buffer, vk::CommandBufferResetFlags::empty())?;

            let info = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

            self.device.begin_command_buffer(command_buffer, &info)?;
            let regions = vk::BufferCopy::default().size(size_bytes as u64);
            self.device
                .cmd_copy_buffer(command_buffer, src.buffer, dst.buffer, &[regions]);
            self.device.end_command_buffer(command_buffer)?;

            let command_buffers = [command_buffer];
            let submit_info = vk::SubmitInfo::default().command_buffers(&command_buffers);
            self.device
                .queue_submit(self.control.queue, &[submit_info], self.control.fence)?;
            Ok(self
                .device
                .wait_for_fences(&[self.control.fence], true, u64::MAX)?)
        }
    }

    fn map_buffer_read<F, T>(&self, buffer: &Buffer, size: usize, f: F) -> Result<(), GpuError>
    where
        F: FnOnce(&[T]),
    {
        // Not all code paths here are fully tested - some actions like flushing memory if memory
        // is not host_coherent might not work as expected.
        let size_bytes = size * std::mem::size_of::<T>();
        if buffer.size < size_bytes {
            return Err("Mapped read buffer is smaller than expected".into());
        }
        let handle_buffer = |buffer: &Buffer| -> VkResult<()> {
            let memory = unsafe {
                self.device.map_memory(
                    buffer.buffer_memory,
                    0,
                    size_bytes as u64,
                    vk::MemoryMapFlags::empty(),
                )?
            };
            if !buffer.host_coherent {
                let invalidate_memory_ranges = vk::MappedMemoryRange::default()
                    .memory(buffer.buffer_memory)
                    .offset(0)
                    .size(size_bytes as u64);
                unsafe {
                    self.device
                        .invalidate_mapped_memory_ranges(&[invalidate_memory_ranges])?
                };
            }
            {
                let mapped_slice = unsafe { slice::from_raw_parts(memory as *const T, size) };
                f(mapped_slice);
            }

            unsafe { self.device.unmap_memory(buffer.buffer_memory) };
            Ok(())
        };

        if buffer.host_visible {
            // If memory is available to the host, copy data directly to the buffer.
            handle_buffer(buffer)?;
            return Ok(());
        }

        let temp_buffer = Self::create_buffer(
            &self.device,
            &self.memory_properties,
            size_bytes,
            BufferType::HostDestination,
        )?;
        let temp_buffer =
            ScopeRollback::new(temp_buffer, |temp_buffer| temp_buffer.destroy(&self.device));

        self.copy_buffer_to_buffer(buffer, &temp_buffer, size_bytes)?;
        handle_buffer(&temp_buffer)?;

        Ok(())
    }

    fn init_vk(entry: &ash::Entry) -> VkResult<ash::Instance> {
        let app_name = c"Cybervision";
        let engine_name = c"cybervision";
        let appinfo = vk::ApplicationInfo::default()
            .application_name(app_name)
            .application_version(0)
            .engine_name(engine_name)
            .engine_version(0)
            .api_version(vk::make_api_version(0, 1, 0, 0));

        let create_flags = vk::InstanceCreateFlags::default();
        let create_info = vk::InstanceCreateInfo::default()
            .application_info(&appinfo)
            .flags(create_flags);
        unsafe { entry.create_instance(&create_info, None) }
    }

    fn find_device(
        instance: &ash::Instance,
        max_buffer_size: usize,
    ) -> Result<(vk::PhysicalDevice, String, u32), GpuError> {
        let devices = unsafe { instance.enumerate_physical_devices()? };
        let device = devices
            .iter()
            .filter_map(|device| {
                let device = *device;
                let props = unsafe { instance.get_physical_device_properties(device) };
                if props.limits.max_push_constants_size < std::mem::size_of::<ShaderParams>() as u32
                    || props.limits.max_bound_descriptor_sets < 2
                    || props.limits.max_per_stage_descriptor_storage_buffers < MAX_BINDINGS
                    || props.limits.max_storage_buffer_range < max_buffer_size as u32
                {
                    return None;
                }
                let queue_index = Self::find_compute_queue(instance, device)?;

                let device_name = props
                    .device_name_as_c_str()
                    .ok()?
                    .to_string_lossy()
                    .into_owned();
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
            return Err("Device not found".into());
        };
        Ok((device, name, queue_index))
    }

    fn find_compute_queue(instance: &ash::Instance, device: vk::PhysicalDevice) -> Option<u32> {
        unsafe {
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
    }

    fn create_device(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        compute_queue_index: u32,
    ) -> Result<ash::Device, GpuError> {
        let queue_info = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(compute_queue_index)
            .queue_priorities(&[1.0f32]);
        let device_create_info =
            vk::DeviceCreateInfo::default().queue_create_infos(std::slice::from_ref(&queue_info));
        unsafe {
            match instance.create_device(physical_device, &device_create_info, None) {
                Ok(device) => Ok(device),
                Err(err) => Err(err.into()),
            }
        }
    }

    fn create_buffers(
        device: &ash::Device,
        memory_properties: &vk::PhysicalDeviceMemoryProperties,
        img1_pixels: usize,
        img2_pixels: usize,
    ) -> Result<DeviceBuffers, GpuError> {
        let max_pixels = img1_pixels.max(img2_pixels);
        let mut buffers = ScopeRollback::new(vec![], |buffers: Vec<Buffer>| unsafe {
            buffers.iter().for_each(|buffer| {
                device.free_memory(buffer.buffer_memory, None);
                device.destroy_buffer(buffer.buffer, None)
            });
        });
        let buffer_img = Self::create_buffer(
            device,
            memory_properties,
            (img1_pixels + img2_pixels) * std::mem::size_of::<f32>(),
            BufferType::GpuDestination,
        )?;
        buffers.push(buffer_img);

        let buffer_internal_img1 = Self::create_buffer(
            device,
            memory_properties,
            (img1_pixels * 2) * std::mem::size_of::<f32>(),
            BufferType::GpuOnly,
        )?;
        buffers.push(buffer_internal_img1);

        let buffer_internal_img2 = Self::create_buffer(
            device,
            memory_properties,
            (img2_pixels * 2) * std::mem::size_of::<f32>(),
            BufferType::GpuOnly,
        )?;
        buffers.push(buffer_internal_img2);

        let buffer_internal_int = Self::create_buffer(
            device,
            memory_properties,
            max_pixels * 4 * std::mem::size_of::<i32>(),
            BufferType::GpuOnly,
        )?;
        buffers.push(buffer_internal_int);

        let buffer_out = Self::create_buffer(
            device,
            memory_properties,
            img1_pixels * 2 * std::mem::size_of::<i32>(),
            BufferType::GpuSource,
        )?;
        buffers.push(buffer_out);

        let buffer_out_reverse = Self::create_buffer(
            device,
            memory_properties,
            img2_pixels * 2 * std::mem::size_of::<i32>(),
            BufferType::GpuOnly,
        )?;
        buffers.push(buffer_out_reverse);

        let buffer_out_corr = Self::create_buffer(
            device,
            memory_properties,
            max_pixels * std::mem::size_of::<f32>(),
            BufferType::GpuSource,
        )?;

        // All is good - defuse the scope rollback.
        buffers.consume();

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
            None => Err("Buffers not initialized".into()),
        }
    }

    fn create_buffer(
        device: &ash::Device,
        memory_properties: &vk::PhysicalDeviceMemoryProperties,
        size: usize,
        buffer_type: BufferType,
    ) -> Result<Buffer, GpuError> {
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
        let buffer_create_info = vk::BufferCreateInfo::default()
            .size(size as u64)
            .usage(extra_usage_flags)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let buffer = unsafe { device.create_buffer(&buffer_create_info, None)? };
        let memory_requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
        // Most vendors provide a sorted list - with less features going first.
        // As soon as the right flag is found, this search will stop, so it should pick a memory
        // type with the closest match.
        let buffer_memory = memory_properties
            .memory_types_as_slice()
            .iter()
            .enumerate()
            .flat_map(|(memory_type_index, memory_type)| {
                if memory_properties.memory_heaps[memory_type.heap_index as usize].size
                    < memory_requirements.size
                {
                    return None;
                }
                if ((1 << memory_type_index) & memory_requirements.memory_type_bits) == 0 {
                    return None;
                }
                let property_flags = memory_type.property_flags;
                if !property_flags.contains(required_memory_properties) {
                    return None;
                }
                let host_visible = property_flags.contains(vk::MemoryPropertyFlags::HOST_VISIBLE);
                let host_coherent = property_flags.contains(vk::MemoryPropertyFlags::HOST_COHERENT);
                let allocate_info = vk::MemoryAllocateInfo::default()
                    .allocation_size(memory_requirements.size)
                    .memory_type_index(memory_type_index as u32);
                // Some buffers may fill up, in this case allocating memory can fail.
                let mem = unsafe { device.allocate_memory(&allocate_info, None).ok()? };

                Some((mem, host_visible, host_coherent))
            })
            .next();

        let (buffer_memory, host_visible, host_coherent) = if let Some(mem) = buffer_memory {
            mem
        } else {
            unsafe { device.destroy_buffer(buffer, None) };
            return Err("Cannot find suitable memory".into());
        };
        let result = Buffer {
            buffer,
            buffer_memory,
            size,
            host_visible,
            host_coherent,
        };
        unsafe { device.bind_buffer_memory(buffer, buffer_memory, 0)? };
        Ok(result)
    }

    fn create_descriptor_sets(device: &ash::Device) -> Result<DescriptorSets, GpuError> {
        let create_layout_bindings = |count| {
            let bindings = (0..count)
                .map(|i| {
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(i as u32)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .descriptor_count(1)
                        .stage_flags(vk::ShaderStageFlags::COMPUTE)
                })
                .collect::<Vec<_>>();
            let layout_info =
                vk::DescriptorSetLayoutCreateInfo::default().bindings(bindings.as_slice());
            unsafe { device.create_descriptor_set_layout(&layout_info, None) }
        };
        let descriptor_pool_size = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(6)];
        let descriptor_pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(1)
            .pool_sizes(&descriptor_pool_size);
        let descriptor_pool =
            unsafe { device.create_descriptor_pool(&descriptor_pool_info, None)? };
        let descriptor_pool = ScopeRollback::new(descriptor_pool, |descriptor_pool| unsafe {
            device.destroy_descriptor_pool(descriptor_pool, None)
        });
        let layout = create_layout_bindings(6)?;
        let layout = ScopeRollback::new(layout, |layout| unsafe {
            device.destroy_descriptor_set_layout(layout, None)
        });
        let layouts = [*layout.deref()];
        let push_constant_ranges = vk::PushConstantRange::default()
            .offset(0)
            .size(std::mem::size_of::<ShaderParams>() as u32)
            .stage_flags(vk::ShaderStageFlags::COMPUTE);
        let pipeline_layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(&layouts)
                    .push_constant_ranges(&[push_constant_ranges]),
                None,
            )?
        };
        let pipeline_layout = ScopeRollback::new(pipeline_layout, |pipeline_layout| unsafe {
            device.destroy_pipeline_layout(pipeline_layout, None)
        });
        let descriptor_set_allocate_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(*descriptor_pool.deref())
            .set_layouts(&layouts);
        let descriptor_sets =
            unsafe { device.allocate_descriptor_sets(&descriptor_set_allocate_info)? };

        Ok(DescriptorSets {
            descriptor_pool: descriptor_pool.consume(),
            layout: layout.consume(),
            pipeline_layout: pipeline_layout.consume(),
            descriptor_sets,
        })
    }

    fn load_shaders(
        device: &ash::Device,
    ) -> Result<Vec<(ShaderModuleType, vk::ShaderModule)>, GpuError> {
        let mut shader_modules = ShaderModuleType::VALUES
            .iter()
            .map(|module_type| {
                let shader = module_type.load(device)?;
                let shader = ScopeRollback::new(shader, |shader| unsafe {
                    device.destroy_shader_module(shader, None)
                });
                Ok((module_type, shader))
            })
            .collect::<Result<Vec<_>, GpuError>>()?;
        let shader_modules = shader_modules
            .drain(..)
            .map(|(module_type, shader)| (module_type.to_owned(), shader.consume()))
            .collect::<Vec<_>>();
        Ok(shader_modules)
    }

    fn create_pipelines(
        device: &ash::Device,
        descriptor_sets: &DescriptorSets,
    ) -> Result<HashMap<ShaderModuleType, ShaderPipeline>, GpuError> {
        let shader_modules = Self::load_shaders(device)?;

        let main_module_name = c"main";

        let pipeline_create_info = shader_modules
            .iter()
            .map(|(_shader_type, module)| {
                let stage_create_info = vk::PipelineShaderStageCreateInfo::default()
                    .module(*module)
                    .name(main_module_name)
                    .stage(vk::ShaderStageFlags::COMPUTE);
                vk::ComputePipelineCreateInfo::default()
                    .stage(stage_create_info)
                    .layout(descriptor_sets.pipeline_layout)
            })
            .collect::<Vec<_>>();
        let pipelines = unsafe {
            match device.create_compute_pipelines(
                vk::PipelineCache::null(),
                pipeline_create_info.as_slice(),
                None,
            ) {
                Ok(pipelines) => pipelines,
                Err(err) => {
                    err.0
                        .iter()
                        .for_each(|pipeline| device.destroy_pipeline(*pipeline, None));
                    shader_modules.iter().for_each(|(_shader_type, module)| {
                        device.destroy_shader_module(*module, None)
                    });
                    return Err(err.1.into());
                }
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

    fn set_buffer_layout(&mut self, shader: &ShaderModuleType) -> Result<(), GpuError> {
        let direction = self.direction;
        let descriptor_sets = &self.descriptor_sets;
        let buffers = &self.buffers()?;
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
        let buffer_list = if matches!(shader, ShaderModuleType::CrossCheckFilter) {
            vec![buffer_out, buffer_out_reverse]
        } else {
            vec![
                buffers.buffer_img,
                buffer_internal_img1,
                buffer_internal_img2,
                buffers.buffer_internal_int,
                buffer_out,
                buffers.buffer_out_corr,
            ]
        };
        let buffer_infos = buffer_list
            .iter()
            .map(|buf| {
                vk::DescriptorBufferInfo::default()
                    .buffer(buf.buffer)
                    .offset(0)
                    .range(vk::WHOLE_SIZE)
            })
            .collect::<Vec<_>>();
        let write_descriptor = vk::WriteDescriptorSet::default()
            .dst_set(descriptor_sets.descriptor_sets[0])
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .buffer_info(buffer_infos.as_slice());
        unsafe {
            self.device.update_descriptor_sets(&[write_descriptor], &[]);
        }
        Ok(())
    }

    fn create_control(device: &ash::Device, queue_family_index: u32) -> Result<Control, GpuError> {
        let queue = unsafe { device.get_device_queue(queue_family_index, 0) };
        let command_pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(queue_family_index)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let command_pool = unsafe { device.create_command_pool(&command_pool_info, None)? };
        let command_pool = ScopeRollback::new(command_pool, |command_pool| unsafe {
            device.destroy_command_pool(command_pool, None)
        });
        let fence_create_info = vk::FenceCreateInfo::default();
        let fence = unsafe { device.create_fence(&fence_create_info, None)? };
        let fence = ScopeRollback::new(fence, |fence| unsafe { device.destroy_fence(fence, None) });
        let command_buffers_info = vk::CommandBufferAllocateInfo::default()
            .command_buffer_count(1)
            .command_pool(*command_pool.deref())
            .level(vk::CommandBufferLevel::PRIMARY);
        let command_buffer = unsafe { device.allocate_command_buffers(&command_buffers_info)?[0] };
        Ok(Control {
            queue,
            command_pool: command_pool.consume(),
            fence: fence.consume(),
            command_buffer,
        })
    }
}

impl super::Device for Device {
    fn set_buffer_direction(&mut self, direction: &CorrelationDirection) -> Result<(), GpuError> {
        direction.clone_into(&mut self.direction);
        Ok(())
    }

    fn run_shader(
        &mut self,
        dimensions: (usize, usize),
        shader_type: ShaderModuleType,
        shader_params: ShaderParams,
    ) -> Result<(), GpuError> {
        let pipeline_config = self.pipelines.get(&shader_type).unwrap();
        let command_buffer = self.control.command_buffer;

        let workgroup_size = ((dimensions.0 + 15) / 16, ((dimensions.1 + 15) / 16));
        unsafe {
            self.device.reset_fences(&[self.control.fence])?;
            self.device
                .reset_command_buffer(command_buffer, vk::CommandBufferResetFlags::empty())?;
            let info = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            self.device.begin_command_buffer(command_buffer, &info)?;

            self.device.cmd_bind_pipeline(
                self.control.command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                pipeline_config.pipeline,
            );
            self.set_buffer_layout(&shader_type)?;
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
            let submit_info = vk::SubmitInfo::default().command_buffers(&command_buffers);
            self.device
                .queue_submit(self.control.queue, &[submit_info], self.control.fence)?;

            self.device
                .wait_for_fences(&[self.control.fence], true, u64::MAX)?;
        }

        Ok(())
    }

    fn transfer_in_images(&self, img1: &Grid<u8>, img2: &Grid<u8>) -> Result<(), GpuError> {
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

    fn save_corr(
        &self,
        correlation_values: &mut Grid<Option<f32>>,
        correlation_threshold: f32,
    ) -> Result<(), GpuError> {
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

    fn save_result(
        &self,
        out_image: &mut Grid<Option<Match>>,
        correlation_values: &Grid<Option<f32>>,
    ) -> Result<(), GpuError> {
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

    fn destroy_buffers(&mut self) {
        if let Some(buffers) = self.buffers.as_ref() {
            buffers.destroy(&self.device)
        }
        self.buffers = None;
    }
}

impl Buffer {
    fn destroy(&self, device: &ash::Device) {
        unsafe {
            device.free_memory(self.buffer_memory, None);
            device.destroy_buffer(self.buffer, None);
        }
    }
}

impl DeviceBuffers {
    fn destroy(&self, device: &ash::Device) {
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
    fn destroy(&self, device: &ash::Device) {
        unsafe {
            let _ =
                device.free_descriptor_sets(self.descriptor_pool, self.descriptor_sets.as_slice());
            device.destroy_pipeline_layout(self.pipeline_layout, None);
            device.destroy_descriptor_set_layout(self.layout, None);
            device.destroy_descriptor_pool(self.descriptor_pool, None);
        }
    }
}

fn destroy_pipelines(device: &ash::Device, pipelines: &HashMap<ShaderModuleType, ShaderPipeline>) {
    pipelines.values().for_each(|shader_pipeline| unsafe {
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

    fn load(&self, device: &ash::Device) -> Result<vk::ShaderModule, GpuError> {
        const SHADER_INIT_OUT_DATA: &[u8] = include_bytes!("shaders/init_out_data.spv");
        const SHADER_PREPARE_INITIALDATA_SEARCHDATA: &[u8] =
            include_bytes!("shaders/prepare_initialdata_searchdata.spv");
        const SHADER_PREPARE_INITIALDATA_CORRELATION: &[u8] =
            include_bytes!("shaders/prepare_initialdata_correlation.spv");
        const SHADER_PREPARE_SEARCHDATA: &[u8] = include_bytes!("shaders/prepare_searchdata.spv");
        const SHADER_CROSS_CORRELATE: &[u8] = include_bytes!("shaders/cross_correlate.spv");
        const SHADER_CROSS_CHECK_FILTER: &[u8] = include_bytes!("shaders/cross_check_filter.spv");

        let shader_module_spv = match self {
            Self::InitOutData => SHADER_INIT_OUT_DATA,
            Self::PrepareInitialdataSearchdata => SHADER_PREPARE_INITIALDATA_SEARCHDATA,
            Self::PrepareInitialdataCorrelation => SHADER_PREPARE_INITIALDATA_CORRELATION,
            Self::PrepareSearchdata => SHADER_PREPARE_SEARCHDATA,
            Self::CrossCorrelate => SHADER_CROSS_CORRELATE,
            Self::CrossCheckFilter => SHADER_CROSS_CHECK_FILTER,
        };
        let shader_code = ash::util::read_spv(&mut std::io::Cursor::new(shader_module_spv))?;
        let shader_module = unsafe {
            device.create_shader_module(
                &vk::ShaderModuleCreateInfo::default()
                    .flags(vk::ShaderModuleCreateFlags::empty())
                    .code(shader_code.as_slice()),
                None,
            )?
        };

        Ok(shader_module)
    }
}

impl Control {
    fn destroy(&self, device: &ash::Device) {
        unsafe {
            device.free_command_buffers(self.command_pool, &[self.command_buffer]);
            device.destroy_command_pool(self.command_pool, None);
            device.destroy_fence(self.fence, None);
        }
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

struct ScopeRollback<T, F>
where
    F: FnOnce(T),
{
    val: Option<T>,
    rollback: Option<F>,
}

impl<T, F> ScopeRollback<T, F>
where
    F: FnOnce(T),
{
    fn new(val: T, rollback: F) -> ScopeRollback<T, F> {
        ScopeRollback {
            val: Some(val),
            rollback: Some(rollback),
        }
    }

    fn consume(mut self) -> T {
        self.rollback = None;
        self.val.take().unwrap()
    }
}

impl<T, F> Deref for ScopeRollback<T, F>
where
    F: FnOnce(T),
{
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.val.as_ref().unwrap()
    }
}

impl<T, F> DerefMut for ScopeRollback<T, F>
where
    F: FnOnce(T),
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.val.as_mut().unwrap()
    }
}

impl<T, F> Drop for ScopeRollback<T, F>
where
    F: FnOnce(T),
{
    fn drop(&mut self) {
        if let Some(val) = self.val.take() {
            if let Some(rb) = self.rollback.take() {
                rb(val)
            }
        }
    }
}

#[derive(Debug)]
pub enum GpuError {
    Internal(&'static str),
    Vk(&'static str, vk::Result),
    Loading(ash::LoadingError),
    Io(io::Error),
}

impl fmt::Display for GpuError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            Self::Internal(msg) => f.write_str(msg),
            Self::Vk(msg, ref e) => {
                if !msg.is_empty() {
                    write!(f, "Vulkan error: {} ({})", msg, e)
                } else {
                    write!(f, "Vulkan error: {}", e)
                }
            }
            Self::Loading(ref e) => {
                write!(f, "Failed to init GPU: {}", e)
            }
            Self::Io(ref e) => {
                write!(f, "IO error: {}", e)
            }
        }
    }
}

impl std::error::Error for GpuError {
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match *self {
            Self::Internal(_msg) => None,
            Self::Vk(_msg, ref e) => Some(e),
            Self::Loading(ref e) => Some(e),
            Self::Io(ref e) => Some(e),
        }
    }
}

impl From<&'static str> for GpuError {
    fn from(msg: &'static str) -> GpuError {
        Self::Internal(msg)
    }
}

impl From<(&'static str, vk::Result)> for GpuError {
    fn from(e: (&'static str, vk::Result)) -> GpuError {
        Self::Vk(e.0, e.1)
    }
}

impl From<vk::Result> for GpuError {
    fn from(err: vk::Result) -> GpuError {
        Self::Vk("", err)
    }
}

impl From<ash::LoadingError> for GpuError {
    fn from(err: ash::LoadingError) -> GpuError {
        Self::Loading(err)
    }
}

impl From<io::Error> for GpuError {
    fn from(err: io::Error) -> GpuError {
        Self::Io(err)
    }
}
