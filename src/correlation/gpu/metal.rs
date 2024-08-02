use std::{collections::HashMap, error, ffi::c_void, fmt, slice};

use metal::objc::rc::autoreleasepool;
use rayon::iter::ParallelIterator;

use crate::{
    correlation::{gpu::ShaderParams, Match},
    data::{Grid, Point2D},
};

use super::{CorrelationDirection, HardwareMode};

// This is optimized to work with built-in Apple Silicon GPUs.
// AMD or built-in Intel GPUs will likely underperform.
// AMD64 devices are no longer available for sale, and testing/validating would require too much
// effort.
pub struct Device {
    device: metal::Device,
    buffers: Option<DeviceBuffers>,
    direction: CorrelationDirection,
    unified_memory: bool,
    pipelines: HashMap<ShaderModuleType, metal::ComputePipelineState>,
    command_queue: metal::CommandQueue,
}

enum BufferType {
    GpuOnly,
    HostVisible,
}

struct DeviceBuffers {
    buffer_img: metal::Buffer,
    buffer_internal_img1: metal::Buffer,
    buffer_internal_img2: metal::Buffer,
    buffer_internal_int: metal::Buffer,
    buffer_out: metal::Buffer,
    buffer_out_reverse: metal::Buffer,
    buffer_out_corr: metal::Buffer,
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

pub struct DeviceContext {
    low_power: bool,
    device: Option<Device>,
}

impl DeviceContext {
    pub fn new(hardware_mode: HardwareMode) -> Result<DeviceContext, GpuError> {
        autoreleasepool(|| {
            if !matches!(hardware_mode, HardwareMode::Gpu | HardwareMode::GpuLowPower) {
                return Err("GPU mode is not enabled".into());
            };
            let low_power = matches!(hardware_mode, HardwareMode::GpuLowPower);
            let device = Device::new()?;
            Ok(DeviceContext {
                low_power,
                device: Some(device),
            })
        })
    }
}

impl super::DeviceContext<Device> for DeviceContext {
    fn is_low_power(&self) -> bool {
        self.low_power
    }

    fn get_device_name(&self) -> Option<String> {
        autoreleasepool(|| {
            self.device()
                .ok()
                .map(|device| device.device.name().to_owned())
        })
    }

    fn prepare_device(
        &mut self,
        img1_dimensions: (usize, usize),
        img2_dimensions: (usize, usize),
    ) -> Result<(), GpuError> {
        let img1_pixels = img1_dimensions.0 * img1_dimensions.1;
        let img2_pixels = img2_dimensions.0 * img2_dimensions.1;

        autoreleasepool(|| {
            let device = self.device_mut()?;
            device.buffers = None;
            let buffers = unsafe { device.create_buffers(img1_pixels, img2_pixels)? };
            device.buffers = Some(buffers);
            device.direction = CorrelationDirection::Forward;
            Ok(())
        })
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

impl Device {
    fn new() -> Result<Device, GpuError> {
        autoreleasepool(|| {
            let device = match metal::Device::system_default() {
                Some(device) => device,
                None => return Err("GPU mode is not enabled".into()),
            };
            let direction = CorrelationDirection::Forward;
            let unified_memory = device.has_unified_memory();
            let pipelines = Self::create_pipelines(&device)?;
            let command_queue = device.new_command_queue();
            Ok(Device {
                device,
                buffers: None,
                direction,
                unified_memory,
                pipelines,
                command_queue,
            })
        })
    }

    unsafe fn create_buffers(
        &self,
        img1_pixels: usize,
        img2_pixels: usize,
    ) -> Result<DeviceBuffers, GpuError> {
        let max_pixels = img1_pixels.max(img2_pixels);
        let buffer_img = self.create_buffer(
            (img1_pixels + img2_pixels) * std::mem::size_of::<f32>(),
            BufferType::HostVisible,
        );

        let buffer_internal_img1 = self.create_buffer(
            (img1_pixels * 2) * std::mem::size_of::<f32>(),
            BufferType::GpuOnly,
        );

        let buffer_internal_img2 = self.create_buffer(
            (img2_pixels * 2) * std::mem::size_of::<f32>(),
            BufferType::GpuOnly,
        );

        let buffer_internal_int = self.create_buffer(
            max_pixels * 4 * std::mem::size_of::<i32>(),
            BufferType::GpuOnly,
        );

        let buffer_out = self.create_buffer(
            img1_pixels * 2 * std::mem::size_of::<i32>(),
            BufferType::HostVisible,
        );

        let buffer_out_reverse = self.create_buffer(
            img2_pixels * 2 * std::mem::size_of::<i32>(),
            BufferType::GpuOnly,
        );

        let buffer_out_corr = self.create_buffer(
            max_pixels * std::mem::size_of::<f32>(),
            BufferType::HostVisible,
        );

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

    unsafe fn create_buffer(&self, size: usize, buffer_type: BufferType) -> metal::Buffer {
        let size = size as u64;
        let options = match buffer_type {
            BufferType::GpuOnly => metal::MTLResourceOptions::StorageModePrivate,
            BufferType::HostVisible => {
                if self.unified_memory {
                    metal::MTLResourceOptions::StorageModeShared
                } else {
                    metal::MTLResourceOptions::StorageModeManaged
                }
            }
        };
        autoreleasepool(|| self.device.new_buffer(size, options))
    }

    fn flush_buffer_to_device(&self, buffer: &metal::Buffer, size: usize) {
        if !self.unified_memory {
            let range = metal::NSRange {
                location: 0,
                length: size as u64,
            };
            buffer.did_modify_range(range);
        }
    }

    fn flush_buffer_to_host(&self, buffer: &metal::Buffer) {
        if !&self.unified_memory {
            let command_buffer = self.command_queue.new_command_buffer();
            let blit_encoder = command_buffer.new_blit_command_encoder();
            blit_encoder.synchronize_resource(buffer);
            blit_encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();
        }
    }

    fn create_pipelines(
        device: &metal::Device,
    ) -> Result<HashMap<ShaderModuleType, metal::ComputePipelineState>, GpuError> {
        autoreleasepool(|| {
            let mut result = HashMap::new();
            let source = include_bytes!("shaders/correlation.metallib");
            let library = device.new_library_with_data(source)?;
            for module_type in ShaderModuleType::VALUES {
                let function = module_type.load(&library)?;
                let pipeline = device.new_compute_pipeline_state_with_function(&function)?;
                result.insert(module_type, pipeline);
            }
            Ok(result)
        })
    }

    fn set_buffer_layout(
        &self,
        shader: &ShaderModuleType,
        command_encoder: &metal::ComputeCommandEncoderRef,
    ) -> Result<(), GpuError> {
        let buffers = &self.buffers()?;
        let (buffer_internal_img1, buffer_internal_img2, buffer_out, buffer_out_reverse) =
            match self.direction {
                CorrelationDirection::Forward => (
                    &buffers.buffer_internal_img1,
                    &buffers.buffer_internal_img2,
                    &buffers.buffer_out,
                    &buffers.buffer_out_reverse,
                ),
                CorrelationDirection::Reverse => (
                    &buffers.buffer_internal_img2,
                    &buffers.buffer_internal_img1,
                    &buffers.buffer_out_reverse,
                    &buffers.buffer_out,
                ),
            };
        // Index 0 is reserved for push_constants.
        if matches!(shader, ShaderModuleType::CrossCheckFilter) {
            command_encoder.set_buffer(1, Some(buffer_out), 0);
            command_encoder.set_buffer(2, Some(buffer_out_reverse), 0);
            return Ok(());
        }

        command_encoder.set_buffer(1, Some(&buffers.buffer_img), 0);
        command_encoder.set_buffer(2, Some(buffer_internal_img1), 0);
        command_encoder.set_buffer(3, Some(buffer_internal_img2), 0);
        command_encoder.set_buffer(4, Some(&buffers.buffer_internal_int), 0);
        command_encoder.set_buffer(5, Some(buffer_out), 0);
        command_encoder.set_buffer(6, Some(&buffers.buffer_out_corr), 0);
        Ok(())
    }
}

impl super::Device for Device {
    fn set_buffer_direction(&mut self, direction: &CorrelationDirection) -> Result<(), GpuError> {
        self.direction.clone_from(&direction.to_owned());
        Ok(())
    }

    unsafe fn run_shader(
        &mut self,
        dimensions: (usize, usize),
        shader_type: ShaderModuleType,
        shader_params: ShaderParams,
    ) -> Result<(), GpuError> {
        let workgroup_size = ((dimensions.0 + 15) / 16, ((dimensions.1 + 15) / 16));
        autoreleasepool(|| {
            let pipeline = self.pipelines.get(&shader_type).unwrap();
            let command_buffer = self.command_queue.new_command_buffer();

            let compute_encoder = command_buffer.new_compute_command_encoder();
            compute_encoder.set_compute_pipeline_state(pipeline);

            let push_constants_data = slice::from_raw_parts(
                &shader_params as *const ShaderParams as *const u8,
                std::mem::size_of::<ShaderParams>(),
            );
            compute_encoder.set_bytes(
                0,
                push_constants_data.len() as u64,
                push_constants_data.as_ptr() as *const c_void,
            );
            self.set_buffer_layout(&shader_type, compute_encoder)?;

            let thread_group_count = metal::MTLSize {
                width: workgroup_size.0 as u64,
                height: workgroup_size.1 as u64,
                depth: 1,
            };
            let thread_group_size = metal::MTLSize {
                width: 16,
                height: 16,
                depth: 1,
            };

            compute_encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
            compute_encoder.end_encoding();

            command_buffer.commit();
            command_buffer.wait_until_completed();

            Ok(())
        })
    }

    unsafe fn transfer_in_images(&self, img1: &Grid<u8>, img2: &Grid<u8>) -> Result<(), GpuError> {
        autoreleasepool(|| {
            let buffers = self.buffers()?;
            let buffer = &buffers.buffer_img;
            let buffer_contents = buffer.contents();
            let img2_offset = img1.width() * img1.height();
            let size = img1.width() * img1.height() + img2.width() * img2.height();
            let img_slice = slice::from_raw_parts_mut(buffer_contents as *mut f32, size);
            img1.iter()
                .for_each(|(x, y, val)| img_slice[y * img1.width() + x] = *val as f32);
            img2.iter().for_each(|(x, y, val)| {
                img_slice[img2_offset + y * img2.width() + x] = *val as f32
            });

            self.flush_buffer_to_device(buffer, std::mem::size_of_val(img_slice));
            Ok(())
        })
    }

    unsafe fn save_corr(
        &self,
        correlation_values: &mut Grid<Option<f32>>,
        correlation_threshold: f32,
    ) -> Result<(), GpuError> {
        autoreleasepool(|| {
            let buffers = self.buffers()?;
            let buffer = &buffers.buffer_out_corr;
            self.flush_buffer_to_host(buffer);
            let buffer_contents = buffer.contents();
            let size = correlation_values.width() * correlation_values.height();
            let width = correlation_values.width();
            let out_corr = slice::from_raw_parts_mut(buffer_contents as *mut f32, size);
            correlation_values
                .par_iter_mut()
                .for_each(|(x, y, out_point)| {
                    let corr = out_corr[y * width + x];
                    if corr > correlation_threshold {
                        *out_point = Some(corr);
                    }
                });

            Ok(())
        })
    }

    unsafe fn save_result(
        &self,
        out_image: &mut Grid<Option<Match>>,
        correlation_values: &Grid<Option<f32>>,
    ) -> Result<(), GpuError> {
        autoreleasepool(|| {
            let buffers = self.buffers()?;
            let buffer = &buffers.buffer_out;
            self.flush_buffer_to_host(buffer);
            let buffer_contents = buffer.contents();
            let size = out_image.width() * out_image.height() * 2;
            let width = out_image.width();
            let out_data = slice::from_raw_parts_mut(buffer_contents as *mut i32, size);
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

            Ok(())
        })
    }

    unsafe fn destroy_buffers(&mut self) {
        autoreleasepool(|| {
            self.buffers = None;
        })
    }
}

impl Drop for DeviceContext {
    fn drop(&mut self) {
        autoreleasepool(|| {
            if let Some(device) = self.device.as_mut() {
                device.pipelines.clear();
                device.buffers = None;
            };
            self.device = None;
        });
    }
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

    fn load(&self, library: &metal::Library) -> Result<metal::Function, GpuError> {
        let function_name = match self {
            Self::InitOutData => "init_out_data",
            Self::PrepareInitialdataSearchdata => "prepare_initialdata_searchdata",
            Self::PrepareInitialdataCorrelation => "prepare_initialdata_correlation",
            Self::PrepareSearchdata => "prepare_searchdata",
            Self::CrossCorrelate => "cross_correlate",
            Self::CrossCheckFilter => "cross_check_filter",
        };
        let function = library.get_function(function_name, None)?;

        Ok(function)
    }
}

#[derive(Debug)]
pub enum GpuError {
    Internal(&'static str),
    Metal(String),
}

impl fmt::Display for GpuError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            Self::Internal(msg) => f.write_str(msg),
            Self::Metal(ref msg) => f.write_str(msg),
        }
    }
}

impl std::error::Error for GpuError {
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match *self {
            Self::Internal(_msg) => None,
            Self::Metal(ref _msg) => None,
        }
    }
}

impl From<String> for GpuError {
    fn from(e: String) -> GpuError {
        Self::Metal(e)
    }
}

impl From<&'static str> for GpuError {
    fn from(msg: &'static str) -> GpuError {
        Self::Internal(msg)
    }
}
