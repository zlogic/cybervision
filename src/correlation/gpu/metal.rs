use std::{collections::HashMap, error, fmt, ptr::NonNull, slice};

use objc2::{
    rc::{Retained, autoreleasepool},
    runtime::ProtocolObject,
};
use objc2_foundation::{NSRange, NSString};
use objc2_metal::{
    MTLBlitCommandEncoder as _, MTLBuffer, MTLCommandBuffer as _, MTLCommandEncoder as _,
    MTLCommandQueue, MTLComputeCommandEncoder, MTLComputePipelineState, MTLDevice, MTLFunction,
    MTLLibrary, MTLResourceOptions, MTLSize,
};
use rayon::iter::ParallelIterator;

use crate::{
    correlation::{Match, gpu::ShaderParams},
    data::{Grid, Point2D},
};

use super::{CorrelationDirection, HardwareMode};

// This is optimized to work with built-in Apple Silicon GPUs.
// AMD or built-in Intel GPUs will likely underperform.
// AMD64 devices are no longer available for sale, and testing/validating would require too much
// effort.
pub struct Device {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    buffers: Option<DeviceBuffers>,
    direction: CorrelationDirection,
    unified_memory: bool,
    pipelines: HashMap<ShaderModuleType, Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    command_queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
}

unsafe impl Sync for Device {}

enum BufferType {
    GpuOnly,
    HostVisible,
}

struct DeviceBuffers {
    buffer_img: Retained<ProtocolObject<dyn MTLBuffer>>,
    buffer_internal_img1: Retained<ProtocolObject<dyn MTLBuffer>>,
    buffer_internal_img2: Retained<ProtocolObject<dyn MTLBuffer>>,
    buffer_internal_int: Retained<ProtocolObject<dyn MTLBuffer>>,
    buffer_out: Retained<ProtocolObject<dyn MTLBuffer>>,
    buffer_out_reverse: Retained<ProtocolObject<dyn MTLBuffer>>,
    buffer_out_corr: Retained<ProtocolObject<dyn MTLBuffer>>,
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
        autoreleasepool(|_| {
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
        autoreleasepool(|_| {
            self.device()
                .ok()
                .map(|device| device.device.name().to_string())
        })
    }

    fn prepare_device(
        &mut self,
        img1_dimensions: (usize, usize),
        img2_dimensions: (usize, usize),
    ) -> Result<(), GpuError> {
        let img1_pixels = img1_dimensions.0 * img1_dimensions.1;
        let img2_pixels = img2_dimensions.0 * img2_dimensions.1;

        autoreleasepool(|_| {
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
        autoreleasepool(|_| {
            let device = match objc2_metal::MTLCreateSystemDefaultDevice() {
                Some(device) => device,
                None => return Err("GPU mode is not enabled".into()),
            };
            let direction = CorrelationDirection::Forward;
            let unified_memory = device.hasUnifiedMemory();
            let pipelines = Self::create_pipelines(&device)?;
            let command_queue = match device.newCommandQueue() {
                Some(command_queue) => command_queue,
                None => return Err("Command queue is not available".into()),
            };
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
        )?;

        let buffer_internal_img1 = self.create_buffer(
            (img1_pixels * 2) * std::mem::size_of::<f32>(),
            BufferType::GpuOnly,
        )?;

        let buffer_internal_img2 = self.create_buffer(
            (img2_pixels * 2) * std::mem::size_of::<f32>(),
            BufferType::GpuOnly,
        )?;

        let buffer_internal_int = self.create_buffer(
            max_pixels * 4 * std::mem::size_of::<i32>(),
            BufferType::GpuOnly,
        )?;

        let buffer_out = self.create_buffer(
            img1_pixels * 2 * std::mem::size_of::<i32>(),
            BufferType::HostVisible,
        )?;

        let buffer_out_reverse = self.create_buffer(
            img2_pixels * 2 * std::mem::size_of::<i32>(),
            BufferType::GpuOnly,
        )?;

        let buffer_out_corr = self.create_buffer(
            max_pixels * std::mem::size_of::<f32>(),
            BufferType::HostVisible,
        )?;

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
        &self,
        size: usize,
        buffer_type: BufferType,
    ) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, GpuError> {
        let options = match buffer_type {
            BufferType::GpuOnly => MTLResourceOptions::StorageModePrivate,
            BufferType::HostVisible => {
                if self.unified_memory {
                    MTLResourceOptions::StorageModeShared
                } else {
                    MTLResourceOptions::StorageModeManaged
                }
            }
        };
        autoreleasepool(
            |_| match self.device.newBufferWithLength_options(size, options) {
                Some(buffer) => Ok(buffer),
                None => Err("Failed to create buffer".into()),
            },
        )
    }

    fn flush_buffer_to_device(
        &self,
        buffer: &Retained<ProtocolObject<dyn MTLBuffer>>,
        size: usize,
    ) {
        if !self.unified_memory {
            let range = NSRange {
                location: 0,
                length: size,
            };
            buffer.didModifyRange(range);
        }
    }

    fn flush_buffer_to_host(
        &self,
        buffer: &Retained<ProtocolObject<dyn MTLBuffer>>,
    ) -> Result<(), GpuError> {
        if !&self.unified_memory {
            let command_buffer = match self.command_queue.commandBuffer() {
                Some(command_buffer) => command_buffer,
                None => return Err("Command queue has no command buffer".into()),
            };
            let blit_encoder = match command_buffer.blitCommandEncoder() {
                Some(blit_encoder) => blit_encoder,
                None => return Err("Command buffer has no blit command encoder".into()),
            };
            blit_encoder.synchronizeResource(buffer.as_ref());
            blit_encoder.endEncoding();
            command_buffer.commit();
            command_buffer.waitUntilCompleted();
        }
        Ok(())
    }

    fn create_pipelines(
        device: &Retained<ProtocolObject<dyn MTLDevice>>,
    ) -> Result<
        HashMap<ShaderModuleType, Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
        GpuError,
    > {
        autoreleasepool(|_| {
            let mut result = HashMap::new();
            let source = include_bytes!("shaders/correlation.metallib");
            let library =
                device.newLibraryWithData_error(&dispatch2::DispatchData::from_bytes(source))?;
            for module_type in ShaderModuleType::VALUES {
                let function = module_type.load(&library)?;
                let pipeline = device.newComputePipelineStateWithFunction_error(&function)?;
                result.insert(module_type, pipeline);
            }
            Ok(result)
        })
    }

    fn set_buffer_layout(
        &self,
        shader: &ShaderModuleType,
        command_encoder: &Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>,
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
            unsafe {
                command_encoder.setBuffer_offset_atIndex(Some(buffer_out), 0, 1);
                command_encoder.setBuffer_offset_atIndex(Some(buffer_out_reverse), 0, 2);
            }
            return Ok(());
        }

        unsafe {
            command_encoder.setBuffer_offset_atIndex(Some(&buffers.buffer_img), 0, 1);
            command_encoder.setBuffer_offset_atIndex(Some(buffer_internal_img1), 0, 2);
            command_encoder.setBuffer_offset_atIndex(Some(buffer_internal_img2), 0, 3);
            command_encoder.setBuffer_offset_atIndex(Some(&buffers.buffer_internal_int), 0, 4);
            command_encoder.setBuffer_offset_atIndex(Some(buffer_out), 0, 5);
            command_encoder.setBuffer_offset_atIndex(Some(&buffers.buffer_out_corr), 0, 6);
        }
        Ok(())
    }
}

impl super::Device for Device {
    fn set_buffer_direction(&mut self, direction: &CorrelationDirection) -> Result<(), GpuError> {
        self.direction.clone_from(&direction.to_owned());
        Ok(())
    }

    fn run_shader(
        &mut self,
        dimensions: (usize, usize),
        shader_type: ShaderModuleType,
        shader_params: ShaderParams,
    ) -> Result<(), GpuError> {
        let workgroup_size = (dimensions.0.div_ceil(16), dimensions.1.div_ceil(16));
        autoreleasepool(|_| {
            let pipeline = self.pipelines.get(&shader_type).unwrap();
            let command_buffer = match self.command_queue.commandBuffer() {
                Some(command_buffer) => command_buffer,
                None => return Err("Command queue has no command buffer".into()),
            };

            let compute_encoder = match command_buffer.computeCommandEncoder() {
                Some(command_encoder) => command_encoder,
                None => return Err("Command buffer has no command encoder".into()),
            };
            compute_encoder.setComputePipelineState(pipeline);

            let push_constants_data = NonNull::from_ref(&shader_params);
            unsafe {
                compute_encoder.setBytes_length_atIndex(
                    push_constants_data.cast(),
                    std::mem::size_of::<ShaderParams>(),
                    0,
                );
            }
            self.set_buffer_layout(&shader_type, &compute_encoder)?;

            let thread_group_count = MTLSize {
                width: workgroup_size.0,
                height: workgroup_size.1,
                depth: 1,
            };
            let thread_group_size = MTLSize {
                width: 16,
                height: 16,
                depth: 1,
            };

            compute_encoder
                .dispatchThreadgroups_threadsPerThreadgroup(thread_group_count, thread_group_size);
            compute_encoder.endEncoding();

            command_buffer.commit();
            command_buffer.waitUntilCompleted();

            Ok(())
        })
    }

    fn transfer_in_images(&self, img1: &Grid<u8>, img2: &Grid<u8>) -> Result<(), GpuError> {
        autoreleasepool(|_| {
            let buffers = self.buffers()?;
            let buffer = &buffers.buffer_img;
            let buffer_contents = buffer.contents();
            let img2_offset = img1.width() * img1.height();
            let size = img1.width() * img1.height() + img2.width() * img2.height();
            if buffer.length() < size {
                return Err("Image buffer is smaller than expected".into());
            }
            let img_slice =
                unsafe { slice::from_raw_parts_mut(buffer_contents.cast::<f32>().as_ptr(), size) };
            img1.iter()
                .for_each(|(x, y, val)| img_slice[y * img1.width() + x] = *val as f32);
            img2.iter().for_each(|(x, y, val)| {
                img_slice[img2_offset + y * img2.width() + x] = *val as f32
            });

            self.flush_buffer_to_device(buffer, std::mem::size_of_val(img_slice));
            Ok(())
        })
    }

    fn save_corr(
        &self,
        correlation_values: &mut Grid<Option<f32>>,
        correlation_threshold: f32,
    ) -> Result<(), GpuError> {
        autoreleasepool(|_| {
            let buffers = self.buffers()?;
            let buffer = &buffers.buffer_out_corr;
            self.flush_buffer_to_host(buffer)?;
            let buffer_contents = buffer.contents();
            let size = correlation_values.width() * correlation_values.height();
            let width = correlation_values.width();
            if buffer.length() < size {
                return Err("Output correlation buffer is smaller than expected".into());
            }
            let out_corr =
                unsafe { slice::from_raw_parts_mut(buffer_contents.cast::<f32>().as_ptr(), size) };
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

    fn save_result(
        &self,
        out_image: &mut Grid<Option<Match>>,
        correlation_values: &Grid<Option<f32>>,
    ) -> Result<(), GpuError> {
        autoreleasepool(|_| {
            let buffers = self.buffers()?;
            let buffer = &buffers.buffer_out;
            self.flush_buffer_to_host(buffer)?;
            let buffer_contents = buffer.contents();
            let size = out_image.width() * out_image.height() * 2;
            let width = out_image.width();
            if buffer.length() < size {
                return Err("Output buffer is smaller than expected".into());
            }
            let out_data =
                unsafe { slice::from_raw_parts_mut(buffer_contents.cast::<i32>().as_ptr(), size) };
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

    fn destroy_buffers(&mut self) {
        autoreleasepool(|_| {
            self.buffers = None;
        })
    }
}

impl Drop for DeviceContext {
    fn drop(&mut self) {
        autoreleasepool(|_| {
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

    fn load(
        &self,
        library: &Retained<ProtocolObject<dyn MTLLibrary>>,
    ) -> Result<Retained<ProtocolObject<dyn MTLFunction>>, GpuError> {
        let function_name = match self {
            Self::InitOutData => "init_out_data",
            Self::PrepareInitialdataSearchdata => "prepare_initialdata_searchdata",
            Self::PrepareInitialdataCorrelation => "prepare_initialdata_correlation",
            Self::PrepareSearchdata => "prepare_searchdata",
            Self::CrossCorrelate => "cross_correlate",
            Self::CrossCheckFilter => "cross_check_filter",
        };
        match library.newFunctionWithName(&NSString::from_str(function_name)) {
            Some(function) => Ok(function),
            None => Err("Function not found in library".into()),
        }
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

impl From<Retained<objc2_foundation::NSError>> for GpuError {
    fn from(e: Retained<objc2_foundation::NSError>) -> GpuError {
        Self::Metal(e.to_string())
    }
}

impl From<&'static str> for GpuError {
    fn from(msg: &'static str) -> GpuError {
        Self::Internal(msg)
    }
}
