use std::{
    borrow::Cow,
    error,
    ffi::{CStr, CString},
    fmt,
};

use ash::{prelude::VkResult, vk};
use nalgebra::Matrix3;

use crate::data::{Grid, Point2D};

use super::{
    CorrelationDirection, CorrelationParameters, ProjectionMode, CORRIDOR_MIN_RANGE,
    CROSS_CHECK_SEARCH_AREA, KERNEL_SIZE, NEIGHBOR_DISTANCE,
};

const MAX_BINDINGS: u32 = 6;
// Decrease when using a low-powered GPU
const CORRIDOR_SEGMENT_LENGTH_HIGHPERFORMANCE: usize = 512;
const SEARCH_AREA_SEGMENT_LENGTH_HIGHPERFORMANCE: usize = 1024;
const CORRIDOR_SEGMENT_LENGTH_LOWPOWER: usize = 8;
const SEARCH_AREA_SEGMENT_LENGTH_LOWPOWER: usize = 128;

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq)]
struct ShaderParams {
    img1_width: u32,
    img1_height: u32,
    img2_width: u32,
    img2_height: u32,
    out_width: u32,
    out_height: u32,
    scale: f32,
    iteration_pass: u32,
    fundamental_matrix: [f32; 3 * 4], // matrices are column-major and each column is aligned to 4-component vectors; should be aligned to 16 bytes
    corridor_offset: i32,
    corridor_start: u32,
    corridor_end: u32,
    kernel_size: u32,
    threshold: f32,
    min_stdev: f32,
    neighbor_distance: u32,
    extend_range: f32,
    min_range: f32,
}

pub struct GpuContext {
    min_stdev: f32,
    correlation_threshold: f32,
    fundamental_matrix: Matrix3<f64>,
    img1_shape: (usize, usize),
    img2_shape: (usize, usize),

    correlation_values: Grid<Option<f32>>,

    corridor_segment_length: usize,
    search_area_segment_length: usize,
    corridor_size: usize,
    corridor_extend_range: f64,

    device_name: String,
    entry: ash::Entry,
    instance: ash::Instance,
    physical_device: vk::PhysicalDevice,
    /*
    device: vk::Device,
    device: wgpu::Device,
    queue: wgpu::Queue,
    shader_module: wgpu::ShaderModule,
    buffer_img: wgpu::Buffer,
    buffer_internal_img1: wgpu::Buffer,
    buffer_internal_img2: wgpu::Buffer,
    buffer_internal_int: wgpu::Buffer,
    buffer_out: wgpu::Buffer,
    buffer_out_reverse: wgpu::Buffer,
    buffer_out_corr: wgpu::Buffer,

    pipeline_configs: HashMap<String, ComputePipelineConfig>,
    */
}

impl GpuContext {
    pub fn new(
        img1_dimensions: (usize, usize),
        img2_dimensions: (usize, usize),
        projection_mode: ProjectionMode,
        fundamental_matrix: Matrix3<f64>,
        low_power: bool,
    ) -> Result<GpuContext, Box<dyn error::Error>> {
        let img1_shape = (img1_dimensions.0, img1_dimensions.1);
        let img2_shape = (img2_dimensions.0, img2_dimensions.1);

        let img1_pixels = img1_dimensions.0 * img1_dimensions.1;
        let img2_pixels = img2_dimensions.0 * img2_dimensions.1;
        let max_pixels = img1_pixels.max(img2_pixels);

        let (search_area_segment_length, corridor_segment_length) = if low_power {
            (
                SEARCH_AREA_SEGMENT_LENGTH_LOWPOWER,
                CORRIDOR_SEGMENT_LENGTH_LOWPOWER,
            )
        } else {
            (
                SEARCH_AREA_SEGMENT_LENGTH_HIGHPERFORMANCE,
                CORRIDOR_SEGMENT_LENGTH_HIGHPERFORMANCE,
            )
        };

        // Ensure there's enough memory for the largest buffer.
        let max_buffer_size = max_pixels * 4 * std::mem::size_of::<i32>();
        // Init adapter.
        let entry = unsafe {
            match ash::Entry::load() {
                Ok(entry) => entry,
                Err(err) => return Err(err.into()),
            }
        };
        let instance = match init_vk(&entry) {
            Ok(instance) => instance,
            Err(err) => return Err(err.into()),
        };
        let (physical_device, device_name) = unsafe {
            match find_device(&instance, max_buffer_size) {
                Ok(dev) => dev,
                Err(err) => return Err(err.into()),
            }
        };
        let device_name = device_name.to_string();
        //let device = instance.create_device(physical_device, None, None)?;

        /*
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("correlation.wgsl"))),
        });

        // Init buffers.
        let buffer_img = init_buffer(
            &device,
            (img1_pixels + img2_pixels) * std::mem::size_of::<f32>(),
            false,
            true,
        );
        let buffer_internal_img1 = init_buffer(
            &device,
            (img1_pixels * 2) * std::mem::size_of::<f32>(),
            true,
            false,
        );
        let buffer_internal_img2 = init_buffer(
            &device,
            (img2_pixels * 2) * std::mem::size_of::<f32>(),
            true,
            false,
        );
        let buffer_internal_int = init_buffer(
            &device,
            max_pixels * 4 * std::mem::size_of::<i32>(),
            true,
            false,
        );
        let buffer_out = init_buffer(
            &device,
            max_pixels * 2 * std::mem::size_of::<i32>(),
            false,
            false,
        );
        let buffer_out_reverse = init_buffer(
            &device,
            img2_pixels * 2 * std::mem::size_of::<i32>(),
            true,
            false,
        );
        let buffer_out_corr = init_buffer(
            &device,
            max_pixels * std::mem::size_of::<f32>(),
            false,
            false,
        );
        */

        let correlation_values = Grid::new(img1_shape.0, img1_shape.1, None);

        let params = CorrelationParameters::for_projection(&projection_mode);
        let result = GpuContext {
            min_stdev: params.min_stdev,
            correlation_threshold: params.correlation_threshold,
            corridor_size: params.corridor_size,
            corridor_extend_range: params.corridor_extend_range,
            fundamental_matrix,
            img1_shape,
            img2_shape,
            correlation_values,
            corridor_segment_length,
            search_area_segment_length,
            device_name,
            entry,
            instance,
            physical_device,
            /*
            device,
            queue,
            shader_module,
            buffer_img,
            buffer_internal_img1,
            buffer_internal_img2,
            buffer_internal_int,
            buffer_out,
            buffer_out_reverse,
            buffer_out_corr,
            pipeline_configs: HashMap::new(),
            */
        };
        Ok(result)
    }

    pub fn get_device_name(&self) -> &str {
        &self.device_name.as_str()
    }

    pub fn cross_check_filter(&mut self, _: f32, _: CorrelationDirection) {}

    pub fn complete_process(
        &mut self,
    ) -> Result<Grid<Option<super::Match>>, Box<dyn error::Error>> {
        todo!()
    }

    pub fn correlate_images<PL: super::ProgressListener>(
        &mut self,
        _: &Grid<u8>,
        _: &Grid<u8>,
        _: f32,
        _: bool,
        _: Option<&PL>,
        _: CorrelationDirection,
    ) -> Result<(), Box<dyn error::Error>> {
        todo!()
    }
}

impl Drop for GpuContext {
    fn drop(&mut self) {
        unsafe {
            //self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

fn init_vk(entry: &ash::Entry) -> VkResult<ash::Instance> {
    let app_name = CString::new("Cybervision").unwrap();
    let engine_name = CString::new("cybervision").unwrap();
    let appinfo = vk::ApplicationInfo::builder()
        .application_name(app_name.as_c_str())
        .application_version(0)
        .engine_name(engine_name.as_c_str())
        .engine_version(0)
        .api_version(vk::make_api_version(0, 1, 0, 0));

    let create_flags = vk::InstanceCreateFlags::default();

    let create_info = vk::InstanceCreateInfo::builder()
        .application_info(&appinfo)
        .flags(create_flags);
    unsafe { entry.create_instance(&create_info, None) }
}

unsafe fn find_device(
    instance: &ash::Instance,
    max_buffer_size: usize,
) -> Result<(vk::PhysicalDevice, &'static str), Box<dyn error::Error>> {
    let devices = instance.enumerate_physical_devices()?;
    let device = devices
        .iter()
        .filter_map(|device| {
            let props = instance.get_physical_device_properties(*device);
            if props.limits.max_push_constants_size < std::mem::size_of::<ShaderParams>() as u32
                || props.limits.max_per_stage_descriptor_storage_buffers < MAX_BINDINGS
                || props.limits.max_storage_buffer_range < max_buffer_size as u32
            {
                return None;
            }

            let device_name = CStr::from_ptr(props.device_name.as_ptr());
            let device_name = device_name.to_str().unwrap();
            // TODO: allow to specify a device name filter/regex?
            let score = match props.device_type {
                vk::PhysicalDeviceType::DISCRETE_GPU => 3,
                vk::PhysicalDeviceType::VIRTUAL_GPU => 2,
                vk::PhysicalDeviceType::INTEGRATED_GPU => 1,
                _ => 0,
            };
            Some((device.to_owned(), device_name, score))
        })
        .max_by_key(|(_device, _name, score)| *score);
    let (device, name) = if let Some((device, name, _score)) = device {
        (device, name)
    } else {
        return Err(GpuError::new("Device not found").into());
    };
    Ok((device, name))
}

#[derive(Debug)]
pub struct GpuError {
    msg: &'static str,
}

impl GpuError {
    fn new(msg: &'static str) -> GpuError {
        GpuError { msg }
    }
}

impl std::error::Error for GpuError {}

impl fmt::Display for GpuError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.msg)
    }
}
