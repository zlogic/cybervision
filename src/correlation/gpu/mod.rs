#[cfg(target_os = "macos")]
mod metal;
#[cfg(not(target_os = "macos"))]
mod vulkan;

use std::{error, fmt};
#[cfg(not(target_os = "macos"))]
use vulkan::ShaderModuleType;

#[cfg(not(target_os = "macos"))]
pub type DefaultDeviceContext = vulkan::DeviceContext;

use crate::data::Grid;
use nalgebra::Matrix3;

use crate::correlation::{
    CorrelationDirection, CorrelationParameters, HardwareMode, ProjectionMode, CORRIDOR_MIN_RANGE,
    CROSS_CHECK_SEARCH_AREA, KERNEL_SIZE, NEIGHBOR_DISTANCE,
};

use super::Match;

// Decrease when using a low-powered GPU
const CORRIDOR_SEGMENT_LENGTH_HIGHPERFORMANCE: usize = 512;
const SEARCH_AREA_SEGMENT_LENGTH_HIGHPERFORMANCE: usize = 1024;
const CORRIDOR_SEGMENT_LENGTH_LOWPOWER: usize = 8;
const SEARCH_AREA_SEGMENT_LENGTH_LOWPOWER: usize = 128;

trait Device {
    unsafe fn run_shader(
        &mut self,
        dimensions: (usize, usize),
        shader_type: ShaderModuleType,
        shader_params: ShaderParams,
    ) -> Result<(), Box<dyn error::Error>>;

    unsafe fn transfer_in_images(
        &self,
        img1: &Grid<u8>,
        img2: &Grid<u8>,
    ) -> Result<(), Box<dyn error::Error>>;

    unsafe fn save_corr(
        &self,
        correlation_values: &mut Grid<Option<f32>>,
        correlation_threshold: f32,
    ) -> Result<(), Box<dyn error::Error>>;

    unsafe fn save_result(
        &self,
        out_image: &mut Grid<Option<Match>>,
        correlation_values: &Grid<Option<f32>>,
    ) -> Result<(), Box<dyn error::Error>>;

    unsafe fn destroy_buffers(&mut self);
}

trait DeviceContext<D>
where
    D: Device,
{
    fn is_low_power(&self) -> bool;

    fn get_device_name(&self) -> Option<String>;

    fn prepare_device(
        &mut self,
        img1_dimensions: (usize, usize),
        img2_dimensions: (usize, usize),
    ) -> Result<(), Box<dyn error::Error>>;

    fn device(&self) -> Result<&D, GpuError>;

    fn device_mut(&mut self) -> Result<&mut D, GpuError>;
}

#[repr(C)]
#[derive(Copy, Clone)]
struct ShaderParams {
    img1_width: u32,
    img1_height: u32,
    img2_width: u32,
    img2_height: u32,
    out_width: u32,
    out_height: u32,
    scale: f32,
    iteration_pass: u32,
    fundamental_matrix: [f32; 3 * 4],
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

pub struct GpuContext<'a> {
    min_stdev: f32,
    correlation_threshold: f32,
    fundamental_matrix: Matrix3<f64>,
    img1_dimensions: (usize, usize),
    img2_dimensions: (usize, usize),

    correlation_values: Grid<Option<f32>>,

    corridor_segment_length: usize,
    search_area_segment_length: usize,
    corridor_size: usize,
    corridor_extend_range: f64,

    device_context: &'a mut DefaultDeviceContext,
}

impl GpuContext<'_> {
    pub fn new(
        device_context: &mut DefaultDeviceContext,
        img1_dimensions: (usize, usize),
        img2_dimensions: (usize, usize),
        projection_mode: ProjectionMode,
        fundamental_matrix: Matrix3<f64>,
    ) -> Result<GpuContext, Box<dyn error::Error>> {
        let (search_area_segment_length, corridor_segment_length) = if device_context.is_low_power()
        {
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

        device_context.prepare_device(img1_dimensions, img2_dimensions)?;
        let correlation_values = Grid::new(img1_dimensions.0, img1_dimensions.1, None);

        let params = CorrelationParameters::for_projection(&projection_mode);
        Ok(GpuContext {
            min_stdev: params.min_stdev,
            correlation_threshold: params.correlation_threshold,
            corridor_size: params.corridor_size,
            corridor_extend_range: params.corridor_extend_range,
            fundamental_matrix,
            img1_dimensions,
            img2_dimensions,
            correlation_values,
            corridor_segment_length,
            search_area_segment_length,
            device_context,
        })
    }

    pub fn get_device_name(&self) -> String {
        self.device_context.get_device_name().map_or(
            String::from("Error: device not initialized"),
            |device_name| device_name,
        )
    }

    pub fn cross_check_filter(
        &mut self,
        scale: f32,
        dir: CorrelationDirection,
    ) -> Result<(), Box<dyn error::Error>> {
        let device = self.device_context.device_mut()?;
        device.set_buffer_direction(&dir)?;
        let (out_dimensions, out_dimensions_reverse) = match dir {
            CorrelationDirection::Forward => (self.img1_dimensions, self.img2_dimensions),
            CorrelationDirection::Reverse => (self.img2_dimensions, self.img1_dimensions),
        };

        let search_area = CROSS_CHECK_SEARCH_AREA * (1.0 / scale).round() as usize;

        // Reuse/repurpose ShaderParams.
        let params = ShaderParams {
            img1_width: out_dimensions.0 as u32,
            img1_height: out_dimensions.1 as u32,
            img2_width: out_dimensions_reverse.0 as u32,
            img2_height: out_dimensions_reverse.1 as u32,
            out_width: 0,
            out_height: 0,
            fundamental_matrix: [0.0; 3 * 4],
            scale: 0.0,
            iteration_pass: 0,
            corridor_offset: 0,
            corridor_start: 0,
            corridor_end: 0,
            kernel_size: 0,
            threshold: 0.0,
            min_stdev: 0.0,
            neighbor_distance: search_area as u32,
            extend_range: 0.0,
            min_range: 0.0,
        };
        unsafe {
            device.run_shader(out_dimensions, ShaderModuleType::CrossCheckFilter, params)?;
        }
        Ok(())
    }

    pub fn complete_process(
        &mut self,
    ) -> Result<Grid<Option<super::Match>>, Box<dyn error::Error>> {
        let device = self.device_context.device_mut()?;
        let mut out_image = Grid::new(self.img1_dimensions.0, self.img1_dimensions.1, None);
        unsafe {
            device.save_result(&mut out_image, &self.correlation_values)?;
            device.destroy_buffers();
        }
        Ok(out_image)
    }

    pub fn correlate_images<PL: super::ProgressListener>(
        &mut self,
        img1: &Grid<u8>,
        img2: &Grid<u8>,
        scale: f32,
        first_pass: bool,
        progress_listener: Option<&PL>,
        dir: CorrelationDirection,
    ) -> Result<(), Box<dyn error::Error>> {
        {
            let device = self.device_context.device()?;
            device.set_buffer_direction(&dir)?;
        }
        let max_width = img1.width().max(img2.width());
        let max_height = img1.height().max(img2.height());
        let max_dimensions = (max_width, max_height);
        let img1_dimensions = (img1.width(), img1.height());
        let out_dimensions = match dir {
            CorrelationDirection::Forward => self.img1_dimensions,
            CorrelationDirection::Reverse => self.img2_dimensions,
        };

        let mut progressbar_completed_percentage = 0.02;
        let send_progress = |value| {
            let value = match dir {
                CorrelationDirection::Forward => value * 0.98 / 2.0,
                CorrelationDirection::Reverse => 0.51 + value * 0.98 / 2.0,
            };
            if let Some(pl) = progress_listener {
                pl.report_status(value);
            }
        };

        let mut params = ShaderParams {
            img1_width: img1.width() as u32,
            img1_height: img1.height() as u32,
            img2_width: img2.width() as u32,
            img2_height: img2.height() as u32,
            out_width: out_dimensions.0 as u32,
            out_height: out_dimensions.1 as u32,
            fundamental_matrix: self.convert_fundamental_matrix(&dir),
            scale,
            iteration_pass: 0,
            corridor_offset: 0,
            corridor_start: 0,
            corridor_end: 0,
            kernel_size: KERNEL_SIZE as u32,
            threshold: self.correlation_threshold,
            min_stdev: self.min_stdev,
            neighbor_distance: NEIGHBOR_DISTANCE as u32,
            extend_range: self.corridor_extend_range as f32,
            min_range: CORRIDOR_MIN_RANGE as f32,
        };

        let device = self.device_context.device_mut()?;

        unsafe { device.transfer_in_images(img1, img2)? };

        if first_pass {
            unsafe {
                device.run_shader(out_dimensions, ShaderModuleType::InitOutData, params)?;
            }
        } else {
            unsafe {
                device.run_shader(
                    out_dimensions,
                    ShaderModuleType::PrepareInitialdataSearchdata,
                    params,
                )?;
            }
            progressbar_completed_percentage = 0.02;
            send_progress(progressbar_completed_percentage);

            let segment_length = self.search_area_segment_length;
            let neighbor_width = (NEIGHBOR_DISTANCE as f32 / scale).ceil() as usize * 2 + 1;
            let neighbor_pixels = neighbor_width * neighbor_width;
            let neighbor_segments = neighbor_pixels / segment_length + 1;

            params.iteration_pass = 0;
            for l in 0u32..neighbor_segments as u32 {
                params.corridor_start = l * segment_length as u32;
                params.corridor_end = (l + 1) * segment_length as u32;
                if params.corridor_end > neighbor_pixels as u32 {
                    params.corridor_end = neighbor_pixels as u32;
                }
                unsafe {
                    device.run_shader(
                        img1_dimensions,
                        ShaderModuleType::PrepareSearchdata,
                        params,
                    )?;
                }

                let percent_complete =
                    progressbar_completed_percentage + 0.09 * (l as f32 / neighbor_segments as f32);
                send_progress(percent_complete);
            }
            progressbar_completed_percentage = 0.11;
            send_progress(progressbar_completed_percentage);

            params.iteration_pass = 1;
            for l in 0u32..neighbor_segments as u32 {
                params.corridor_start = l * segment_length as u32;
                params.corridor_end = (l + 1) * segment_length as u32;
                if params.corridor_end > neighbor_pixels as u32 {
                    params.corridor_end = neighbor_pixels as u32;
                }
                unsafe {
                    device.run_shader(
                        img1_dimensions,
                        ShaderModuleType::PrepareSearchdata,
                        params,
                    )?;
                }

                let percent_complete =
                    progressbar_completed_percentage + 0.09 * (l as f32 / neighbor_segments as f32);
                send_progress(percent_complete);
            }

            progressbar_completed_percentage = 0.20;
        }
        send_progress(progressbar_completed_percentage);
        params.iteration_pass = if first_pass { 0 } else { 1 };

        unsafe {
            device.run_shader(
                max_dimensions,
                ShaderModuleType::PrepareInitialdataCorrelation,
                params,
            )?;
        }

        let corridor_size = self.corridor_size;
        let corridor_stripes = 2 * corridor_size + 1;
        let max_length = img2.width().max(img2.height());
        let segment_length = self.corridor_segment_length;
        let corridor_length = max_length - (KERNEL_SIZE * 2);
        let corridor_segments = corridor_length / segment_length + 1;
        for corridor_offset in -(corridor_size as i32)..=corridor_size as i32 {
            for l in 0u32..corridor_segments as u32 {
                params.corridor_offset = corridor_offset;
                params.corridor_start = l * segment_length as u32;
                params.corridor_end = (l + 1) * segment_length as u32;
                if params.corridor_end > corridor_length as u32 {
                    params.corridor_end = corridor_length as u32;
                }
                unsafe {
                    device.run_shader(img1_dimensions, ShaderModuleType::CrossCorrelate, params)?;
                }

                let corridor_complete = params.corridor_end as f32 / corridor_length as f32;
                let percent_complete = progressbar_completed_percentage
                    + (1.0 - progressbar_completed_percentage)
                        * (corridor_offset as f32 + corridor_size as f32 + corridor_complete)
                        / corridor_stripes as f32;
                send_progress(percent_complete);
            }
        }

        if matches!(dir, CorrelationDirection::Forward) {
            unsafe { device.save_corr(&mut self.correlation_values, self.correlation_threshold)? };
        }
        Ok(())
    }

    fn convert_fundamental_matrix(&self, dir: &CorrelationDirection) -> [f32; 3 * 4] {
        let fundamental_matrix = match dir {
            CorrelationDirection::Forward => self.fundamental_matrix,
            CorrelationDirection::Reverse => self.fundamental_matrix.transpose(),
        };
        let mut f = [0f32; 3 * 4];
        // Matrix layout in GLSL (OpenGL) is pure madness: https://www.opengl.org/archives/resources/faq/technical/transformations.htm.
        // "Column major" means that vectors are vertical and a matrix multiplies a vector.
        // "Row major" means a horizontal vector multiplies a matrix.
        // This says nothing about how the matrix is stored in memory.
        for row in 0..3 {
            for col in 0..3 {
                f[col * 4 + row] = fundamental_matrix[(row, col)] as f32;
            }
        }
        f
    }
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
