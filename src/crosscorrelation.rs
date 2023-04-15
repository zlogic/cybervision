use crate::correlation;
use nalgebra::{DMatrix, Matrix3, Vector3};
use rayon::prelude::*;
use std::{cell::RefCell, error, ops::Range, sync::atomic::AtomicUsize, sync::atomic::Ordering};

const SCALE_MIN_SIZE: usize = 64;
const KERNEL_SIZE: usize = 5;
const KERNEL_WIDTH: usize = KERNEL_SIZE * 2 + 1;
const KERNEL_POINT_COUNT: usize = KERNEL_WIDTH * KERNEL_WIDTH;

const THRESHOLD_AFFINE: f32 = 0.6;
const THRESHOLD_PERSPECTIVE: f32 = 0.6;
const MIN_STDEV_AFFINE: f32 = 1.0;
const MIN_STDEV_PERSPECTIVE: f32 = 5.0;
const CORRIDOR_SIZE: usize = 2;
// Decrease when using a low-powered GPU
const CORRIDOR_SEGMENT_LENGTH_HIGHPERFORMANCE: usize = 512;
const SEARCH_AREA_SEGMENT_LENGTH_HIGHPERFORMANCE: usize = 1024;
const CORRIDOR_SEGMENT_LENGTH_LOWPOWER: usize = 8;
const SEARCH_AREA_SEGMENT_LENGTH_LOWPOWER: usize = 128;
const NEIGHBOR_DISTANCE: usize = 10;
const CORRIDOR_EXTEND_RANGE: f64 = 1.0;
const CORRIDOR_MIN_RANGE: f64 = 2.5;
const CROSS_CHECK_SEARCH_AREA: usize = 2;

type Match = (u32, u32);

#[derive(Debug, Clone, Copy)]
pub enum ProjectionMode {
    Affine,
    Perspective,
}

#[derive(Debug, Clone, Copy)]
pub enum HardwareMode {
    Gpu,
    GpuLowPower,
    Cpu,
}

pub trait ProgressListener
where
    Self: Sync + Sized,
{
    fn report_status(&self, pos: f32);
}

pub struct PointCorrelations {
    pub correlated_points: DMatrix<Option<Match>>,
    correlated_points_reverse: DMatrix<Option<Match>>,
    first_pass: bool,
    min_stdev: f32,
    correlation_threshold: f32,
    fundamental_matrix: Matrix3<f64>,
    gpu_context: Option<gpu::GpuContext>,
    selected_hardware: String,
}

enum CorrelationDirection {
    Forward,
    Reverse,
}

struct EpipolarLine {
    coeff: (f64, f64),
    add: (f64, f64),
    corridor_offset: (isize, isize),
}

struct BestMatch {
    pos: Option<Match>,
    corr: Option<f32>,
}

struct CorrelationStep<'a> {
    scale: f32,
    fundamental_matrix: Matrix3<f64>,
    correlated_points: &'a DMatrix<Option<Match>>,
    img1: &'a DMatrix<u8>,
    img2: &'a DMatrix<u8>,
    img2_data: ImagePointData,
}

impl PointCorrelations {
    pub fn new(
        img1_dimensions: (u32, u32),
        img2_dimensions: (u32, u32),
        fundamental_matrix: Matrix3<f64>,
        projection_mode: ProjectionMode,
        hardware_mode: HardwareMode,
    ) -> PointCorrelations {
        let (min_stdev, correlation_threshold) = match projection_mode {
            ProjectionMode::Affine => (MIN_STDEV_AFFINE, THRESHOLD_AFFINE),
            ProjectionMode::Perspective => (MIN_STDEV_PERSPECTIVE, THRESHOLD_PERSPECTIVE),
        };
        let selected_hardware;
        let gpu_context = if matches!(hardware_mode, HardwareMode::Gpu | HardwareMode::GpuLowPower)
        {
            let low_power = matches!(hardware_mode, HardwareMode::GpuLowPower);
            match gpu::GpuContext::new(
                (img1_dimensions.0 as usize, img1_dimensions.1 as usize),
                (img2_dimensions.0 as usize, img2_dimensions.1 as usize),
                min_stdev,
                correlation_threshold,
                fundamental_matrix,
                low_power,
            ) {
                Ok(gpu_context) => {
                    selected_hardware = format!("GPU {}", gpu_context.get_device_name());
                    Some(gpu_context)
                }
                Err(err) => {
                    selected_hardware = format!("CPU fallback ({})", err);
                    None
                }
            }
        } else {
            selected_hardware = "CPU".to_string();
            None
        };

        // Height specifies rows, width specifies columns.
        let (correlated_points, correlated_points_reverse) = match gpu_context {
            Some(_) => (
                DMatrix::from_element(0, 0, None),
                DMatrix::from_element(0, 0, None),
            ),
            None => (
                DMatrix::from_element(img1_dimensions.1 as usize, img1_dimensions.0 as usize, None),
                DMatrix::from_element(img2_dimensions.1 as usize, img2_dimensions.0 as usize, None),
            ),
        };
        PointCorrelations {
            correlated_points,
            correlated_points_reverse,
            first_pass: true,
            min_stdev,
            correlation_threshold,
            fundamental_matrix,
            gpu_context,
            selected_hardware,
        }
    }

    pub fn get_selected_hardware(&self) -> &String {
        &self.selected_hardware
    }

    pub fn complete(&mut self) -> Result<(), Box<dyn error::Error>> {
        self.correlated_points_reverse = DMatrix::from_element(0, 0, None);
        if let Some(gpu_context) = &mut self.gpu_context {
            match gpu_context.complete_process() {
                Ok(correlated_points) => self.correlated_points = correlated_points,
                Err(err) => return Err(err),
            };
            self.gpu_context = None;
        }
        Ok(())
    }

    pub fn correlate_images<PL: ProgressListener>(
        &mut self,
        img1: DMatrix<u8>,
        img2: DMatrix<u8>,
        scale: f32,
        progress_listener: Option<&PL>,
    ) {
        self.correlate_images_step(
            &img1,
            &img2,
            scale,
            progress_listener,
            CorrelationDirection::Forward,
        );
        self.correlate_images_step(
            &img2,
            &img1,
            scale,
            progress_listener,
            CorrelationDirection::Reverse,
        );

        self.cross_check_filter(scale, CorrelationDirection::Forward);
        self.cross_check_filter(scale, CorrelationDirection::Reverse);

        self.first_pass = false;
    }

    fn correlate_images_step<PL: ProgressListener>(
        &mut self,
        img1: &DMatrix<u8>,
        img2: &DMatrix<u8>,
        scale: f32,
        progress_listener: Option<&PL>,
        dir: CorrelationDirection,
    ) {
        if let Some(gpu_context) = &mut self.gpu_context {
            let dir = match dir {
                CorrelationDirection::Forward => gpu::CorrelationDirection::Forward,
                CorrelationDirection::Reverse => gpu::CorrelationDirection::Reverse,
            };
            gpu_context.correlate_images(
                img1,
                img2,
                scale,
                self.first_pass,
                progress_listener,
                dir,
            );
            return;
        };
        let img2_data = compute_image_point_data(img2);
        let mut out_data: DMatrix<Option<Match>> =
            DMatrix::from_element(img1.shape().0, img1.shape().1, None);

        let correlated_points = match dir {
            CorrelationDirection::Forward => &self.correlated_points,
            CorrelationDirection::Reverse => &self.correlated_points_reverse,
        };

        let fundamental_matrix = match dir {
            CorrelationDirection::Forward => self.fundamental_matrix,
            CorrelationDirection::Reverse => self.fundamental_matrix.transpose(),
        };

        let corelation_step = CorrelationStep {
            scale,
            fundamental_matrix,
            correlated_points,
            img1,
            img2,
            img2_data,
        };

        let counter = AtomicUsize::new(0);
        let out_data_cols = out_data.ncols().saturating_sub(KERNEL_SIZE * 2) as f32;
        let (nrows, ncols) = out_data.shape();
        out_data
            .column_iter_mut()
            .enumerate()
            .par_bridge()
            .for_each(|(col, mut out_col)| {
                if col < KERNEL_SIZE || col >= ncols - KERNEL_SIZE {
                    return;
                }
                if let Some(pl) = progress_listener {
                    let value = counter.fetch_add(1, Ordering::Relaxed) as f32 / out_data_cols;
                    let value = match dir {
                        CorrelationDirection::Forward => value / 2.0,
                        CorrelationDirection::Reverse => 0.5 + value / 2.0,
                    };
                    pl.report_status(value);
                }
                out_col.iter_mut().enumerate().for_each(|(row, out_point)| {
                    if row < KERNEL_SIZE || row >= nrows - KERNEL_SIZE {
                        return;
                    }
                    self.correlate_point(&corelation_step, row, col, out_point);
                })
            });

        let correlated_points = match dir {
            CorrelationDirection::Forward => &mut self.correlated_points,
            CorrelationDirection::Reverse => &mut self.correlated_points_reverse,
        };

        for row in 0..nrows {
            for col in 0..ncols {
                let point = out_data[(row, col)];
                if point.is_none() {
                    continue;
                }
                let out_row = (row as f32 / scale) as usize;
                let out_col = (col as f32 / scale) as usize;
                correlated_points[(out_row, out_col)] = point;
            }
        }
    }

    fn correlate_point(
        &self,
        correlation_step: &CorrelationStep,
        row: usize,
        col: usize,
        out_point: &mut Option<Match>,
    ) {
        let img1 = &correlation_step.img1;
        let img2 = &correlation_step.img2;
        let p1_data =
            correlation::compute_point_data::<KERNEL_SIZE, KERNEL_POINT_COUNT>(img1, row, col);
        let p1_data = match p1_data {
            Some(p) => p,
            None => return,
        };
        if !p1_data.stdev.is_finite() || p1_data.stdev.abs() < self.min_stdev {
            return;
        }

        let e_line = PointCorrelations::get_epipolar_line(correlation_step, row, col);
        if !e_line.coeff.0.is_finite()
            || !e_line.coeff.1.is_finite()
            || !e_line.add.0.is_finite()
            || !e_line.add.1.is_finite()
        {
            return;
        }
        const CORRIDOR_START: usize = KERNEL_SIZE;
        let corridor_end = match e_line.coeff.1.abs() > e_line.coeff.0.abs() {
            true => img2.ncols().saturating_sub(KERNEL_SIZE),
            false => img2.nrows().saturating_sub(KERNEL_SIZE),
        };
        let corridor_range = match self.first_pass {
            true => Some(CORRIDOR_START..corridor_end),
            false => self.estimate_search_range(
                correlation_step,
                row,
                col,
                &e_line,
                CORRIDOR_START,
                corridor_end,
            ),
        };
        let corridor_range = match corridor_range {
            Some(cr) => cr,
            None => return,
        };

        let mut best_match = BestMatch {
            pos: None,
            corr: None,
        };

        for corridor_offset in -(CORRIDOR_SIZE as isize)..=CORRIDOR_SIZE as isize {
            self.correlate_corridor_area(
                correlation_step,
                &e_line,
                &p1_data,
                &mut best_match,
                corridor_offset,
                corridor_range.clone(),
            );
        }
        *out_point = best_match.pos
    }

    fn get_epipolar_line(
        correlation_step: &CorrelationStep,
        row: usize,
        col: usize,
    ) -> EpipolarLine {
        let scale = correlation_step.scale;
        let p1 = Vector3::new(col as f64 / scale as f64, row as f64 / scale as f64, 1.0);
        let f_p1 = correlation_step.fundamental_matrix * p1;
        if f_p1[0].abs() > f_p1[1].abs() {
            return EpipolarLine {
                coeff: (1.0, -f_p1[1] / f_p1[0]),
                add: (0.0, -scale as f64 * f_p1[2] / f_p1[0]),
                corridor_offset: (0, 1),
            };
        }
        EpipolarLine {
            coeff: (-f_p1[0] / f_p1[1], 1.0),
            add: (-scale as f64 * f_p1[2] / f_p1[1], 0.0),
            corridor_offset: (1, 0),
        }
    }

    fn correlate_corridor_area(
        &self,
        correlation_step: &CorrelationStep,
        e_line: &EpipolarLine,
        p1_data: &correlation::PointData<KERNEL_POINT_COUNT>,
        best_match: &mut BestMatch,
        corridor_offset: isize,
        corridor_range: Range<usize>,
    ) {
        let scale = correlation_step.scale;
        let img2 = &correlation_step.img2;
        let img2_data = &correlation_step.img2_data;
        for i in corridor_range {
            let row2 = (e_line.coeff.0 * i as f64 + e_line.add.0)
                + (corridor_offset * e_line.corridor_offset.0) as f64;
            let col2 = (e_line.coeff.1 * i as f64 + e_line.add.1)
                + (corridor_offset * e_line.corridor_offset.1) as f64;
            let row2 = row2.floor() as usize;
            let col2 = col2.floor() as usize;
            if row2 < KERNEL_SIZE
                || row2 >= img2.nrows() - KERNEL_SIZE
                || col2 < KERNEL_SIZE
                || col2 >= img2.ncols() - KERNEL_SIZE
            {
                continue;
            }
            let avg2 = img2_data.avg[(row2, col2)];
            let stdev2 = img2_data.stdev[(row2, col2)];
            if !stdev2.is_finite() || stdev2.abs() < self.min_stdev {
                continue;
            }
            let mut corr = 0.0;
            for c in 0..KERNEL_WIDTH {
                for r in 0..KERNEL_WIDTH {
                    let delta1 = p1_data.delta[r * KERNEL_WIDTH + c];
                    let delta2 = img2[(
                        (row2 + r).saturating_sub(KERNEL_SIZE),
                        (col2 + c).saturating_sub(KERNEL_SIZE),
                    )] as f32
                        - avg2;
                    corr += delta1 * delta2;
                }
            }
            corr /= p1_data.stdev * stdev2 * KERNEL_POINT_COUNT as f32;

            if corr >= self.correlation_threshold
                && best_match.corr.map_or(true, |best_corr| corr > best_corr)
            {
                best_match.pos = Some((
                    (row2 as f32 / scale).round() as u32,
                    (col2 as f32 / scale).round() as u32,
                ));
                best_match.corr = Some(corr);
            }
        }
    }

    fn estimate_search_range(
        &self,
        correlation_step: &CorrelationStep,
        row1: usize,
        col1: usize,
        e_line: &EpipolarLine,
        corridor_start: usize,
        corridor_end: usize,
    ) -> Option<Range<usize>> {
        let scale = correlation_step.scale;
        thread_local! {static STDEV_RANGE: RefCell<Vec<f64>> = RefCell::new(Vec::new())};
        let mut mid_corridor = 0.0;
        let mut neighbor_count: usize = 0;

        let row_min = (row1.saturating_sub(NEIGHBOR_DISTANCE) as f32 / scale).floor() as usize;
        let row_max = ((row1 + NEIGHBOR_DISTANCE) as f32 / scale).ceil() as usize;
        let col_min = (col1.saturating_sub(NEIGHBOR_DISTANCE) as f32 / scale).floor() as usize;
        let col_max = ((col1 + NEIGHBOR_DISTANCE) as f32 / scale).ceil() as usize;
        let corridor_vertical = e_line.coeff.0.abs() > e_line.coeff.1.abs();

        let data = correlation_step.correlated_points;
        let row_min = row_min.clamp(0, data.nrows());
        let row_max = row_max.clamp(0, data.nrows());
        let col_min = col_min.clamp(0, data.ncols());
        let col_max = col_max.clamp(0, data.ncols());
        STDEV_RANGE.with(|stdev_range| {
            stdev_range
                .borrow_mut()
                .resize((row_max - row_min) * (col_max - col_min), 0.0)
        });
        for r in row_min..row_max {
            for c in col_min..col_max {
                let current_point = match data[(r, c)] {
                    Some(p) => p,
                    None => continue,
                };
                let row2 = scale as f64 * current_point.0 as f64;
                let col2 = scale as f64 * current_point.1 as f64;

                let corridor_pos = match corridor_vertical {
                    true => (row2 - e_line.add.0) / e_line.coeff.0,
                    false => (col2 - e_line.add.1) / e_line.coeff.1,
                };
                STDEV_RANGE
                    .with(|stdev_range| stdev_range.borrow_mut()[neighbor_count] = corridor_pos);
                neighbor_count += 1;
                mid_corridor += corridor_pos;
            }
        }
        if neighbor_count == 0 {
            return None;
        }

        mid_corridor /= neighbor_count as f64;
        let mut range_stdev = 0.0;
        for i in 0..neighbor_count {
            let delta = STDEV_RANGE.with(|stdev_range| stdev_range.borrow()[i] - mid_corridor);
            range_stdev += delta * delta;
        }
        range_stdev = (range_stdev / neighbor_count as f64).sqrt();

        let corridor_center = mid_corridor.round() as usize;
        let corridor_length =
            (CORRIDOR_MIN_RANGE + range_stdev * CORRIDOR_EXTEND_RANGE).round() as usize;
        let corridor_start = corridor_center
            .saturating_sub(corridor_length)
            .clamp(corridor_start, corridor_end);
        let corridor_end = corridor_center
            .saturating_add(corridor_length)
            .clamp(corridor_start, corridor_end);
        Some(corridor_start..corridor_end)
    }

    pub fn optimal_scale_steps(dimensions: (u32, u32)) -> usize {
        // TODO: replace this with log2
        let min_dimension = dimensions.1.min(dimensions.0) as usize;
        let mut scale = 0;
        while min_dimension / (1 << scale) > SCALE_MIN_SIZE {
            scale += 1;
        }
        scale - 1
    }

    fn cross_check_filter(&mut self, scale: f32, dir: CorrelationDirection) {
        if let Some(gpu_context) = &mut self.gpu_context {
            let dir = match dir {
                CorrelationDirection::Forward => gpu::CorrelationDirection::Forward,
                CorrelationDirection::Reverse => gpu::CorrelationDirection::Reverse,
            };
            gpu_context.cross_check_filter(scale, dir);
            return;
        };

        let (correlated_points, correlated_points_reverse) = match dir {
            CorrelationDirection::Forward => {
                (&mut self.correlated_points, &self.correlated_points_reverse)
            }
            CorrelationDirection::Reverse => {
                (&mut self.correlated_points_reverse, &self.correlated_points)
            }
        };
        let search_area = CROSS_CHECK_SEARCH_AREA * (1.0 / scale).round() as usize;
        correlated_points
            .column_iter_mut()
            .enumerate()
            .par_bridge()
            .for_each(|(col, mut out_col)| {
                out_col.iter_mut().enumerate().for_each(|(row, out_point)| {
                    if let Some(m) = out_point {
                        if !PointCorrelations::cross_check_point(
                            correlated_points_reverse,
                            search_area,
                            row,
                            col,
                            *m,
                        ) {
                            *out_point = None;
                        }
                    }
                })
            });
    }

    #[inline]
    fn cross_check_point(
        reverse: &DMatrix<Option<Match>>,
        search_area: usize,
        row: usize,
        col: usize,
        m: (u32, u32),
    ) -> bool {
        let min_row = (m.0 as usize)
            .saturating_sub(search_area)
            .clamp(0, reverse.nrows());
        let max_row = (m.0 as usize)
            .saturating_add(search_area + 1)
            .clamp(0, reverse.nrows());
        let min_col = (m.1 as usize)
            .saturating_sub(search_area + 1)
            .clamp(0, reverse.ncols());
        let max_col = (m.1 as usize)
            .saturating_add(search_area)
            .clamp(0, reverse.ncols());

        let r_min_row = row.saturating_sub(search_area);
        let r_max_row = row.saturating_add(search_area + 1);
        let r_min_col = col.saturating_sub(search_area);
        let r_max_col = col.saturating_add(search_area + 1);

        for srow in min_row..max_row {
            for scol in min_col..max_col {
                if let Some(rm) = reverse[(srow, scol)] {
                    let (rrow, rcol) = (rm.0 as usize, rm.1 as usize);
                    if rrow >= r_min_row
                        && rrow < r_max_row
                        && rcol >= r_min_col
                        && rcol < r_max_col
                    {
                        return true;
                    }
                }
            }
        }
        false
    }
}

struct PointData {
    avg: f32,
    stdev: f32,
}

struct ImagePointData {
    avg: DMatrix<f32>,
    stdev: DMatrix<f32>,
}

fn compute_image_point_data(img: &DMatrix<u8>) -> ImagePointData {
    let mut data = ImagePointData {
        avg: DMatrix::from_element(img.shape().0, img.shape().1, f32::NAN),
        stdev: DMatrix::from_element(img.shape().0, img.shape().1, f32::NAN),
    };
    data.avg
        .column_iter_mut()
        .zip(data.stdev.column_iter_mut())
        .enumerate()
        .par_bridge()
        .for_each(|(col, (mut avg, mut stdev))| {
            for (row, (avg, stdev)) in avg.iter_mut().zip(stdev.iter_mut()).enumerate() {
                let p = match compute_compact_point_data(img, row, col) {
                    Some(p) => p,
                    None => continue,
                };
                *avg = p.avg;
                *stdev = p.stdev;
            }
        });
    data
}

#[inline]
fn compute_compact_point_data(img: &DMatrix<u8>, row: usize, col: usize) -> Option<PointData> {
    if !correlation::point_inside_bounds::<KERNEL_SIZE>(img.shape(), row, col) {
        return None;
    };
    let mut result = PointData {
        avg: 0.0,
        stdev: 0.0,
    };
    for r in 0..KERNEL_WIDTH {
        let srow = (row + r).saturating_sub(KERNEL_SIZE);
        for c in 0..KERNEL_WIDTH {
            let scol = (col + c).saturating_sub(KERNEL_SIZE);
            let value = img[(srow, scol)];
            result.avg += value as f32;
        }
    }
    result.avg /= KERNEL_POINT_COUNT as f32;

    for r in 0..KERNEL_WIDTH {
        let srow = (row + r).saturating_sub(KERNEL_SIZE);
        for c in 0..KERNEL_WIDTH {
            let scol = (col + c).saturating_sub(KERNEL_SIZE);
            let value = img[(srow, scol)];
            let delta = value as f32 - result.avg;
            result.stdev += delta * delta;
        }
    }
    result.stdev = (result.stdev / KERNEL_POINT_COUNT as f32).sqrt();

    Some(result)
}

mod gpu {
    const MAX_BINDINGS: u32 = 5;

    use std::{borrow::Cow, collections::HashMap, error, fmt};

    use bytemuck::{Pod, Zeroable};
    use nalgebra::{DMatrix, Matrix3};
    use pollster::FutureExt;
    use std::sync::mpsc;

    use super::{
        CORRIDOR_EXTEND_RANGE, CORRIDOR_MIN_RANGE, CORRIDOR_SEGMENT_LENGTH_HIGHPERFORMANCE,
        CORRIDOR_SEGMENT_LENGTH_LOWPOWER, CORRIDOR_SIZE, CROSS_CHECK_SEARCH_AREA, KERNEL_SIZE,
        NEIGHBOR_DISTANCE, SEARCH_AREA_SEGMENT_LENGTH_HIGHPERFORMANCE,
        SEARCH_AREA_SEGMENT_LENGTH_LOWPOWER,
    };

    #[repr(C)]
    #[derive(Copy, Clone, Debug, PartialEq, Pod, Zeroable)]
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

        corridor_segment_length: usize,
        search_area_segment_length: usize,

        device_name: String,
        device: wgpu::Device,
        queue: wgpu::Queue,
        shader_module: wgpu::ShaderModule,
        buffer_img: wgpu::Buffer,
        buffer_internal_img1: wgpu::Buffer,
        buffer_internal_img2: wgpu::Buffer,
        buffer_internal_int: wgpu::Buffer,
        buffer_out: wgpu::Buffer,
        buffer_out_reverse: wgpu::Buffer,

        pipeline_configs: HashMap<String, ComputePipelineConfig>,
    }

    pub enum CorrelationDirection {
        Forward,
        Reverse,
    }

    struct ComputePipelineConfig {
        pipeline: wgpu::ComputePipeline,
        cross_correlation_bind_group: wgpu::BindGroup,
        cross_check_bind_group: wgpu::BindGroup,
    }

    impl GpuContext {
        pub fn new(
            img1_dimensions: (usize, usize),
            img2_dimensions: (usize, usize),
            min_stdev: f32,
            correlation_threshold: f32,
            fundamental_matrix: Matrix3<f64>,
            low_power: bool,
        ) -> Result<GpuContext, Box<dyn error::Error>> {
            let img1_shape = (img1_dimensions.1, img1_dimensions.0);
            let img2_shape = (img2_dimensions.1, img2_dimensions.0);

            let img1_pixels = img1_dimensions.0 * img1_dimensions.1;
            let img2_pixels = img2_dimensions.0 * img2_dimensions.1;

            // Init adapter.
            let instance = wgpu::Instance::default();
            let adapter_options = wgpu::RequestAdapterOptions {
                power_preference: if low_power {
                    wgpu::PowerPreference::LowPower
                } else {
                    wgpu::PowerPreference::HighPerformance
                },
                force_fallback_adapter: false,
                compatible_surface: None,
            };
            let adapter = instance.request_adapter(&adapter_options).block_on();
            let adapter = if let Some(adapter) = adapter {
                adapter
            } else {
                return Err(GpuError::new("Adapter not found").into());
            };

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

            let mut limits = wgpu::Limits::downlevel_defaults();
            limits.max_bindings_per_bind_group = MAX_BINDINGS;
            limits.max_storage_buffers_per_shader_stage = MAX_BINDINGS;
            // Ensure there's enough memory for the largest buffer.
            let max_buffer_size = (img1_pixels * 3 + img2_pixels * 2) * std::mem::size_of::<f32>();
            limits.max_storage_buffer_binding_size = max_buffer_size as u32;
            limits.max_buffer_size = max_buffer_size as u64;
            limits.max_push_constant_size = std::mem::size_of::<ShaderParams>() as u32;
            let (device, queue) = adapter
                .request_device(
                    &wgpu::DeviceDescriptor {
                        label: None,
                        features: wgpu::Features::PUSH_CONSTANTS,
                        limits,
                    },
                    None,
                )
                .block_on()?;

            let info = adapter.get_info();
            let device_name = format!("{:?} - {}", info.backend, info.name);

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
                (img1_pixels * 4) * std::mem::size_of::<f32>(),
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
                img1_pixels * 4 * std::mem::size_of::<i32>(),
                true,
                false,
            );
            let buffer_out = init_buffer(
                &device,
                img1_pixels * 2 * std::mem::size_of::<i32>(),
                false,
                false,
            );
            let buffer_out_reverse = init_buffer(
                &device,
                img2_pixels * 2 * std::mem::size_of::<i32>(),
                true,
                false,
            );

            let result = GpuContext {
                min_stdev,
                correlation_threshold,
                fundamental_matrix,
                img1_shape,
                img2_shape,
                corridor_segment_length,
                search_area_segment_length,
                device_name,
                device,
                queue,
                shader_module,
                buffer_img,
                buffer_internal_img1,
                buffer_internal_img2,
                buffer_internal_int,
                buffer_out,
                buffer_out_reverse,
                pipeline_configs: HashMap::new(),
            };
            Ok(result)
        }

        pub fn get_device_name(&self) -> &String {
            &self.device_name
        }

        pub fn correlate_images<PL: super::ProgressListener>(
            &mut self,
            img1: &DMatrix<u8>,
            img2: &DMatrix<u8>,
            scale: f32,
            first_pass: bool,
            progress_listener: Option<&PL>,
            dir: CorrelationDirection,
        ) {
            let max_width = img1.ncols().max(img2.ncols());
            let max_height = img1.nrows().max(img2.nrows());
            let max_shape = (max_height, max_width);
            let img1_shape = img1.shape();
            let out_shape = match dir {
                CorrelationDirection::Forward => self.img1_shape,
                CorrelationDirection::Reverse => self.img2_shape,
            };

            let mut progressbar_completed_percentage = 0.02;
            let send_progress = |value| {
                let value = match dir {
                    CorrelationDirection::Forward => value / 2.0,
                    CorrelationDirection::Reverse => 0.5 + value / 2.0,
                };
                if let Some(pl) = progress_listener {
                    pl.report_status(value);
                }
            };

            let mut params = ShaderParams {
                img1_width: img1.ncols() as u32,
                img1_height: img1.nrows() as u32,
                img2_width: img2.ncols() as u32,
                img2_height: img2.nrows() as u32,
                out_width: out_shape.1 as u32,
                out_height: out_shape.0 as u32,
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
                extend_range: CORRIDOR_EXTEND_RANGE as f32,
                min_range: CORRIDOR_MIN_RANGE as f32,
            };

            self.transfer_in_images(img1, img2);

            if first_pass {
                self.run_shader(out_shape, &dir, "init_out_data", params);
            } else {
                self.run_shader(max_shape, &dir, "prepare_initialdata_searchdata", params);
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
                    self.run_shader(img1_shape, &dir, "prepare_searchdata", params);

                    let percent_complete = progressbar_completed_percentage
                        + 0.09 * (l as f32 / neighbor_segments as f32);
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
                    self.run_shader(img1_shape, &dir, "prepare_searchdata", params);

                    let percent_complete = progressbar_completed_percentage
                        + 0.09 * (l as f32 / neighbor_segments as f32);
                    send_progress(percent_complete);
                }

                progressbar_completed_percentage = 0.20;
            }
            send_progress(progressbar_completed_percentage);
            params.iteration_pass = if first_pass { 0 } else { 1 };

            self.run_shader(max_shape, &dir, "prepare_initialdata_correlation", params);

            let corridor_stripes = 2 * CORRIDOR_SIZE + 1;
            let max_length = img2.nrows().max(img2.ncols());
            let segment_length = self.corridor_segment_length;
            let corridor_length = max_length - (KERNEL_SIZE * 2);
            let corridor_segments = corridor_length / segment_length + 1;
            for corridor_offset in -(CORRIDOR_SIZE as i32)..=CORRIDOR_SIZE as i32 {
                for l in 0u32..corridor_segments as u32 {
                    params.corridor_offset = corridor_offset;
                    params.corridor_start = l * segment_length as u32;
                    params.corridor_end = (l + 1) * segment_length as u32;
                    if params.corridor_end > corridor_length as u32 {
                        params.corridor_end = corridor_length as u32;
                    }
                    self.run_shader(img1_shape, &dir, "cross_correlate", params);

                    let corridor_complete = params.corridor_end as f32 / corridor_length as f32;
                    let percent_complete = progressbar_completed_percentage
                        + (1.0 - progressbar_completed_percentage)
                            * (corridor_offset as f32 + CORRIDOR_SIZE as f32 + corridor_complete)
                            / corridor_stripes as f32;
                    send_progress(percent_complete);
                }
            }
        }

        pub fn cross_check_filter(&mut self, scale: f32, dir: CorrelationDirection) {
            let (out_shape, out_shape_reverse) = match dir {
                CorrelationDirection::Forward => (self.img1_shape, self.img2_shape),
                CorrelationDirection::Reverse => (self.img2_shape, self.img1_shape),
            };

            let search_area = CROSS_CHECK_SEARCH_AREA * (1.0 / scale).round() as usize;

            // Reuse/repurpose ShaderParams.
            let params = ShaderParams {
                img1_width: out_shape.1 as u32,
                img1_height: out_shape.0 as u32,
                img2_width: out_shape_reverse.1 as u32,
                img2_height: out_shape_reverse.0 as u32,
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
            self.run_shader(out_shape, &dir, "cross_check_filter", params);
        }

        fn run_shader(
            &mut self,
            shape: (usize, usize),
            dir: &CorrelationDirection,
            entry_point: &str,
            shader_params: ShaderParams,
        ) {
            let dir_name = match dir {
                CorrelationDirection::Forward => "forward",
                CorrelationDirection::Reverse => "reverse",
            };
            let config_key = format!("{}-{}", entry_point, dir_name);
            if !self.pipeline_configs.contains_key(&config_key) {
                let pipeline_config = self.create_pipeline_config(entry_point, dir);
                self.pipeline_configs
                    .insert(config_key.to_string(), pipeline_config);
            }
            let pipeline_config = self.pipeline_configs.get(&config_key).unwrap();

            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            {
                let workgroup_size = ((shape.1 + 15) / 16, ((shape.0 + 15) / 16));
                let mut cpass =
                    encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
                cpass.set_pipeline(&pipeline_config.pipeline);
                cpass.set_push_constants(0, bytemuck::cast_slice(&[shader_params]));
                cpass.set_bind_group(0, &pipeline_config.cross_correlation_bind_group, &[]);
                cpass.set_bind_group(1, &pipeline_config.cross_check_bind_group, &[]);
                cpass.dispatch_workgroups(workgroup_size.0 as u32, workgroup_size.1 as u32, 1);
            }

            self.queue.submit(Some(encoder.finish()));
            self.device.poll(wgpu::Maintain::Wait);
        }

        fn convert_fundamental_matrix(&self, dir: &CorrelationDirection) -> [f32; 3 * 4] {
            let fundamental_matrix = match dir {
                CorrelationDirection::Forward => self.fundamental_matrix,
                CorrelationDirection::Reverse => self.fundamental_matrix.transpose(),
            };
            let mut f = [0f32; 3 * 4];
            for row in 0..3 {
                for col in 0..3 {
                    f[col * 4 + row] = fundamental_matrix[(row, col)] as f32;
                }
            }
            f
        }

        fn transfer_in_images(&self, img1: &DMatrix<u8>, img2: &DMatrix<u8>) {
            let mut img_slice =
                Vec::with_capacity(img1.nrows() * img1.ncols() + img2.nrows() * img2.ncols());
            for row in 0..img1.nrows() {
                for col in 0..img1.ncols() {
                    img_slice.push(img1[(row, col)] as f32);
                }
            }
            for row in 0..img2.nrows() {
                for col in 0..img2.ncols() {
                    img_slice.push(img2[(row, col)] as f32);
                }
            }
            self.queue.write_buffer(
                &self.buffer_img,
                0,
                bytemuck::cast_slice(img_slice.as_slice()),
            );
        }

        pub fn complete_process(
            &mut self,
        ) -> Result<DMatrix<Option<super::Match>>, Box<dyn error::Error>> {
            self.buffer_img.destroy();
            self.buffer_internal_img1.destroy();
            self.buffer_internal_img2.destroy();
            self.buffer_internal_int.destroy();
            self.buffer_out_reverse.destroy();

            let mut out_image = DMatrix::from_element(self.img1_shape.0, self.img1_shape.1, None);

            let out_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: self.buffer_out.size(),
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let out_buffer_slice = out_buffer.slice(..);
            {
                let mut encoder = self
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                encoder.copy_buffer_to_buffer(
                    &self.buffer_out,
                    0,
                    &out_buffer,
                    0,
                    self.buffer_out.size(),
                );
                self.queue.submit(Some(encoder.finish()));
                let (sender, receiver) = mpsc::channel();
                out_buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());
                self.device.poll(wgpu::Maintain::Wait);

                if let Err(err) = receiver.recv() {
                    return Err(err.into());
                }
            }

            let out_buffer_slice_mapped = out_buffer_slice.get_mapped_range();
            let out_data: &[i32] = bytemuck::cast_slice(&out_buffer_slice_mapped);
            for col in 0..out_image.ncols() {
                for row in 0..out_image.nrows() {
                    let pos = 2 * (row * out_image.ncols() + col);
                    let point_match = (out_data[pos], out_data[pos + 1]);
                    out_image[(row, col)] = if point_match.0 > 0 && point_match.1 > 0 {
                        Some((point_match.1 as u32, point_match.0 as u32))
                    } else {
                        None
                    };
                }
            }
            drop(out_buffer_slice_mapped);
            out_buffer.unmap();
            Ok(out_image)
        }

        fn create_pipeline_config(
            &self,
            entry_point: &str,
            dir: &CorrelationDirection,
        ) -> ComputePipelineConfig {
            let (buffer_out, buffer_out_reverse) = match dir {
                CorrelationDirection::Forward => (&self.buffer_out, &self.buffer_out_reverse),
                CorrelationDirection::Reverse => (&self.buffer_out_reverse, &self.buffer_out),
            };

            let correlation_layout =
                self.device
                    .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        label: None,
                        entries: &[
                            wgpu::BindGroupLayoutEntry {
                                binding: 0,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                                    has_dynamic_offset: false,
                                    min_binding_size: wgpu::BufferSize::new(self.buffer_img.size()),
                                },
                                count: None,
                            },
                            wgpu::BindGroupLayoutEntry {
                                binding: 1,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                                    has_dynamic_offset: false,
                                    min_binding_size: wgpu::BufferSize::new(
                                        self.buffer_internal_img1.size(),
                                    ),
                                },
                                count: None,
                            },
                            wgpu::BindGroupLayoutEntry {
                                binding: 2,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                                    has_dynamic_offset: false,
                                    min_binding_size: wgpu::BufferSize::new(
                                        self.buffer_internal_img2.size(),
                                    ),
                                },
                                count: None,
                            },
                            wgpu::BindGroupLayoutEntry {
                                binding: 3,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                                    has_dynamic_offset: false,
                                    min_binding_size: wgpu::BufferSize::new(
                                        self.buffer_internal_int.size(),
                                    ),
                                },
                                count: None,
                            },
                            wgpu::BindGroupLayoutEntry {
                                binding: 4,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                                    has_dynamic_offset: false,
                                    min_binding_size: wgpu::BufferSize::new(buffer_out.size()),
                                },
                                count: None,
                            },
                        ],
                    });

            let cross_check_layout =
                self.device
                    .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        label: None,
                        entries: &[
                            wgpu::BindGroupLayoutEntry {
                                binding: 0,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                                    has_dynamic_offset: false,
                                    min_binding_size: wgpu::BufferSize::new(buffer_out.size()),
                                },
                                count: None,
                            },
                            wgpu::BindGroupLayoutEntry {
                                binding: 1,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                                    has_dynamic_offset: false,
                                    min_binding_size: wgpu::BufferSize::new(
                                        buffer_out_reverse.size(),
                                    ),
                                },
                                count: None,
                            },
                        ],
                    });
            let pipeline_layout =
                self.device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: None,
                        bind_group_layouts: &[&correlation_layout, &cross_check_layout],
                        push_constant_ranges: &[wgpu::PushConstantRange {
                            stages: wgpu::ShaderStages::COMPUTE,
                            range: 0..std::mem::size_of::<ShaderParams>() as u32,
                        }],
                    });

            let pipeline = self
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: None,
                    layout: Some(&pipeline_layout),
                    module: &self.shader_module,
                    entry_point,
                });

            let cross_correlation_bind_group =
                self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &correlation_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: self.buffer_img.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: self.buffer_internal_img1.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: self.buffer_internal_img2.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: self.buffer_internal_int.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: buffer_out.as_entire_binding(),
                        },
                    ],
                });

            let cross_check_bind_group =
                self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &cross_check_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: buffer_out.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: buffer_out_reverse.as_entire_binding(),
                        },
                    ],
                });

            ComputePipelineConfig {
                pipeline,
                cross_correlation_bind_group,
                cross_check_bind_group,
            }
        }
    }

    fn init_buffer(
        device: &wgpu::Device,
        size: usize,
        gpuonly: bool,
        readonly: bool,
    ) -> wgpu::Buffer {
        let size = size as wgpu::BufferAddress;

        let buffer_usage = if gpuonly {
            wgpu::BufferUsages::STORAGE
        } else if readonly {
            wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE
        } else {
            wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::STORAGE
        };
        device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: buffer_usage,
            mapped_at_creation: false,
        })
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
}
