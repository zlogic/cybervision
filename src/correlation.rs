use crate::data::{Grid, Point2D};
use nalgebra::{Matrix3, Vector3};
use rayon::iter::ParallelIterator;
use std::{cell::RefCell, error, ops::Range, sync::atomic::AtomicUsize, sync::atomic::Ordering};

const SCALE_MIN_SIZE: usize = 64;
const KERNEL_SIZE: usize = 5;
const KERNEL_WIDTH: usize = KERNEL_SIZE * 2 + 1;
const KERNEL_POINT_COUNT: usize = KERNEL_WIDTH * KERNEL_WIDTH;

const THRESHOLD_AFFINE: f32 = 0.6;
const THRESHOLD_PERSPECTIVE: f32 = 0.7;
const MIN_STDEV_AFFINE: f32 = 1.0;
const MIN_STDEV_PERSPECTIVE: f32 = 3.0;
const CORRIDOR_SIZE_AFFINE: usize = 2;
const CORRIDOR_SIZE_PERSPECTIVE: usize = 3;
// Decrease when using a low-powered GPU
const CORRIDOR_SEGMENT_LENGTH_HIGHPERFORMANCE: usize = 512;
const SEARCH_AREA_SEGMENT_LENGTH_HIGHPERFORMANCE: usize = 1024;
const CORRIDOR_SEGMENT_LENGTH_LOWPOWER: usize = 8;
const SEARCH_AREA_SEGMENT_LENGTH_LOWPOWER: usize = 128;
const NEIGHBOR_DISTANCE: usize = 10;
const CORRIDOR_EXTEND_RANGE_AFFINE: f64 = 1.0;
const CORRIDOR_EXTEND_RANGE_PERSPECTIVE: f64 = 1.0;
const CORRIDOR_MIN_RANGE: f64 = 2.5;
const CROSS_CHECK_SEARCH_AREA: usize = 2;

type Match = (Point2D<u32>, f32);

#[derive(Debug)]
pub struct PointData<const KPC: usize> {
    pub delta: [f32; KPC],
    pub stdev: f32,
}

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
    pub correlated_points: Grid<Option<Match>>,
    correlated_points_reverse: Grid<Option<Match>>,
    first_pass: bool,
    min_stdev: f32,
    corridor_size: usize,
    correlation_threshold: f32,
    corridor_extend_range: f64,
    fundamental_matrix: Matrix3<f64>,
    gpu_context: Option<gpu::GpuContext>,
    selected_hardware: String,
}

enum CorrelationDirection {
    Forward,
    Reverse,
}

struct EpipolarLine {
    coeff: Point2D<f64>,
    add: Point2D<f64>,
    corridor_offset: Point2D<isize>,
}

struct BestMatch {
    pos: Option<Point2D<u32>>,
    corr: Option<f32>,
}

struct CorrelationStep<'a> {
    scale: f32,
    fundamental_matrix: Matrix3<f64>,
    correlated_points: &'a Grid<Option<Match>>,
    img1: &'a Grid<u8>,
    img2: &'a Grid<u8>,
    img2_data: ImagePointData,
}

struct CorrelationParameters {
    min_stdev: f32,
    corridor_size: usize,
    correlation_threshold: f32,
    corridor_extend_range: f64,
}

impl CorrelationParameters {
    fn for_projection(projection_mode: &ProjectionMode) -> CorrelationParameters {
        let (min_stdev, correlation_threshold, corridor_size, corridor_extend_range) =
            match projection_mode {
                ProjectionMode::Affine => (
                    MIN_STDEV_AFFINE,
                    THRESHOLD_AFFINE,
                    CORRIDOR_SIZE_AFFINE,
                    CORRIDOR_EXTEND_RANGE_AFFINE,
                ),
                ProjectionMode::Perspective => (
                    MIN_STDEV_PERSPECTIVE,
                    THRESHOLD_PERSPECTIVE,
                    CORRIDOR_SIZE_PERSPECTIVE,
                    CORRIDOR_EXTEND_RANGE_PERSPECTIVE,
                ),
            };
        CorrelationParameters {
            min_stdev,
            corridor_size,
            correlation_threshold,
            corridor_extend_range,
        }
    }
}

impl PointCorrelations {
    pub fn new(
        img1_dimensions: (u32, u32),
        img2_dimensions: (u32, u32),
        fundamental_matrix: Matrix3<f64>,
        projection_mode: ProjectionMode,
        hardware_mode: HardwareMode,
    ) -> PointCorrelations {
        let selected_hardware;
        let gpu_context = if matches!(hardware_mode, HardwareMode::Gpu | HardwareMode::GpuLowPower)
        {
            let low_power = matches!(hardware_mode, HardwareMode::GpuLowPower);
            match gpu::GpuContext::new(
                (img1_dimensions.0 as usize, img1_dimensions.1 as usize),
                (img2_dimensions.0 as usize, img2_dimensions.1 as usize),
                &projection_mode,
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
            Some(_) => (Grid::new(0, 0, None), Grid::new(0, 0, None)),
            None => (
                Grid::new(img1_dimensions.0 as usize, img1_dimensions.1 as usize, None),
                Grid::new(img2_dimensions.0 as usize, img2_dimensions.1 as usize, None),
            ),
        };

        let params = CorrelationParameters::for_projection(&projection_mode);
        PointCorrelations {
            correlated_points,
            correlated_points_reverse,
            first_pass: true,
            min_stdev: params.min_stdev,
            corridor_size: params.corridor_size,
            correlation_threshold: params.correlation_threshold,
            corridor_extend_range: params.corridor_extend_range,
            fundamental_matrix,
            gpu_context,
            selected_hardware,
        }
    }

    pub fn get_selected_hardware(&self) -> &String {
        &self.selected_hardware
    }

    pub fn complete(&mut self) -> Result<(), Box<dyn error::Error>> {
        self.correlated_points_reverse = Grid::new(0, 0, None);
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
        img1: Grid<u8>,
        img2: Grid<u8>,
        scale: f32,
        progress_listener: Option<&PL>,
    ) -> Result<(), Box<dyn error::Error>> {
        self.correlate_images_step(
            &img1,
            &img2,
            scale,
            progress_listener,
            CorrelationDirection::Forward,
        )?;
        self.correlate_images_step(
            &img2,
            &img1,
            scale,
            progress_listener,
            CorrelationDirection::Reverse,
        )?;

        self.cross_check_filter(scale, CorrelationDirection::Forward);
        self.cross_check_filter(scale, CorrelationDirection::Reverse);

        self.first_pass = false;

        Ok(())
    }

    fn correlate_images_step<PL: ProgressListener>(
        &mut self,
        img1: &Grid<u8>,
        img2: &Grid<u8>,
        scale: f32,
        progress_listener: Option<&PL>,
        dir: CorrelationDirection,
    ) -> Result<(), Box<dyn error::Error>> {
        if let Some(gpu_context) = &mut self.gpu_context {
            let dir = match dir {
                CorrelationDirection::Forward => gpu::CorrelationDirection::Forward,
                CorrelationDirection::Reverse => gpu::CorrelationDirection::Reverse,
            };
            return gpu_context.correlate_images(
                img1,
                img2,
                scale,
                self.first_pass,
                progress_listener,
                dir,
            );
        };
        let img2_data = compute_image_point_data(img2);
        let mut out_data = Grid::<Option<Match>>::new(img1.width(), img1.height(), None);

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
        let out_data_height = out_data.height().saturating_sub(KERNEL_SIZE * 2) as f32;
        let (max_width, max_height) = (
            out_data.width() - KERNEL_SIZE,
            out_data.height() - KERNEL_SIZE,
        );
        out_data.par_iter_mut().for_each(|(x, y, out_point)| {
            if x == 0 {
                if let Some(pl) = progress_listener {
                    let value = counter.fetch_add(1, Ordering::Relaxed) as f32 / out_data_height;
                    let value = match dir {
                        CorrelationDirection::Forward => value / 2.0,
                        CorrelationDirection::Reverse => 0.5 + value / 2.0,
                    };
                    pl.report_status(value);
                }
            }
            if x < KERNEL_SIZE || y < KERNEL_SIZE || x >= max_width || y >= max_height {
                return;
            }
            let point = Point2D::new(x, y);
            self.correlate_point(&corelation_step, point, out_point);
        });

        let correlated_points = match dir {
            CorrelationDirection::Forward => &mut self.correlated_points,
            CorrelationDirection::Reverse => &mut self.correlated_points_reverse,
        };

        out_data.iter().for_each(|(x, y, point)| {
            let out_x = (x as f32 / scale) as usize;
            let out_y = (y as f32 / scale) as usize;
            let out_point = correlated_points.val_mut(out_x, out_y);
            *out_point = *point;
        });

        Ok(())
    }

    fn correlate_point(
        &self,
        correlation_step: &CorrelationStep,
        point: Point2D<usize>,
        out_point: &mut Option<Match>,
    ) {
        let img1 = &correlation_step.img1;
        let img2 = &correlation_step.img2;
        let p1_data = compute_point_data::<KERNEL_SIZE, KERNEL_POINT_COUNT>(img1, &point);
        let p1_data = match p1_data {
            Some(p) => p,
            None => return,
        };
        if !p1_data.stdev.is_finite() || p1_data.stdev.abs() < self.min_stdev {
            return;
        }

        let e_line = PointCorrelations::get_epipolar_line(correlation_step, &point);
        if !e_line.coeff.x.is_finite()
            || !e_line.coeff.y.is_finite()
            || !e_line.add.x.is_finite()
            || !e_line.add.y.is_finite()
        {
            return;
        }
        const CORRIDOR_START: usize = KERNEL_SIZE;
        let corridor_end = match e_line.coeff.x.abs() > e_line.coeff.y.abs() {
            true => img2.width().saturating_sub(KERNEL_SIZE),
            false => img2.height().saturating_sub(KERNEL_SIZE),
        };
        let corridor_range = match self.first_pass {
            true => Some(CORRIDOR_START..corridor_end),
            false => self.estimate_search_range(
                correlation_step,
                &point,
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

        let corridor_size = self.corridor_size as isize;
        for corridor_offset in -corridor_size..=corridor_size {
            self.correlate_corridor_area(
                correlation_step,
                &e_line,
                &p1_data,
                &mut best_match,
                corridor_offset,
                corridor_range.clone(),
            );
        }
        *out_point = best_match
            .pos
            .and_then(|m| best_match.corr.map(|corr| (m, corr)))
    }

    fn get_epipolar_line(
        correlation_step: &CorrelationStep,
        point: &Point2D<usize>,
    ) -> EpipolarLine {
        let scale = correlation_step.scale;
        let p1 = Vector3::new(
            point.x as f64 / scale as f64,
            point.y as f64 / scale as f64,
            1.0,
        );
        let f_p1 = correlation_step.fundamental_matrix * p1;
        if f_p1.x.abs() > f_p1.y.abs() {
            return EpipolarLine {
                coeff: Point2D::new(-f_p1[1] / f_p1[0], 1.0),
                add: Point2D::new(-scale as f64 * f_p1[2] / f_p1[0], 0.0),
                corridor_offset: Point2D::new(1, 0),
            };
        }
        EpipolarLine {
            coeff: Point2D::new(1.0, -f_p1[0] / f_p1[1]),
            add: Point2D::new(0.0, -scale as f64 * f_p1[2] / f_p1[1]),
            corridor_offset: Point2D::new(0, 1),
        }
    }

    fn correlate_corridor_area(
        &self,
        correlation_step: &CorrelationStep,
        e_line: &EpipolarLine,
        p1_data: &PointData<KERNEL_POINT_COUNT>,
        best_match: &mut BestMatch,
        corridor_offset: isize,
        corridor_range: Range<usize>,
    ) {
        let scale = correlation_step.scale;
        let img2 = &correlation_step.img2;
        let img2_data = &correlation_step.img2_data;
        for i in corridor_range {
            let x2 = (e_line.coeff.x * i as f64 + e_line.add.x)
                + (corridor_offset * e_line.corridor_offset.x) as f64;
            let y2 = (e_line.coeff.y * i as f64 + e_line.add.y)
                + (corridor_offset * e_line.corridor_offset.y) as f64;
            let x2 = x2.floor() as usize;
            let y2 = y2.floor() as usize;
            if x2 < KERNEL_SIZE
                || x2 >= img2.width() - KERNEL_SIZE
                || y2 < KERNEL_SIZE
                || y2 >= img2.height() - KERNEL_SIZE
            {
                continue;
            }
            let avg2 = img2_data.avg.val(x2, y2);
            let stdev2 = img2_data.stdev.val(x2, y2);
            if !stdev2.is_finite() || stdev2.abs() < self.min_stdev {
                continue;
            }
            let mut corr = 0.0;
            for y in 0..KERNEL_WIDTH {
                for x in 0..KERNEL_WIDTH {
                    let delta1 = p1_data.delta[y * KERNEL_WIDTH + x];
                    let delta2 = *img2.val(
                        (x2 + x).saturating_sub(KERNEL_SIZE),
                        (y2 + y).saturating_sub(KERNEL_SIZE),
                    ) as f32
                        - avg2;
                    corr += delta1 * delta2;
                }
            }
            corr /= p1_data.stdev * stdev2 * KERNEL_POINT_COUNT as f32;

            if corr >= self.correlation_threshold
                && best_match.corr.map_or(true, |best_corr| corr > best_corr)
            {
                best_match.pos = Some(Point2D::new(
                    (x2 as f32 / scale).round() as u32,
                    (y2 as f32 / scale).round() as u32,
                ));
                best_match.corr = Some(corr);
            }
        }
    }

    fn estimate_search_range(
        &self,
        correlation_step: &CorrelationStep,
        point1: &Point2D<usize>,
        e_line: &EpipolarLine,
        corridor_start: usize,
        corridor_end: usize,
    ) -> Option<Range<usize>> {
        let scale = correlation_step.scale;
        thread_local! {static STDEV_RANGE: RefCell<Vec<f64>> = RefCell::new(Vec::new())};
        let mut mid_corridor = 0.0;
        let mut neighbor_count: usize = 0;

        let x_min = (point1.x.saturating_sub(NEIGHBOR_DISTANCE) as f32 / scale).floor() as usize;
        let x_max = ((point1.x + NEIGHBOR_DISTANCE) as f32 / scale).ceil() as usize;
        let y_min = (point1.y.saturating_sub(NEIGHBOR_DISTANCE) as f32 / scale).floor() as usize;
        let y_max = ((point1.y + NEIGHBOR_DISTANCE) as f32 / scale).ceil() as usize;
        let corridor_vertical = e_line.coeff.y.abs() > e_line.coeff.x.abs();

        let data = correlation_step.correlated_points;
        let x_min = x_min.clamp(0, data.width());
        let x_max = x_max.clamp(0, data.width());
        let y_min = y_min.clamp(0, data.height());
        let y_max = y_max.clamp(0, data.height());
        STDEV_RANGE.with(|stdev_range| {
            stdev_range
                .borrow_mut()
                .resize((x_max - x_min) * (y_max - y_min), 0.0)
        });
        for y in y_min..y_max {
            for x in x_min..x_max {
                let current_point = match data.val(x, y) {
                    Some(p) => p,
                    None => continue,
                };
                let point2 = Point2D::new(
                    scale as f64 * current_point.0.x as f64,
                    scale as f64 * current_point.0.y as f64,
                );

                let corridor_pos = match corridor_vertical {
                    true => (point2.y - e_line.add.y) / e_line.coeff.y,
                    false => (point2.x - e_line.add.x) / e_line.coeff.x,
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
            (CORRIDOR_MIN_RANGE + range_stdev * self.corridor_extend_range).round() as usize;
        let corridor_start = corridor_center
            .saturating_sub(corridor_length)
            .clamp(corridor_start, corridor_end);
        let corridor_end = corridor_center
            .saturating_add(corridor_length)
            .clamp(corridor_start, corridor_end);
        Some(corridor_start..corridor_end)
    }

    pub fn optimal_scale_steps(dimensions: (u32, u32)) -> usize {
        let min_dimension = dimensions.1.min(dimensions.0) as usize;
        if min_dimension <= SCALE_MIN_SIZE {
            return 0;
        }
        (min_dimension as f64 / SCALE_MIN_SIZE as f64)
            .log2()
            .floor() as usize
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
            .par_iter_mut()
            .for_each(|(x, y, out_point)| {
                if let Some(m) = out_point {
                    if !PointCorrelations::cross_check_point(
                        correlated_points_reverse,
                        search_area,
                        Point2D::new(x, y),
                        *m,
                    ) {
                        *out_point = None;
                    }
                }
            });
    }

    #[inline]
    fn cross_check_point(
        reverse: &Grid<Option<Match>>,
        search_area: usize,
        point: Point2D<usize>,
        m: Match,
    ) -> bool {
        let min_x = (m.0.x as usize)
            .saturating_sub(search_area)
            .clamp(0, reverse.width());
        let max_x = (m.0.x as usize)
            .saturating_add(search_area + 1)
            .clamp(0, reverse.width());
        let min_y = (m.0.y as usize)
            .saturating_sub(search_area)
            .clamp(0, reverse.height());
        let max_y = (m.0.y as usize)
            .saturating_add(search_area + 1)
            .clamp(0, reverse.height());

        let r_min_x = point.x.saturating_sub(search_area);
        let r_max_x = point.x.saturating_add(search_area + 1);
        let r_min_y = point.y.saturating_sub(search_area);
        let r_max_y = point.y.saturating_add(search_area + 1);

        for s_y in min_y..max_y {
            for s_x in min_x..max_x {
                if let Some(rm) = reverse.val(s_x, s_y) {
                    let (r_x, r_y) = (rm.0.x as usize, rm.0.y as usize);
                    if r_x >= r_min_x && r_x < r_max_x && r_y >= r_min_y && r_y < r_max_y {
                        return true;
                    }
                }
            }
        }
        false
    }
}

struct PointDataCompact {
    avg: f32,
    stdev: f32,
}

struct ImagePointData {
    avg: Grid<f32>,
    stdev: Grid<f32>,
}

fn compute_image_point_data(img: &Grid<u8>) -> ImagePointData {
    let mut data = ImagePointData {
        avg: Grid::new(img.width(), img.height(), f32::NAN),
        stdev: Grid::new(img.width(), img.height(), f32::NAN),
    };
    data.avg
        .iter_mut()
        .zip(data.stdev.iter_mut())
        .for_each(|((x, y, avg), (_x, _y, stdev))| {
            let point = Point2D::new(x, y);
            let p = match compute_compact_point_data(img, &point) {
                Some(p) => p,
                None => return,
            };
            *avg = p.avg;
            *stdev = p.stdev;
        });
    data
}

#[inline]
fn compute_compact_point_data(img: &Grid<u8>, point: &Point2D<usize>) -> Option<PointDataCompact> {
    if !point_inside_bounds::<KERNEL_SIZE>(img, point) {
        return None;
    };
    let mut result = PointDataCompact {
        avg: 0.0,
        stdev: 0.0,
    };
    for y in 0..KERNEL_WIDTH {
        let s_y = (point.y + y).saturating_sub(KERNEL_SIZE);
        for x in 0..KERNEL_WIDTH {
            let s_x = (point.x + x).saturating_sub(KERNEL_SIZE);
            let value = img.val(s_x, s_y);
            result.avg += *value as f32;
        }
    }
    result.avg /= KERNEL_POINT_COUNT as f32;

    for y in 0..KERNEL_WIDTH {
        let s_y = (point.y + y).saturating_sub(KERNEL_SIZE);
        for x in 0..KERNEL_WIDTH {
            let s_x = (point.x + x).saturating_sub(KERNEL_SIZE);
            let value = img.val(s_x, s_y);
            let delta = *value as f32 - result.avg;
            result.stdev += delta * delta;
        }
    }
    result.stdev = (result.stdev / KERNEL_POINT_COUNT as f32).sqrt();

    Some(result)
}

#[inline]
pub fn point_inside_bounds<const KS: usize>(img: &Grid<u8>, point: &Point2D<usize>) -> bool {
    point.x >= KS && point.y >= KS && point.x + KS < img.width() && point.y + KS < img.height()
}

#[inline]
pub fn compute_point_data<const KS: usize, const KPC: usize>(
    img: &Grid<u8>,
    point: &Point2D<usize>,
) -> Option<PointData<KPC>> {
    if !point_inside_bounds::<KS>(img, point) {
        return None;
    };
    let kernel_width = KS * 2 + 1;
    let mut result = PointData::<KPC> {
        delta: [0.0; KPC],
        stdev: 0.0,
    };
    let mut avg = 0.0;
    for y in 0..=KS * 2 {
        let s_y = (point.y + y).saturating_sub(KS);
        for x in 0..=KS * 2 {
            let s_x = (point.x + x).saturating_sub(KS);
            let value = img.val(s_x, s_y);
            let delta_pos = y * kernel_width + x;
            result.delta[delta_pos] = (*value).into();
            avg += *value as f32;
        }
    }
    avg /= KPC as f32;

    for i in 0..KPC {
        let delta = result.delta[i] - avg;
        result.delta[i] = delta;
        result.stdev += delta * delta;
    }
    result.stdev = (result.stdev / KPC as f32).sqrt();

    Some(result)
}

mod gpu {
    const MAX_BINDINGS: u32 = 6;

    use std::{borrow::Cow, collections::HashMap, error, fmt};

    use bytemuck::{Pod, Zeroable};
    use nalgebra::Matrix3;
    use pollster::FutureExt;
    use rayon::iter::ParallelIterator;
    use std::sync::mpsc;

    use crate::data::{Grid, Point2D};

    use super::{
        CorrelationParameters, ProjectionMode, CORRIDOR_MIN_RANGE,
        CORRIDOR_SEGMENT_LENGTH_HIGHPERFORMANCE, CORRIDOR_SEGMENT_LENGTH_LOWPOWER,
        CROSS_CHECK_SEARCH_AREA, KERNEL_SIZE, NEIGHBOR_DISTANCE,
        SEARCH_AREA_SEGMENT_LENGTH_HIGHPERFORMANCE, SEARCH_AREA_SEGMENT_LENGTH_LOWPOWER,
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

        correlation_values: Grid<Option<f32>>,

        corridor_segment_length: usize,
        search_area_segment_length: usize,
        corridor_size: usize,
        corridor_extend_range: f64,

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
        buffer_out_corr: wgpu::Buffer,

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
            projection_mode: &ProjectionMode,
            fundamental_matrix: Matrix3<f64>,
            low_power: bool,
        ) -> Result<GpuContext, Box<dyn error::Error>> {
            let img1_shape = (img1_dimensions.0, img1_dimensions.1);
            let img2_shape = (img2_dimensions.0, img2_dimensions.1);

            let img1_pixels = img1_dimensions.0 * img1_dimensions.1;
            let img2_pixels = img2_dimensions.0 * img2_dimensions.1;
            let max_pixels = img1_pixels.max(img2_pixels);

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
            let max_buffer_size = max_pixels * 4 * std::mem::size_of::<i32>();
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

            let correlation_values = Grid::new(img1_shape.0, img1_shape.1, None);

            let params = CorrelationParameters::for_projection(projection_mode);
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
            };
            Ok(result)
        }

        pub fn get_device_name(&self) -> &String {
            &self.device_name
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
            let max_width = img1.width().max(img2.width());
            let max_height = img1.height().max(img2.height());
            let max_shape = (max_width, max_height);
            let img1_shape = (img1.width(), img1.height());
            let out_shape = match dir {
                CorrelationDirection::Forward => self.img1_shape,
                CorrelationDirection::Reverse => self.img2_shape,
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
                out_width: out_shape.0 as u32,
                out_height: out_shape.1 as u32,
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

            self.transfer_in_images(img1, img2);

            if first_pass {
                self.run_shader(out_shape, &dir, "init_out_data", params);
            } else {
                self.run_shader(out_shape, &dir, "prepare_initialdata_searchdata", params);
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
                    self.run_shader(img1_shape, &dir, "cross_correlate", params);

                    let corridor_complete = params.corridor_end as f32 / corridor_length as f32;
                    let percent_complete = progressbar_completed_percentage
                        + (1.0 - progressbar_completed_percentage)
                            * (corridor_offset as f32 + corridor_size as f32 + corridor_complete)
                            / corridor_stripes as f32;
                    send_progress(percent_complete);
                }
            }

            self.save_corr(&dir)
        }

        pub fn cross_check_filter(&mut self, scale: f32, dir: CorrelationDirection) {
            let (out_shape, out_shape_reverse) = match dir {
                CorrelationDirection::Forward => (self.img1_shape, self.img2_shape),
                CorrelationDirection::Reverse => (self.img2_shape, self.img1_shape),
            };

            let search_area = CROSS_CHECK_SEARCH_AREA * (1.0 / scale).round() as usize;

            // Reuse/repurpose ShaderParams.
            let params = ShaderParams {
                img1_width: out_shape.0 as u32,
                img1_height: out_shape.1 as u32,
                img2_width: out_shape_reverse.0 as u32,
                img2_height: out_shape_reverse.1 as u32,
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
                let workgroup_size = ((shape.0 + 15) / 16, ((shape.1 + 15) / 16));
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: None,
                });
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

        fn transfer_in_images(&self, img1: &Grid<u8>, img2: &Grid<u8>) {
            let img2_offset = img1.width() * img1.height();
            let mut img_slice =
                vec![0.0f32; img1.width() * img1.height() + img2.width() * img2.height()];
            img1.iter()
                .for_each(|(x, y, val)| img_slice[y * img1.width() + x] = *val as f32);
            img2.iter().for_each(|(x, y, val)| {
                img_slice[img2_offset + y * img2.width() + x] = *val as f32
            });
            self.queue.write_buffer(
                &self.buffer_img,
                0,
                bytemuck::cast_slice(img_slice.as_slice()),
            );
        }

        fn save_corr(&mut self, dir: &CorrelationDirection) -> Result<(), Box<dyn error::Error>> {
            if !matches!(dir, CorrelationDirection::Forward) {
                return Ok(());
            }
            let out_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: self.buffer_out_corr.size(),
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let out_buffer_slice = out_buffer.slice(..);
            {
                let mut encoder = self
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
                encoder.copy_buffer_to_buffer(
                    &self.buffer_out_corr,
                    0,
                    &out_buffer,
                    0,
                    self.buffer_out_corr.size(),
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
            let out_data: &[f32] = bytemuck::cast_slice(&out_buffer_slice_mapped);

            let width = self.correlation_values.width();
            self.correlation_values
                .par_iter_mut()
                .for_each(|(x, y, out_point)| {
                    let corr = out_data[y * width + x];
                    if corr > self.correlation_threshold {
                        *out_point = Some(corr);
                    }
                });
            drop(out_buffer_slice_mapped);
            out_buffer.unmap();

            Ok(())
        }

        pub fn complete_process(
            &mut self,
        ) -> Result<Grid<Option<super::Match>>, Box<dyn error::Error>> {
            self.buffer_img.destroy();
            self.buffer_internal_img1.destroy();
            self.buffer_internal_img2.destroy();
            self.buffer_internal_int.destroy();
            self.buffer_out_reverse.destroy();
            self.buffer_out_corr.destroy();

            let mut out_image = Grid::new(self.img1_shape.0, self.img1_shape.1, None);

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
            let width = out_image.width();
            out_image.par_iter_mut().for_each(|(x, y, out_point)| {
                let pos = 2 * (y * width + x);
                let (match_x, match_y) = (out_data[pos], out_data[pos + 1]);
                if let Some(corr) = self.correlation_values.val(x, y) {
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
                            wgpu::BindGroupLayoutEntry {
                                binding: 5,
                                visibility: wgpu::ShaderStages::COMPUTE,
                                ty: wgpu::BindingType::Buffer {
                                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                                    has_dynamic_offset: false,
                                    min_binding_size: wgpu::BufferSize::new(
                                        self.buffer_out_corr.size(),
                                    ),
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
                        wgpu::BindGroupEntry {
                            binding: 5,
                            resource: self.buffer_out_corr.as_entire_binding(),
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
