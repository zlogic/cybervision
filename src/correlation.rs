#[cfg(target_os = "macos")]
mod metalcorrelation;
#[cfg(not(target_os = "macos"))]
mod vkcorrelation;

#[cfg(target_os = "macos")]
use metalcorrelation as gpu;
#[cfg(not(target_os = "macos"))]
use vkcorrelation as gpu;

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
                projection_mode,
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

struct ImagePointData {
    avg: Grid<f32>,
    stdev: Grid<f32>,
}

fn compute_image_point_data(img: &Grid<u8>) -> ImagePointData {
    let mut data = ImagePointData {
        avg: Grid::new(img.width(), img.height(), f32::NAN),
        stdev: Grid::new(img.width(), img.height(), f32::NAN),
    };
    data.avg.par_iter_mut().for_each(|(x, y, avg)| {
        let point = Point2D::new(x, y);
        let point_avg = match compute_point_avg(img, &point) {
            Some(p) => p,
            None => return,
        };
        *avg = point_avg;
    });
    data.stdev.par_iter_mut().for_each(|(x, y, stdev)| {
        let point = Point2D::new(x, y);
        let avg = data.avg.val(x, y);
        let point_stdev = match compute_point_stdev(img, &point, *avg) {
            Some(p) => p,
            None => return,
        };
        *stdev = point_stdev;
    });
    data
}

#[inline]
fn compute_point_avg(img: &Grid<u8>, point: &Point2D<usize>) -> Option<f32> {
    if !point_inside_bounds::<KERNEL_SIZE>(img, point) {
        return None;
    };
    let mut avg = 0.0f32;
    for y in 0..KERNEL_WIDTH {
        let s_y = (point.y + y).saturating_sub(KERNEL_SIZE);
        for x in 0..KERNEL_WIDTH {
            let s_x = (point.x + x).saturating_sub(KERNEL_SIZE);
            let value = img.val(s_x, s_y);
            avg += *value as f32;
        }
    }
    avg /= KERNEL_POINT_COUNT as f32;
    Some(avg)
}

#[inline]
fn compute_point_stdev(img: &Grid<u8>, point: &Point2D<usize>, avg: f32) -> Option<f32> {
    if !point_inside_bounds::<KERNEL_SIZE>(img, point) {
        return None;
    };
    let mut stdev = 0.0f32;

    for y in 0..KERNEL_WIDTH {
        let s_y = (point.y + y).saturating_sub(KERNEL_SIZE);
        for x in 0..KERNEL_WIDTH {
            let s_x = (point.x + x).saturating_sub(KERNEL_SIZE);
            let value = img.val(s_x, s_y);
            let delta = *value as f32 - avg;
            stdev += delta * delta;
        }
    }
    stdev = (stdev / KERNEL_POINT_COUNT as f32).sqrt();

    Some(stdev)
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
