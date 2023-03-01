use crate::correlation;
use nalgebra::{DMatrix, Matrix3, Vector3};
use rayon::prelude::*;
use std::{cell::RefCell, ops::Range, sync::atomic::AtomicUsize, sync::atomic::Ordering};

const SCALE_MIN_SIZE: usize = 64;
const KERNEL_SIZE: usize = 5;
const KERNEL_WIDTH: usize = KERNEL_SIZE * 2 + 1;
const KERNEL_POINT_COUNT: usize = (KERNEL_WIDTH * KERNEL_WIDTH) as usize;

const THRESHOLD_AFFINE: f32 = 0.6;
const THRESHOLD_PERSPECTIVE: f32 = 0.7;
const MIN_STDEV_AFFINE: f32 = 1.0;
const MIN_STDEV_PERSPECTIVE: f32 = 25.0;
const CORRIDOR_SIZE: usize = 20;
// Decrease when using a low-powered GPU
const CORRIDOR_SEGMENT_LENGTH: usize = 256;
const SEARCH_AREA_SEGMENT_LENGTH: usize = 8;
const NEIGHBOR_DISTANCE: usize = 10;
const CORRIDOR_EXTEND_RANGE: f64 = 1.0;
const CORRIDOR_MIN_RANGE: f64 = 2.5;

type Match = (u32, u32);

#[derive(Debug)]
pub enum ProjectionMode {
    Affine,
    Perspective,
}

pub trait ProgressListener
where
    Self: Sync + Sized,
{
    fn report_status(&self, pos: f32);
}

pub struct PointCorrelations {
    pub correlated_points: DMatrix<Option<Match>>,
    first_pass: bool,
    min_stdev: f32,
    correlation_threshold: f32,
    fundamental_matrix: Matrix3<f64>,
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

struct CorrelationStep {
    scale: f32,
    img1: DMatrix<u8>,
    img2: DMatrix<u8>,
    img2_data: ImagePointData,
}

impl PointCorrelations {
    pub fn new(
        dimensions: (u32, u32),
        fundamental_matrix: Matrix3<f64>,
        projection_mode: ProjectionMode,
    ) -> PointCorrelations {
        // Height specifies rows, width specifies columns.
        let correlated_points =
            DMatrix::from_element(dimensions.1 as usize, dimensions.0 as usize, None);
        let (min_stdev, correlation_threshold) = match projection_mode {
            ProjectionMode::Affine => (MIN_STDEV_AFFINE, THRESHOLD_AFFINE),
            ProjectionMode::Perspective => (MIN_STDEV_PERSPECTIVE, THRESHOLD_PERSPECTIVE),
        };
        return PointCorrelations {
            correlated_points,
            first_pass: true,
            min_stdev,
            correlation_threshold,
            fundamental_matrix,
        };
    }

    pub fn correlate_images<PL: ProgressListener>(
        &mut self,
        img1: DMatrix<u8>,
        img2: DMatrix<u8>,
        scale: f32,
        progress_listener: Option<&PL>,
    ) {
        let img2_data = compute_image_point_data(&img2);
        let mut out_data: DMatrix<Option<Match>> =
            DMatrix::from_element(img1.shape().0, img1.shape().1, None);

        let corelation_step = CorrelationStep {
            scale,
            img1: img1,
            img2: img2,
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
                    pl.report_status(value);
                }
                out_col.iter_mut().enumerate().for_each(|(row, out_point)| {
                    if row < KERNEL_SIZE || row >= nrows - KERNEL_SIZE {
                        return;
                    }
                    self.correlate_point(&corelation_step, row, col, out_point);
                })
            });

        for row in 0..nrows {
            for col in 0..ncols {
                let point = out_data[(row, col)];
                if point.is_none() {
                    continue;
                }
                let out_row = (row as f32 / scale) as usize;
                let out_col = (col as f32 / scale) as usize;
                self.correlated_points[(out_row, out_col)] = point;
            }
        }
        self.first_pass = false;
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

        let e_line = self.get_epipolar_line(correlation_step, row, col);
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

        for corridor_offset in -(CORRIDOR_SIZE as isize)..CORRIDOR_SIZE as isize + 1 {
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
        &self,
        correlation_step: &CorrelationStep,
        row: usize,
        col: usize,
    ) -> EpipolarLine {
        let scale = correlation_step.scale;
        let p1 = Vector3::new(col as f64 / scale as f64, row as f64 / scale as f64, 1.0);
        let f_p1 = self.fundamental_matrix * p1;
        if f_p1[0].abs() > f_p1[1].abs() {
            return EpipolarLine {
                coeff: (1.0, -f_p1[1] / f_p1[0]),
                add: (0.0, -scale as f64 * f_p1[2] / f_p1[0]),
                corridor_offset: (0, 1),
            };
        }
        return EpipolarLine {
            coeff: (-f_p1[0] / f_p1[1], 1.0),
            add: (-scale as f64 * f_p1[2] / f_p1[1], 0.0),
            corridor_offset: (1, 0),
        };
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
            let avg2 = unsafe { img2_data.avg.get_unchecked((row2, col2)) };
            let stdev2 = unsafe { img2_data.stdev.get_unchecked((row2, col2)) };
            if !stdev2.is_finite() || stdev2.abs() < self.min_stdev {
                continue;
            }
            let mut corr = 0.0;
            for c in 0..KERNEL_WIDTH {
                for r in 0..KERNEL_WIDTH {
                    let delta1 = p1_data.delta[r * KERNEL_WIDTH + c];
                    let delta2 = unsafe {
                        *img2.get_unchecked((
                            (row2 + r).saturating_sub(KERNEL_SIZE),
                            (col2 + c).saturating_sub(KERNEL_SIZE),
                        )) as f32
                    } - avg2;
                    corr += delta1 * delta2;
                }
            }
            corr = corr / (p1_data.stdev * stdev2 * KERNEL_POINT_COUNT as f32);

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
        let data = &self.correlated_points;
        thread_local! {static STDEV_RANGE: RefCell<Vec<f64>> = RefCell::new(Vec::new())};
        let mut mid_corridor = 0.0;
        let mut neighbor_count: usize = 0;
        if row1 < NEIGHBOR_DISTANCE
            || col1 < NEIGHBOR_DISTANCE
            || row1 + NEIGHBOR_DISTANCE >= data.nrows()
            || col1 + NEIGHBOR_DISTANCE >= data.ncols()
        {
            return None;
        }
        let row_min = ((row1 - NEIGHBOR_DISTANCE) as f32 / scale).floor() as usize;
        let row_max = ((row1 + NEIGHBOR_DISTANCE) as f32 / scale).ceil() as usize;
        let col_min = ((col1 - NEIGHBOR_DISTANCE) as f32 / scale).floor() as usize;
        let col_max = ((col1 + NEIGHBOR_DISTANCE) as f32 / scale).ceil() as usize;
        let corridor_vertical = e_line.coeff.0.abs() > e_line.coeff.1.abs();

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
        return Some(corridor_start..corridor_end);
    }

    pub fn optimal_scale_steps(dimensions: (u32, u32)) -> usize {
        // TODO: replace this with log2
        let min_dimension = dimensions.1.min(dimensions.0) as usize;
        let mut scale = 0;
        while min_dimension / (1 << scale) > SCALE_MIN_SIZE {
            scale += 1;
        }
        return scale - 1;
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
    return data;
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

    return Some(result);
}
