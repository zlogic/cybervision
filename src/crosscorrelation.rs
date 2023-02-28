use crate::correlation;
use nalgebra::{DMatrix, Matrix3, Vector3};
use rayon::prelude::*;
use std::{ops::Range, sync::atomic::AtomicUsize, sync::atomic::Ordering};

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
const CORRIDOR_EXTEND_RANGE: f32 = 1.0;
const CORRIDOR_MIN_RANGE: f32 = 2.5;

type Match = (usize, usize);

#[derive(Debug)]
pub enum ProjectionMode {
    Affine,
    Perspective,
}

pub struct DepthImage {
    first_pass: bool,
    pub data: DMatrix<Option<Match>>,
}

impl DepthImage {
    pub fn new(dimensions: (u32, u32)) -> DepthImage {
        // Height specifies rows, width specifies columns.
        let data = DMatrix::from_element(dimensions.1 as usize, dimensions.0 as usize, None);
        return DepthImage {
            first_pass: true,
            data,
        };
    }
}

pub fn correlate_images<P>(
    img1: &DMatrix<u8>,
    img2: &DMatrix<u8>,
    fundamental_matrix: &Matrix3<f64>,
    scale: f32,
    projection_mode: ProjectionMode,
    destination: &mut DepthImage,
    pb: Option<P>,
) where
    P: Fn(f32) + Sync + Send,
{
    let inv_scale = 1.0 / scale;
    //let neighbor_size = 2 * (NEIGHBOR_DISTANCE as f32 * inv_scale).ceil() as usize + 1;
    let (min_stdev, correlation_threshold) = match projection_mode {
        ProjectionMode::Affine => (MIN_STDEV_AFFINE, THRESHOLD_AFFINE),
        ProjectionMode::Perspective => (MIN_STDEV_PERSPECTIVE, THRESHOLD_PERSPECTIVE),
    };
    let img2_data = compute_image_point_data(img2);
    let ct = CorrelationTask {
        correlation_threshold,
        min_stdev,
        scale,
        inv_scale,
        fundamental_matrix: *fundamental_matrix,
        img1: &img1,
        img2: &img2,
        img2_data,
        destination: &destination,
    };

    let mut out_data: DMatrix<Option<Match>> =
        DMatrix::from_element(img1.shape().0, img1.shape().1, None);

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
            pb.as_ref().map(|pb| {
                let it = (counter.fetch_add(1, Ordering::Relaxed)) as f32 / out_data_cols;
                pb(it);
            });
            out_col.iter_mut().enumerate().for_each(|(row, out_point)| {
                if row < KERNEL_SIZE || row >= nrows - KERNEL_SIZE {
                    return;
                }
                ct.correlate_point(row, col, out_point);
            })
        });

    for row in 0..nrows {
        for col in 0..ncols {
            let point = out_data[(row, col)];
            if point.is_none() {
                continue;
            }
            let out_row = (row as f32 * inv_scale) as usize;
            let out_col = (col as f32 * inv_scale) as usize;
            destination.data[(out_row, out_col)] = point;
        }
    }
    destination.first_pass = false;
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
                let p = match compute_compact_point_data(img, &(row, col)) {
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
fn compute_compact_point_data(img: &DMatrix<u8>, p: &(usize, usize)) -> Option<PointData> {
    if !correlation::point_inside_bounds::<KERNEL_SIZE>(img.shape(), p.0, p.1) {
        return None;
    };
    const KERNEL_WIDTH: usize = KERNEL_SIZE * 2 + 1;
    const KERNEL_POINT_COUNT: f32 = (KERNEL_WIDTH * KERNEL_WIDTH) as f32;
    let mut result = PointData {
        avg: 0.0,
        stdev: 0.0,
    };
    for r in 0..KERNEL_SIZE * 2 + 1 {
        let row = (p.0 + KERNEL_SIZE).saturating_sub(r);
        for c in 0..KERNEL_SIZE * 2 + 1 {
            let col = (p.1 + KERNEL_SIZE).saturating_sub(c);
            let value = img[(row, col)];
            result.avg += value as f32;
        }
    }
    result.avg /= KERNEL_POINT_COUNT;

    for r in 0..KERNEL_SIZE * 2 + 1 {
        let row = (p.0 + KERNEL_SIZE).saturating_sub(r);
        for c in 0..KERNEL_SIZE * 2 + 1 {
            let col = (p.1 + KERNEL_SIZE).saturating_sub(c);
            let value = img[(row, col)];
            let delta = value as f32 - result.avg;
            result.stdev += delta * delta;
        }
    }
    result.stdev = (result.stdev / KERNEL_POINT_COUNT).sqrt();

    return Some(result);
}

struct CorrelationTask<'i> {
    min_stdev: f32,
    correlation_threshold: f32,
    scale: f32,
    inv_scale: f32,
    fundamental_matrix: Matrix3<f64>,
    img1: &'i DMatrix<u8>,
    img2: &'i DMatrix<u8>,
    img2_data: ImagePointData,
    destination: &'i DepthImage,
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

impl CorrelationTask<'_> {
    fn correlate_point(&self, row: usize, col: usize, out_point: &mut Option<Match>) {
        let p1_data = correlation::compute_point_data::<KERNEL_SIZE, KERNEL_POINT_COUNT>(
            &self.img1, row, col,
        );
        let p1_data = match p1_data {
            Some(p) => p,
            None => return,
        };
        if !p1_data.stdev.is_finite() || p1_data.stdev.abs() < self.min_stdev {
            return;
        }

        let e_line = self.get_epipolar_line(row, col);
        if !e_line.coeff.0.is_finite()
            || !e_line.coeff.1.is_finite()
            || !e_line.add.0.is_finite()
            || !e_line.add.1.is_finite()
        {
            return;
        }
        if !self.destination.first_pass {
            // use thread_local!(static CONTENTS: RefCell<String> = RefCell::new(String::new()));
            // and estimate search range
        }
        const CORRIDOR_START: usize = KERNEL_SIZE;
        let corridor_end = match e_line.coeff.1.abs() > e_line.coeff.0.abs() {
            true => self.img2.ncols().saturating_sub(KERNEL_SIZE),
            false => self.img2.nrows().saturating_sub(KERNEL_SIZE),
        };

        let mut best_match = BestMatch {
            pos: None,
            corr: None,
        };

        for corridor_offset in -(CORRIDOR_SIZE as isize)..CORRIDOR_SIZE as isize + 1 {
            self.correlate_corridor_area(
                &e_line,
                &p1_data,
                &mut best_match,
                corridor_offset,
                CORRIDOR_START..corridor_end,
            );
        }
        *out_point = best_match.pos
    }

    fn get_epipolar_line(&self, row: usize, col: usize) -> EpipolarLine {
        let p1 = Vector3::new(col as f64, row as f64, 1.0);
        let f_p1 = self.fundamental_matrix * p1;
        if f_p1[0].abs() > f_p1[1].abs() {
            return EpipolarLine {
                coeff: (1.0, -f_p1[1] / f_p1[0]),
                add: (0.0, -self.scale as f64 * f_p1[2] / f_p1[0]),
                corridor_offset: (0, 1),
            };
        }
        return EpipolarLine {
            coeff: (-f_p1[0] / f_p1[1], 1.0),
            add: (-self.scale as f64 * f_p1[2] / f_p1[1], 0.0),
            corridor_offset: (1, 0),
        };
    }

    fn correlate_corridor_area(
        &self,
        e_line: &EpipolarLine,
        p1_data: &correlation::PointData<KERNEL_POINT_COUNT>,
        best_match: &mut BestMatch,
        corridor_offset: isize,
        corridor_range: Range<usize>,
    ) {
        for i in corridor_range {
            let row2 = (e_line.coeff.0 * i as f64 + e_line.add.0)
                + (corridor_offset * e_line.corridor_offset.0) as f64;
            let col2 = (e_line.coeff.1 * i as f64 + e_line.add.1)
                + (corridor_offset * e_line.corridor_offset.1) as f64;
            let row2 = row2.floor() as usize;
            let col2 = col2.floor() as usize;
            if row2 < KERNEL_SIZE
                || row2 >= self.img2.nrows() - KERNEL_SIZE
                || col2 < KERNEL_SIZE
                || col2 >= self.img2.ncols() - KERNEL_SIZE
            {
                continue;
            }
            let avg2 = self.img2_data.avg[(row2, col2)];
            let stdev2 = self.img2_data.stdev[(row2, col2)];
            if !stdev2.is_finite() || stdev2.abs() < self.min_stdev {
                continue;
            }
            let mut corr = 0.0;
            for c in 0..KERNEL_WIDTH {
                for r in 0..KERNEL_WIDTH {
                    let delta1 = p1_data.delta[r * KERNEL_WIDTH + c];
                    let delta2 = self.img2[(
                        (row2 + r).saturating_sub(KERNEL_SIZE),
                        (col2 + c).saturating_sub(KERNEL_SIZE),
                    )] as f32
                        - avg2;
                    corr += delta1 * delta2;
                }
            }
            corr = corr / (p1_data.stdev * stdev2 * KERNEL_POINT_COUNT as f32);

            if corr >= self.correlation_threshold
                && best_match.corr.map_or(true, |best_corr| corr >= best_corr)
            {
                best_match.pos = Some((
                    (self.inv_scale * row2 as f32).round() as usize,
                    (self.inv_scale * col2 as f32).round() as usize,
                ));
                best_match.corr = Some(corr);
            }
        }
    }
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
