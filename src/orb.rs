use nalgebra::{DMatrix, Matrix3};
use rayon::prelude::*;

type Point = (usize, usize);

const FAST_CIRCLE_PIXELS: [(i8, i8); 16] = [
    (0, -3),
    (1, -3),
    (2, -2),
    (3, -1),
    (3, 0),
    (3, 1),
    (2, 2),
    (1, 3),
    (0, 3),
    (-1, 3),
    (-2, 2),
    (-3, 1),
    (-3, 0),
    (-3, -1),
    (-2, -2),
    (-1, -3),
];

// TODO: allow to override configuration?
const FAST_KERNEL_SIZE: usize = 3;
// TODO: update to match results from previous C version
const FAST_THRESHOLD: u8 = 15;
const KEYPOINT_SCALE_MIN_SIZE: usize = 256;
const FAST_NUM_POINTS: usize = 12;
const FAST_CIRCLE_LENGTH: usize = FAST_CIRCLE_PIXELS.len() + FAST_NUM_POINTS - 1;
const HARRIS_K: f64 = 0.04;
const HARRIS_CORNER_THRESHOLD: f64 = 0.2;
const MAX_KEYPOINTS: usize = 10000;

pub struct ORB {
    points: Vec<Point>,
}

impl ORB {
    /// Find ORB feature points (without BRIEF descriptors).
    pub fn new(img: &DMatrix<u8>) -> ORB {
        let mut img = img.clone();
        adjust_contrast(&mut img);
        let harris_corners = ORB::harris_corners(&img);
        let keypoints = ORB::find_fast_keypoints(&img);
        let mut keypoints = keypoints
            .iter()
            .map(|(col, row)| ((*row, *col), harris_corners[(*row, *col)]))
            .filter(|(_, corner_score)| *corner_score > HARRIS_CORNER_THRESHOLD)
            .collect::<Vec<_>>();

        keypoints.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap().reverse());
        let keypoints = keypoints
            .iter()
            .take(MAX_KEYPOINTS)
            .map(|(keypoint, _)| *keypoint)
            .collect();
        ORB { points: keypoints }
    }

    fn find_fast_keypoints(img: &DMatrix<u8>) -> Vec<Point> {
        // Detect points
        let keypoints: Vec<(usize, usize)> = (FAST_KERNEL_SIZE..(img.nrows() - FAST_KERNEL_SIZE))
            .into_par_iter()
            .map(|row| {
                let kp: Vec<(usize, usize)> = (FAST_KERNEL_SIZE..(img.ncols() - FAST_KERNEL_SIZE))
                    .filter_map(
                        |col| match is_keypoint(&img, FAST_THRESHOLD.into(), row, col) {
                            true => Some((row, col)),
                            false => None,
                        },
                    )
                    .collect();
                kp
            })
            .flatten()
            .collect();
        // Add scores
        let scores: Vec<u8> = keypoints
            .par_iter()
            .map(|p| {
                let mut threshold_min = FAST_THRESHOLD as i16;
                let mut threshold_max = std::u8::MAX as i16;
                let mut threshold = (threshold_max + threshold_min) / 2;
                while threshold_max > threshold_min + 1 {
                    if is_keypoint(&img, threshold, p.0, p.1) {
                        threshold_min = threshold;
                    } else {
                        threshold_max = threshold;
                    }
                    threshold = (threshold_min + threshold_max) / 2;
                }
                threshold_min as u8
            })
            .collect();
        // Choose points with best scores
        (0..keypoints.len())
            .into_par_iter()
            .filter_map(|i| {
                let p1 = &keypoints[i];
                let score1 = scores[i];
                if i > 0 && keypoints[i - 1] == (p1.0, p1.1 - 1) && scores[i - 1] >= score1 {
                    // Left point has better score
                    return None;
                }
                if i < keypoints.len() - 1
                    && keypoints[i + 1] == (p1.0, p1.1 + 1)
                    && scores[i + 1] >= score1
                {
                    // Right point has better score
                    return None;
                }
                // Search for point above current
                for j in (0..i).rev() {
                    let p2 = &keypoints[j];
                    if p2.0 < p1.0 - 1 {
                        break;
                    }
                    if p2.0 == p1.0 - 1
                        && p2.1 >= p1.1 - 1
                        && p2.1 <= p1.1 + 1
                        && scores[j] >= score1
                    {
                        return None;
                    }
                }
                // Search for point below current
                for j in i + 1..keypoints.len() {
                    let p2 = &keypoints[j];
                    if p2.0 > p1.0 + 1 {
                        break;
                    }
                    if p2.0 == p1.0 + 1
                        && p2.1 >= p1.1 - 1
                        && p2.1 <= p1.1 + 1
                        && scores[j] >= score1
                    {
                        return None;
                    }
                }
                Some((p1.1, p1.0))
            })
            .collect()
    }

    fn convolve_kernel(img: &DMatrix<f64>, kernel: Matrix3<f64>) -> DMatrix<f64> {
        const KERNEL_SIZE: usize = 1;
        const KERNEL_WIDTH: usize = KERNEL_SIZE * 2 + 1;
        let mut result = DMatrix::<f64>::zeros(img.nrows(), img.ncols());
        let (nrows, ncols) = (result.nrows(), result.ncols());
        result
            .column_iter_mut()
            .enumerate()
            .par_bridge()
            .for_each(|(col, mut out_col)| {
                if col < KERNEL_SIZE || col + KERNEL_SIZE >= ncols {
                    return;
                }
                out_col.iter_mut().enumerate().for_each(|(row, out_point)| {
                    if row < KERNEL_SIZE || row + KERNEL_SIZE >= nrows {
                        return;
                    }

                    let mut sum = 0.0;
                    for k_row in 0..KERNEL_WIDTH {
                        for k_col in 0..KERNEL_WIDTH {
                            sum += kernel[(k_row, k_col)]
                                * img[(row + k_row - KERNEL_SIZE, col + k_col - KERNEL_SIZE)];
                        }
                    }
                    *out_point = sum;
                })
            });

        result
    }

    fn harris_corners(img: &DMatrix<u8>) -> DMatrix<f64> {
        const KERNEL_SOBEL_X: Matrix3<f64> =
            Matrix3::new(-1.0, -2.0, -1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0);
        const KERNEL_SOBEL_Y: Matrix3<f64> =
            Matrix3::new(-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0);
        const KERNEL_GAUSS: Matrix3<f64> = Matrix3::new(
            1.0 / 16.0,
            2.0 / 16.0,
            1.0 / 16.0,
            2.0 / 16.0,
            4.0 / 16.0,
            2.0 / 16.0,
            1.0 / 16.0,
            2.0 / 16.0,
            1.0 / 16.0,
        );
        let mut img_float = DMatrix::<f64>::zeros(img.nrows(), img.ncols());
        img_float
            .column_iter_mut()
            .enumerate()
            .for_each(|(col, mut out_col)| {
                out_col.iter_mut().enumerate().for_each(|(row, out_point)| {
                    *out_point = img[(row, col)] as f64 / 255.0;
                })
            });
        let img = img_float;

        let mut dx = ORB::convolve_kernel(&img, KERNEL_SOBEL_X);
        let mut dy = ORB::convolve_kernel(&img, KERNEL_SOBEL_Y);
        let (nrows, ncols) = (img.nrows(), img.ncols());
        drop(img);
        let mut dx_dy = DMatrix::<f64>::zeros(nrows, ncols);
        dx_dy
            .column_iter_mut()
            .enumerate()
            .for_each(|(col, mut out_col)| {
                out_col.iter_mut().enumerate().for_each(|(row, out_point)| {
                    *out_point = dx[(row, col)] * dy[(row, col)];
                })
            });
        dx.apply(|val| *val = val.powi(2));
        dy.apply(|val| *val = val.powi(2));

        let g_dx2 = ORB::convolve_kernel(&dx, KERNEL_GAUSS);
        let g_dy2 = ORB::convolve_kernel(&dy, KERNEL_GAUSS);
        let g_dx_dy = ORB::convolve_kernel(&dx_dy, KERNEL_GAUSS);
        drop(dx);
        drop(dy);
        drop(dx_dy);

        let mut corner_map = DMatrix::<f64>::zeros(nrows, ncols);
        corner_map
            .column_iter_mut()
            .enumerate()
            .for_each(|(col, mut out_col)| {
                out_col.iter_mut().enumerate().for_each(|(row, out_point)| {
                    let det = g_dx2[(row, col)] * g_dy2[(row, col)] - g_dx_dy[(row, col)].powi(2);
                    let trace = g_dx2[(row, col)] + g_dy2[(row, col)];
                    *out_point = det - HARRIS_K * trace.powi(2);
                })
            });

        corner_map
    }

    pub fn keypoints(&self) -> &Vec<Point> {
        &self.points
    }

    pub fn optimal_scale_steps(dimensions: (u32, u32)) -> usize {
        let min_dimension = dimensions.1.min(dimensions.0) as usize;
        if min_dimension <= KEYPOINT_SCALE_MIN_SIZE {
            return 0;
        }
        (min_dimension as f64 / KEYPOINT_SCALE_MIN_SIZE as f64)
            .log2()
            .floor() as usize
    }
}

#[inline]
fn get_pixel_offset(img: &DMatrix<u8>, row: usize, col: usize, offset: (i8, i8)) -> i16 {
    let row_new = row.saturating_add_signed(offset.1 as isize);
    let col_new = col.saturating_add_signed(offset.0 as isize);
    img[(row_new, col_new)] as i16
}

#[inline]
fn is_keypoint(img: &DMatrix<u8>, threshold: i16, row: usize, col: usize) -> bool {
    let val: i16 = img[(row, col)] as i16;
    let mut last_more_pos: Option<usize> = None;
    let mut last_less_pos: Option<usize> = None;
    let mut max_length = 0;

    for i in 0..FAST_CIRCLE_LENGTH {
        let p = FAST_CIRCLE_PIXELS[i % FAST_CIRCLE_PIXELS.len()];
        let c_val = get_pixel_offset(img, row, col, p);
        if c_val > val + threshold {
            last_more_pos = last_more_pos.or(Some(i));
            let length = last_more_pos.map(|p| i - p).unwrap_or(0) + 1;
            max_length = max_length.max(length);
        } else {
            last_more_pos = None;
        }
        if c_val < val - threshold {
            last_less_pos = last_less_pos.or(Some(i));
            let length = last_less_pos.map(|p| i - p).unwrap_or(0) + 1;
            max_length = max_length.max(length);
        } else {
            last_less_pos = None;
        }
        if max_length >= FAST_NUM_POINTS {
            return true;
        }
    }
    false
}

fn adjust_contrast(img: &mut DMatrix<u8>) {
    let mut min: u8 = core::u8::MAX;
    let mut max: u8 = core::u8::MIN;

    for p in img.iter() {
        min = min.min(*p);
        max = max.max(*p);
    }

    if min >= max {
        return;
    }

    let coeff = core::u8::MAX as f32 / ((max - min) as f32);
    for p in img.iter_mut() {
        *p = (coeff * ((*p - min) as f32)).round() as u8;
    }
}
