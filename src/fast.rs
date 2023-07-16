use nalgebra::DMatrix;
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
const KERNEL_SIZE: usize = 3;
// TODO: update to match results from previous C version
const FAST_THRESHOLD: u8 = 15;
const KEYPOINT_SCALE_MIN_SIZE: usize = 512;
const FAST_NUM_POINTS: usize = 12;
const FAST_CIRCLE_LENGTH: usize = FAST_CIRCLE_PIXELS.len() + FAST_NUM_POINTS - 1;

pub struct Fast {
    points: Vec<Point>,
}

impl Fast {
    /// Extract FAST features.
    pub fn new(img: &DMatrix<u8>) -> Fast {
        let mut img = img.clone();
        adjust_contrast(&mut img);
        // Detect points
        let keypoints: Vec<(usize, usize)> = (KERNEL_SIZE..(img.nrows() - KERNEL_SIZE))
            .into_par_iter()
            .map(|row| {
                let kp: Vec<(usize, usize)> = (KERNEL_SIZE..(img.ncols() - KERNEL_SIZE))
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
        let keypoints = (0..keypoints.len())
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
            .collect();
        Fast { points: keypoints }
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
