use image::GrayImage;

use rayon::prelude::*;

type Point = (u32, u32);

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
const KERNEL_SIZE: u32 = 3;
// TODO: update to match results from previous C version
const FAST_THRESHOLD: u8 = 15;
const KEYPOINT_SCALE_MIN_SIZE: u32 = 512;
const FAST_NUM_POINTS: u8 = 12;
const FAST_CIRCLE_LENGTH: usize = FAST_CIRCLE_PIXELS.len() + FAST_NUM_POINTS as usize - 1;

/// Extract FAST features.
pub fn find_points(img: &GrayImage) -> Vec<Point> {
    let kernel_size = KERNEL_SIZE;
    let mut img = img.clone();
    adjust_contrast(&mut img);
    // Detect points
    let keypoints: Vec<(u32, u32)> = (kernel_size..(img.height() - kernel_size))
        .into_par_iter()
        .map(|y| {
            let kp: Vec<(u32, u32)> = (kernel_size..(img.width() - kernel_size))
                .into_iter()
                .filter_map(|x| match is_keypoint(&img, FAST_THRESHOLD.into(), x, y) {
                    true => Some((x, y)),
                    false => None,
                })
                .collect();
            return kp;
        })
        .flatten()
        .collect();
    // Add scores
    let scores: Vec<u8> = keypoints
        .par_iter()
        .map(|p| {
            let mut threshold_min = FAST_THRESHOLD as i16;
            let mut threshold_max = std::u8::MAX as i16;
            let mut threshold = (threshold_max as i16 + threshold_min as i16) / 2;
            while threshold_max > threshold_min + 1 {
                if is_keypoint(&img, threshold, p.0, p.1) {
                    threshold_min = threshold;
                } else {
                    threshold_max = threshold;
                }
                threshold = (threshold_min + threshold_max) / 2;
            }
            return threshold_min as u8;
        })
        .collect();
    // Choose points with best scores
    return (0..keypoints.len())
        .into_par_iter()
        .filter_map(|i| {
            let p1 = &keypoints[i];
            let score1 = scores[i];
            if i > 0 && keypoints[i - 1] == (p1.0 - 1, p1.1) && scores[i - 1] >= score1 {
                // Left point has better score
                return None;
            }
            if i < keypoints.len() - 1
                && keypoints[i + 1] == (p1.0 + 1, p1.1)
                && scores[i + 1] >= score1
            {
                // Right point has better score
                return None;
            }
            // Search for point above current
            for j in (0..i).rev() {
                let p2 = &keypoints[j];
                if p2.1 < p1.1 - 1 {
                    break;
                }
                if p2.1 == p1.1 - 1 && p2.0 >= p1.0 - 1 && p2.0 <= p1.0 + 1 && scores[j] >= score1 {
                    return None;
                }
            }
            // Search for point below current
            for j in i + 1..keypoints.len() {
                let p2 = &keypoints[j];
                if p2.1 > p1.1 + 1 {
                    break;
                }
                if p2.1 == p1.1 + 1 && p2.0 >= p1.0 - 1 && p2.0 <= p1.0 + 1 && scores[j] >= score1 {
                    return None;
                }
            }
            return Some(*p1);
        })
        .collect();
}

#[inline]
fn get_pixel_offset(img: &GrayImage, x: u32, y: u32, offset: (i8, i8)) -> i16 {
    let x_new = x.saturating_add_signed(offset.0 as i32);
    let y_new = y.saturating_add_signed(offset.1 as i32);
    return img.get_pixel(x_new, y_new)[0] as i16;
}

#[inline]
fn is_keypoint(img: &GrayImage, threshold: i16, x: u32, y: u32) -> bool {
    let val: i16 = img.get_pixel(x, y)[0] as i16;
    let mut last_more_pos: Option<usize> = None;
    let mut last_less_pos: Option<usize> = None;
    let mut max_length = 0;

    for i in 0..FAST_CIRCLE_LENGTH {
        let p = FAST_CIRCLE_PIXELS[i % FAST_CIRCLE_PIXELS.len()];
        let c_val = get_pixel_offset(&img, x, y, p);
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
        if max_length >= FAST_NUM_POINTS.into() {
            return true;
        }
    }
    return false;
}

pub fn optimal_keypoint_scale(img: &GrayImage) -> f32 {
    let min_dimension = img.width().min(img.height());
    let mut scale = 0;
    while min_dimension / (1 << scale) > KEYPOINT_SCALE_MIN_SIZE {
        scale += 1;
    }
    scale = (scale - 1).max(0);
    return 1.0 / ((1 << scale) as f32);
}

fn adjust_contrast(img: &mut GrayImage) {
    let mut min: u8 = core::u8::MAX;
    let mut max: u8 = core::u8::MIN;

    for p in img.pixels() {
        min = min.min(p[0]);
        max = max.max(p[0]);
    }

    if min >= max {
        return;
    }

    let coeff = core::u8::MAX as f32 / ((max - min) as f32);
    for p in img.pixels_mut() {
        p[0] = (coeff * ((p[0] - min) as f32)).round() as u8;
    }
}
