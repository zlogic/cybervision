use std::i8;

use image::imageops::FilterType;
use image::GenericImageView;
use image::GrayImage;

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
const KERNEL_SIZE: u32 = 3;

pub struct FastExtractor {
    threshold: u8,
    num_points: u8,
    keypoint_scale_min_size: u32,
    circle_points: [(i8, i8); 16],
    circle_length: usize,
}

impl FastExtractor {
    pub fn new() -> FastExtractor {
        // TODO: allow to override configuration?
        // TODO: update threshold since to match results from previous C version
        let num_points = 12;
        return FastExtractor {
            threshold: 15,
            num_points: num_points,
            keypoint_scale_min_size: 512,
            circle_points: FAST_CIRCLE_PIXELS,
            circle_length: FAST_CIRCLE_PIXELS.len() + num_points as usize - 1,
        };
    }

    /// Extract FAST features.
    pub fn find_points(&self, img: &GrayImage) -> Vec<Point> {
        let img = self.adjust_image(img);
        let kernel_size = KERNEL_SIZE;
        let mut keypoints = Vec::<Point>::new();
        let mut scores = Vec::<u8>::new();
        // Detect points
        for y in kernel_size..(img.height() - kernel_size) {
            for x in kernel_size..(img.width() - kernel_size) {
                if self.is_keypoint(&img, self.threshold.into(), x, y) {
                    keypoints.push((x, y));
                }
            }
        }
        // Add scores
        for (x, y) in &keypoints {
            let mut threshold_min = self.threshold as i16;
            let mut threshold_max = std::u8::MAX as i16;
            let mut threshold = (threshold_max as i16 + threshold_min as i16) / 2;
            while threshold_max > threshold_min + 1 {
                if self.is_keypoint(&img, threshold, *x, *y) {
                    threshold_min = threshold;
                } else {
                    threshold_max = threshold;
                }
                threshold = (threshold_min + threshold_max) / 2;
            }
            scores.push(threshold_min as u8);
        }
        // Choose points with best scores
        let mut filtered_points = Vec::<Point>::new();
        'kp: for i in 0..keypoints.len() {
            let p1 = &keypoints[i];
            let score1 = scores[i];
            if i > 0 && keypoints[i - 1] == (p1.0 - 1, p1.1) && scores[i - 1] >= score1 {
                // Left point has better score
                continue;
            }
            if i < keypoints.len() - 1
                && keypoints[i + 1] == (p1.0 + 1, p1.1)
                && scores[i + 1] >= score1
            {
                // Right point has better score
                continue;
            }
            // Search for point above current
            for j in (0..i).rev() {
                let p2 = &keypoints[j];
                if p2.1 < p1.1 - 1 {
                    break;
                }
                if p2.1 == p1.1 - 1 && p2.0 >= p1.0 - 1 && p2.0 <= p1.0 + 1 && scores[j] >= score1 {
                    continue 'kp;
                }
            }
            // Search for point below current
            for j in i + 1..keypoints.len() {
                let p2 = &keypoints[j];
                if p2.1 > p1.1 + 1 {
                    break;
                }
                if p2.1 == p1.1 + 1 && p2.0 >= p1.0 - 1 && p2.0 <= p1.0 + 1 && scores[j] >= score1 {
                    continue 'kp;
                }
            }
            filtered_points.push(*p1);
        }
        return filtered_points;
    }

    #[inline]
    unsafe fn get_pixel_offset(img: &GrayImage, x: u32, y: u32, offset: (i8, i8)) -> i16 {
        let x_new = x.saturating_add_signed(offset.0 as i32);
        let y_new = y.saturating_add_signed(offset.1 as i32);
        return img.unsafe_get_pixel(x_new, y_new)[0] as i16;
    }

    #[inline]
    fn is_keypoint(&self, img: &GrayImage, threshold: i16, x: u32, y: u32) -> bool {
        let val: i16 = unsafe { img.unsafe_get_pixel(x, y)[0] as i16 };
        let mut last_more_pos: Option<usize> = None;
        let mut last_less_pos: Option<usize> = None;
        let mut max_length = 0;

        for i in 0..self.circle_length {
            let p = self.circle_points[i % self.circle_points.len()];
            let c_val = unsafe { FastExtractor::get_pixel_offset(&img, x, y, p) };
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
            if max_length >= self.num_points.into() {
                return true;
            }
        }
        return false;
    }

    fn adjust_image(&self, img: &GrayImage) -> GrayImage {
        let optimal_scale = self.optimal_keypoint_scale(&img);
        let mut img = image::imageops::resize(
            img,
            (img.width() as f32 * optimal_scale) as u32,
            (img.height() as f32 * optimal_scale) as u32,
            FilterType::Lanczos3,
        );
        self.adjust_contrast(&mut img);
        return img;
    }

    fn optimal_keypoint_scale(&self, img: &GrayImage) -> f32 {
        let min_dimension = img.width().min(img.height());
        let mut scale = 0;
        while min_dimension / (1 << scale) > self.keypoint_scale_min_size {
            scale += 1;
        }
        scale = (scale - 1).max(0);
        return 1.0 / ((1 << scale) as f32);
    }

    fn adjust_contrast(&self, img: &mut GrayImage) {
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
}
