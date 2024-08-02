use crate::data::{Grid, Point2D};
use std::f64::consts::PI;
use std::sync::atomic::{AtomicUsize, Ordering};

use rayon::prelude::*;

type Point = Point2D<usize>;
type Offset = Point2D<i8>;
type Keypoint = (Point, [u32; 8]);

const FAST_CIRCLE_PIXELS: [Offset; 16] = [
    Offset::new(0, -3),
    Offset::new(1, -3),
    Offset::new(2, -2),
    Offset::new(3, -1),
    Offset::new(3, 0),
    Offset::new(3, 1),
    Offset::new(2, 2),
    Offset::new(1, 3),
    Offset::new(0, 3),
    Offset::new(-1, 3),
    Offset::new(-2, 2),
    Offset::new(-3, 1),
    Offset::new(-3, 0),
    Offset::new(-3, -1),
    Offset::new(-2, -2),
    Offset::new(-1, -3),
];

const FAST_KERNEL_SIZE: usize = 3;
const FAST_THRESHOLD: u8 = 15;
const KEYPOINT_SCALE_MIN_SIZE: usize = 256;
const FAST_NUM_POINTS: usize = 9;
const FAST_CIRCLE_LENGTH: usize = FAST_CIRCLE_PIXELS.len() + FAST_NUM_POINTS - 1;
const HARRIS_KERNEL_SIZE: usize = 3;
const HARRIS_KERNEL_WIDTH: usize = HARRIS_KERNEL_SIZE * 2 + 1;
const HARRIS_K: f64 = 0.04;
const ORB_GAUSS_KERNEL_WIDTH: usize = 11;
const ORB_PATCH_WIDTH: usize = 31;
const ORB_PATCH_SIZE: usize = ORB_PATCH_WIDTH / 2;
const MAX_KEYPOINTS: usize = 10_000;

pub trait ProgressListener
where
    Self: Sync + Sized,
{
    fn report_status(&self, pos: f32);
}

pub fn extract_points<PL: ProgressListener>(
    img: &Grid<u8>,
    progress_listener: Option<&PL>,
) -> Vec<Keypoint> {
    let mut img_adjusted = img.to_owned();
    adjust_contrast(&mut img_adjusted);
    let keypoints = find_fast_keypoints(&img_adjusted, progress_listener);

    let harris_gaussian_kernel = gaussian_kernel::<HARRIS_KERNEL_WIDTH>();
    let counter = AtomicUsize::new(0);
    let mut keypoints = keypoints
        .par_iter()
        .filter_map(|point| {
            if let Some(pl) = progress_listener {
                let value = 0.35
                    + 0.35
                        * (counter.fetch_add(1, Ordering::Relaxed) as f32 / keypoints.len() as f32);
                pl.report_status(value);
            }
            Some((
                *point,
                harris_response(img, &harris_gaussian_kernel, *point)?,
            ))
        })
        .collect::<Vec<_>>();

    keypoints.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap().reverse());
    let keypoints = keypoints
        .iter()
        .take(MAX_KEYPOINTS)
        .map(|(keypoint, _)| *keypoint)
        .collect();

    extract_brief_descriptors(img, keypoints, progress_listener)
}

fn find_fast_keypoints<PL: ProgressListener>(
    img: &Grid<u8>,
    progress_listener: Option<&PL>,
) -> Vec<Point> {
    // Detect points
    let total_rows = img.height() - FAST_KERNEL_SIZE * 2;
    let counter = AtomicUsize::new(0);
    let keypoints: Vec<Point> = (FAST_KERNEL_SIZE..(img.height() - FAST_KERNEL_SIZE))
        .into_par_iter()
        .map(|y| {
            if let Some(pl) = progress_listener {
                let value =
                    0.20 * (counter.fetch_add(1, Ordering::Relaxed) as f32 / total_rows as f32);
                pl.report_status(value);
            }
            let kp: Vec<Point> = (FAST_KERNEL_SIZE..(img.width() - FAST_KERNEL_SIZE))
                .filter_map(|x| match is_keypoint(img, FAST_THRESHOLD.into(), x, y) {
                    true => Some(Point::new(x, y)),
                    false => None,
                })
                .collect();
            kp
        })
        .flatten()
        .collect();
    // Add scores
    let counter = AtomicUsize::new(0);
    let scores: Vec<u8> = keypoints
        .par_iter()
        .map(|p| {
            if let Some(pl) = progress_listener {
                let value = 0.20
                    + 0.05
                        * (counter.fetch_add(1, Ordering::Relaxed) as f32 / keypoints.len() as f32);
                pl.report_status(value);
            }
            let mut threshold_min = FAST_THRESHOLD as i16;
            let mut threshold_max = u8::MAX as i16;
            let mut threshold = (threshold_max + threshold_min) / 2;
            while threshold_max > threshold_min + 1 {
                if is_keypoint(img, threshold, p.x, p.y) {
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
    let counter = AtomicUsize::new(0);
    (0..keypoints.len())
        .into_par_iter()
        .filter_map(|i| {
            if let Some(pl) = progress_listener {
                let value = 0.25
                    + 0.10
                        * (counter.fetch_add(1, Ordering::Relaxed) as f32 / keypoints.len() as f32);
                pl.report_status(value);
            }
            let p1 = &keypoints[i];
            let score1 = scores[i];
            if i > 0
                && keypoints[i - 1].x == p1.x - 1
                && keypoints[i - 1].y == p1.y
                && scores[i - 1] >= score1
            {
                // Left point has better score
                return None;
            }
            if i < keypoints.len() - 1
                && keypoints[i + 1].x == p1.x + 1
                && keypoints[i + 1].y == p1.y
                && scores[i + 1] >= score1
            {
                // Right point has better score
                return None;
            }
            // Search for point above current
            for j in (0..i).rev() {
                let p2 = &keypoints[j];
                if p2.y < p1.y - 1 {
                    break;
                }
                if p2.y == p1.y - 1 && p2.x >= p1.x - 1 && p2.x <= p1.x + 1 && scores[j] >= score1 {
                    return None;
                }
            }
            // Search for point below current
            for j in i + 1..keypoints.len() {
                let p2 = &keypoints[j];
                if p2.y > p1.y + 1 {
                    break;
                }
                if p2.y == p1.y + 1 && p2.x >= p1.x - 1 && p2.x <= p1.x + 1 && scores[j] >= score1 {
                    return None;
                }
            }
            Some(*p1)
        })
        .collect()
}

fn gaussian_kernel<const KERNEL_WIDTH: usize>() -> [f64; KERNEL_WIDTH] {
    let sigma = (KERNEL_WIDTH - 1) as f64 / 6.0;
    let sigma_2 = sigma.powi(2);
    let divider = (2.0 * PI).sqrt() * sigma;
    let center = (KERNEL_WIDTH / 2) as f64;
    let mut kernel = [0.0; KERNEL_WIDTH];

    for (i, kernel_out) in kernel.iter_mut().enumerate() {
        *kernel_out = (-(i as f64 - center).powi(2) / (2.0 * sigma_2)).exp() / divider;
    }

    kernel
}

fn convolve_kernel<const KERNEL_WIDTH: usize, const KERNEL_PIXELS_COUNT: usize>(
    img: &Grid<u8>,
    point: Point,
    kernel: &'static [f64; KERNEL_PIXELS_COUNT],
) -> Option<f64> {
    let kernel_size = KERNEL_WIDTH / 2;
    let (x, y) = (point.x, point.y);
    if x < kernel_size
        || y < kernel_size
        || x + kernel_size >= img.width()
        || y + kernel_size >= img.height()
    {
        return None;
    }

    let mut result = 0.0;
    for (i, kernel_val) in kernel.iter().enumerate() {
        let k_x = i % KERNEL_WIDTH;
        let k_y = i / KERNEL_WIDTH;
        result +=
            kernel_val * (*img.val(x + k_x - kernel_size, y + k_y - kernel_size)) as f64 / 255.0;
    }

    Some(result)
}

fn harris_response<const KERNEL_WIDTH: usize>(
    img: &Grid<u8>,
    kernel_gauss: &[f64; KERNEL_WIDTH],
    point: Point,
) -> Option<f64> {
    const KERNEL_SOBEL_X: [f64; 9] = [-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0];
    const KERNEL_SOBEL_Y: [f64; 9] = [-1.0, -2.0, -1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0];

    let kernel_size = KERNEL_WIDTH / 2;
    let (x, y) = (point.x, point.y);
    if x < kernel_size
        || y < kernel_size
        || x + kernel_size >= img.width()
        || y + kernel_size >= img.height()
    {
        return None;
    }

    let mut g_dx2 = 0.0;
    let mut g_dy2 = 0.0;
    let mut g_dx_dy = 0.0;
    for k_y in 0..KERNEL_WIDTH {
        for k_x in 0..KERNEL_WIDTH {
            let point = Point::new(x + k_x - HARRIS_KERNEL_SIZE, y + k_y - HARRIS_KERNEL_SIZE);
            let dx = convolve_kernel::<KERNEL_WIDTH, 9>(img, point, &KERNEL_SOBEL_X)?;
            let dy = convolve_kernel::<KERNEL_WIDTH, 9>(img, point, &KERNEL_SOBEL_Y)?;

            let gauss_mul = kernel_gauss[k_x] * kernel_gauss[k_y];
            g_dx2 += dx * dx * gauss_mul;
            g_dy2 += dy * dy * gauss_mul;
            g_dx_dy += dx * dy * gauss_mul;
        }
    }

    let det = g_dx2 * g_dy2 - g_dx_dy.powi(2);
    let trace = g_dx2 + g_dy2;
    let harris_response = det - HARRIS_K * trace.powi(2);

    Some(harris_response)
}

fn gaussian_blur<const KERNEL_WIDTH: usize>(
    img: &Grid<u8>,
    kernel_gauss: &[f64; KERNEL_WIDTH],
) -> Grid<Option<f64>> {
    let mut result = Grid::new(img.width(), img.height(), None);
    let kernel_size = KERNEL_WIDTH / 2;

    result.par_iter_mut().for_each(|(x, y, out_point)| {
        if y < kernel_size || y + kernel_size >= img.height() {
            return;
        }
        if x < kernel_size || x + kernel_size >= img.width() {
            return;
        }
        let mut sum = 0.0;
        for (i, kernel_val) in kernel_gauss.iter().enumerate() {
            sum += kernel_val * (*img.val(x + i - kernel_size, y)) as f64;
        }
        *out_point = Some(sum);
    });

    let img = result;
    let mut result = Grid::new(img.width(), img.width(), None);
    result.par_iter_mut().for_each(|(x, y, out_point)| {
        if y < kernel_size || y + kernel_size >= img.height() {
            return;
        }
        if x < kernel_size || x + kernel_size >= img.width() {
            return;
        }
        let mut sum = 0.0;
        for (i, kernel_val) in kernel_gauss.iter().enumerate() {
            let val = if let Some(val) = img.val(x, y + i - kernel_size) {
                val
            } else {
                return;
            };
            sum += kernel_val * val;
        }
        *out_point = Some(sum);
    });

    result
}

fn get_brief_orientation(img: &Grid<Option<f64>>, point: &Point) -> Option<f64> {
    let (x, y) = (point.x, point.y);
    if x < ORB_PATCH_SIZE
        || y < ORB_PATCH_SIZE
        || x + ORB_PATCH_SIZE >= img.width()
        || y + ORB_PATCH_SIZE >= img.height()
    {
        return None;
    }

    let mut m_00 = 0;
    let mut m_01 = 0;
    let mut m_10 = 0;
    for m_y in 0..ORB_PATCH_WIDTH {
        for m_x in 0..ORB_PATCH_WIDTH {
            let (s_x, s_y) = (x + m_x - ORB_PATCH_SIZE, y + m_y - ORB_PATCH_SIZE);
            let val = (*img.val(s_x, s_y))?.clamp(0.0, 255.0) as usize;
            m_00 += val;
            m_10 += s_x * val;
            m_01 += s_y * val;
        }
    }

    let centroid_x = m_10 as f64 / m_00 as f64;
    let centroid_y = m_01 as f64 / m_00 as f64;
    let angle = (centroid_y - y as f64).atan2(centroid_x - x as f64);

    Some(angle)
}

fn extract_brief_descriptors<PL: ProgressListener>(
    img: &Grid<u8>,
    points: Vec<Point>,
    progress_listener: Option<&PL>,
) -> Vec<Keypoint> {
    let kernel_gauss = gaussian_kernel::<ORB_GAUSS_KERNEL_WIDTH>();

    let img = gaussian_blur(img, &kernel_gauss);

    let counter = AtomicUsize::new(0);
    points
        .par_iter()
        .filter_map(|coords| {
            if let Some(pl) = progress_listener {
                let value = 0.70
                    + 0.30 * (counter.fetch_add(1, Ordering::Relaxed) as f32 / points.len() as f32);
                pl.report_status(value);
            }
            let angle = get_brief_orientation(&img, coords)?;
            let angle_sin = angle.sin();
            let angle_cos = angle.cos();
            let mut orb_descriptor = [0_u32; 8];
            for i in 0..ORB_MATCH_PATTERN.len() {
                let (offset1_x, offset1_y) = ORB_MATCH_PATTERN[i].0;
                let (offset2_x, offset2_y) = ORB_MATCH_PATTERN[i].1;
                let offset1 = Point2D::new(
                    (offset1_y as f64 * angle_cos - offset1_x as f64 * angle_sin).round() as isize,
                    (offset1_y as f64 * angle_sin + offset1_x as f64 * angle_cos).round() as isize,
                );
                let offset2 = Point2D::new(
                    (offset2_y as f64 * angle_cos - offset2_x as f64 * angle_sin).round() as isize,
                    (offset2_y as f64 * angle_sin + offset2_x as f64 * angle_cos).round() as isize,
                );
                let p1_coords = Point2D::new(
                    coords.x.saturating_add_signed(offset1.x),
                    coords.y.saturating_add_signed(offset1.y),
                );
                let p2_coords = Point2D::new(
                    coords.x.saturating_add_signed(offset2.x),
                    coords.y.saturating_add_signed(offset2.y),
                );
                if p1_coords.x == 0
                    || p2_coords.x == 0
                    || p1_coords.x + 1 >= img.width()
                    || p2_coords.x + 1 >= img.width()
                    || p1_coords.y + 1 >= img.height()
                    || p2_coords.y + 1 >= img.height()
                {
                    return None;
                }
                let p1 = (*img.val(p1_coords.x, p1_coords.y))?;
                let p2 = (*img.val(p2_coords.x, p2_coords.y))?;
                let dst_block = &mut orb_descriptor[i / 32];
                let tau = if p1 < p2 { 1 } else { 0 };
                *dst_block |= tau << (i % 32);
            }
            Some((*coords, orb_descriptor))
        })
        .collect()
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

#[inline]
fn get_pixel_offset(img: &Grid<u8>, x: usize, y: usize, offset: Offset) -> i16 {
    let x_new = x.saturating_add_signed(offset.x as isize);
    let y_new = y.saturating_add_signed(offset.y as isize);
    *img.val(x_new, y_new) as i16
}

#[inline]
fn is_keypoint(img: &Grid<u8>, threshold: i16, x: usize, y: usize) -> bool {
    let val: i16 = *img.val(x, y) as i16;
    let mut last_more_pos: Option<usize> = None;
    let mut last_less_pos: Option<usize> = None;
    let mut max_length = 0;

    for i in 0..FAST_CIRCLE_LENGTH {
        let p = FAST_CIRCLE_PIXELS[i % FAST_CIRCLE_PIXELS.len()];
        let c_val = get_pixel_offset(img, x, y, p);
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

fn adjust_contrast(img: &mut Grid<u8>) {
    let mut min: u8 = u8::MAX;
    let mut max: u8 = u8::MIN;

    for (_x, _y, p) in img.iter() {
        min = min.min(*p);
        max = max.max(*p);
    }

    if min >= max {
        return;
    }

    let coeff = u8::MAX as f32 / ((max - min) as f32);
    for (_x, _y, p) in img.iter_mut() {
        *p = (coeff * ((*p - min) as f32)).round() as u8;
    }
}

// Extracted from OpenCV orb.cpp bit_pattern_31_
const ORB_MATCH_PATTERN: [((i8, i8), (i8, i8)); 256] = [
    ((8, -3), (9, 5)),
    ((4, 2), (7, -12)),
    ((-11, 9), (-8, 2)),
    ((7, -12), (12, -13)),
    ((2, -13), (2, 12)),
    ((1, -7), (1, 6)),
    ((-2, -10), (-2, -4)),
    ((-13, -13), (-11, -8)),
    ((-13, -3), (-12, -9)),
    ((10, 4), (11, 9)),
    ((-13, -8), (-8, -9)),
    ((-11, 7), (-9, 12)),
    ((7, 7), (12, 6)),
    ((-4, -5), (-3, 0)),
    ((-13, 2), (-12, -3)),
    ((-9, 0), (-7, 5)),
    ((12, -6), (12, -1)),
    ((-3, 6), (-2, 12)),
    ((-6, -13), (-4, -8)),
    ((11, -13), (12, -8)),
    ((4, 7), (5, 1)),
    ((5, -3), (10, -3)),
    ((3, -7), (6, 12)),
    ((-8, -7), (-6, -2)),
    ((-2, 11), (-1, -10)),
    ((-13, 12), (-8, 10)),
    ((-7, 3), (-5, -3)),
    ((-4, 2), (-3, 7)),
    ((-10, -12), (-6, 11)),
    ((5, -12), (6, -7)),
    ((5, -6), (7, -1)),
    ((1, 0), (4, -5)),
    ((9, 11), (11, -13)),
    ((4, 7), (4, 12)),
    ((2, -1), (4, 4)),
    ((-4, -12), (-2, 7)),
    ((-8, -5), (-7, -10)),
    ((4, 11), (9, 12)),
    ((0, -8), (1, -13)),
    ((-13, -2), (-8, 2)),
    ((-3, -2), (-2, 3)),
    ((-6, 9), (-4, -9)),
    ((8, 12), (10, 7)),
    ((0, 9), (1, 3)),
    ((7, -5), (11, -10)),
    ((-13, -6), (-11, 0)),
    ((10, 7), (12, 1)),
    ((-6, -3), (-6, 12)),
    ((10, -9), (12, -4)),
    ((-13, 8), (-8, -12)),
    ((-13, 0), (-8, -4)),
    ((3, 3), (7, 8)),
    ((5, 7), (10, -7)),
    ((-1, 7), (1, -12)),
    ((3, -10), (5, 6)),
    ((2, -4), (3, -10)),
    ((-13, 0), (-13, 5)),
    ((-13, -7), (-12, 12)),
    ((-13, 3), (-11, 8)),
    ((-7, 12), (-4, 7)),
    ((6, -10), (12, 8)),
    ((-9, -1), (-7, -6)),
    ((-2, -5), (0, 12)),
    ((-12, 5), (-7, 5)),
    ((3, -10), (8, -13)),
    ((-7, -7), (-4, 5)),
    ((-3, -2), (-1, -7)),
    ((2, 9), (5, -11)),
    ((-11, -13), (-5, -13)),
    ((-1, 6), (0, -1)),
    ((5, -3), (5, 2)),
    ((-4, -13), (-4, 12)),
    ((-9, -6), (-9, 6)),
    ((-12, -10), (-8, -4)),
    ((10, 2), (12, -3)),
    ((7, 12), (12, 12)),
    ((-7, -13), (-6, 5)),
    ((-4, 9), (-3, 4)),
    ((7, -1), (12, 2)),
    ((-7, 6), (-5, 1)),
    ((-13, 11), (-12, 5)),
    ((-3, 7), (-2, -6)),
    ((7, -8), (12, -7)),
    ((-13, -7), (-11, -12)),
    ((1, -3), (12, 12)),
    ((2, -6), (3, 0)),
    ((-4, 3), (-2, -13)),
    ((-1, -13), (1, 9)),
    ((7, 1), (8, -6)),
    ((1, -1), (3, 12)),
    ((9, 1), (12, 6)),
    ((-1, -9), (-1, 3)),
    ((-13, -13), (-10, 5)),
    ((7, 7), (10, 12)),
    ((12, -5), (12, 9)),
    ((6, 3), (7, 11)),
    ((5, -13), (6, 10)),
    ((2, -12), (2, 3)),
    ((3, 8), (4, -6)),
    ((2, 6), (12, -13)),
    ((9, -12), (10, 3)),
    ((-8, 4), (-7, 9)),
    ((-11, 12), (-4, -6)),
    ((1, 12), (2, -8)),
    ((6, -9), (7, -4)),
    ((2, 3), (3, -2)),
    ((6, 3), (11, 0)),
    ((3, -3), (8, -8)),
    ((7, 8), (9, 3)),
    ((-11, -5), (-6, -4)),
    ((-10, 11), (-5, 10)),
    ((-5, -8), (-3, 12)),
    ((-10, 5), (-9, 0)),
    ((8, -1), (12, -6)),
    ((4, -6), (6, -11)),
    ((-10, 12), (-8, 7)),
    ((4, -2), (6, 7)),
    ((-2, 0), (-2, 12)),
    ((-5, -8), (-5, 2)),
    ((7, -6), (10, 12)),
    ((-9, -13), (-8, -8)),
    ((-5, -13), (-5, -2)),
    ((8, -8), (9, -13)),
    ((-9, -11), (-9, 0)),
    ((1, -8), (1, -2)),
    ((7, -4), (9, 1)),
    ((-2, 1), (-1, -4)),
    ((11, -6), (12, -11)),
    ((-12, -9), (-6, 4)),
    ((3, 7), (7, 12)),
    ((5, 5), (10, 8)),
    ((0, -4), (2, 8)),
    ((-9, 12), (-5, -13)),
    ((0, 7), (2, 12)),
    ((-1, 2), (1, 7)),
    ((5, 11), (7, -9)),
    ((3, 5), (6, -8)),
    ((-13, -4), (-8, 9)),
    ((-5, 9), (-3, -3)),
    ((-4, -7), (-3, -12)),
    ((6, 5), (8, 0)),
    ((-7, 6), (-6, 12)),
    ((-13, 6), (-5, -2)),
    ((1, -10), (3, 10)),
    ((4, 1), (8, -4)),
    ((-2, -2), (2, -13)),
    ((2, -12), (12, 12)),
    ((-2, -13), (0, -6)),
    ((4, 1), (9, 3)),
    ((-6, -10), (-3, -5)),
    ((-3, -13), (-1, 1)),
    ((7, 5), (12, -11)),
    ((4, -2), (5, -7)),
    ((-13, 9), (-9, -5)),
    ((7, 1), (8, 6)),
    ((7, -8), (7, 6)),
    ((-7, -4), (-7, 1)),
    ((-8, 11), (-7, -8)),
    ((-13, 6), (-12, -8)),
    ((2, 4), (3, 9)),
    ((10, -5), (12, 3)),
    ((-6, -5), (-6, 7)),
    ((8, -3), (9, -8)),
    ((2, -12), (2, 8)),
    ((-11, -2), (-10, 3)),
    ((-12, -13), (-7, -9)),
    ((-11, 0), (-10, -5)),
    ((5, -3), (11, 8)),
    ((-2, -13), (-1, 12)),
    ((-1, -8), (0, 9)),
    ((-13, -11), (-12, -5)),
    ((-10, -2), (-10, 11)),
    ((-3, 9), (-2, -13)),
    ((2, -3), (3, 2)),
    ((-9, -13), (-4, 0)),
    ((-4, 6), (-3, -10)),
    ((-4, 12), (-2, -7)),
    ((-6, -11), (-4, 9)),
    ((6, -3), (6, 11)),
    ((-13, 11), (-5, 5)),
    ((11, 11), (12, 6)),
    ((7, -5), (12, -2)),
    ((-1, 12), (0, 7)),
    ((-4, -8), (-3, -2)),
    ((-7, 1), (-6, 7)),
    ((-13, -12), (-8, -13)),
    ((-7, -2), (-6, -8)),
    ((-8, 5), (-6, -9)),
    ((-5, -1), (-4, 5)),
    ((-13, 7), (-8, 10)),
    ((1, 5), (5, -13)),
    ((1, 0), (10, -13)),
    ((9, 12), (10, -1)),
    ((5, -8), (10, -9)),
    ((-1, 11), (1, -13)),
    ((-9, -3), (-6, 2)),
    ((-1, -10), (1, 12)),
    ((-13, 1), (-8, -10)),
    ((8, -11), (10, -6)),
    ((2, -13), (3, -6)),
    ((7, -13), (12, -9)),
    ((-10, -10), (-5, -7)),
    ((-10, -8), (-8, -13)),
    ((4, -6), (8, 5)),
    ((3, 12), (8, -13)),
    ((-4, 2), (-3, -3)),
    ((5, -13), (10, -12)),
    ((4, -13), (5, -1)),
    ((-9, 9), (-4, 3)),
    ((0, 3), (3, -9)),
    ((-12, 1), (-6, 1)),
    ((3, 2), (4, -8)),
    ((-10, -10), (-10, 9)),
    ((8, -13), (12, 12)),
    ((-8, -12), (-6, -5)),
    ((2, 2), (3, 7)),
    ((10, 6), (11, -8)),
    ((6, 8), (8, -12)),
    ((-7, 10), (-6, 5)),
    ((-3, -9), (-3, 9)),
    ((-1, -13), (-1, 5)),
    ((-3, -7), (-3, 4)),
    ((-8, -2), (-8, 3)),
    ((4, 2), (12, 12)),
    ((2, -5), (3, 11)),
    ((6, -9), (11, -13)),
    ((3, -1), (7, 12)),
    ((11, -1), (12, 4)),
    ((-3, 0), (-3, 6)),
    ((4, -11), (4, 12)),
    ((2, -4), (2, 1)),
    ((-10, -6), (-8, 1)),
    ((-13, 7), (-11, 1)),
    ((-13, 12), (-11, -13)),
    ((6, 0), (11, -13)),
    ((0, -1), (1, 4)),
    ((-13, 3), (-9, -2)),
    ((-9, 8), (-6, -3)),
    ((-13, -6), (-8, -2)),
    ((5, -9), (8, 10)),
    ((2, 7), (3, -9)),
    ((-1, -6), (-1, -1)),
    ((9, 5), (11, -2)),
    ((11, -3), (12, -8)),
    ((3, 0), (3, 5)),
    ((-1, 4), (0, 10)),
    ((3, -6), (4, 5)),
    ((-13, 0), (-10, 5)),
    ((5, 8), (12, 11)),
    ((8, 9), (9, -6)),
    ((7, -4), (8, -12)),
    ((-10, 4), (-10, 9)),
    ((7, 3), (12, 4)),
    ((9, -7), (10, -2)),
    ((7, 0), (12, -2)),
    ((-1, -6), (0, -11)),
];
