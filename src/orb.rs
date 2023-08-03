use std::f64::consts::PI;
use std::sync::atomic::{AtomicUsize, Ordering};

use nalgebra::{DMatrix, Matrix3, SMatrix, SVector};
use rayon::prelude::*;

type Point = ((usize, usize), [u32; 8]);

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

const FAST_KERNEL_SIZE: usize = 3;
const FAST_THRESHOLD: u8 = 15;
const KEYPOINT_SCALE_MIN_SIZE: usize = 256;
const FAST_NUM_POINTS: usize = 9;
const FAST_CIRCLE_LENGTH: usize = FAST_CIRCLE_PIXELS.len() + FAST_NUM_POINTS - 1;
const HARRIS_KERNEL_SIZE: usize = 3;
const HARRIS_KERNEL_WIDTH: usize = HARRIS_KERNEL_SIZE * 2 + 1;
const ORB_GAUSS_KERNEL_WIDTH: usize = 11;
const HARRIS_K: f64 = 0.04;
const MAX_KEYPOINTS: usize = 5_000;

pub trait ProgressListener
where
    Self: Sync + Sized,
{
    fn report_status(&self, pos: f32);
}

pub fn extract_points<PL: ProgressListener>(
    img: &DMatrix<u8>,
    progress_listener: Option<&PL>,
) -> Vec<Point> {
    let mut img_adjusted = img.clone();
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
                harris_response(&img, &harris_gaussian_kernel, *point)?,
            ))
        })
        .collect::<Vec<_>>();

    keypoints.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap().reverse());
    let keypoints = keypoints
        .iter()
        .take(MAX_KEYPOINTS)
        .map(|(keypoint, _)| *keypoint)
        .collect();

    extract_brief_descriptors(&img, keypoints, progress_listener)
}

fn find_fast_keypoints<PL: ProgressListener>(
    img: &DMatrix<u8>,
    progress_listener: Option<&PL>,
) -> Vec<(usize, usize)> {
    // Detect points
    let total_rows = img.nrows() - FAST_KERNEL_SIZE * 2;
    let counter = AtomicUsize::new(0);
    let keypoints: Vec<(usize, usize)> = (FAST_KERNEL_SIZE..(img.nrows() - FAST_KERNEL_SIZE))
        .into_par_iter()
        .map(|row| {
            if let Some(pl) = progress_listener {
                let value =
                    0.20 * (counter.fetch_add(1, Ordering::Relaxed) as f32 / total_rows as f32);
                pl.report_status(value);
            }
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
                if p2.0 == p1.0 - 1 && p2.1 >= p1.1 - 1 && p2.1 <= p1.1 + 1 && scores[j] >= score1 {
                    return None;
                }
            }
            // Search for point below current
            for j in i + 1..keypoints.len() {
                let p2 = &keypoints[j];
                if p2.0 > p1.0 + 1 {
                    break;
                }
                if p2.0 == p1.0 + 1 && p2.1 >= p1.1 - 1 && p2.1 <= p1.1 + 1 && scores[j] >= score1 {
                    return None;
                }
            }
            Some((p1.0, p1.1))
        })
        .collect()
}

fn gaussian_kernel<const KERNEL_WIDTH: usize>() -> SVector<f64, KERNEL_WIDTH> {
    let sigma = (KERNEL_WIDTH - 1) as f64 / 6.0;
    let sigma_2 = sigma.powi(2);
    let divider = (2.0 * PI).sqrt() * sigma;
    let center = (KERNEL_WIDTH / 2) as f64;
    let mut kernel = SVector::<f64, KERNEL_WIDTH>::zeros();

    for i in 0..KERNEL_WIDTH {
        kernel[i] = (-(i as f64 - center).powi(2) / (2.0 * sigma_2)).exp() / divider;
    }

    kernel
}

fn convolve_kernel<const KERNEL_WIDTH: usize>(
    img: &DMatrix<u8>,
    point: (usize, usize),
    kernel: SMatrix<f64, KERNEL_WIDTH, KERNEL_WIDTH>,
) -> Option<f64> {
    let kernel_size = KERNEL_WIDTH / 2;
    let (row, col) = point;
    if col < kernel_size
        || row < kernel_size
        || col + kernel_size >= img.ncols()
        || row + kernel_size >= img.nrows()
    {
        return None;
    }

    let mut result = 0.0;
    for k_row in 0..KERNEL_WIDTH {
        for k_col in 0..KERNEL_WIDTH {
            result += kernel[(k_row, k_col)]
                * img[(row + k_row - kernel_size, col + k_col - kernel_size)] as f64
                / 255.0;
        }
    }

    Some(result)
}

fn harris_response<const KERNEL_WIDTH: usize>(
    img: &DMatrix<u8>,
    kernel_gauss: &SVector<f64, KERNEL_WIDTH>,
    point: (usize, usize),
) -> Option<f64> {
    const KERNEL_SOBEL_X: Matrix3<f64> =
        Matrix3::new(-1.0, -2.0, -1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0);
    const KERNEL_SOBEL_Y: Matrix3<f64> =
        Matrix3::new(-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0);

    let kernel_size = KERNEL_WIDTH / 2;
    let (row, col) = point;
    if col < kernel_size
        || row < kernel_size
        || col + kernel_size >= img.ncols()
        || row + kernel_size >= img.nrows()
    {
        return None;
    }

    let mut g_dx2 = 0.0;
    let mut g_dy2 = 0.0;
    let mut g_dx_dy = 0.0;
    for k_row in 0..KERNEL_WIDTH {
        for k_col in 0..KERNEL_WIDTH {
            let dx = convolve_kernel(
                img,
                (
                    row + k_row - HARRIS_KERNEL_SIZE,
                    col + k_col - HARRIS_KERNEL_SIZE,
                ),
                KERNEL_SOBEL_X,
            )?;
            let dy = convolve_kernel(
                img,
                (
                    row + k_row - HARRIS_KERNEL_SIZE,
                    col + k_col - HARRIS_KERNEL_SIZE,
                ),
                KERNEL_SOBEL_Y,
            )?;

            let gauss_mul = kernel_gauss[k_row] * kernel_gauss[k_col];
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
    img: &DMatrix<u8>,
    kernel_gauss: &SVector<f64, KERNEL_WIDTH>,
) -> DMatrix<Option<f64>> {
    let mut result = DMatrix::from_element(img.nrows(), img.ncols(), None);
    let (nrows, ncols) = result.shape();
    let kernel_size = KERNEL_WIDTH / 2;

    result
        .column_iter_mut()
        .enumerate()
        .par_bridge()
        .for_each(|(col, mut out_col)| {
            if col < kernel_size || col + kernel_size >= ncols {
                return;
            }
            out_col.iter_mut().enumerate().for_each(|(row, out_point)| {
                if row < kernel_size || row + kernel_size >= nrows {
                    return;
                }
                let mut sum = 0.0;
                for i in 0..KERNEL_WIDTH {
                    sum += kernel_gauss[i] * img[(row + i - kernel_size, col)] as f64;
                }
                *out_point = Some(sum);
            })
        });

    let img = result;
    let mut result = DMatrix::from_element(nrows, ncols, None);
    result
        .column_iter_mut()
        .enumerate()
        .par_bridge()
        .for_each(|(col, mut out_col)| {
            if col < kernel_size || col + kernel_size >= ncols {
                return;
            }
            out_col.iter_mut().enumerate().for_each(|(row, out_point)| {
                if row < kernel_size || row + kernel_size >= nrows {
                    return;
                }
                let mut sum = 0.0;
                for i in 0..KERNEL_WIDTH {
                    let val = if let Some(val) = img[(row, col + i - kernel_size)] {
                        val
                    } else {
                        return;
                    };
                    sum += kernel_gauss[i] * val;
                }
                *out_point = Some(sum);
            })
        });

    result
}

fn extract_brief_descriptors<PL: ProgressListener>(
    img: &DMatrix<u8>,
    points: Vec<(usize, usize)>,
    progress_listener: Option<&PL>,
) -> Vec<Point> {
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
            let mut orb_descriptor = [0 as u32; 8];
            for i in 0..ORB_MATCH_PATTERN.len() {
                let offset1 = ORB_MATCH_PATTERN[i].0;
                let offset2 = ORB_MATCH_PATTERN[i].1;
                let p1_coords = (
                    coords.0.saturating_add_signed(offset1.1 as isize),
                    coords.1.saturating_add_signed(offset1.0 as isize),
                );
                let p2_coords = (
                    coords.0.saturating_add_signed(offset2.1 as isize),
                    coords.1.saturating_add_signed(offset2.0 as isize),
                );
                if p1_coords.0 == 0
                    || p2_coords.0 == 0
                    || p1_coords.0 + 1 >= img.nrows()
                    || p2_coords.0 + 1 >= img.nrows()
                    || p1_coords.1 + 1 >= img.ncols()
                    || p2_coords.1 + 1 >= img.ncols()
                {
                    return None;
                }
                let p1 = img[p1_coords]?;
                let p2 = img[p2_coords]?;
                let dst_block = &mut orb_descriptor[i / 32];
                let tau = if p1 < p2 { 1 } else { 0 };
                *dst_block = *dst_block | (tau << (i % 32));
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

// Extracted from OpenCV orb.cpp bit_pattern_31_
const ORB_MATCH_PATTERN: [((i8, i8), (i8, i8)); 256] = [
    ((8, -3), (9, 5)),       /*mean (0), correlation (0)*/
    ((4, 2), (7, -12)),      /*mean (1.12461e-05), correlation (0.0437584)*/
    ((-11, 9), (-8, 2)),     /*mean (3.37382e-05), correlation (0.0617409)*/
    ((7, -12), (12, -13)),   /*mean (5.62303e-05), correlation (0.0636977)*/
    ((2, -13), (2, 12)),     /*mean (0.000134953), correlation (0.085099)*/
    ((1, -7), (1, 6)),       /*mean (0.000528565), correlation (0.0857175)*/
    ((-2, -10), (-2, -4)),   /*mean (0.0188821), correlation (0.0985774)*/
    ((-13, -13), (-11, -8)), /*mean (0.0363135), correlation (0.0899616)*/
    ((-13, -3), (-12, -9)),  /*mean (0.121806), correlation (0.099849)*/
    ((10, 4), (11, 9)),      /*mean (0.122065), correlation (0.093285)*/
    ((-13, -8), (-8, -9)),   /*mean (0.162787), correlation (0.0942748)*/
    ((-11, 7), (-9, 12)),    /*mean (0.21561), correlation (0.0974438)*/
    ((7, 7), (12, 6)),       /*mean (0.160583), correlation (0.130064)*/
    ((-4, -5), (-3, 0)),     /*mean (0.228171), correlation (0.132998)*/
    ((-13, 2), (-12, -3)),   /*mean (0.00997526), correlation (0.145926)*/
    ((-9, 0), (-7, 5)),      /*mean (0.198234), correlation (0.143636)*/
    ((12, -6), (12, -1)),    /*mean (0.0676226), correlation (0.16689)*/
    ((-3, 6), (-2, 12)),     /*mean (0.166847), correlation (0.171682)*/
    ((-6, -13), (-4, -8)),   /*mean (0.101215), correlation (0.179716)*/
    ((11, -13), (12, -8)),   /*mean (0.200641), correlation (0.192279)*/
    ((4, 7), (5, 1)),        /*mean (0.205106), correlation (0.186848)*/
    ((5, -3), (10, -3)),     /*mean (0.234908), correlation (0.192319)*/
    ((3, -7), (6, 12)),      /*mean (0.0709964), correlation (0.210872)*/
    ((-8, -7), (-6, -2)),    /*mean (0.0939834), correlation (0.212589)*/
    ((-2, 11), (-1, -10)),   /*mean (0.127778), correlation (0.20866)*/
    ((-13, 12), (-8, 10)),   /*mean (0.14783), correlation (0.206356)*/
    ((-7, 3), (-5, -3)),     /*mean (0.182141), correlation (0.198942)*/
    ((-4, 2), (-3, 7)),      /*mean (0.188237), correlation (0.21384)*/
    ((-10, -12), (-6, 11)),  /*mean (0.14865), correlation (0.23571)*/
    ((5, -12), (6, -7)),     /*mean (0.222312), correlation (0.23324)*/
    ((5, -6), (7, -1)),      /*mean (0.229082), correlation (0.23389)*/
    ((1, 0), (4, -5)),       /*mean (0.241577), correlation (0.215286)*/
    ((9, 11), (11, -13)),    /*mean (0.00338507), correlation (0.251373)*/
    ((4, 7), (4, 12)),       /*mean (0.131005), correlation (0.257622)*/
    ((2, -1), (4, 4)),       /*mean (0.152755), correlation (0.255205)*/
    ((-4, -12), (-2, 7)),    /*mean (0.182771), correlation (0.244867)*/
    ((-8, -5), (-7, -10)),   /*mean (0.186898), correlation (0.23901)*/
    ((4, 11), (9, 12)),      /*mean (0.226226), correlation (0.258255)*/
    ((0, -8), (1, -13)),     /*mean (0.0897886), correlation (0.274827)*/
    ((-13, -2), (-8, 2)),    /*mean (0.148774), correlation (0.28065)*/
    ((-3, -2), (-2, 3)),     /*mean (0.153048), correlation (0.283063)*/
    ((-6, 9), (-4, -9)),     /*mean (0.169523), correlation (0.278248)*/
    ((8, 12), (10, 7)),      /*mean (0.225337), correlation (0.282851)*/
    ((0, 9), (1, 3)),        /*mean (0.226687), correlation (0.278734)*/
    ((7, -5), (11, -10)),    /*mean (0.00693882), correlation (0.305161)*/
    ((-13, -6), (-11, 0)),   /*mean (0.0227283), correlation (0.300181)*/
    ((10, 7), (12, 1)),      /*mean (0.125517), correlation (0.31089)*/
    ((-6, -3), (-6, 12)),    /*mean (0.131748), correlation (0.312779)*/
    ((10, -9), (12, -4)),    /*mean (0.144827), correlation (0.292797)*/
    ((-13, 8), (-8, -12)),   /*mean (0.149202), correlation (0.308918)*/
    ((-13, 0), (-8, -4)),    /*mean (0.160909), correlation (0.310013)*/
    ((3, 3), (7, 8)),        /*mean (0.177755), correlation (0.309394)*/
    ((5, 7), (10, -7)),      /*mean (0.212337), correlation (0.310315)*/
    ((-1, 7), (1, -12)),     /*mean (0.214429), correlation (0.311933)*/
    ((3, -10), (5, 6)),      /*mean (0.235807), correlation (0.313104)*/
    ((2, -4), (3, -10)),     /*mean (0.00494827), correlation (0.344948)*/
    ((-13, 0), (-13, 5)),    /*mean (0.0549145), correlation (0.344675)*/
    ((-13, -7), (-12, 12)),  /*mean (0.103385), correlation (0.342715)*/
    ((-13, 3), (-11, 8)),    /*mean (0.134222), correlation (0.322922)*/
    ((-7, 12), (-4, 7)),     /*mean (0.153284), correlation (0.337061)*/
    ((6, -10), (12, 8)),     /*mean (0.154881), correlation (0.329257)*/
    ((-9, -1), (-7, -6)),    /*mean (0.200967), correlation (0.33312)*/
    ((-2, -5), (0, 12)),     /*mean (0.201518), correlation (0.340635)*/
    ((-12, 5), (-7, 5)),     /*mean (0.207805), correlation (0.335631)*/
    ((3, -10), (8, -13)),    /*mean (0.224438), correlation (0.34504)*/
    ((-7, -7), (-4, 5)),     /*mean (0.239361), correlation (0.338053)*/
    ((-3, -2), (-1, -7)),    /*mean (0.240744), correlation (0.344322)*/
    ((2, 9), (5, -11)),      /*mean (0.242949), correlation (0.34145)*/
    ((-11, -13), (-5, -13)), /*mean (0.244028), correlation (0.336861)*/
    ((-1, 6), (0, -1)),      /*mean (0.247571), correlation (0.343684)*/
    ((5, -3), (5, 2)),       /*mean (0.000697256), correlation (0.357265)*/
    ((-4, -13), (-4, 12)),   /*mean (0.00213675), correlation (0.373827)*/
    ((-9, -6), (-9, 6)),     /*mean (0.0126856), correlation (0.373938)*/
    ((-12, -10), (-8, -4)),  /*mean (0.0152497), correlation (0.364237)*/
    ((10, 2), (12, -3)),     /*mean (0.0299933), correlation (0.345292)*/
    ((7, 12), (12, 12)),     /*mean (0.0307242), correlation (0.366299)*/
    ((-7, -13), (-6, 5)),    /*mean (0.0534975), correlation (0.368357)*/
    ((-4, 9), (-3, 4)),      /*mean (0.099865), correlation (0.372276)*/
    ((7, -1), (12, 2)),      /*mean (0.117083), correlation (0.364529)*/
    ((-7, 6), (-5, 1)),      /*mean (0.126125), correlation (0.369606)*/
    ((-13, 11), (-12, 5)),   /*mean (0.130364), correlation (0.358502)*/
    ((-3, 7), (-2, -6)),     /*mean (0.131691), correlation (0.375531)*/
    ((7, -8), (12, -7)),     /*mean (0.160166), correlation (0.379508)*/
    ((-13, -7), (-11, -12)), /*mean (0.167848), correlation (0.353343)*/
    ((1, -3), (12, 12)),     /*mean (0.183378), correlation (0.371916)*/
    ((2, -6), (3, 0)),       /*mean (0.228711), correlation (0.371761)*/
    ((-4, 3), (-2, -13)),    /*mean (0.247211), correlation (0.364063)*/
    ((-1, -13), (1, 9)),     /*mean (0.249325), correlation (0.378139)*/
    ((7, 1), (8, -6)),       /*mean (0.000652272), correlation (0.411682)*/
    ((1, -1), (3, 12)),      /*mean (0.00248538), correlation (0.392988)*/
    ((9, 1), (12, 6)),       /*mean (0.0206815), correlation (0.386106)*/
    ((-1, -9), (-1, 3)),     /*mean (0.0364485), correlation (0.410752)*/
    ((-13, -13), (-10, 5)),  /*mean (0.0376068), correlation (0.398374)*/
    ((7, 7), (10, 12)),      /*mean (0.0424202), correlation (0.405663)*/
    ((12, -5), (12, 9)),     /*mean (0.0942645), correlation (0.410422)*/
    ((6, 3), (7, 11)),       /*mean (0.1074), correlation (0.413224)*/
    ((5, -13), (6, 10)),     /*mean (0.109256), correlation (0.408646)*/
    ((2, -12), (2, 3)),      /*mean (0.131691), correlation (0.416076)*/
    ((3, 8), (4, -6)),       /*mean (0.165081), correlation (0.417569)*/
    ((2, 6), (12, -13)),     /*mean (0.171874), correlation (0.408471)*/
    ((9, -12), (10, 3)),     /*mean (0.175146), correlation (0.41296)*/
    ((-8, 4), (-7, 9)),      /*mean (0.183682), correlation (0.402956)*/
    ((-11, 12), (-4, -6)),   /*mean (0.184672), correlation (0.416125)*/
    ((1, 12), (2, -8)),      /*mean (0.191487), correlation (0.386696)*/
    ((6, -9), (7, -4)),      /*mean (0.192668), correlation (0.394771)*/
    ((2, 3), (3, -2)),       /*mean (0.200157), correlation (0.408303)*/
    ((6, 3), (11, 0)),       /*mean (0.204588), correlation (0.411762)*/
    ((3, -3), (8, -8)),      /*mean (0.205904), correlation (0.416294)*/
    ((7, 8), (9, 3)),        /*mean (0.213237), correlation (0.409306)*/
    ((-11, -5), (-6, -4)),   /*mean (0.243444), correlation (0.395069)*/
    ((-10, 11), (-5, 10)),   /*mean (0.247672), correlation (0.413392)*/
    ((-5, -8), (-3, 12)),    /*mean (0.24774), correlation (0.411416)*/
    ((-10, 5), (-9, 0)),     /*mean (0.00213675), correlation (0.454003)*/
    ((8, -1), (12, -6)),     /*mean (0.0293635), correlation (0.455368)*/
    ((4, -6), (6, -11)),     /*mean (0.0404971), correlation (0.457393)*/
    ((-10, 12), (-8, 7)),    /*mean (0.0481107), correlation (0.448364)*/
    ((4, -2), (6, 7)),       /*mean (0.050641), correlation (0.455019)*/
    ((-2, 0), (-2, 12)),     /*mean (0.0525978), correlation (0.44338)*/
    ((-5, -8), (-5, 2)),     /*mean (0.0629667), correlation (0.457096)*/
    ((7, -6), (10, 12)),     /*mean (0.0653846), correlation (0.445623)*/
    ((-9, -13), (-8, -8)),   /*mean (0.0858749), correlation (0.449789)*/
    ((-5, -13), (-5, -2)),   /*mean (0.122402), correlation (0.450201)*/
    ((8, -8), (9, -13)),     /*mean (0.125416), correlation (0.453224)*/
    ((-9, -11), (-9, 0)),    /*mean (0.130128), correlation (0.458724)*/
    ((1, -8), (1, -2)),      /*mean (0.132467), correlation (0.440133)*/
    ((7, -4), (9, 1)),       /*mean (0.132692), correlation (0.454)*/
    ((-2, 1), (-1, -4)),     /*mean (0.135695), correlation (0.455739)*/
    ((11, -6), (12, -11)),   /*mean (0.142904), correlation (0.446114)*/
    ((-12, -9), (-6, 4)),    /*mean (0.146165), correlation (0.451473)*/
    ((3, 7), (7, 12)),       /*mean (0.147627), correlation (0.456643)*/
    ((5, 5), (10, 8)),       /*mean (0.152901), correlation (0.455036)*/
    ((0, -4), (2, 8)),       /*mean (0.167083), correlation (0.459315)*/
    ((-9, 12), (-5, -13)),   /*mean (0.173234), correlation (0.454706)*/
    ((0, 7), (2, 12)),       /*mean (0.18312), correlation (0.433855)*/
    ((-1, 2), (1, 7)),       /*mean (0.185504), correlation (0.443838)*/
    ((5, 11), (7, -9)),      /*mean (0.185706), correlation (0.451123)*/
    ((3, 5), (6, -8)),       /*mean (0.188968), correlation (0.455808)*/
    ((-13, -4), (-8, 9)),    /*mean (0.191667), correlation (0.459128)*/
    ((-5, 9), (-3, -3)),     /*mean (0.193196), correlation (0.458364)*/
    ((-4, -7), (-3, -12)),   /*mean (0.196536), correlation (0.455782)*/
    ((6, 5), (8, 0)),        /*mean (0.1972), correlation (0.450481)*/
    ((-7, 6), (-6, 12)),     /*mean (0.199438), correlation (0.458156)*/
    ((-13, 6), (-5, -2)),    /*mean (0.211224), correlation (0.449548)*/
    ((1, -10), (3, 10)),     /*mean (0.211718), correlation (0.440606)*/
    ((4, 1), (8, -4)),       /*mean (0.213034), correlation (0.443177)*/
    ((-2, -2), (2, -13)),    /*mean (0.234334), correlation (0.455304)*/
    ((2, -12), (12, 12)),    /*mean (0.235684), correlation (0.443436)*/
    ((-2, -13), (0, -6)),    /*mean (0.237674), correlation (0.452525)*/
    ((4, 1), (9, 3)),        /*mean (0.23962), correlation (0.444824)*/
    ((-6, -10), (-3, -5)),   /*mean (0.248459), correlation (0.439621)*/
    ((-3, -13), (-1, 1)),    /*mean (0.249505), correlation (0.456666)*/
    ((7, 5), (12, -11)),     /*mean (0.00119208), correlation (0.495466)*/
    ((4, -2), (5, -7)),      /*mean (0.00372245), correlation (0.484214)*/
    ((-13, 9), (-9, -5)),    /*mean (0.00741116), correlation (0.499854)*/
    ((7, 1), (8, 6)),        /*mean (0.0208952), correlation (0.499773)*/
    ((7, -8), (7, 6)),       /*mean (0.0220085), correlation (0.501609)*/
    ((-7, -4), (-7, 1)),     /*mean (0.0233806), correlation (0.496568)*/
    ((-8, 11), (-7, -8)),    /*mean (0.0236505), correlation (0.489719)*/
    ((-13, 6), (-12, -8)),   /*mean (0.0268781), correlation (0.503487)*/
    ((2, 4), (3, 9)),        /*mean (0.0323324), correlation (0.501938)*/
    ((10, -5), (12, 3)),     /*mean (0.0399235), correlation (0.494029)*/
    ((-6, -5), (-6, 7)),     /*mean (0.0420153), correlation (0.486579)*/
    ((8, -3), (9, -8)),      /*mean (0.0548021), correlation (0.484237)*/
    ((2, -12), (2, 8)),      /*mean (0.0616622), correlation (0.496642)*/
    ((-11, -2), (-10, 3)),   /*mean (0.0627755), correlation (0.498563)*/
    ((-12, -13), (-7, -9)),  /*mean (0.0829622), correlation (0.495491)*/
    ((-11, 0), (-10, -5)),   /*mean (0.0843342), correlation (0.487146)*/
    ((5, -3), (11, 8)),      /*mean (0.0929937), correlation (0.502315)*/
    ((-2, -13), (-1, 12)),   /*mean (0.113327), correlation (0.48941)*/
    ((-1, -8), (0, 9)),      /*mean (0.132119), correlation (0.467268)*/
    ((-13, -11), (-12, -5)), /*mean (0.136269), correlation (0.498771)*/
    ((-10, -2), (-10, 11)),  /*mean (0.142173), correlation (0.498714)*/
    ((-3, 9), (-2, -13)),    /*mean (0.144141), correlation (0.491973)*/
    ((2, -3), (3, 2)),       /*mean (0.14892), correlation (0.500782)*/
    ((-9, -13), (-4, 0)),    /*mean (0.150371), correlation (0.498211)*/
    ((-4, 6), (-3, -10)),    /*mean (0.152159), correlation (0.495547)*/
    ((-4, 12), (-2, -7)),    /*mean (0.156152), correlation (0.496925)*/
    ((-6, -11), (-4, 9)),    /*mean (0.15749), correlation (0.499222)*/
    ((6, -3), (6, 11)),      /*mean (0.159211), correlation (0.503821)*/
    ((-13, 11), (-5, 5)),    /*mean (0.162427), correlation (0.501907)*/
    ((11, 11), (12, 6)),     /*mean (0.16652), correlation (0.497632)*/
    ((7, -5), (12, -2)),     /*mean (0.169141), correlation (0.484474)*/
    ((-1, 12), (0, 7)),      /*mean (0.169456), correlation (0.495339)*/
    ((-4, -8), (-3, -2)),    /*mean (0.171457), correlation (0.487251)*/
    ((-7, 1), (-6, 7)),      /*mean (0.175), correlation (0.500024)*/
    ((-13, -12), (-8, -13)), /*mean (0.175866), correlation (0.497523)*/
    ((-7, -2), (-6, -8)),    /*mean (0.178273), correlation (0.501854)*/
    ((-8, 5), (-6, -9)),     /*mean (0.181107), correlation (0.494888)*/
    ((-5, -1), (-4, 5)),     /*mean (0.190227), correlation (0.482557)*/
    ((-13, 7), (-8, 10)),    /*mean (0.196739), correlation (0.496503)*/
    ((1, 5), (5, -13)),      /*mean (0.19973), correlation (0.499759)*/
    ((1, 0), (10, -13)),     /*mean (0.204465), correlation (0.49873)*/
    ((9, 12), (10, -1)),     /*mean (0.209334), correlation (0.49063)*/
    ((5, -8), (10, -9)),     /*mean (0.211134), correlation (0.503011)*/
    ((-1, 11), (1, -13)),    /*mean (0.212), correlation (0.499414)*/
    ((-9, -3), (-6, 2)),     /*mean (0.212168), correlation (0.480739)*/
    ((-1, -10), (1, 12)),    /*mean (0.212731), correlation (0.502523)*/
    ((-13, 1), (-8, -10)),   /*mean (0.21327), correlation (0.489786)*/
    ((8, -11), (10, -6)),    /*mean (0.214159), correlation (0.488246)*/
    ((2, -13), (3, -6)),     /*mean (0.216993), correlation (0.50287)*/
    ((7, -13), (12, -9)),    /*mean (0.223639), correlation (0.470502)*/
    ((-10, -10), (-5, -7)),  /*mean (0.224089), correlation (0.500852)*/
    ((-10, -8), (-8, -13)),  /*mean (0.228666), correlation (0.502629)*/
    ((4, -6), (8, 5)),       /*mean (0.22906), correlation (0.498305)*/
    ((3, 12), (8, -13)),     /*mean (0.233378), correlation (0.503825)*/
    ((-4, 2), (-3, -3)),     /*mean (0.234323), correlation (0.476692)*/
    ((5, -13), (10, -12)),   /*mean (0.236392), correlation (0.475462)*/
    ((4, -13), (5, -1)),     /*mean (0.236842), correlation (0.504132)*/
    ((-9, 9), (-4, 3)),      /*mean (0.236977), correlation (0.497739)*/
    ((0, 3), (3, -9)),       /*mean (0.24314), correlation (0.499398)*/
    ((-12, 1), (-6, 1)),     /*mean (0.243297), correlation (0.489447)*/
    ((3, 2), (4, -8)),       /*mean (0.00155196), correlation (0.553496)*/
    ((-10, -10), (-10, 9)),  /*mean (0.00239541), correlation (0.54297)*/
    ((8, -13), (12, 12)),    /*mean (0.0034413), correlation (0.544361)*/
    ((-8, -12), (-6, -5)),   /*mean (0.003565), correlation (0.551225)*/
    ((2, 2), (3, 7)),        /*mean (0.00835583), correlation (0.55285)*/
    ((10, 6), (11, -8)),     /*mean (0.00885065), correlation (0.540913)*/
    ((6, 8), (8, -12)),      /*mean (0.0101552), correlation (0.551085)*/
    ((-7, 10), (-6, 5)),     /*mean (0.0102227), correlation (0.533635)*/
    ((-3, -9), (-3, 9)),     /*mean (0.0110211), correlation (0.543121)*/
    ((-1, -13), (-1, 5)),    /*mean (0.0113473), correlation (0.550173)*/
    ((-3, -7), (-3, 4)),     /*mean (0.0140913), correlation (0.554774)*/
    ((-8, -2), (-8, 3)),     /*mean (0.017049), correlation (0.55461)*/
    ((4, 2), (12, 12)),      /*mean (0.01778), correlation (0.546921)*/
    ((2, -5), (3, 11)),      /*mean (0.0224022), correlation (0.549667)*/
    ((6, -9), (11, -13)),    /*mean (0.029161), correlation (0.546295)*/
    ((3, -1), (7, 12)),      /*mean (0.0303081), correlation (0.548599)*/
    ((11, -1), (12, 4)),     /*mean (0.0355151), correlation (0.523943)*/
    ((-3, 0), (-3, 6)),      /*mean (0.0417904), correlation (0.543395)*/
    ((4, -11), (4, 12)),     /*mean (0.0487292), correlation (0.542818)*/
    ((2, -4), (2, 1)),       /*mean (0.0575124), correlation (0.554888)*/
    ((-10, -6), (-8, 1)),    /*mean (0.0594242), correlation (0.544026)*/
    ((-13, 7), (-11, 1)),    /*mean (0.0597391), correlation (0.550524)*/
    ((-13, 12), (-11, -13)), /*mean (0.0608974), correlation (0.55383)*/
    ((6, 0), (11, -13)),     /*mean (0.065126), correlation (0.552006)*/
    ((0, -1), (1, 4)),       /*mean (0.074224), correlation (0.546372)*/
    ((-13, 3), (-9, -2)),    /*mean (0.0808592), correlation (0.554875)*/
    ((-9, 8), (-6, -3)),     /*mean (0.0883378), correlation (0.551178)*/
    ((-13, -6), (-8, -2)),   /*mean (0.0901035), correlation (0.548446)*/
    ((5, -9), (8, 10)),      /*mean (0.0949843), correlation (0.554694)*/
    ((2, 7), (3, -9)),       /*mean (0.0994152), correlation (0.550979)*/
    ((-1, -6), (-1, -1)),    /*mean (0.10045), correlation (0.552714)*/
    ((9, 5), (11, -2)),      /*mean (0.100686), correlation (0.552594)*/
    ((11, -3), (12, -8)),    /*mean (0.101091), correlation (0.532394)*/
    ((3, 0), (3, 5)),        /*mean (0.101147), correlation (0.525576)*/
    ((-1, 4), (0, 10)),      /*mean (0.105263), correlation (0.531498)*/
    ((3, -6), (4, 5)),       /*mean (0.110785), correlation (0.540491)*/
    ((-13, 0), (-10, 5)),    /*mean (0.112798), correlation (0.536582)*/
    ((5, 8), (12, 11)),      /*mean (0.114181), correlation (0.555793)*/
    ((8, 9), (9, -6)),       /*mean (0.117431), correlation (0.553763)*/
    ((7, -4), (8, -12)),     /*mean (0.118522), correlation (0.553452)*/
    ((-10, 4), (-10, 9)),    /*mean (0.12094), correlation (0.554785)*/
    ((7, 3), (12, 4)),       /*mean (0.122582), correlation (0.555825)*/
    ((9, -7), (10, -2)),     /*mean (0.124978), correlation (0.549846)*/
    ((7, 0), (12, -2)),      /*mean (0.127002), correlation (0.537452)*/
    ((-1, -6), (0, -11)),    /*mean (0.127148), correlation (0.547401)*/
];
