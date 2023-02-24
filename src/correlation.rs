use image::{GenericImageView, GrayImage};

use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};

type Point = (u32, u32);

const KEYPOINT_THRESHOLD: f32 = 0.95;
const KEYPOINT_KERNEL_SIZE: usize = 7;

const KEYPOINT_KERNEL_WIDTH: usize = KEYPOINT_KERNEL_SIZE * 2 + 1;
const KEYPOINT_KERNEL_POINT_COUNT: usize = (KEYPOINT_KERNEL_WIDTH * KEYPOINT_KERNEL_WIDTH) as usize;

pub struct Correlator {}

struct PointData<const K: usize> {
    avg: f32,
    delta: [f32; K],
    stdev: f32,
}

impl Correlator {
    pub fn new() -> Correlator {
        return Correlator {};
    }

    pub fn match_points<P>(
        &self,
        img1: &GrayImage,
        img2: &GrayImage,
        points1: &Vec<Point>,
        points2: &Vec<Point>,
        pb: Option<P>,
    ) -> Vec<(Point, Point)>
    where
        P: Fn(f32) + Sync + Send,
    {
        const K: usize = KEYPOINT_KERNEL_POINT_COUNT as usize;
        let data1: Vec<Option<PointData<K>>> = compute_points_data(img1, points1);
        let data2: Vec<Option<PointData<K>>> = compute_points_data(img2, points2);
        let counter = AtomicUsize::new(0);
        let matches: Vec<((u32, u32), (u32, u32))> = points1
            .into_par_iter()
            .enumerate()
            .flat_map(|(i1, p1)| {
                pb.as_ref().map(|pb| {
                    pb(counter.fetch_add(1, Ordering::Relaxed) as f32 / points1.len() as f32);
                });
                let data1 = match &data1[i1] {
                    Some(it) => it,
                    None => return vec![],
                };
                let points2 = &points2;
                let matches: Vec<((u32, u32), (u32, u32))> = points2
                    .into_iter()
                    .enumerate()
                    .filter_map(|(i2, p2)| {
                        let data2 = match &data2[i2] {
                            Some(it) => it,
                            None => return None,
                        };
                        return correlate_points(data1, data2)
                            .filter(|corr| *corr > KEYPOINT_THRESHOLD)
                            .map(|_| (*p1, *p2));
                    })
                    .collect();
                return matches;
            })
            .collect();
        return matches;
    }
}

#[inline]
fn point_inside_bounds<const K: usize>(dimensions: (u32, u32), p: &Point) -> bool {
    return p.0 as usize >= K
        && p.1 as usize >= K
        && p.0 as usize + K < dimensions.0 as usize
        && p.1 as usize + K < dimensions.1 as usize;
}

#[inline]
fn compute_points_data(
    img: &GrayImage,
    points: &Vec<Point>,
) -> Vec<Option<PointData<KEYPOINT_KERNEL_POINT_COUNT>>> {
    return points
        .into_par_iter()
        .map(|p| compute_point_data::<KEYPOINT_KERNEL_SIZE, KEYPOINT_KERNEL_POINT_COUNT>(&img, p))
        .collect();
}

#[inline]
fn compute_point_data<const KS: usize, const KPC: usize>(
    img: &GrayImage,
    p: &Point,
) -> Option<PointData<KPC>> {
    if !point_inside_bounds::<KS>(img.dimensions(), p) {
        return None;
    };
    let kernel_size = KS as i32;
    let kernel_width = KS * 2 + 1;
    let mut result = PointData::<KPC> {
        avg: 0.0,
        delta: [0.0; KPC],
        stdev: 0.0,
    };
    for j in 0..KS * 2 + 1 {
        let y = p.1.saturating_add_signed(-(j as i32) + kernel_size);
        for i in 0..KS * 2 + 1 {
            let x = p.0.saturating_add_signed(-(i as i32) + kernel_size);
            let value = unsafe { img.unsafe_get_pixel(x, y)[0] };
            let delta_pos = j * kernel_width + i;
            result.delta[delta_pos] = value.into();
            result.avg += value as f32;
        }
    }
    result.avg /= KPC as f32;

    for i in 0..KPC {
        let delta = result.delta[i] - result.avg;
        result.delta[i] = delta;
        result.stdev += delta * delta;
    }
    let kernel_point_count = KPC as f32;
    result.stdev = (result.stdev / kernel_point_count).sqrt();

    return Some(result);
}

#[inline]
fn correlate_points(
    data1: &PointData<KEYPOINT_KERNEL_POINT_COUNT>,
    data2: &PointData<KEYPOINT_KERNEL_POINT_COUNT>,
) -> Option<f32> {
    let mut corr = 0.0;
    for i in 0..KEYPOINT_KERNEL_POINT_COUNT {
        corr += data1.delta[i] * data2.delta[i];
    }
    corr = corr / (data1.stdev * data2.stdev * KEYPOINT_KERNEL_POINT_COUNT as f32);
    if corr.is_nan() {
        return None;
    }
    return Some(corr);
}
