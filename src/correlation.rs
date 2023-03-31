use nalgebra::DMatrix;
use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};

type Point = (usize, usize);

const THRESHOLD: f32 = 0.90;
const KERNEL_SIZE: usize = 5;
const KERNEL_WIDTH: usize = KERNEL_SIZE * 2 + 1;
const KERNEL_POINT_COUNT: usize = KERNEL_WIDTH * KERNEL_WIDTH;

#[derive(Debug)]
pub struct PointData<const KPC: usize> {
    pub delta: [f32; KPC],
    pub stdev: f32,
}

type KeypointPointData = PointData<KERNEL_POINT_COUNT>;

pub struct KeypointMatching {
    pub matches: Vec<(Point, Point)>,
}

pub trait ProgressListener
where
    Self: Sync + Sized,
{
    fn report_status(&self, pos: f32);
}

impl KeypointMatching {
    pub fn new<PL: ProgressListener>(
        img1: &DMatrix<u8>,
        img2: &DMatrix<u8>,
        points1: &Vec<Point>,
        points2: &Vec<Point>,
        progress_listener: Option<&PL>,
    ) -> KeypointMatching {
        let matches =
            KeypointMatching::match_points(img1, img2, points1, points2, progress_listener);
        KeypointMatching { matches }
    }

    fn match_points<PL: ProgressListener>(
        img1: &DMatrix<u8>,
        img2: &DMatrix<u8>,
        points1: &Vec<Point>,
        points2: &Vec<Point>,
        progress_listener: Option<&PL>,
    ) -> Vec<(Point, Point)> {
        let data1: Vec<Option<KeypointPointData>> = compute_points_data(img1, points1);
        let data2: Vec<Option<KeypointPointData>> = compute_points_data(img2, points2);
        let counter = AtomicUsize::new(0);
        points1
            .into_par_iter()
            .enumerate()
            .flat_map(|(i1, p1)| {
                if let Some(pl) = progress_listener {
                    let value =
                        counter.fetch_add(1, Ordering::Relaxed) as f32 / points1.len() as f32;
                    pl.report_status(value);
                }
                let data1 = match &data1[i1] {
                    Some(it) => it,
                    None => return vec![],
                };
                let points2 = &points2;
                points2
                    .iter()
                    .enumerate()
                    .filter_map(|(i2, p2)| {
                        let data2 = match &data2[i2] {
                            Some(it) => it,
                            None => return None,
                        };
                        correlate_points(data1, data2)
                            .filter(|corr| *corr > THRESHOLD)
                            .map(|_| (*p1, *p2))
                    })
                    .collect::<Vec<_>>()
            })
            .collect()
    }
}

#[inline]
pub fn point_inside_bounds<const KS: usize>(shape: (usize, usize), row: usize, col: usize) -> bool {
    row >= KS && col >= KS && row + KS < shape.0 && col + KS < shape.1
}

#[inline]
fn compute_points_data(img: &DMatrix<u8>, points: &Vec<Point>) -> Vec<Option<KeypointPointData>> {
    points
        .into_par_iter()
        .map(|p| compute_point_data::<KERNEL_SIZE, KERNEL_POINT_COUNT>(img, p.1, p.0))
        .collect()
}

#[inline]
pub fn compute_point_data<const KS: usize, const KPC: usize>(
    img: &DMatrix<u8>,
    row: usize,
    col: usize,
) -> Option<PointData<KPC>> {
    if !point_inside_bounds::<KS>(img.shape(), row, col) {
        return None;
    };
    let kernel_width = KS * 2 + 1;
    let mut result = PointData::<KPC> {
        delta: [0.0; KPC],
        stdev: 0.0,
    };
    let mut avg = 0.0;
    for r in 0..=KS * 2 {
        let row = (row + r).saturating_sub(KERNEL_SIZE);
        for c in 0..=KS * 2 {
            let col = (col + c).saturating_sub(KERNEL_SIZE);
            let value = img[(row, col)];
            let delta_pos = r * kernel_width + c;
            result.delta[delta_pos] = value.into();
            avg += value as f32;
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

#[inline]
fn correlate_points(
    data1: &PointData<KERNEL_POINT_COUNT>,
    data2: &PointData<KERNEL_POINT_COUNT>,
) -> Option<f32> {
    let mut corr = 0.0;
    for i in 0..KERNEL_POINT_COUNT {
        corr += data1.delta[i] * data2.delta[i];
    }
    corr /= data1.stdev * data2.stdev * KERNEL_POINT_COUNT as f32;
    if corr.is_nan() {
        return None;
    }
    Some(corr)
}
