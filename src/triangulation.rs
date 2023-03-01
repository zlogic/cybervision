use nalgebra::DMatrix;
use rayon::prelude::*;

const HISTOGRAM_FILTER_BINS: usize = 100;
const HISTOGRAM_FILTER_DISCARD_PERCENTILE: f32 = 0.025;
const HISTOGRAM_FILTER_EPSILON: f32 = 0.001;

#[derive(Debug)]
pub enum ProjectionMode {
    Affine,
    Perspective,
}

pub struct Surface {
    pub points: DMatrix<Option<f32>>,
}

type Match = (u32, u32);

impl Surface {
    pub fn new(
        correlated_points: &DMatrix<Option<Match>>,
        projection_mode: ProjectionMode,
        scale: (f32, f32, f32),
    ) -> Surface {
        match projection_mode {
            ProjectionMode::Affine => triangulate_affine(correlated_points, scale),
            ProjectionMode::Perspective => unimplemented!(),
        }
    }
}

fn triangulate_affine(
    correlated_points: &DMatrix<Option<Match>>,
    scale: (f32, f32, f32),
) -> Surface {
    let mut points = DMatrix::<Option<f32>>::from_element(
        correlated_points.nrows(),
        correlated_points.ncols(),
        None,
    );

    let depth_scale = scale.2 * ((scale.0 + scale.1) / 2.0);

    points
        .column_iter_mut()
        .enumerate()
        .par_bridge()
        .for_each(|(col, mut out_col)| {
            out_col.iter_mut().enumerate().for_each(|(row, out_point)| {
                *out_point = triangulate_point_affine((row, col), correlated_points[(row, col)])
                    .map(|depth| depth * depth_scale);
            })
        });
    filter_histogram(&mut points);
    return Surface { points };
}

fn triangulate_point_affine(p1: (usize, usize), p2: Option<Match>) -> Option<f32> {
    p2.map(|p2| ((p1.0 as f32 - p2.0 as f32).powi(2) + (p1.1 as f32 - p2.1 as f32).powi(2)).sqrt())
}

fn filter_histogram(points: &mut DMatrix<Option<f32>>) {
    let (min, max) = points
        .iter()
        .flat_map(|v| v)
        .fold((f32::MAX, f32::MIN), |acc, v| {
            (acc.0.min(*v), acc.1.max(*v))
        });

    let mut histogram_sum = 0usize;
    let mut histogram = [0usize; HISTOGRAM_FILTER_BINS];
    points.iter().for_each(|p| {
        let p = match p {
            Some(p) => p,
            None => return,
        };
        let pos = ((p - min) * HISTOGRAM_FILTER_BINS as f32 / (max - min)).round();
        let pos = (pos as usize).clamp(0, HISTOGRAM_FILTER_BINS - 1);
        histogram[pos] += 1;
        histogram_sum += 1;
    });

    let mut current_histogram_sum = 0;
    let mut min_depth = min;
    for (i, bin) in histogram.iter().enumerate() {
        current_histogram_sum += bin;
        if (current_histogram_sum as f32 / histogram_sum as f32)
            > HISTOGRAM_FILTER_DISCARD_PERCENTILE
        {
            break;
        }
        min_depth = min
            + (i as f32 / HISTOGRAM_FILTER_BINS as f32 - HISTOGRAM_FILTER_EPSILON) * (max - min);
    }
    let mut current_histogram_sum = 0;
    let mut max_depth = max;
    for (i, bin) in histogram.iter().enumerate().rev() {
        current_histogram_sum += bin;
        if (current_histogram_sum as f32 / histogram_sum as f32)
            > HISTOGRAM_FILTER_DISCARD_PERCENTILE
        {
            break;
        }
        max_depth = min
            + (i as f32 / HISTOGRAM_FILTER_BINS as f32 - HISTOGRAM_FILTER_EPSILON) * (max - min);
    }

    points.iter_mut().for_each(|p| {
        let p_value = match p {
            Some(p) => p,
            None => return,
        };
        if *p_value < min_depth || *p_value > max_depth {
            *p = None;
        }
    })
}
