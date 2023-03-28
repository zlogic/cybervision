use nalgebra::{DMatrix, Matrix3, Matrix3x4, Vector3};
use rand::{rngs::SmallRng, seq::SliceRandom, SeedableRng};
use rayon::prelude::*;

use crate::fundamentalmatrix::FundamentalMatrix;

const HISTOGRAM_FILTER_BINS: usize = 100;
const HISTOGRAM_FILTER_DISCARD_PERCENTILE: f64 = 0.025;
const HISTOGRAM_FILTER_EPSILON: f64 = 0.001;
const PERSPECTIVE_PROJECTION_OPTIMIZATION_MAX_POINTS: usize = 1000000;
const OPTIMIZE_PROJECTION_MATRIX: bool = true;
// TODO: this should be relative to the image size
const OPTIMIZATION_REPROJECTION_THRESHOLD: f64 = 1.0;
const TRIANGULATION_REPROJECTION_THRESHOLD: f64 = 10.0;

pub struct Point {
    pub original: (usize, usize),
    pub reconstructed: Vector3<f64>,
}

impl Point {
    fn new(original: (usize, usize), reconstructed: Vector3<f64>) -> Point {
        Point {
            original,
            reconstructed,
        }
    }
}

pub type Surface = Vec<Point>;

type Match = (u32, u32);

pub fn triangulate_affine(
    correlated_points: &DMatrix<Option<Match>>,
    scale: (f32, f32, f32),
) -> Surface {
    let mut points = correlated_points
        .column_iter()
        .enumerate()
        .par_bridge()
        .flat_map(|(col, out_col)| {
            let triangulated_points: Vec<Point> = out_col
                .iter()
                .enumerate()
                .filter_map(|(row, matched_point)| {
                    triangulate_point_affine((row, col), matched_point)
                })
                .collect();
            triangulated_points
        })
        .collect();
    filter_histogram(&mut points);

    let depth_scale = (scale.2 * ((scale.0 + scale.1) / 2.0)) as f64;
    points
        .iter_mut()
        .for_each(|p| p.reconstructed.z *= depth_scale);
    points
}

pub fn triangulate_perspective(
    correlated_points: &DMatrix<Option<Match>>,
    p2: &Matrix3x4<f64>,
    scale: (f32, f32, f32),
) -> Surface {
    let mut point_matches: Vec<((usize, usize), (usize, usize))> = correlated_points
        .column_iter()
        .enumerate()
        .par_bridge()
        .flat_map(|(col, out_col)| {
            let col_points: Vec<((usize, usize), (usize, usize))> = out_col
                .iter()
                .enumerate()
                .flat_map(|(row, _)| {
                    let point1 = (col, row);
                    let point2 = correlated_points[(row, col)]?;
                    let point2 = (point1.1 as usize, point2.0 as usize);
                    Some((point1, point2))
                })
                .collect();
            col_points
        })
        .collect();
    let mut rng = &mut SmallRng::from_rng(rand::thread_rng()).unwrap();
    point_matches.shuffle(&mut rng);

    // TODO: show progress
    let mut points = point_matches
        .chunks(100)
        .par_bridge()
        .flat_map(|chunk| {
            let points3d = if let Ok(points3d) =
                FundamentalMatrix::optimize_triangulate_points(&p2, chunk, false)
            {
                points3d
            } else {
                vec![]
            };
            points3d
                .iter()
                .enumerate()
                .filter_map(|(i, (point3d, err))| {
                    if *err < TRIANGULATION_REPROJECTION_THRESHOLD {
                        Some(Point::new(
                            chunk[i].0,
                            Vector3::new(chunk[i].0 .0 as f64, chunk[i].0 .1 as f64, point3d.z),
                            //*point3d,
                        ))
                    } else {
                        None
                    }
                })
                .collect::<Vec<Point>>()
        })
        .collect();

    filter_histogram(&mut points);

    let depth_scale = scale.2 as f64;
    points
        .iter_mut()
        .for_each(|p| p.reconstructed.z *= depth_scale);
    points
}

#[inline]
fn triangulate_point_affine(p1: (usize, usize), p2: &Option<Match>) -> Option<Point> {
    if let Some(p2) = p2 {
        let dx = p1.1 as f64 - p2.1 as f64;
        let dy = p1.0 as f64 - p2.0 as f64;
        let distance = (dx * dx + dy * dy).sqrt();
        let point3d = Vector3::new(p1.1 as f64, p1.0 as f64, distance);

        return Some(Point::new((p1.1, p1.0), point3d));
    }
    None
}

fn filter_histogram(points: &mut Surface) {
    let (min, max) = points.iter().fold((f64::MAX, f64::MIN), |acc, p| {
        let depth = p.reconstructed.z;
        (acc.0.min(depth), acc.1.max(depth))
    });

    let mut histogram_sum = 0usize;
    let mut histogram = [0usize; HISTOGRAM_FILTER_BINS];
    points.iter().for_each(|p| {
        let depth = p.reconstructed.z;
        let pos = ((depth - min) * HISTOGRAM_FILTER_BINS as f64 / (max - min)).round();
        let pos = (pos as usize).clamp(0, HISTOGRAM_FILTER_BINS - 1);
        histogram[pos] += 1;
        histogram_sum += 1;
    });

    let mut current_histogram_sum = 0;
    let mut min_depth = min;
    for (i, bin) in histogram.iter().enumerate() {
        current_histogram_sum += bin;
        if (current_histogram_sum as f64 / histogram_sum as f64)
            > HISTOGRAM_FILTER_DISCARD_PERCENTILE
        {
            break;
        }
        min_depth = min
            + (i as f64 / HISTOGRAM_FILTER_BINS as f64 - HISTOGRAM_FILTER_EPSILON) * (max - min);
    }
    let mut current_histogram_sum = 0;
    let mut max_depth = max;
    for (i, bin) in histogram.iter().enumerate().rev() {
        current_histogram_sum += bin;
        if (current_histogram_sum as f64 / histogram_sum as f64)
            > HISTOGRAM_FILTER_DISCARD_PERCENTILE
        {
            break;
        }
        max_depth = min
            + (i as f64 / HISTOGRAM_FILTER_BINS as f64 + HISTOGRAM_FILTER_EPSILON) * (max - min);
    }

    points.retain(|point| {
        let depth = point.reconstructed.z;
        depth >= min_depth && depth <= max_depth
    })
}

pub fn f_to_projection_matrix(
    f: &Matrix3<f64>,
    correlated_points: &DMatrix<Option<Match>>,
) -> Option<Matrix3x4<f64>> {
    let p2 = FundamentalMatrix::f_to_projection_matrix(f)?;
    if !OPTIMIZE_PROJECTION_MATRIX {
        return Some(p2);
    }

    // Select points with a low reprojection error.
    let point_matches: Vec<((usize, usize), (usize, usize))> = correlated_points
        .column_iter()
        .enumerate()
        .par_bridge()
        .flat_map(|(col, out_col)| {
            let col_points: Vec<((usize, usize), (usize, usize))> = out_col
                .iter()
                .enumerate()
                .flat_map(|(row, _)| {
                    let point1 = (col, row);
                    let point2 = correlated_points[(row, col)]?;
                    let point2 = (point2.1 as usize, point2.0 as usize);
                    let m = (point1, point2);
                    let res =
                        FundamentalMatrix::optimize_triangulate_points(&p2, &[m], false).ok()?;
                    let err = res[0].1;
                    if err < OPTIMIZATION_REPROJECTION_THRESHOLD {
                        Some((point1, point2))
                    } else {
                        None
                    }
                })
                .collect();
            col_points
        })
        .collect();

    let mut rng = &mut SmallRng::from_rng(rand::thread_rng()).unwrap();
    const SELECT_RANDOM_MATCHES: usize = PERSPECTIVE_PROJECTION_OPTIMIZATION_MAX_POINTS;
    let point_matches = if point_matches.len() > SELECT_RANDOM_MATCHES {
        point_matches
            .choose_multiple(&mut rng, SELECT_RANDOM_MATCHES)
            .map(|v| v.to_owned())
            .collect()
    } else {
        point_matches
    };

    FundamentalMatrix::optimize_f_to_projection(f, &point_matches)
}
