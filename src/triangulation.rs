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
const TRIANGULATION_REPROJECTION_THRESHOLD: f64 = 5.0;

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
    // TODO: show progress
    let mut points: Vec<Point> = correlated_points
        .column_iter()
        .enumerate()
        .par_bridge()
        .flat_map(|(col, out_col)| {
            let col_points: Vec<Point> = out_col
                .iter()
                .enumerate()
                .filter_map(|(row, _)| {
                    let point1 = (col as usize, row as usize);
                    let point2 = correlated_points[(row, col)]?;
                    let point2 = (point2.1 as usize, point2.0 as usize);
                    let m = (point1, point2);
                    let (point3d, err) =
                        FundamentalMatrix::optimize_triangulate_point(&p2, &m).ok()?;
                    if err < TRIANGULATION_REPROJECTION_THRESHOLD {
                        Some(Point::new(
                            (col, row),
                            //Vector3::new(col as f64, row as f64, point3d.z),
                            point3d,
                        ))
                    } else {
                        None
                    }
                })
                .collect();
            col_points
        })
        .collect();

    /*
    let mut points: Vec<Point> = correlated_points
        .column_iter()
        .enumerate()
        .par_bridge()
        .flat_map(|(col, out_col)| {
            let col_points: Vec<Point> = out_col
                .iter()
                .enumerate()
                .flat_map(|(row, _)| {
                    let x1 = col as f64;
                    let y1 = row as f64;
                    if let Some(point2) = correlated_points[(row, col)] {
                        let x2 = point2.1 as f64;
                        let y2 = point2.0 as f64;
                        let point3d = triangulate_point_perspective(&p2, (x1, y1), (x2, y2))?;
                        Some(Point::new((col, row), point3d))
                    } else {
                        None
                    }
                })
                .collect();
            col_points
        })
        .collect();
     */

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

#[inline]
fn triangulate_point_perspective(
    p2: &Matrix3x4<f64>,
    point1: (f64, f64),
    point2: (f64, f64),
) -> Option<Vector3<f64>> {
    let p1 = Matrix3x4::identity();
    let mut point3d = FundamentalMatrix::triangulate_point(p2, point1, point2);
    point3d.unscale_mut(point3d.w);

    let mut projection1 = p1 * point3d;
    let mut projection2 = p2 * point3d;
    projection1.unscale_mut(projection1[2]);
    projection2.unscale_mut(projection2[2]);
    projection1.x -= point1.0 as f64;
    projection1.y -= point1.1 as f64;
    projection2.x -= point2.0 as f64;
    projection2.y -= point2.1 as f64;

    let projection_error = (projection1.x * projection1.x
        + projection1.y * projection1.y
        + projection2.x * projection2.x
        + projection2.y * projection2.y)
        .sqrt();

    if projection_error > TRIANGULATION_REPROJECTION_THRESHOLD {
        return None;
    }

    /*
    if point3d.w.abs() < TRIANGULATION_MIN_SCALE {
        return None;
    }
    */

    /*
    if point3d.z > 0.0 {
        Some(Vector3::new(point3d.x, point3d.y, point3d.z))
    } else {
        None
    }
    */
    Some(Vector3::new(point3d.x, point3d.y, point3d.z))
}

pub fn find_projection_matrix(
    fundamental_matrix: &Matrix3<f64>,
    correlated_points: &DMatrix<Option<Match>>,
) -> Option<Matrix3x4<f64>> {
    // Create essential matrix and camera matrices.
    let k = Matrix3::new(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
    let essential_matrix = k.tr_mul(fundamental_matrix) * k;

    // Create camera matrices and find one where
    let svd = essential_matrix.svd(true, true);
    let u = svd.u?;
    let vt = svd.v_t?;
    let u3 = u.column(2);
    const W: Matrix3<f64> = Matrix3::new(0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);

    let mut p2_1 = (u * (W) * vt).insert_column(3, 0.0);
    let mut p2_2 = (u * (W) * vt).insert_column(3, 0.0);
    let mut p2_3 = (u * (W.transpose()) * vt).insert_column(3, 0.0);
    let mut p2_4 = (u * (W.transpose()) * vt).insert_column(3, 0.0);

    // Solve chirality and find the matrix that the most points in front of the image.
    p2_1.column_mut(3).copy_from(&u3);
    p2_2.column_mut(3).copy_from(&-u3);
    p2_3.column_mut(3).copy_from(&u3);
    p2_4.column_mut(3).copy_from(&-u3);
    let p2 = [p2_1, p2_2, p2_3, p2_4]
        .into_iter()
        .map(|p2| {
            let points_count = validate_projection_matrix(p2, correlated_points);
            (p2, points_count)
        })
        .max_by(|r1, r2| r1.1.cmp(&r2.1))
        .map(|(p2, _)| p2)?;

    let point_matches: Vec<((usize, usize), (usize, usize))> = correlated_points
        .column_iter()
        .enumerate()
        .par_bridge()
        .flat_map(|(col, out_col)| {
            let col_points: Vec<((usize, usize), (usize, usize))> = out_col
                .iter()
                .enumerate()
                .flat_map(|(row, _)| {
                    let point1 = (col as usize, row as usize);
                    if let Some(point2) = correlated_points[(row, col)] {
                        let point2 = (point2.1 as usize, point2.0 as usize);
                        Some((point1, point2))
                    } else {
                        None
                    }
                })
                .collect();
            col_points
        })
        .collect();

    let f = FundamentalMatrix::projection_matrix_to_f(&p2);

    FundamentalMatrix::optimize_f_to_projection(&f, &point_matches)
}

fn validate_projection_matrix(
    p2: Matrix3x4<f64>,
    correlated_points: &DMatrix<Option<Match>>,
) -> usize {
    correlated_points
        .column_iter()
        .enumerate()
        .par_bridge()
        .map(move |(col, out_col)| {
            out_col
                .iter()
                .enumerate()
                .filter(|(row, _)| {
                    let point1 = ((*row as f64), col as f64);
                    let point2 = correlated_points[(*row, col)];
                    let point2 = if let Some(point2) = point2 {
                        (point2.0 as f64, point2.1 as f64)
                    } else {
                        return false;
                    };
                    let mut point4d = FundamentalMatrix::triangulate_point(&p2, point1, point2);
                    point4d.unscale_mut(point4d.w);
                    point4d.z > 0.0 && (p2 * point4d).z > 0.0
                })
                .count()
        })
        .sum()
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
                    let (point3d, err) =
                        FundamentalMatrix::optimize_triangulate_point(&p2, &m).ok()?;
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
