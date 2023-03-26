use nalgebra::{DMatrix, Matrix3, Matrix3x4, Vector3};
use rand::{rngs::SmallRng, seq::SliceRandom, SeedableRng};
use rayon::prelude::*;

use crate::fundamentalmatrix::FundamentalMatrix;

const HISTOGRAM_FILTER_BINS: usize = 100;
const HISTOGRAM_FILTER_DISCARD_PERCENTILE: f32 = 0.025;
const HISTOGRAM_FILTER_EPSILON: f32 = 0.001;
const PERSPECTIVE_PROJECTION_OPTIMIZATION_MAX_POINTS: usize = 100000;
const TRIANGULATION_MIN_SCALE: f64 = 0.0001;

pub struct Surface {
    pub points: DMatrix<Option<f32>>,
}

type Match = (u32, u32);

pub fn triangulate_affine(
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
    Surface { points }
}

pub fn triangulate_perspective(
    correlated_points: &DMatrix<Option<Match>>,
    p2: &Matrix3x4<f64>,
    scale: (f32, f32, f32),
) -> Surface {
    let mut points = DMatrix::<Option<f32>>::from_element(
        correlated_points.nrows(),
        correlated_points.ncols(),
        None,
    );
    let depth_scale = scale.2;

    /*
    // TODO: show progress
    let points3d: Vec<Vector3<f64>> = correlated_points
        .column_iter()
        .enumerate()
        .par_bridge()
        .flat_map(|(col, out_col)| {
            let col_points: Vec<Vector3<f64>> = out_col
                .iter()
                .enumerate()
                .filter_map(|(row, _)| {
                    let point1 = (col as usize, row as usize);
                    let point2 = correlated_points[(row, col)]?;
                    let point2 = (point2.1 as usize, point2.0 as usize);
                    let m = (point1, point2);
                    if let Ok(res) = FundamentalMatrix::optimize_triangulate_point(p2, &m) {
                        Some(Vector3::new(point1.0 as f64, point1.1 as f64, res.z))
                        //Some(res)
                    } else {
                        None
                    }
                })
                .collect();
            col_points
        })
        .collect();
    */

    let points3d: Vec<Vector3<f64>> = points
        .column_iter()
        .enumerate()
        .par_bridge()
        .flat_map(|(col, out_col)| {
            let col_points: Vec<Vector3<f64>> = out_col
                .iter()
                .enumerate()
                .flat_map(|(row, _)| {
                    let x1 = col as f64;
                    let y1 = row as f64;
                    if let Some(point2) = correlated_points[(row, col)] {
                        let x2 = point2.1 as f64;
                        let y2 = point2.0 as f64;
                        let z2 = triangulate_point_perspective(&p2, (x1, y1), (x2, y2))?;
                        Some(Vector3::new(x1, y1, z2.z))
                        //Some(z2)
                    } else {
                        None
                    }
                })
                .collect();
            col_points
        })
        .collect();

    // TODO: refactor this
    let x_min = points3d
        .iter()
        .map(|point| point.x)
        .min_by(|p1, p2| p1.partial_cmp(&p2).unwrap())
        .unwrap();
    let x_max = points3d
        .iter()
        .map(|point| point.x)
        .max_by(|p1, p2| p1.partial_cmp(&p2).unwrap())
        .unwrap();
    let y_min = points3d
        .iter()
        .map(|point| point.y)
        .min_by(|p1, p2| p1.partial_cmp(&p2).unwrap())
        .unwrap();
    let y_max = points3d
        .iter()
        .map(|point| point.y)
        .max_by(|p1, p2| p1.partial_cmp(&p2).unwrap())
        .unwrap();

    let z_min = points3d
        .iter()
        .map(|point| point.z)
        .min_by(|p1, p2| p1.partial_cmp(&p2).unwrap())
        .unwrap();
    let z_max = points3d
        .iter()
        .map(|point| point.z)
        .max_by(|p1, p2| p1.partial_cmp(&p2).unwrap())
        .unwrap();

    println!(
        "x = {} {} y= {} {} z={} {}",
        x_max, x_min, y_max, y_min, z_max, z_min
    );

    let max_scale = (x_max - x_min).max(y_max - y_min);
    println!("scale={}", max_scale);

    points3d.into_iter().for_each(|point| {
        let coord_x = correlated_points.ncols() as f64 * (point.x - x_min) / (x_max - x_min);
        let coord_y = correlated_points.nrows() as f64 * (point.y - y_min) / (y_max - y_min);
        let coord_x_signed = coord_x.round() as i32;
        let coord_y_signed = coord_y.round() as i32;
        if coord_x_signed >= points.ncols() as i32 || coord_y_signed >= points.nrows() as i32 {
            return;
        }
        let row = coord_y_signed.clamp(0, points.nrows() as i32 - 1) as usize;
        let col = coord_x_signed.clamp(0, points.ncols() as i32 - 1) as usize;
        let depth = (point.z - z_min) * correlated_points.nrows() as f64 * depth_scale as f64
            / (z_max - z_min);
        points[(row, col)] = Some(depth as f32);
    });

    filter_histogram(&mut points);
    Surface { points }
}

#[inline]
fn triangulate_point_affine(p1: (usize, usize), p2: Option<Match>) -> Option<f32> {
    if let Some(p2) = p2 {
        let dx = p1.1 as f32 - p2.1 as f32;
        let dy = p1.0 as f32 - p2.0 as f32;
        return Some((dx * dx + dy * dy).sqrt());
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

    // TODO: extract this into a constant
    if projection_error > 25.0 {
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
                    let point1 = (row as usize, col as usize);
                    if let Some(point2) = correlated_points[(row, col)] {
                        let point2 = (point2.0 as usize, point2.1 as usize);
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

fn filter_histogram(points: &mut DMatrix<Option<f32>>) {
    let (min, max) = points
        .iter()
        .flatten()
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
            + (i as f32 / HISTOGRAM_FILTER_BINS as f32 + HISTOGRAM_FILTER_EPSILON) * (max - min);
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

pub fn f_to_projection_matrix(
    f: &Matrix3<f64>,
    correlated_points: &DMatrix<Option<Match>>,
) -> Option<Matrix3x4<f64>> {
    //FundamentalMatrix::f_to_projection_matrix(f)

    const SELECT_RANDOM_MATCHES: usize = PERSPECTIVE_PROJECTION_OPTIMIZATION_MAX_POINTS;

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

    let mut rng = &mut SmallRng::from_rng(rand::thread_rng()).unwrap();
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
