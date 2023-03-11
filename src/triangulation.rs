use nalgebra::{DMatrix, Matrix3, Matrix3x4, Matrix4, Vector3, Vector4};
use rayon::prelude::*;

const HISTOGRAM_FILTER_BINS: usize = 100;
const HISTOGRAM_FILTER_DISCARD_PERCENTILE: f32 = 0.025;
const HISTOGRAM_FILTER_EPSILON: f32 = 0.001;
const TRIANGULATION_MIN_SCALE: f64 = 0.01;

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

    /*points
    .column_iter_mut()
    .enumerate()
    .par_bridge()
    .for_each(|(col, mut out_col)| {
        out_col.iter_mut().enumerate().for_each(|(row, out_point)| {
            *out_point =
                triangulate_point_perspective(p2, (row, col), correlated_points[(row, col)])
                    .map(|depth| depth * depth_scale);
        })
    });*/
    let points3d: Vec<Vector3<f32>> = points
        .column_iter()
        .enumerate()
        .par_bridge()
        .flat_map(|(col, out_col)| {
            let col_points: Vec<Vector3<f32>> = out_col
                .iter()
                .enumerate()
                .flat_map(|(row, _)| {
                    let depth = triangulate_point_perspective(
                        &p2,
                        (row, col),
                        correlated_points[(row, col)],
                    )?;
                    Some(Vector3::new(row as f32, col as f32, depth))
                })
                .collect();
            col_points
        })
        .collect();

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

    points3d.into_iter().for_each(|point| {
        let coord_x = correlated_points.nrows() as f32 * (point.x - x_min) / (x_max - x_min);
        let coord_y = correlated_points.ncols() as f32 * (point.y - y_min) / (y_max - y_min);
        let coord_x_signed = coord_x.round() as i32;
        let coord_y_signed = coord_y.round() as i32;
        let row = coord_x_signed.clamp(0, points.nrows() as i32 - 1) as usize;
        let col = coord_y_signed.clamp(0, points.ncols() as i32 - 1) as usize;
        points[(row, col)] = Some(point.z * depth_scale);
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
    point1: (usize, usize),
    point2: Option<Match>,
) -> Option<f32> {
    let point2 = point2?;

    let point = triangulate_match_perspective(p2, &(point1.0 as u32, point1.1 as u32), &point2);

    if point.w.abs() < TRIANGULATION_MIN_SCALE {
        return None;
    }

    let point = point.unscale(point.w);
    if point.z > 0.0 {
        Some(point.z as f32)
    } else {
        None
    }
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
    [p2_1, p2_2, p2_3, p2_4]
        .into_iter()
        .map(|p2| {
            let points_count: usize = correlated_points
                .column_iter()
                .enumerate()
                .par_bridge()
                .map(move |(col, out_col)| {
                    out_col
                        .iter()
                        .enumerate()
                        .filter(|(row, _)| {
                            let point1 = (*row as u32, col as u32);
                            let point2 = correlated_points[(*row, col)];
                            let point2 = if let Some(point2) = point2 {
                                point2
                            } else {
                                return false;
                            };
                            let mut point4d = triangulate_match_perspective(&p2, &point1, &point2);
                            point4d.unscale_mut(point4d.w);
                            point4d.z > 0.0 && (p2 * point4d).z > 0.0
                        })
                        .count()
                })
                .sum();
            (p2, points_count)
        })
        .max_by(|r1, r2| r1.1.cmp(&r2.1))
        .map(|(p2, _)| p2)
}

pub fn triangulate_match_perspective(
    p2: &Matrix3x4<f64>,
    point1: &Match,
    point2: &Match,
) -> Vector4<f64> {
    let p1: Matrix3x4<f64> = Matrix3x4::identity();

    let mut a = Matrix4::<f64>::zeros();
    a.row_mut(0)
        .copy_from(&(p1.row(2) * point1.0 as f64 - p1.row(0)));
    a.row_mut(1)
        .copy_from(&(p1.row(2) * point1.1 as f64 - p1.row(1)));
    a.row_mut(2)
        .copy_from(&(p2.row(2) * point2.0 as f64 - p2.row(0)));
    a.row_mut(3)
        .copy_from(&(p2.row(2) * point2.1 as f64 - p2.row(1)));

    let usv = a.svd(false, true);
    let vt = usv.v_t.unwrap();
    let point4d = vt.row(vt.nrows() - 1).transpose();
    point4d
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
