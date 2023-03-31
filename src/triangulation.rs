use nalgebra::{DMatrix, Matrix3x4, Vector3};

use rayon::prelude::*;

use crate::fundamentalmatrix::FundamentalMatrix;

const PERSPECTIVE_VALUE_RANGE: f64 = 100.0;

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
    let mut points: Vec<Point> = correlated_points
        .column_iter()
        .enumerate()
        .par_bridge()
        .flat_map(|(col, out_col)| {
            out_col
                .iter()
                .enumerate()
                .filter_map(|(row, matched_point)| {
                    triangulate_point_affine((row, col), matched_point)
                })
                .collect::<Vec<_>>()
        })
        .collect();

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
    let mut points: Vec<Point> = correlated_points
        .column_iter()
        .enumerate()
        .par_bridge()
        .flat_map(|(col, out_col)| {
            out_col
                .iter()
                .enumerate()
                .flat_map(|(row, _)| {
                    let x1 = col as f64;
                    let y1 = row as f64;
                    if let Some(point2) = correlated_points[(row, col)] {
                        let x2 = point2.1 as f64;
                        let y2 = point2.0 as f64;
                        let point3d = triangulate_point_perspective(p2, (x1, y1), (x2, y2))?;
                        Some(Point::new((col, row), point3d))
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>()
        })
        .collect();
    scale_points(
        &mut points,
        (scale.0 as f64, scale.1 as f64, scale.2 as f64),
    );
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
    let mut point3d = FundamentalMatrix::triangulate_point(p2, point1, point2);
    point3d.unscale_mut(point3d.w);

    if point3d.z < 0.0 {
        point3d.unscale_mut(-1.0);
    }

    Some(Vector3::new(point3d.x, point3d.y, point3d.z))
}

fn scale_points(points: &mut Surface, scale: (f64, f64, f64)) {
    let (min_x, max_x, min_y, max_y, min_z, max_z) = points.iter().fold(
        (f64::MAX, f64::MIN, f64::MAX, f64::MIN, f64::MAX, f64::MIN),
        |acc, p| {
            let x = p.reconstructed.x;
            let y = p.reconstructed.y;
            let z = p.reconstructed.z;
            (
                acc.0.min(x),
                acc.1.max(x),
                acc.2.min(y),
                acc.3.max(y),
                acc.4.min(z),
                acc.5.max(z),
            )
        },
    );
    points.iter_mut().for_each(|point| {
        let point = &mut point.reconstructed;
        point.x = scale.0 * (point.x - min_x) * PERSPECTIVE_VALUE_RANGE / (max_x - min_x);
        point.y = scale.1 * (point.y - min_y) * PERSPECTIVE_VALUE_RANGE / (max_y - min_y);
        point.z = scale.2 * (point.z - min_z) * PERSPECTIVE_VALUE_RANGE / (max_z - min_z);
    })
}
