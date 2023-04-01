use nalgebra::{DMatrix, Matrix3x4, Vector3};

use rayon::prelude::*;

use crate::{correlation, fundamentalmatrix::FundamentalMatrix};

const PERSPECTIVE_VALUE_RANGE: f64 = 100.0;
const OUTLIER_FILTER_STDEV_THRESHOLD: f64 = 1.0;
const OUTLIER_FILTER_SEARCH_AREA: usize = 5;
const OUTLIER_FILTER_MIN_NEIGHBORS: usize = 10;

#[derive(Clone, Copy)]
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
    let points: Vec<Point> = correlated_points
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

    let mut points = filter_outliers(correlated_points.shape(), &points);

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

fn filter_outliers(shape: (usize, usize), points: &Vec<Point>) -> Vec<Point> {
    // TODO: replace this with something better?
    let mut point_depths: DMatrix<Option<f64>> = DMatrix::from_element(shape.0, shape.1, None);

    points
        .iter()
        .for_each(|p| point_depths[(p.original.1, p.original.0)] = Some(p.reconstructed.z));

    points
        .par_iter()
        .filter_map(|p| {
            if point_not_outlier(&point_depths, p.original.1, p.original.0) {
                Some(*p)
            } else {
                None
            }
        })
        .collect()
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
    let optimal_scale = 1.0 / (max_x - min_x).min(max_y - min_y);
    points.iter_mut().for_each(|point| {
        let point = &mut point.reconstructed;
        point.x = scale.0 * (point.x - min_x) * PERSPECTIVE_VALUE_RANGE * optimal_scale;
        point.y = scale.1 * (point.y - min_y) * PERSPECTIVE_VALUE_RANGE * optimal_scale;
        point.z = scale.2 * (point.z - min_z) * PERSPECTIVE_VALUE_RANGE / (max_z - min_z);
    })
}

#[inline]
fn point_not_outlier(img: &DMatrix<Option<f64>>, row: usize, col: usize) -> bool {
    const SEARCH_RADIUS: usize = OUTLIER_FILTER_SEARCH_AREA;
    const SEARCH_WIDTH: usize = SEARCH_RADIUS * 2 + 1;
    if !correlation::point_inside_bounds::<SEARCH_RADIUS>(img.shape(), row, col) {
        return false;
    };
    let point_distance = if let Some(v) = img[(row, col)] {
        v
    } else {
        return false;
    };
    let mut avg = 0.0;
    let mut stdev = 0.0;
    let mut count = 0;
    for r in 0..SEARCH_WIDTH {
        let srow = (row + r).saturating_sub(SEARCH_RADIUS);
        for c in 0..SEARCH_WIDTH {
            let scol = (col + c).saturating_sub(SEARCH_RADIUS);
            if srow == row && scol == col {
                continue;
            }
            let value = if let Some(v) = img[(srow, scol)] {
                v
            } else {
                continue;
            };
            avg += value;
            count += 1;
        }
    }
    if count < OUTLIER_FILTER_MIN_NEIGHBORS {
        return false;
    }

    avg /= count as f64;

    for r in 0..SEARCH_WIDTH {
        let srow = (row + r).saturating_sub(SEARCH_RADIUS);
        for c in 0..SEARCH_WIDTH {
            let scol = (col + c).saturating_sub(SEARCH_RADIUS);
            if srow == row && scol == col {
                continue;
            }
            let value = if let Some(v) = img[(srow, scol)] {
                v
            } else {
                continue;
            };
            let delta = value - avg;
            stdev += delta * delta;
        }
    }
    stdev = (stdev / count as f64).sqrt();

    (point_distance - avg).abs() < stdev * OUTLIER_FILTER_STDEV_THRESHOLD
}
