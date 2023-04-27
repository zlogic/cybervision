use std::fmt;

use nalgebra::{DMatrix, Matrix3, Matrix3x4, Matrix4, Vector3, Vector4};

use rand::seq::SliceRandom;
use rand::{rngs::SmallRng, SeedableRng};
use rayon::prelude::*;

use crate::correlation;

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

pub trait Triangulation {
    fn triangulate(
        &mut self,
        correlated_points: &DMatrix<Option<Match>>,
        fundamental_matrix: &Matrix3<f64>,
    ) -> Result<(), TriangulationError>;
    fn get_surface(&self) -> Surface;
}

pub struct AffineTriangulation {
    surface: Surface,
    scale: (f64, f64, f64),
}

impl AffineTriangulation {
    pub fn new(scale: (f64, f64, f64)) -> AffineTriangulation {
        AffineTriangulation {
            surface: vec![],
            scale,
        }
    }

    #[inline]
    fn triangulate_point(p1: (usize, usize), p2: &Option<Match>) -> Option<Point> {
        if let Some(p2) = p2 {
            let dx = p1.1 as f64 - p2.1 as f64;
            let dy = p1.0 as f64 - p2.0 as f64;
            let distance = (dx * dx + dy * dy).sqrt();
            let point3d = Vector3::new(p1.1 as f64, p1.0 as f64, distance);

            return Some(Point::new((p1.1, p1.0), point3d));
        }
        None
    }

    fn scale_points(&self) -> Surface {
        let scale = &self.scale;

        let depth_scale = (scale.2 * ((scale.0 + scale.1) / 2.0)) as f64;
        self.surface
            .iter()
            .map(|point| {
                let mut point = *point;
                point.reconstructed.z *= depth_scale;
                point
            })
            .collect()
    }
}

impl Triangulation for AffineTriangulation {
    fn triangulate(
        &mut self,
        correlated_points: &DMatrix<Option<Match>>,
        _: &Matrix3<f64>,
    ) -> Result<(), TriangulationError> {
        let points3d = correlated_points
            .column_iter()
            .enumerate()
            .par_bridge()
            .flat_map(|(col, out_col)| {
                out_col
                    .iter()
                    .enumerate()
                    .filter_map(|(row, matched_point)| {
                        AffineTriangulation::triangulate_point((row, col), matched_point)
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        self.surface = points3d;

        Ok(())
    }

    fn get_surface(&self) -> Surface {
        // TODO: drop unused items?
        self.scale_points()
    }
}

pub struct PerspectiveTriangulation {
    surface: Surface,
    scale: (f64, f64, f64),
}

impl Triangulation for PerspectiveTriangulation {
    fn triangulate(
        &mut self,
        correlated_points: &DMatrix<Option<Match>>,
        fundamental_matrix: &Matrix3<f64>,
    ) -> Result<(), TriangulationError> {
        let p2 = match PerspectiveTriangulation::f_to_projection_matrix(&fundamental_matrix) {
            Some(p2) => p2,
            None => return Err(TriangulationError::new("Unable to find projection matrix")),
        };

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
                            let point3d = PerspectiveTriangulation::triangulate_point(
                                &p2,
                                (x1, y1),
                                (x2, y2),
                            );
                            Some(Point::new((col, row), point3d))
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .collect();

        let mut points = filter_outliers(correlated_points.shape(), &points);
        self.surface.append(&mut points);
        self.previous_matches = correlated_points.to_owned();

        Ok(())
    }

    fn get_surface(&self) -> Surface {
        // TODO: drop unused items?
        self.scale_points()
    }
}

impl PerspectiveTriangulation {
    pub fn new(scale: (f64, f64, f64)) -> PerspectiveTriangulation {
        PerspectiveTriangulation {
            surface: vec![],
            scale,
        }
    }

    #[inline]
    fn triangulate_point(
        p2: &Matrix3x4<f64>,
        point1: (f64, f64),
        point2: (f64, f64),
    ) -> Vector3<f64> {
        let mut point3d =
            PerspectiveTriangulation::triangulate_point_unnormalized(p2, point1, point2);
        point3d.unscale_mut(point3d.w);

        if point3d.z < 0.0 {
            point3d.unscale_mut(-1.0);
        }

        Vector3::new(point3d.x, point3d.y, point3d.z)
    }

    #[inline]
    fn triangulate_point_unnormalized(
        p2: &Matrix3x4<f64>,
        point1: (f64, f64),
        point2: (f64, f64),
    ) -> Vector4<f64> {
        let p1: Matrix3x4<f64> = Matrix3x4::identity();

        let mut a = Matrix4::<f64>::zeros();

        a.row_mut(0).copy_from(&(p1.row(2) * point1.0 - p1.row(0)));
        a.row_mut(1).copy_from(&(p1.row(2) * point1.1 - p1.row(1)));
        a.row_mut(2).copy_from(&(p2.row(2) * point2.0 - p2.row(0)));
        a.row_mut(3).copy_from(&(p2.row(2) * point2.1 - p2.row(1)));

        let usv = a.svd(false, true);
        let vt = usv.v_t.unwrap();
        let point4d = vt.row(vt.nrows() - 1).transpose();
        point4d
    }

    fn f_to_projection_matrix(f: &Matrix3<f64>) -> Option<Matrix3x4<f64>> {
        let usv = f.svd(true, false);
        let u = usv.u?;
        let e2 = u.column(2);
        let e2_skewsymmetric =
            Matrix3::new(0.0, -e2[2], e2[1], e2[2], 0.0, -e2[0], -e2[1], e2[0], 0.0);
        let e2s_f = e2_skewsymmetric * f;

        let mut p2 = Matrix3x4::zeros();
        for row in 0..3 {
            for col in 0..3 {
                p2[(row, col)] = e2s_f[(row, col)];
            }
            p2[(row, 3)] = e2[row];
        }

        Some(p2)
    }

    fn scale_points(&self) -> Surface {
        let scale = &self.scale;
        let points = &self.surface;

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
        let range_xy = (max_x - min_x).min(max_y - min_y);
        points
            .iter()
            .map(|point| {
                let mut point = *point;
                let point3d = &mut point.reconstructed;
                point3d.x = scale.0 * PERSPECTIVE_VALUE_RANGE * (point3d.x - min_x) / range_xy;
                point3d.y = scale.1 * PERSPECTIVE_VALUE_RANGE * (point3d.y - min_y) / range_xy;
                point3d.z =
                    scale.2 * PERSPECTIVE_VALUE_RANGE * (point3d.z - min_z) / (max_z - min_z);

                point
            })
            .collect()
    }
}

pub fn find_projection_matrix(
    fundamental_matrix: &Matrix3<f64>,
    correlated_points: &DMatrix<Option<Match>>,
) -> Option<Matrix3x4<f64>> {
    // Create essential matrix and camera matrices.
    let k = Matrix3::new(
        1.0,
        0.0,
        correlated_points.ncols() as f64 / 2.0,
        0.0,
        1.0,
        correlated_points.nrows() as f64 / 2.0,
        0.0,
        0.0,
        1.0,
    );
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
                            let point1 = (col as f64, *row as f64);
                            let point2 = correlated_points[(*row, col)];
                            let point2 = if let Some(point2) = point2 {
                                (point2.1 as f64, point2.0 as f64)
                            } else {
                                return false;
                            };
                            let mut point4d =
                                PerspectiveTriangulation::triangulate_point_unnormalized(
                                    &p2, point1, point2,
                                );
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

pub fn recover_relative_pose(
    correlated_points: &DMatrix<Option<Match>>,
    points: &Vec<Point>,
) -> (Matrix3<f64>, Vector3<f64>) {
    // Gaku Nakano, "A Simple Direct Solution to the Perspective-Three-Point Problem," BMVC2019
    const RANSAC_N: usize = 3;
    const RANSAC_K: usize = 10000;
    const RANSAC_T: f64 = 5.0;
    // TODO: add ransac_d
    // TODO: add progressbar
    let result = (0..RANSAC_K)
        .into_iter()
        .flat_map(|_| {
            let rng = &mut SmallRng::from_rng(rand::thread_rng()).unwrap();

            // Select points
            let inliers: Vec<_> = points
                .choose_multiple(rng, RANSAC_N)
                .filter_map(|point3d| {
                    let p1 = point3d.original;
                    let p2 = correlated_points[(p1.1, p1.0)]?;
                    //let p1 = Vector3::new(p1.1 as f64, p1.0 as f64, 1.0).normalize();
                    let p2 = Vector3::new(p2.1 as f64, p2.0 as f64, 1.0).normalize();
                    Some((p2, point3d.reconstructed))
                })
                .collect();
            if inliers.len() != RANSAC_N {
                return Vec::new();
            }

            // TODO: check if points are collinear?

            recover_pose_from_points(inliers)
                .into_iter()
                .map(|(r, t)| {
                    let mut p2 = Matrix3x4::zeros();
                    p2.fixed_view_mut::<3, 3>(0, 0).copy_from(&r);
                    p2.column_mut(3).copy_from(&t);

                    let (count, error) = points
                        .iter()
                        .filter_map(|point| {
                            let mut projection = p2 * point.reconstructed.insert_row(3, 1.0);
                            projection.unscale_mut(projection.z);
                            let p1 = point.original;
                            let p2 = correlated_points[(p1.1, p1.0)]?;
                            let dx = projection.x - p2.1 as f64;
                            let dy = projection.y - p2.0 as f64;
                            let error = (dx * dx + dy * dy).sqrt();
                            if error < RANSAC_T {
                                Some(error)
                            } else {
                                None
                            }
                        })
                        .fold((0, 0.0), |(count, error), match_error| {
                            (count + 1, error + match_error)
                        });
                    ((r, t), count, error / (count as f64))
                })
                .collect()
        })
        .reduce(|(pose1, count1, error1), (pose2, count2, error2)| {
            if count1 > count2 || (count1 == count2 && error1 < error2) {
                (pose1, count1, error1)
            } else {
                (pose2, count2, error2)
            }
        });

    // TODO: return error if not found
    let result = result.unwrap();

    result.0
}

fn recover_pose_from_points(
    mut inliers: Vec<(Vector3<f64>, Vector3<f64>)>,
) -> Vec<(Matrix3<f64>, Vector3<f64>)> {
    {
        // Rearrange inliers so that 0-1 has the largest distance
        let dist_01 = inliers[0].1.metric_distance(&inliers[1].1);
        let dist_12 = inliers[1].1.metric_distance(&inliers[2].1);
        let dist_02 = inliers[0].1.metric_distance(&inliers[2].1);
        if dist_12 > dist_01 && dist_12 > dist_02 {
            inliers.rotate_left(1)
        } else if dist_02 > dist_01 && dist_02 > dist_12 {
            inliers.swap(1, 2);
        }
    }

    let x10 = inliers[1].1 - inliers[0].1;
    let x20 = inliers[2].1 - inliers[0].1;
    let nx = x10.normalize();
    let nz = nx.cross(&x20).normalize();
    let ny = nz.cross(&nx).normalize();
    let mut n = Matrix3::zeros();
    n.column_mut(0).copy_from(&nx);
    n.column_mut(1).copy_from(&ny);
    n.column_mut(2).copy_from(&nz);

    let a = nx.tr_mul(&x10)[0];
    let b = nx.tr_mul(&x20)[0];
    let c = ny.tr_mul(&x20)[0];

    let m01 = inliers[0].0.tr_mul(&inliers[1].0)[0];
    let m02 = inliers[0].0.tr_mul(&inliers[2].0)[0];
    let m12 = inliers[1].0.tr_mul(&inliers[2].0)[0];

    let p = b / a;
    let q = (b * b + c * c) / (a * a);

    let f = [p, -m12, 0.0, -m01 * (2.0 * p - 1.0), m02, p - 1.0];
    let g = [q, 0.0, -1.0, -2.0 * m01 * q, 2.0 * m02, q - 1.0];

    let h = [
        -f[0] * f[0] + g[0] * f[1] * f[1],
        f[1] * f[1] * g[3] - 2.0 * f[0] * f[3] - 2.0 * f[0] * f[1] * f[4]
            + 2.0 * f[1] * f[4] * g[0],
        f[4] * f[4] * g[0] - 2.0 * f[0] * f[4] * f[4] - 2.0 * f[0] * f[5] + f[1] * f[1] * g[5]
            - f[3] * f[3]
            - 2.0 * f[1] * f[3] * f[4]
            + 2.0 * f[1] * f[4] * g[3],
        f[4] * f[4] * g[3]
            - 2.0 * f[3] * f[4] * f[4]
            - 2.0 * f[3] * f[5]
            - 2.0 * f[1] * f[4] * f[5]
            + 2.0 * f[1] * f[4] * g[5],
        -2.0 * f[4] * f[4] * f[5] + g[5] * f[4] * f[4] - f[5] * f[5],
    ];

    let mut xy = solve_quartic(h)
        .into_iter()
        .filter_map(|x| {
            if !x.is_finite() {
                return None;
            }
            let y = -((f[0] * x + f[3]) * x + f[5]) / (f[4] + f[1] * x);
            Some((x, y))
        })
        .collect::<Vec<_>>();

    polish_roots(f, g, &mut xy);

    let a_vector = Matrix3::new(
        -inliers[0].0.x,
        -inliers[0].0.y,
        -inliers[0].0.z,
        inliers[1].0.x,
        inliers[1].0.y,
        inliers[1].0.z,
        0.0,
        0.0,
        0.0,
    )
    .transpose();
    let b_vector = Matrix3::new(
        -inliers[0].0.x,
        -inliers[0].0.y,
        -inliers[0].0.z,
        0.0,
        0.0,
        0.0,
        inliers[2].0.x,
        inliers[2].0.y,
        inliers[2].0.z,
    )
    .transpose();
    let c_vector = b_vector - p * a_vector;

    xy.iter()
        .map(|xy| {
            let lambda = Vector3::new(1.0, xy.0, xy.1);
            let s = (a_vector * lambda).norm() / a;
            let d = lambda / s;
            let r1 = (a_vector * d) / a;
            let r2 = (c_vector * d) / c;
            let r3 = r1.cross(&r2);
            let mut rc = Matrix3::zeros();
            rc.column_mut(0).copy_from(&r1);
            rc.column_mut(1).copy_from(&r2);
            rc.column_mut(2).copy_from(&r3);

            let tc = d[0] * inliers[0].0;
            let r = rc * n.transpose();
            let t = tc - rc * n.transpose() * inliers[0].1;

            (r, t)
        })
        .collect()
}

fn solve_quartic(factors: [f64; 5]) -> [f64; 4] {
    let a = factors[0];
    let b = factors[1];
    let c = factors[2];
    let d = factors[3];
    let e = factors[4];

    let a_pw2 = a * a;
    let b_pw2 = b * b;
    let a_pw3 = a_pw2 * a;
    let b_pw3 = b_pw2 * b;
    let a_pw4 = a_pw3 * a;
    let b_pw4 = b_pw3 * b;

    let alpha = -3.0 * b_pw2 / (8.0 * a_pw2) + c / a;
    let beta = b_pw3 / (8.0 * a_pw3) - b * c / (2.0 * a_pw2) + d / a;
    let gamma =
        -3.0 * b_pw4 / (256.0 * a_pw4) + b_pw2 * c / (16.0 * a_pw3) - b * d / (4.0 * a_pw2) + e / a;

    let alpha_pw2 = alpha * alpha;
    let alpha_pw3 = alpha_pw2 * alpha;

    let p = -alpha_pw2 / 12.0 - gamma;
    let q = -alpha_pw3 / 108.0 + alpha * gamma / 3.0 - beta * beta / 8.0;
    let r = -q / 2.0 + (q * q / 4.0 + p * p * p / 27.0).sqrt();
    let u = r.powf(1.0 / 3.0);

    let y = if u.abs() < f64::EPSILON {
        -5.0 * alpha / 6.0 - q.powf(1.0 / 3.0)
    } else {
        -5.0 * alpha / 6.0 - p / (3.0 * u) + u
    };
    let w = (alpha + 2.0 * y).sqrt();
    [
        -b / (4.0 * a) + 0.5 * (w + (-(3.0 * alpha + 2.0 * y + 2.0 * beta / w)).sqrt()),
        -b / (4.0 * a) + 0.5 * (w - (-(3.0 * alpha + 2.0 * y + 2.0 * beta / w)).sqrt()),
        -b / (4.0 * a) + 0.5 * (-w + (-(3.0 * alpha + 2.0 * y - 2.0 * beta / w)).sqrt()),
        -b / (4.0 * a) + 0.5 * (-w - (-(3.0 * alpha + 2.0 * y - 2.0 * beta / w)).sqrt()),
    ]
}

fn polish_roots(f: [f64; 6], g: [f64; 6], xy: &mut Vec<(f64, f64)>) {
    const MAX_ITER: usize = 50;
    for _ in 0..MAX_ITER {
        for (x_target, y_target) in xy.iter_mut() {
            let x = *x_target;
            let y = *y_target;
            let x2 = x * x;
            let y2 = y * y;
            let x_y = x * y;

            let fv = f[0] * x2 + f[1] * x_y + f[3] * x + f[4] * y + f[5];
            let gv = g[0] * x2 - y2 + g[3] * x + g[4] * y + g[5];

            if fv.abs() < f64::EPSILON && gv.abs() < f64::EPSILON {
                continue;
            }

            let dfdx = 2.0 * f[0] * x + f[1] * y + f[3];
            let dfdy = f[1] * x + f[4];
            let dgdx = 2.0 * g[0] * x + g[3];
            let dgdy = -2.0 * y + g[4];

            let inv_det_j = 1.0 / (dfdx * dgdy - dfdy * dgdx);

            let dx = (dgdy * fv - dfdy * gv) * inv_det_j;
            let dy = (-dgdx * fv + dfdx * gv) * inv_det_j;

            *x_target -= dx;
            *y_target -= dy;
        }
    }
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

#[derive(Debug)]
pub struct TriangulationError {
    msg: &'static str,
}

impl TriangulationError {
    fn new(msg: &'static str) -> TriangulationError {
        TriangulationError { msg }
    }
}

impl std::error::Error for TriangulationError {}

impl fmt::Display for TriangulationError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.msg)
    }
}
