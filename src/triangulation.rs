use levenberg_marquardt::{LeastSquaresProblem, LevenbergMarquardt};
use nalgebra::{
    DMatrix, Dyn, Matrix3, Matrix3x4, Matrix4, OMatrix, OVector, Owned, Vector3, Vector4,
};
use rand::{rngs::SmallRng, seq::SliceRandom, SeedableRng};
use rayon::prelude::*;

const HISTOGRAM_FILTER_BINS: usize = 100;
const HISTOGRAM_FILTER_DISCARD_PERCENTILE: f32 = 0.025;
const HISTOGRAM_FILTER_EPSILON: f32 = 0.001;
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
    let mut point_matches: Vec<(Match, Match)> = correlated_points
        .column_iter()
        .enumerate()
        .par_bridge()
        .flat_map(|(col, out_col)| {
            let col_points: Vec<(Match, Match)> = out_col
                .iter()
                .enumerate()
                .flat_map(|(row, _)| {
                    let point1 = (row as u32, col as u32);
                    if let Some(point2) = correlated_points[(row, col)] {
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
    point_matches.shuffle(&mut rng);

    // TODO: customize this
    // TODO: show progress
    let points3d: Vec<Vector3<f32>> = point_matches
        .chunks(100)
        .par_bridge()
        .flat_map(|chunk| {
            let problem = ReprojectionErrorMinimization::new(p2.clone(), chunk, true).unwrap();
            let (result, report) = LevenbergMarquardt::new().minimize(problem);
            if !report.termination.was_successful() {
                panic!("LM failed")
            }
            result.extract_points3d()
        })
        .collect();
    */

    /*
    points
        .column_iter_mut()
        .enumerate()
        .par_bridge()
        .for_each(|(col, mut out_col)| {
            out_col.iter_mut().enumerate().for_each(|(row, out_point)| {
                let x1 = 2.0 * row as f64 / correlated_points.nrows() as f64 - 1.0;
                let y1 = 2.0 * col as f64 / correlated_points.ncols() as f64 - 1.0;
                let point2 = correlated_points[(row, col)];
                if let Some(point2) = correlated_points[(row, col)] {
                    let x2 = 2.0 * point2.0 as f64 / correlated_points.nrows() as f64 - 1.0;
                    let y2 = 2.0 * point2.1 as f64 / correlated_points.ncols() as f64 - 1.0;
                    *out_point = triangulate_point_perspective(p2, (x1, y1), (x2, y2))
                        .map(|depth| depth * depth_scale);
                } else {
                    *out_point = None
                };
            })
        });
        */

    /*
    let h = Matrix3::<f64>::new(
        2.0 / correlated_points.nrows() as f64,
        0.0,
        -1.0,
        0.0,
        2.0 / correlated_points.ncols() as f64,
        -1.0,
        0.0,
        0.0,
        1.0,
    );
    let p1: Matrix3x4<f64> = h * Matrix3x4::identity();

    let p2 = h * p2;
    */
    let p1: Matrix3x4<f64> = Matrix3x4::identity();

    let points3d: Vec<Vector3<f32>> = points
        .column_iter()
        .enumerate()
        .par_bridge()
        .flat_map(|(col, out_col)| {
            let col_points: Vec<Vector3<f32>> = out_col
                .iter()
                .enumerate()
                .flat_map(|(row, _)| {
                    let x1 = row as f32;
                    let y1 = col as f32;
                    if let Some(point2) = correlated_points[(row, col)] {
                        let x2 = point2.0 as f32;
                        let y2 = point2.1 as f32;
                        let z2 = triangulate_point_perspective(&p1, &p2, (x1, y1), (x2, y2))?;
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
        let coord_x = correlated_points.nrows() as f32 * (point.x - x_min) / (x_max - x_min);
        let coord_y = correlated_points.ncols() as f32 * (point.y - y_min) / (y_max - y_min);
        let coord_x_signed = coord_x.round() as i32;
        let coord_y_signed = coord_y.round() as i32;
        if coord_x_signed >= points.nrows() as i32 || coord_y_signed >= points.ncols() as i32 {
            return;
        }
        let row = coord_x_signed.clamp(0, points.nrows() as i32 - 1) as usize;
        let col = coord_y_signed.clamp(0, points.ncols() as i32 - 1) as usize;
        points[(row, col)] = Some(
            (point.z - z_min) * correlated_points.nrows() as f32 * depth_scale / (z_max - z_min),
        );
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
    p1: &Matrix3x4<f64>,
    p2: &Matrix3x4<f64>,
    point1: (f32, f32),
    point2: (f32, f32),
) -> Option<Vector3<f32>> {
    let mut point3d = triangulate_match_perspective(p1, p2, &point1, &point2);
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

    if projection_error > 10.0 {
        return None;
    }

    /*
    if point3d.w.abs() < TRIANGULATION_MIN_SCALE {
        return None;
    }
    */

    /*if point3d.z > 0.0 {
        Some(Vector3::new(
            point3d.x as f32,
            point3d.y as f32,
            point3d.z as f32,
        ))
    } else {
        Some(Vector3::new(
            point3d.x as f32,
            point3d.y as f32,
            point3d.z as f32,
        ))
    }*/
    Some(Vector3::new(
        point3d.x as f32,
        point3d.y as f32,
        point3d.z as f32,
    ))
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

    let p1: Matrix3x4<f64> = Matrix3x4::identity();

    // Solve chirality and find the matrix that the most points in front of the image.
    p2_1.column_mut(3).copy_from(&u3);
    p2_2.column_mut(3).copy_from(&-u3);
    p2_3.column_mut(3).copy_from(&u3);
    p2_4.column_mut(3).copy_from(&-u3);
    let p2 = [p2_1, p2_2, p2_3, p2_4]
        .into_iter()
        .map(|p2| {
            let points_count = validate_projection_matrix(p1, p2, correlated_points);
            (p2, points_count)
        })
        .max_by(|r1, r2| r1.1.cmp(&r2.1))
        .map(|(p2, _)| p2)?;

    let point_matches: Vec<(Match, Match)> = correlated_points
        .column_iter()
        .enumerate()
        .par_bridge()
        .flat_map(|(col, out_col)| {
            let col_points: Vec<(Match, Match)> = out_col
                .iter()
                .enumerate()
                .flat_map(|(row, _)| {
                    let point1 = (row as u32, col as u32);
                    if let Some(point2) = correlated_points[(row, col)] {
                        Some((point1, point2))
                    } else {
                        None
                    }
                })
                .collect();
            col_points
        })
        .collect();

    let problem = ReprojectionErrorMinimization::new(p2.clone(), &point_matches, false).unwrap();
    let (result, report) = LevenbergMarquardt::new().minimize(problem);
    if !report.termination.was_successful() {
        panic!("LM failed {:?}", report.termination)
    }

    Some(result.extract_p2())
}

fn validate_projection_matrix(
    p1: Matrix3x4<f64>,
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
                    let point1 = ((*row as f32), col as f32);
                    let point2 = correlated_points[(*row, col)];
                    let point2 = if let Some(point2) = point2 {
                        (point2.0 as f32, point2.1 as f32)
                    } else {
                        return false;
                    };
                    let mut point4d = triangulate_match_perspective(&p1, &p2, &point1, &point2);
                    point4d.unscale_mut(point4d.w);
                    point4d.z > 0.0 && (p2 * point4d).z > 0.0
                })
                .count()
        })
        .sum()
}

pub fn triangulate_match_perspective(
    p1: &Matrix3x4<f64>,
    p2: &Matrix3x4<f64>,
    point1: &(f32, f32),
    point2: &(f32, f32),
) -> Vector4<f64> {
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

struct ReprojectionErrorMinimization<'a> {
    /// Optimization parameters vector.
    /// First 12 items are columns of P2; followed by 3D coordinates of triangulated points.
    params: OVector<f64, Dyn>,
    point_matches: &'a [(Match, Match)],
    optimize_points: bool,
}

impl ReprojectionErrorMinimization<'_> {
    pub fn new<'a>(
        p2: Matrix3x4<f64>,
        point_matches: &[(Match, Match)],
        optimize_points: bool,
    ) -> Option<ReprojectionErrorMinimization> {
        let parameters_len = if optimize_points {
            12 + point_matches.len() * 3
        } else {
            12
        };
        let p1 = Matrix3x4::identity();
        let mut params = OVector::<f64, Dyn>::zeros(parameters_len);
        for col in 0..4 {
            for row in 0..3 {
                params[col * 3 + row] = p2[(row, col)];
            }
        }
        if optimize_points {
            for m_i in 0..point_matches.len() {
                let m = &point_matches[m_i];
                let m = triangulate_match_perspective(
                    &p1,
                    &p2,
                    &(m.0 .0 as f32, m.0 .1 as f32),
                    &(m.1 .0 as f32, m.1 .1 as f32),
                );
                for m_c in 0..3 {
                    params[12 + m_i * 3 + m_c] = m[m_c];
                }
            }
        }
        Some(ReprojectionErrorMinimization {
            point_matches,
            params,
            optimize_points,
        })
    }

    #[inline]
    fn projection_error(
        p1: Matrix3x4<f64>,
        p2: Matrix3x4<f64>,
        point3d: Vector4<f64>,
        point1: (u32, u32),
        point2: (u32, u32),
    ) -> [f64; 4] {
        let mut projection1 = p1 * point3d;
        let mut projection2 = p2 * point3d;
        projection1.unscale_mut(projection1[2]);
        projection2.unscale_mut(projection2[2]);
        [
            point1.0 as f64 - projection1[0],
            point1.1 as f64 - projection1[1],
            point2.0 as f64 - projection2[0],
            point2.1 as f64 - projection2[1],
        ]
    }

    #[inline]
    fn extract_p2(&self) -> Matrix3x4<f64> {
        let mut p2 = Matrix3x4::zeros();
        for col in 0..4 {
            for row in 0..3 {
                p2[(row, col)] = self.params[col * 3 + row];
            }
        }
        p2
    }

    #[inline]
    fn extract_points3d(&self) -> Vec<Vector3<f32>> {
        let p1 = Matrix3x4::identity();
        let p2 = self.extract_p2();
        (0..self.point_matches.len())
            .into_iter()
            .flat_map(|m_i| {
                let m = &self.point_matches[m_i];
                let mut point3d = Vector4::zeros();
                for m_c in 0..3 {
                    point3d[m_c] = self.params[12 + m_i * 3 + m_c];
                }
                point3d[3] = 1.0;
                let projection_error =
                    ReprojectionErrorMinimization::projection_error(p1, p2, point3d, m.0, m.1)
                        .iter()
                        .map(|e| e * e)
                        .sum::<f64>()
                        .sqrt();
                if projection_error < 25.0 {
                    Some(Vector3::new(
                        point3d.x as f32,
                        point3d.y as f32,
                        point3d.z as f32,
                    ))
                } else {
                    None
                }
            })
            .collect()
    }
}

impl LeastSquaresProblem<f64, Dyn, Dyn> for ReprojectionErrorMinimization<'_> {
    type ParameterStorage = Owned<f64, Dyn>;
    type ResidualStorage = Owned<f64, Dyn>;
    type JacobianStorage = Owned<f64, Dyn, Dyn>;

    fn set_params(&mut self, params: &OVector<f64, Dyn>) {
        self.params.copy_from(params);
    }

    fn params(&self) -> OVector<f64, Dyn> {
        self.params.clone_owned()
    }

    fn residuals(&self) -> Option<OVector<f64, Dyn>> {
        let p1 = Matrix3x4::identity();
        let p2 = self.extract_p2();
        // Residuals contain point reprojection errors.
        let mut residuals = OVector::<f64, Dyn>::zeros(self.point_matches.len() * 4);
        for m_i in 0..self.point_matches.len() {
            let m = &self.point_matches[m_i];
            let point3d = if self.optimize_points {
                let mut point3d = Vector4::zeros();
                for m_c in 0..3 {
                    point3d[m_c] = self.params[12 + m_i * 3 + m_c];
                }
                point3d[3] = 1.0;
                point3d
            } else {
                triangulate_match_perspective(
                    &p1,
                    &p2,
                    &(m.0 .0 as f32, m.0 .1 as f32),
                    &(m.1 .0 as f32, m.1 .1 as f32),
                )
            };
            let projection_error =
                ReprojectionErrorMinimization::projection_error(p1, p2, point3d, m.0, m.1);
            for r_c in 0..4 {
                residuals[m_i * 4 + r_c] = projection_error[r_c];
            }
        }
        Some(residuals)
    }

    fn jacobian(&self) -> Option<OMatrix<f64, Dyn, Dyn>> {
        let parameters_len = if self.optimize_points {
            12 + self.point_matches.len() * 3
        } else {
            12
        };
        // TODO: use sparse matrix?
        let mut jac = OMatrix::<f64, Dyn, Dyn>::zeros(self.point_matches.len() * 4, parameters_len);
        let p1 = Matrix3x4::identity();
        let p2 = self.extract_p2();
        // Write a row for each residual (reprojection error).
        // Using a symbolic formula (not finite differences/central difference), check the Rust LM library for more info.
        for m_i in 0..self.point_matches.len() {
            // Read data from parameters
            let p3d = if self.optimize_points {
                Vector4::new(
                    self.params[12 + m_i * 3 + 0],
                    self.params[12 + m_i * 3 + 1],
                    self.params[12 + m_i * 3 + 2],
                    1.0,
                )
            } else {
                let m = &self.point_matches[m_i];
                triangulate_match_perspective(
                    &p1,
                    &p2,
                    &(m.0 .0 as f32, m.0 .1 as f32),
                    &(m.1 .0 as f32, m.1 .1 as f32),
                )
            };
            let r1 = m_i * 4 + 0;
            let r2 = m_i * 4 + 1;
            let r3 = m_i * 4 + 2;
            let r4 = m_i * 4 + 3;
            // P is the projection matrix for image (1 or 2)
            // Prc is the r-th row, c-th column of P
            // X is a 3D coordinate (4-component vector [x y z 1])

            // Image 2 contains r3 and r4 (residuals for x2 and y2), affected by p2 and the point coordinates.
            // r3 = -(P11*x+P12*y+P13*z+P14)/(P31*x+P32*y+P33*z+P34)
            // r4 = -(P21*x+P22*y+P23*z+P24)/(P31*x+P32*y+P33*z+P34)
            // To keep things sane, create some aliases
            // Poi -> i-th element of 3D point e.g. x, y, z, or w
            // Pr1 = P11*x+P12*y+P13*z+P14
            // Pr2 = P21*x+P22*y+P23*z+P24
            // Pr3 = P31*x+P32*y+P33*z+P34
            let p_r = p2 * &p3d;
            // dr3/dP1i = -Poi/(P31*x+P32*y+P33*z+P34) = -Poi/Pr3
            for p_col in 0..4 {
                jac[(r3, p_col * 3 + 0)] = -p3d[p_col] / p_r[2];
            }
            // dr4/dP2i = -Poi/(P31*x+P32*y+P33*z+P34) = -Poi/Pr3
            for p_col in 0..4 {
                jac[(r4, p_col * 3 + 1)] = -p3d[p_col] / p_r[2];
            }
            // dr3/dP3i = Poi*(P11*x+P12*y+P13*z+P14)/((P31*x+P32*y+P33*z+P34)^2) = Poi*Pr1/(Pr3^2)
            for p_col in 0..4 {
                jac[(r3, p_col * 3 + 2)] = p3d[p_col] * p_r[0] / (p_r[2] * p_r[2]);
            }
            // dr4/dP3i = Poi*(P21*x+P22*y+P23*z+P24)/((P31*x+P32*y+P33*z+P34)^2) = Poi*Pr2/(Pr3^2)
            for p_col in 0..4 {
                jac[(r4, p_col * 3 + 2)] = p3d[p_col] * p_r[1] / (p_r[2] * p_r[2]);
            }
            // Skip 3D point coordinate optimization if not required.
            if !self.optimize_points {
                continue;
            }
            // dr3/dx = -(P11*(P32*y+P33*z+P34)-P31*(P12*y+P13*z+P14))/(Pr3^2) = -(P11*Pr3[x=0]-P31*Pr1[x=0])/(Pr3^2)
            // dr3/di = -(P1i*Pr3[i=0]-P3i*Pr1[i=0])/(Pr3^2)
            // dr4/dx = -(P21*(P32*y+P33*z+P34)-P31*(P22*y+P23*z+P24))/(Pr3^2) = -(P21*Pr3[x=0]-P31*Pr2[x=0])/(Pr3^2)
            // dr3/di = -(P2i*Pr3[i=0]-P3i*Pr2[i=0])/(Pr3^2)
            for coord in 0..3 {
                // Create a vector where coord = 0
                let mut vec_diff = p3d;
                vec_diff[coord] = 0.0;
                // Create projection where coord = 0
                let p_r_diff = p2 * vec_diff;
                jac[(r3, 12 + m_i * 3 + coord)] = -(p2[(0, coord)] * p_r_diff[2]
                    - p2[(2, coord)] * p_r_diff[0])
                    / (p_r[2] * p_r[2]);
                jac[(r4, 12 + m_i * 3 + coord)] = -(p2[(1, coord)] * p_r_diff[2]
                    - p2[(2, coord)] * p_r_diff[1])
                    / (p_r[2] * p_r[2]);
            }

            // Image 1 contains r1 and r2 (residuals for x1 and y1)
            // r1 = -(P11*x+P12*y+P13*z+P14)/(P31*x+P32*y+P33*z+P34) = -(x/z)
            // r2 = -(P21*x+P22*y+P23*z+P24)/(P31*x+P32*y+P33*z+P34) = -(y/z)
            // dr1/dx = -1/z; dr1/dz = x/(z^2)
            jac[(r1, 12 + m_i * 3 + 0)] = -1.0 / p3d.z;
            jac[(r1, 12 + m_i * 3 + 2)] = p3d.x / (p3d.z * p3d.z);
            // dr2/dy = -1/z; dr2/dz = y/(z^2)
            jac[(r2, 12 + m_i * 3 + 1)] = -1.0 / p3d.z;
            jac[(r2, 12 + m_i * 3 + 2)] = p3d.y / (p3d.z * p3d.z);
        }

        Some(jac)
    }
}

pub fn f_to_projection_matrix(
    f: &Matrix3<f64>,
    correlated_points: &DMatrix<Option<Match>>,
) -> Option<Matrix3x4<f64>> {
    let usv = f.svd(true, true);
    let u = usv.u?;
    let e2 = u.row(2);
    let e2_skewsymmetric = Matrix3::new(0.0, -e2[2], e2[1], e2[2], 0.0, -e2[0], -e2[1], e2[0], 0.0);
    let e2s_f = -e2_skewsymmetric * f;

    let mut p2 = Matrix3x4::zeros();
    for row in 0..3 {
        for col in 0..3 {
            p2[(row, col)] = e2s_f[(row, col)];
        }
        p2[(row, 3)] = e2[row];
    }

    //Some(p2)

    let point_matches: Vec<(Match, Match)> = correlated_points
        .column_iter()
        .enumerate()
        .par_bridge()
        .flat_map(|(col, out_col)| {
            let col_points: Vec<(Match, Match)> = out_col
                .iter()
                .enumerate()
                .flat_map(|(row, _)| {
                    let point1 = (row as u32, col as u32);
                    if let Some(point2) = correlated_points[(row, col)] {
                        Some((point1, point2))
                    } else {
                        None
                    }
                })
                .collect();
            col_points
        })
        .collect();

    let problem = ReprojectionErrorMinimization::new(p2.clone(), &point_matches, false).unwrap();
    let (result, report) = LevenbergMarquardt::new().minimize(problem);
    if !report.termination.was_successful() {
        panic!("LM failed {:?}", report.termination)
    }

    Some(result.extract_p2())
}
