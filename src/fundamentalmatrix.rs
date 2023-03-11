use levenberg_marquardt::{LeastSquaresProblem, LevenbergMarquardt};
use nalgebra::{
    Dyn, Matrix3, Matrix3x4, Matrix4, OMatrix, OVector, Owned, SMatrix, Vector3, Vector4,
};
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rayon::prelude::*;
use std::cmp::Ordering;
use std::{fmt, sync::atomic::AtomicUsize, sync::atomic::Ordering as AtomicOrdering};

const MATCH_GRID_SIZE: usize = 8;
const RANSAC_K_AFFINE: usize = 1_000_000;
const RANSAC_K_PERSPECTIVE: usize = 1_000_000;
const RANSAC_N_AFFINE: usize = 4;
const RANSAC_N_PERSPECTIVE: usize = 8;
const RANSAC_T_AFFINE: f64 = 0.1;
const RANSAC_T_PERSPECTIVE: f64 = 0.5;
const RANSAC_D: usize = 10;
const RANSAC_D_EARLY_EXIT_AFFINE: usize = 1000;
const RANSAC_D_EARLY_EXIT_PERSPECTIVE: usize = 1000;
const RANSAC_CHECK_INTERVAL: usize = 50_000;
const PERSPECTIVE_OPTIMIZE_F: bool = true;
const PERSPECTIVE_OPTIMIZE_POINTS: bool = true;

type Point = (usize, usize);
type Match = (Point, Point);

#[derive(Debug, PartialEq)]
pub enum ProjectionMode {
    Affine,
    Perspective,
}

pub trait ProgressListener
where
    Self: Sync + Sized,
{
    fn report_status(&self, pos: f32);
    fn report_matches(&self, matches_count: usize);
}

#[derive(Debug)]
struct RansacIterationResult {
    f: Matrix3<f64>,
    p2: Option<Matrix3x4<f64>>,
    matches_count: usize,
    best_error: f64,
}

pub struct FundamentalMatrix {
    pub f: Matrix3<f64>,
    pub p2: Option<Matrix3x4<f64>>,
    pub matches_count: usize,
    projection: ProjectionMode,
    ransac_k: usize,
    ransac_n: usize,
    ransac_t: f64,
    ransac_d_early_exit: usize,
}

impl FundamentalMatrix {
    pub fn matches_to_buckets(point_matches: &[Match], dimensions: (u32, u32)) -> Vec<Vec<Match>> {
        let width = dimensions.0;
        let height = dimensions.1;
        let mut match_buckets: Vec<Vec<Match>> = Vec::new();
        match_buckets.resize(MATCH_GRID_SIZE * MATCH_GRID_SIZE, vec![]);
        point_matches.iter().for_each(|(p1, p2)| {
            let x_i = (MATCH_GRID_SIZE * p1.0) / (width as usize);
            let y_i = (MATCH_GRID_SIZE * p1.1) / (height as usize);
            let pos = y_i * MATCH_GRID_SIZE + x_i;
            match_buckets[pos].push((*p1, *p2));
        });
        match_buckets
            .into_iter()
            .filter(|l| !l.is_empty())
            .collect()
    }

    pub fn new<PL: ProgressListener>(
        projection: ProjectionMode,
        match_buckets: &Vec<Vec<Match>>,
        progress_listener: Option<&PL>,
    ) -> Result<FundamentalMatrix, RansacError> {
        let ransac_k = match projection {
            ProjectionMode::Affine => RANSAC_K_AFFINE,
            ProjectionMode::Perspective => RANSAC_K_PERSPECTIVE,
        };
        let ransac_n = match projection {
            ProjectionMode::Affine => RANSAC_N_AFFINE,
            ProjectionMode::Perspective => RANSAC_N_PERSPECTIVE,
        };
        let ransac_t = match projection {
            ProjectionMode::Affine => RANSAC_T_AFFINE,
            // TODO: should the scale be used here, like it was done in the C version?
            ProjectionMode::Perspective => RANSAC_T_PERSPECTIVE,
        };
        let ransac_d_early_exit = match projection {
            ProjectionMode::Affine => RANSAC_D_EARLY_EXIT_AFFINE,
            ProjectionMode::Perspective => RANSAC_D_EARLY_EXIT_PERSPECTIVE,
        };
        let mut fm = FundamentalMatrix {
            f: Matrix3::from_element(f64::NAN),
            p2: None,
            matches_count: 0,
            projection,
            ransac_k,
            ransac_n,
            ransac_t,
            ransac_d_early_exit,
        };
        match fm.find_ransac(match_buckets, progress_listener) {
            Ok(res) => {
                fm.f = res.f;
                fm.p2 = res.p2;
                fm.matches_count = res.matches_count;
                Ok(fm)
            }
            Err(err) => Err(err),
        }
    }

    fn find_ransac<PL: ProgressListener>(
        &self,
        match_buckets: &Vec<Vec<Match>>,
        progress_listener: Option<&PL>,
    ) -> Result<RansacIterationResult, RansacError> {
        if match_buckets.len() < self.ransac_n {
            return Err(RansacError::new("Not enough match buckets"));
        }
        let matches_count: usize = match_buckets.iter().map(|b| b.len()).sum();
        if matches_count < RANSAC_D + self.ransac_n {
            return Err(RansacError::new("Not enough matches"));
        }

        let ransac_outer = self.ransac_k / RANSAC_CHECK_INTERVAL;
        let mut best_result = None;
        let counter = AtomicUsize::new(0);
        let max_matches = AtomicUsize::new(0);
        for _ in 0..ransac_outer {
            let result = (0..RANSAC_CHECK_INTERVAL)
                .into_par_iter()
                .filter_map(|_| {
                    if let Some(pl) = progress_listener {
                        let value = counter.fetch_add(1, AtomicOrdering::Relaxed) as f32
                            / self.ransac_k as f32;
                        pl.report_status(value);
                    }
                    self.ransac_iteration(match_buckets)
                })
                .inspect(|m| {
                    if let Some(pl) = progress_listener {
                        let count = max_matches.fetch_max(m.matches_count, AtomicOrdering::Relaxed);
                        pl.report_matches(count);
                    }
                })
                .max();
            best_result = best_result.max(result);
            if best_result
                .as_ref()
                .filter(|best_result| best_result.matches_count > self.ransac_d_early_exit)
                .is_some()
            {
                break;
            }
        }
        match best_result {
            Some(res) => Ok(res),
            None => Err(RansacError::new("No reliable matches found")),
        }
    }

    fn ransac_iteration(&self, match_buckets: &Vec<Vec<Match>>) -> Option<RansacIterationResult> {
        let rng = &mut SmallRng::from_rng(rand::thread_rng()).unwrap();
        let inliers: Vec<Match> = match_buckets
            .choose_multiple(rng, self.ransac_n)
            .into_iter()
            .filter_map(|bucket| bucket.choose(rng).map(|m| m.to_owned()))
            .collect();
        if inliers.len() < self.ransac_n {
            return None;
        }

        let mut f = match self.projection {
            ProjectionMode::Affine => FundamentalMatrix::calculate_model_affine(&inliers)?,
            ProjectionMode::Perspective => {
                FundamentalMatrix::calculate_model_perspective(&inliers)?
            }
        };
        if !f.iter().all(|v| v.is_finite()) {
            return None;
        }
        let mut p2 = None;
        if self.projection == ProjectionMode::Perspective {
            p2 = FundamentalMatrix::f_to_projection_matrix(&f);
            if PERSPECTIVE_OPTIMIZE_F {
                p2 = FundamentalMatrix::optimize_projection_matrix(&p2?, &inliers);
                match p2 {
                    Some(p2) => f = FundamentalMatrix::projection_matrix_to_f(&p2),
                    None => return None,
                }
            }
        }
        let inliers_pass = inliers
            .into_iter()
            .all(|m| self.fits_model(&f, &m).is_some());
        if !inliers_pass {
            return None;
        }
        let all_inliers: (usize, f64) = match_buckets
            .iter()
            .map(|bucket| {
                bucket
                    .iter()
                    .filter_map(|m| self.fits_model(&f, m))
                    .fold((0, 0.0), |(count, error), match_error| {
                        (count + 1, error + match_error)
                    })
            })
            .fold((0, 0.0), |acc, err| (acc.0 + err.0, acc.1 + err.1));

        if all_inliers.0 < RANSAC_D + self.ransac_n {
            return None;
        }

        let matches_count = all_inliers.0;
        let inliers_error = all_inliers.1 / matches_count as f64;
        Some(RansacIterationResult {
            f,
            p2,
            matches_count,
            best_error: inliers_error,
        })
    }

    #[inline]
    fn calculate_model_affine(inliers: &[Match]) -> Option<Matrix3<f64>> {
        let mut a = SMatrix::<f64, RANSAC_N_AFFINE, 4>::zeros();
        for i in 0..RANSAC_N_AFFINE {
            let inlier = &inliers[i];
            a[(i, 0)] = inlier.1 .0 as f64;
            a[(i, 1)] = inlier.1 .1 as f64;
            a[(i, 2)] = inlier.0 .0 as f64;
            a[(i, 3)] = inlier.0 .1 as f64;
        }
        let mean = a.row_mean();
        a.row_iter_mut().for_each(|mut r| r -= mean);
        let usv = a.svd(false, true);
        let vt = &usv.v_t?;

        let vtc = vt.row(vt.nrows() - 1);
        let e = vtc.dot(&mean);
        let f = Matrix3::new(0.0, 0.0, vtc[0], 0.0, 0.0, vtc[1], vtc[2], vtc[3], -e);
        Some(f)
    }

    #[inline]
    fn calculate_model_perspective(inliers: &Vec<Match>) -> Option<Matrix3<f64>> {
        let (m1, m2) = FundamentalMatrix::normalize_points_perspective(inliers);

        let mut a = SMatrix::<f64, RANSAC_N_PERSPECTIVE, 9>::zeros();
        for i in 0..RANSAC_N_PERSPECTIVE {
            let inlier = inliers[i];
            let p1 = m1 * Vector3::new(inlier.0 .0 as f64, inlier.0 .1 as f64, 1.0);
            let p2 = m2 * Vector3::new(inlier.1 .0 as f64, inlier.1 .1 as f64, 1.0);
            a[(i, 0)] = p2[0] * p1[0];
            a[(i, 1)] = p2[0] * p1[1];
            a[(i, 2)] = p2[0];
            a[(i, 3)] = p2[1] * p1[0];
            a[(i, 4)] = p2[1] * p1[1];
            a[(i, 5)] = p2[1];
            a[(i, 6)] = p1[0];
            a[(i, 7)] = p1[1];
            a[(i, 8)] = 1.0;
        }
        let usv = a.svd(false, true);
        let vt = &usv.v_t?;

        let vtc = vt.row(vt.nrows() - 1);
        let f = Matrix3::new(
            vtc[0], vtc[1], vtc[2], vtc[3], vtc[4], vtc[5], vtc[6], vtc[7], vtc[8],
        );
        let usv = f.svd(true, true);
        let u = usv.u?;
        let s = usv.singular_values;
        let vt = &usv.v_t?;
        let s = Vector3::new(s[0], s[1], 0.0);
        let s = Matrix3::from_diagonal(&s);
        let f = u * s * vt;

        // Scale back to image coordinates.
        Some(m2.tr_mul(&f) * m1)
    }

    #[inline]
    fn normalize_points_perspective(inliers: &Vec<Match>) -> (Matrix3<f64>, Matrix3<f64>) {
        let mut center1 = (0.0, 0.0);
        let mut center2 = (0.0, 0.0);
        for inlier in inliers {
            center1.0 += inlier.0 .0 as f64;
            center1.1 += inlier.0 .1 as f64;
            center2.0 += inlier.1 .0 as f64;
            center2.1 += inlier.1 .1 as f64;
        }
        let matches_count = inliers.len() as f64;
        center1 = (center1.0 / matches_count, center1.1 / matches_count);
        center2 = (center2.0 / matches_count, center2.1 / matches_count);
        let mut scale1 = 0.0;
        let mut scale2 = 0.0;
        for inlier in inliers {
            let dist1 = (
                center1.0 - inlier.0 .0 as f64,
                center1.1 - inlier.0 .1 as f64,
            );
            let dist2 = (
                center2.0 - inlier.1 .0 as f64,
                center2.1 - inlier.1 .1 as f64,
            );
            scale1 += (dist1.0 * dist1.0 + dist1.1 * dist1.1).sqrt();
            scale2 += (dist2.0 * dist2.0 + dist2.1 * dist2.1).sqrt();
        }
        scale1 = 2.0f64.sqrt() / (scale1 / matches_count);
        scale2 = 2.0f64.sqrt() / (scale2 / matches_count);

        let m1 = Matrix3::new(
            scale1,
            0.0,
            -center1.0 * scale1,
            0.0,
            scale1,
            -center1.1 * scale1,
            0.0,
            0.0,
            1.0,
        );

        let m2 = Matrix3::new(
            scale2,
            0.0,
            -center2.0 * scale2,
            0.0,
            scale2,
            -center2.1 * scale2,
            0.0,
            0.0,
            1.0,
        );
        (m1, m2)
    }

    #[inline]
    fn f_to_projection_matrix(f: &Matrix3<f64>) -> Option<Matrix3x4<f64>> {
        let usv = f.svd(true, true);
        let u = usv.u?;
        let e2 = u.row(2);
        let e2_skewsymmetric =
            Matrix3::new(0.0, -e2[2], e2[1], e2[2], 0.0, -e2[0], -e2[1], e2[0], 0.0).transpose();
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

    #[inline]
    fn projection_matrix_to_f(p2: &Matrix3x4<f64>) -> Matrix3<f64> {
        let t = p2.column(3);
        let t_skewsymmetric =
            Matrix3::new(0.0, -t[2], t[1], t[2], 0.0, -t[0], -t[1], t[0], 0.0).transpose();

        t_skewsymmetric * p2.fixed_view(0, 0)
    }

    fn optimize_projection_matrix(
        p2: &Matrix3x4<f64>,
        inliers: &Vec<Match>,
    ) -> Option<Matrix3x4<f64>> {
        let problem =
            ReprojectionErrorMinimization::new(p2.clone(), inliers, PERSPECTIVE_OPTIMIZE_POINTS)?;
        let (result, report) = LevenbergMarquardt::new().minimize(problem);
        if !report.termination.was_successful() {
            return None;
        }
        return Some(result.extract_p2());
    }

    #[inline]
    fn fits_model(&self, f: &Matrix3<f64>, m: &Match) -> Option<f64> {
        let p1 = Vector3::new(m.0 .0 as f64, m.0 .1 as f64, 1.0);
        let p2 = Vector3::new(m.1 .0 as f64, m.1 .1 as f64, 1.0);
        let p2t_f_p1 = p2.tr_mul(f) * p1;
        let f_p1 = f * p1;
        let ft_p2 = f.tr_mul(&p2);
        let nominator = (p2t_f_p1[0]) * (p2t_f_p1[0]);
        let denominator =
            f_p1[0] * f_p1[0] + f_p1[1] * f_p1[1] + ft_p2[0] * ft_p2[0] + ft_p2[1] * ft_p2[1];
        let err = nominator / denominator;
        if !err.is_finite() || err.abs() > self.ransac_t {
            return None;
        }
        Some(err)
    }

    pub fn triangulate_point(p2: &Matrix3x4<f64>, m: &Match) -> Vector4<f64> {
        let p1: Matrix3x4<f64> = Matrix3x4::identity();

        let mut a = Matrix4::<f64>::zeros();
        a.row_mut(0)
            .copy_from(&(p1.row(2) * m.0 .0 as f64 - p1.row(0)));
        a.row_mut(1)
            .copy_from(&(p1.row(2) * m.0 .1 as f64 - p1.row(1)));
        a.row_mut(2)
            .copy_from(&(p2.row(2) * m.1 .0 as f64 - p2.row(0)));
        a.row_mut(3)
            .copy_from(&(p2.row(2) * m.1 .1 as f64 - p2.row(1)));

        let usv = a.svd(false, true);
        let vt = usv.v_t.unwrap();
        let point4d = vt.row(vt.nrows() - 1).transpose();
        point4d
    }
}

impl Ord for RansacIterationResult {
    fn cmp(&self, other: &Self) -> Ordering {
        let mc_cmp = self.matches_count.cmp(&other.matches_count);
        if mc_cmp != Ordering::Equal {
            return mc_cmp;
        }
        // TODO: check if this is actually valid
        self.best_error
            .partial_cmp(&other.best_error)
            .unwrap_or_else(|| {
                // Errors should be minimized, so a normal value is preferred (should be greater)
                // But the condition is reversed
                if self.best_error.is_finite() {
                    return Ordering::Less;
                } else if other.best_error.is_finite() {
                    return Ordering::Greater;
                }
                Ordering::Equal
            })
            .reverse()
    }
}

struct ReprojectionErrorMinimization<'a> {
    /// Optimization parameters vector.
    /// First 12 items are columns of P2; followed by 3D coordinates of triangulated points.
    params: OVector<f64, Dyn>,
    point_matches: &'a Vec<Match>,
    optimize_points: bool,
}

impl ReprojectionErrorMinimization<'_> {
    pub fn new<'a>(
        p2: Matrix3x4<f64>,
        point_matches: &'a Vec<Match>,
        optimize_points: bool,
    ) -> Option<ReprojectionErrorMinimization> {
        let mut params = OVector::<f64, Dyn>::zeros(12 + point_matches.len() * 3);
        for col in 0..4 {
            for row in 0..3 {
                params[col * 3 + row] = p2[(row, col)];
            }
        }
        for m_i in 0..point_matches.len() {
            let m = FundamentalMatrix::triangulate_point(&p2, &point_matches[m_i]);
            for m_c in 0..3 {
                params[12 + m_i * 3 + m_c] = m[m_c];
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
        p2: Matrix3x4<f64>,
        point3d: Vector4<f64>,
        point1: (usize, usize),
        point2: (usize, usize),
    ) -> [f64; 4] {
        let p1 = Matrix3x4::<f64>::identity();
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
        let p2 = self.extract_p2();
        // Residuals contain point reprojection errors.
        let mut residuals = OVector::<f64, Dyn>::zeros(self.point_matches.len() * 4);
        for m_i in 0..self.point_matches.len() {
            let m = &self.point_matches[m_i];
            let mut point3d = Vector4::zeros();
            for m_c in 0..3 {
                point3d[m_c] = self.params[12 + m_i * 3 + m_c];
            }
            point3d[3] = 1.0;
            let projection_error =
                ReprojectionErrorMinimization::projection_error(p2, point3d, m.0, m.1);
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
        let p2 = self.extract_p2();
        // Write a row for each residual (reprojection error).
        // Using a symbolic formula (not finite differences/central difference), check the Rust LM library for more info.
        for m_i in 0..self.point_matches.len() {
            // Read data from parameters
            let p3d = Vector4::new(
                self.params[12 + m_i * 3 + 0],
                self.params[12 + m_i * 3 + 1],
                self.params[12 + m_i * 3 + 2],
                1.0,
            );
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
            // Pr1 = P11*x+P12*y+P13*z+P14
            // Pr2 = P21*x+P22*y+P23*z+P24
            // Pr3 = P31*x+P32*y+P33*z+P34
            let p_r = p2 * &p3d;
            // dr3/dP1i = -P1i/(P31*x+P32*y+P33*z+P34) = -P1i/Pr3
            for p_col in 0..4 {
                jac[(r3, p_col * 3 + 0)] = -p2[(0, p_col)] / p_r[2];
            }
            // dr4/dP2i = -P2i/(P31*x+P32*y+P33*z+P34) = -P2i/Pr3
            for p_col in 0..4 {
                jac[(r4, p_col * 3 + 1)] = -p2[(1, p_col)] / p_r[2];
            }
            // dr3/dP3i = -P3i*(P11*x+P12*y+P13*z+P14)/((P31*x+P32*y+P33*z+P34)^2) = -P3i*Pr1/(Pr3^2)
            for p_col in 0..4 {
                jac[(r3, p_col * 3 + 2)] = -p2[(2, p_col)] * p_r[0] / (p_r[2] * p_r[2]);
            }
            // dr4/dP3i = -P3i*(P21*x+P22*y+P23*z+P24)/((P31*x+P32*y+P33*z+P34)^2) = -P3i*Pr2/(Pr3^2)
            for p_col in 0..4 {
                jac[(r4, p_col * 3 + 2)] = -p2[(2, p_col)] * p_r[1] / (p_r[2] * p_r[2]);
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
            // dr1/dx = -z; dr1/dz = -x/(z^2)
            jac[(r1, 12 + m_i * 3 + 0)] = -p3d.z;
            jac[(r1, 12 + m_i * 3 + 2)] = -p3d.x / (p3d.z * p3d.z);
            // dr2/dy = -z; dr2/dz = -y/(z^2)
            jac[(r2, 12 + m_i * 3 + 1)] = -p3d.z;
            jac[(r2, 12 + m_i * 3 + 2)] = -p3d.y / (p3d.z * p3d.z);
        }

        Some(jac)
    }
}

impl PartialOrd for RansacIterationResult {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for RansacIterationResult {
    fn eq(&self, other: &Self) -> bool {
        self.matches_count == other.matches_count && self.best_error == other.best_error
    }
}

impl Eq for RansacIterationResult {}

#[derive(Debug)]
pub struct RansacError {
    msg: &'static str,
}

impl RansacError {
    fn new(msg: &'static str) -> RansacError {
        RansacError { msg }
    }
}

impl std::error::Error for RansacError {}

impl fmt::Display for RansacError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.msg)
    }
}
