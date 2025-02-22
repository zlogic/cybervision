use crate::data::Point2D;
use nalgebra::allocator::Allocator;
use nalgebra::{
    ArrayStorage, DVector, DefaultAllocator, Dim, DimMin, DimMinimum, Dyn, Matrix, Matrix3,
    OMatrix, OVector, SMatrix, SVector, U1, U7, VecStorage, Vector3,
};
use rand::{Rng, SeedableRng, rngs::SmallRng};
use rayon::prelude::*;
use roots::find_roots_cubic;
use std::cmp::Ordering;
use std::{fmt, sync::atomic::AtomicUsize, sync::atomic::Ordering as AtomicOrdering};

type MatrixXx7<T> = Matrix<T, Dyn, U7, VecStorage<T, Dyn, U7>>;
type Vector7<T> = Matrix<T, U7, U1, ArrayStorage<T, 7, 1>>;

const TOP_INLIERS: usize = 5_000;
const MIN_INLIER_DISTANCE: usize = 10;
const RANSAC_K_AFFINE: usize = 1_000_000;
const RANSAC_K_PERSPECTIVE: usize = 1_000_000;
const RANSAC_N_AFFINE: usize = 4;
const RANSAC_N_PERSPECTIVE: usize = 7;
const RANSAC_T_AFFINE: f64 = 0.1;
const RANSAC_T_PERSPECTIVE: f64 = 10.0 / 1000.0;
const RANSAC_D_AFFINE: usize = 10;
const RANSAC_D_PERSPECTIVE: usize = 200;
const RANSAC_D_EARLY_EXIT_AFFINE: usize = 1000;
const RANSAC_D_EARLY_EXIT_PERSPECTIVE: usize = 50_000;
const RANSAC_CHECK_INTERVAL: usize = 50_000;
const RANSAC_RANK_EPSILON_AFFINE: f64 = 0.001;
const RANSAC_RANK_EPSILON_PERSPECTIVE: f64 = 0.001;

type Point = Point2D<usize>;
type Match = (Point, Point);

#[derive(Debug, PartialEq, Copy, Clone)]
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
    matches_count: usize,
    best_error: f64,
}

#[derive(Debug)]
pub struct FundamentalMatrixResult {
    pub f: Matrix3<f64>,
    pub inliers: Vec<Match>,
}

pub struct FundamentalMatrix {
    projection: ProjectionMode,
    ransac_k: usize,
    ransac_n: usize,
    ransac_t: f64,
    ransac_d: usize,
    ransac_d_early_exit: usize,
}

impl FundamentalMatrix {
    pub fn new(projection: ProjectionMode, max_dimension: f64) -> FundamentalMatrix {
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
            ProjectionMode::Perspective => RANSAC_T_PERSPECTIVE * max_dimension,
        };
        let ransac_d = match projection {
            ProjectionMode::Affine => RANSAC_D_AFFINE,
            ProjectionMode::Perspective => RANSAC_D_PERSPECTIVE,
        };
        let ransac_d_early_exit = match projection {
            ProjectionMode::Affine => RANSAC_D_EARLY_EXIT_AFFINE,
            ProjectionMode::Perspective => RANSAC_D_EARLY_EXIT_PERSPECTIVE,
        };
        FundamentalMatrix {
            projection,
            ransac_k,
            ransac_n,
            ransac_t,
            ransac_d,
            ransac_d_early_exit,
        }
    }

    pub fn find_ransac<PL: ProgressListener>(
        &self,
        point_matches: &[Match],
        progress_listener: Option<&PL>,
    ) -> Result<FundamentalMatrixResult, RansacError> {
        if point_matches.len() < self.ransac_d + self.ransac_n {
            return Err("Not enough matches".into());
        }

        let ransac_outer = self.ransac_k / RANSAC_CHECK_INTERVAL;
        let mut best_result = None;
        let counter = AtomicUsize::new(0);
        let max_matches = AtomicUsize::new(0);
        for _ in 0..ransac_outer {
            let result = (0..RANSAC_CHECK_INTERVAL)
                .into_par_iter()
                .flat_map(|_| {
                    if let Some(pl) = progress_listener {
                        let value = counter.fetch_add(1, AtomicOrdering::Relaxed) as f32
                            / self.ransac_k as f32;
                        pl.report_status(value);
                    }
                    self.ransac_iteration(point_matches)
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
            Some(res) => Ok(self.optimize_result(res, point_matches)),
            None => Err("No reliable matches found".into()),
        }
    }

    #[inline]
    fn dist(x: usize, y: usize) -> usize {
        x.max(y).saturating_sub(x.min(y))
    }

    #[inline]
    fn choose_inliers(&self, point_matches: &[Match]) -> Vec<Match> {
        let rng = &mut SmallRng::from_rng(&mut rand::rng());
        let mut inliers: Vec<Match> = vec![];

        let inliers_limit = point_matches.len().min(TOP_INLIERS);

        while inliers.len() < self.ransac_n {
            let next_index = rng.random_range(0..inliers_limit);
            let next_match = point_matches[next_index];
            let close_to_existing = inliers.iter().any(|check_match| {
                Self::dist(next_match.0.x, check_match.0.x) < MIN_INLIER_DISTANCE
                    || Self::dist(next_match.0.y, check_match.0.y) < MIN_INLIER_DISTANCE
                    || Self::dist(next_match.1.x, check_match.1.x) < MIN_INLIER_DISTANCE
                    || Self::dist(next_match.1.y, check_match.1.y) < MIN_INLIER_DISTANCE
            });
            if !close_to_existing {
                inliers.push(next_match);
            }
        }
        inliers
    }

    fn ransac_iteration(&self, point_matches: &[Match]) -> Vec<RansacIterationResult> {
        let inliers = self.choose_inliers(point_matches);
        if inliers.len() < self.ransac_n {
            return vec![];
        }

        let f = match self.projection {
            ProjectionMode::Affine => Self::calculate_model_affine(&inliers),
            ProjectionMode::Perspective => Self::calculate_model_perspective(&inliers),
        };
        f.iter()
            .filter_map(|f| self.validate_f(f.to_owned(), &inliers, point_matches))
            .collect::<Vec<_>>()
    }

    fn validate_f(
        &self,
        f: Matrix3<f64>,
        inliers: &[Match],
        point_matches: &[Match],
    ) -> Option<RansacIterationResult> {
        if !f.iter().all(|v| v.is_finite()) {
            return None;
        }
        let f = if self.projection == ProjectionMode::Perspective {
            Self::optimize_perspective_f(&f, inliers)?
        } else {
            f
        };
        let inliers_pass = inliers.iter().all(|m| self.fits_model(&f, m).is_some());
        if !inliers_pass {
            return None;
        }
        let all_inliers: (usize, f64) = point_matches
            .iter()
            .filter_map(|point_match| {
                self.fits_model(&f, point_match)
                    .map(|match_error| (1, match_error))
            })
            .fold((0, 0.0), |acc, err| (acc.0 + err.0, acc.1 + err.1));

        if all_inliers.0 < self.ransac_d + self.ransac_n {
            return None;
        }

        let matches_count = all_inliers.0;
        let inliers_error = all_inliers.1 / matches_count as f64;
        Some(RansacIterationResult {
            f: f.to_owned(),
            matches_count,
            best_error: inliers_error,
        })
    }

    fn optimize_result(
        &self,
        res: RansacIterationResult,
        point_matches: &[Match],
    ) -> FundamentalMatrixResult {
        let inliers = point_matches
            .iter()
            .filter_map(|point_match| self.fits_model(&res.f, point_match).map(|_| *point_match))
            .collect::<Vec<_>>();
        if self.projection == ProjectionMode::Affine {
            return FundamentalMatrixResult { f: res.f, inliers };
        }

        let optimal_f = Self::optimize_perspective_f(&res.f, inliers.as_slice()).unwrap_or(res.f);

        let inliers = point_matches
            .iter()
            .filter_map(|point_match| {
                self.fits_model(&optimal_f, point_match)
                    .map(|_| *point_match)
            })
            .collect::<Vec<_>>();
        FundamentalMatrixResult {
            f: optimal_f,
            inliers,
        }
    }

    #[inline]
    fn calculate_model_affine(inliers: &[Match]) -> Vec<Matrix3<f64>> {
        let mut a = SMatrix::<f64, RANSAC_N_AFFINE, 4>::zeros();
        for i in 0..RANSAC_N_AFFINE {
            let inlier = &inliers[i];
            a[(i, 0)] = inlier.1.x as f64;
            a[(i, 1)] = inlier.1.y as f64;
            a[(i, 2)] = inlier.0.x as f64;
            a[(i, 3)] = inlier.0.y as f64;
        }
        let mean = a.row_mean();
        a.row_iter_mut().for_each(|mut r| r -= mean);
        let usv = a.svd(false, true);
        let s = usv.singular_values;
        if s[1].abs() < RANSAC_RANK_EPSILON_AFFINE {
            return vec![];
        }
        let vt = if let Some(v_t) = &usv.v_t {
            v_t
        } else {
            return vec![];
        };

        let vtc = vt.row(vt.nrows() - 1);
        let e = vtc.dot(&mean);
        let f = Matrix3::new(0.0, 0.0, vtc[0], 0.0, 0.0, vtc[1], vtc[2], vtc[3], -e);
        vec![f / f[(2, 2)]]
    }

    #[inline]
    fn calculate_model_perspective(inliers: &[Match]) -> Vec<Matrix3<f64>> {
        let mut a = SMatrix::<f64, RANSAC_N_PERSPECTIVE, 9>::zeros();
        let mut x1 = SMatrix::<f64, 3, RANSAC_N_PERSPECTIVE>::zeros();
        let mut x2 = SMatrix::<f64, 3, RANSAC_N_PERSPECTIVE>::zeros();
        for i in 0..RANSAC_N_PERSPECTIVE {
            let inlier = inliers[i];
            let p1 = Vector3::new(inlier.0.x as f64, inlier.0.y as f64, 1.0);
            let p2 = Vector3::new(inlier.1.x as f64, inlier.1.y as f64, 1.0);
            a[(i, 0)] = p2[0] * p1[0];
            a[(i, 1)] = p2[0] * p1[1];
            a[(i, 2)] = p2[0];
            a[(i, 3)] = p2[1] * p1[0];
            a[(i, 4)] = p2[1] * p1[1];
            a[(i, 5)] = p2[1];
            a[(i, 6)] = p1[0];
            a[(i, 7)] = p1[1];
            a[(i, 8)] = 1.0;
            x1.set_column(i, &p1);
            x2.set_column(i, &p2);
        }
        let usv = a.svd(false, true);
        let v_t = if let Some(v_t) = &usv.v_t {
            v_t
        } else {
            return vec![];
        };

        let vtc = v_t.row(v_t.nrows() - 2);
        let f1 = Matrix3::new(
            vtc[0], vtc[1], vtc[2], vtc[3], vtc[4], vtc[5], vtc[6], vtc[7], vtc[8],
        );
        let vtc = v_t.row(v_t.nrows() - 1);
        let f2 = Matrix3::new(
            vtc[0], vtc[1], vtc[2], vtc[3], vtc[4], vtc[5], vtc[6], vtc[7], vtc[8],
        );

        let f = [f1, f2];

        // Based on vgg_singF_from_FF.
        let mut d = [[[0.0; 2]; 2]; 2];
        for i in 0..2 {
            for j in 0..2 {
                for k in 0..2 {
                    let det =
                        Matrix3::from_columns(&[f[i].column(0), f[j].column(1), f[k].column(2)]);
                    d[i][j][k] = det.determinant();
                }
            }
        }

        let mut coeffs = [0.0; 4];
        coeffs[0] = -d[1][0][0] + d[0][1][1] + d[0][0][0] + d[1][1][0] + d[1][0][1]
            - d[0][1][0]
            - d[0][0][1]
            - d[1][1][1];
        coeffs[1] = d[0][0][1] - 2.0 * d[0][1][1] - 2.0 * d[1][0][1] + d[1][0][0]
            - 2.0 * d[1][1][0]
            + d[0][1][0]
            + 3.0 * d[1][1][1];
        coeffs[2] = d[1][1][0] + d[0][1][1] + d[1][0][1] - 3.0 * d[1][1][1];
        coeffs[3] = d[1][1][1];

        let roots = find_roots_cubic(coeffs[0], coeffs[1], coeffs[2], coeffs[3]);
        let roots = match roots {
            roots::Roots::No(r) => r.to_vec(),
            roots::Roots::One(r) => r.to_vec(),
            roots::Roots::Two(r) => r.to_vec(),
            roots::Roots::Three(r) => r.to_vec(),
            roots::Roots::Four(r) => r.to_vec(),
        };

        roots
            .iter()
            .flat_map(|root| {
                let mut f = root.to_owned() * f1 + (1.0 - root) * f2;

                let usv = f.transpose().svd(false, true);
                let s = usv.singular_values;
                if s[1].abs() < RANSAC_RANK_EPSILON_PERSPECTIVE
                    || s[2].abs() > RANSAC_RANK_EPSILON_PERSPECTIVE
                {
                    return None;
                }

                // Normalize by last element of F.
                f.unscale_mut(f[(2, 2)]);

                // Check sign consistency.
                let v_t = usv.v_t?;
                let e1 = v_t.row(v_t.nrows() - 1).transpose();
                let e1_skewsymmetric = e1.cross_matrix();
                let l1 = e1_skewsymmetric * x1;
                let s = (f * x2).component_mul(&l1).column_sum();
                if s.iter().all(|s_c| *s_c > 0.0) || s.iter().all(|s_c| *s_c < 0.0) {
                    Some(f)
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
    }

    fn optimize_perspective_f(f: &Matrix3<f64>, inliers: &[Match]) -> Option<Matrix3<f64>> {
        let f_params = Self::params_from_perspective_f(f);
        let f_residuals = |p: &Vector7<f64>| {
            let f = Self::f_from_perspective_params(p);
            let mut residuals = DVector::<f64>::zeros(inliers.len());
            for i in 0..inliers.len() {
                residuals[i] = Self::reprojection_error(&f, &inliers[i]);
            }
            residuals
        };
        let f_jacobian = |p: &Vector7<f64>| {
            let mut jacobian = MatrixXx7::zeros(inliers.len());
            let f = Self::f_from_perspective_params(p);
            for (inlier_i, inlier) in inliers.iter().enumerate() {
                let inlier_jacobian = Self::f_jacobian(&f, inlier);
                jacobian
                    .row_mut(inlier_i)
                    .copy_from(&inlier_jacobian.transpose());
            }
            jacobian
        };

        let f = match least_squares(f_params, f_residuals, f_jacobian) {
            Ok(f_params) => Self::f_from_perspective_params(&f_params),
            Err(_) => return None,
        };

        let usv = f.transpose().svd(false, true);
        let s = usv.singular_values;
        if s[1].abs() < RANSAC_RANK_EPSILON_PERSPECTIVE
            || s[2].abs() > RANSAC_RANK_EPSILON_PERSPECTIVE
        {
            return None;
        }
        Some(f)
    }

    #[inline]
    fn params_from_perspective_f(f: &Matrix3<f64>) -> SVector<f64, 7> {
        // Use the first 7 elements of f as degrees of freedom.
        SVector::from([
            f[(0, 0)],
            f[(0, 1)],
            f[(0, 2)],
            f[(1, 0)],
            f[(1, 1)],
            f[(1, 2)],
            f[(2, 0)],
        ])
    }

    #[inline]
    fn f_from_perspective_params(p: &SVector<f64, 7>) -> Matrix3<f64> {
        // Parametrize f so that det(f)=0 (to enforce rank<=2) and f has 7 degrees of freedom.
        let x = -(-p[0] * p[4] + p[6] * p[2] * p[4] + p[3] * p[1] - p[6] * p[1] * p[5])
            / (-p[3] * p[2] + p[0] * p[5]);

        Matrix3::new(p[0], p[1], p[2], p[3], p[4], p[5], p[6], x, 1.0)
    }

    #[inline]
    fn fits_model(&self, f: &Matrix3<f64>, m: &Match) -> Option<f64> {
        let err = Self::reprojection_error(f, m);
        if !err.is_finite() || err.abs() > self.ransac_t {
            return None;
        }
        Some(err)
    }

    #[inline]
    fn reprojection_error(f: &Matrix3<f64>, m: &Match) -> f64 {
        let p1 = Vector3::new(m.0.x as f64, m.0.y as f64, 1.0);
        let p2 = Vector3::new(m.1.x as f64, m.1.y as f64, 1.0);
        let p2t_f_p1 = p2.tr_mul(f) * p1;
        let f_p1 = f * p1;
        let ft_p2 = f.tr_mul(&p2);
        let nominator = (p2t_f_p1[0]) * (p2t_f_p1[0]);
        let denominator =
            f_p1[0] * f_p1[0] + f_p1[1] * f_p1[1] + ft_p2[0] * ft_p2[0] + ft_p2[1] * ft_p2[1];
        nominator / denominator
    }

    fn f_jacobian(f: &Matrix3<f64>, m: &Match) -> Vector7<f64> {
        let mut result = Vector7::<f64>::zeros();
        for i in 0..7 {
            let row = i / 3;
            let col = i % 3;
            let p1 = Vector3::new(m.0.x as f64, m.0.y as f64, 1.0);
            let p2 = Vector3::new(m.1.x as f64, m.1.y as f64, 1.0);
            // Return a residual for reprojection error.
            // Using a symbolic formula (not finite differences/central difference).
            // Nominator:
            // d/dx (p2'*F*p1) = d/dx (x2*Fi1+y2*Fi2+Fi3)*[x1 y1 1] =
            // d/dx {p2'*F*p1} (Fij = 1 for matching row/col, Fij=0 otherwise) +
            // {p2'*F*p1} (Fij = 0 for matching row/column, original Fij otherwise)
            // d/dx (ax+b)^2 = 2a*(ax+b)
            // Full equation:
            // d/dx (ax+b)^2/((cx)^2+d) = (2a*(ax+b)*(cx*cx+d)-(2c*cx)*(ax+b)^2)/((cx)^2+d)^2 =
            // d/dx 2*(ax+b)*(acx*cx+ad-acx*cx-bc*cx)/((cx)^2+d)^2 =
            // d/dx 2*(ax+b)*(ad-bc*cx)/((cx)^2+d)^2
            let mut f_mask = Matrix3::from_element(0.0);
            f_mask[(row, col)] = 1.0;
            let nominator_diff_a = p2.tr_mul(&f_mask) * p1;
            let f_p1 = f * p1;
            let ft_p2 = f.tr_mul(&p2);
            let denominator_diff_c = f_p1[0] + f_p1[1] + ft_p2[0] + ft_p2[1];
            let mut f_mask = f.to_owned();
            f_mask[(row, col)] = 0.0;
            let nominator_const_b = p2.tr_mul(&f_mask) * p1;
            let f_p1 = f * p1;
            let ft_p2 = f.tr_mul(&p2);
            let denominator_diff_d = f_p1[0] + f_p1[1] + ft_p2[0] + ft_p2[1];
            let x = f[(row, col)];
            let jacobian_i = 2.0
                * (nominator_diff_a * x + nominator_const_b)
                * (nominator_diff_a * denominator_diff_d
                    - nominator_const_b * denominator_diff_c * denominator_diff_c * x)
                / (denominator_diff_c * denominator_diff_c * x * x + denominator_diff_d);
            result[i] = jacobian_i[0]
        }
        result
    }
}

fn least_squares<NP, NR, RF, JF>(
    params: OVector<f64, NP>,
    calc_residuals: RF,
    calc_jacobian: JF,
) -> Result<OVector<f64, NP>, RansacError>
where
    NR: Dim,
    NP: Dim + DimMin<NR> + DimMin<NP, Output = NP>,
    RF: Fn(&OVector<f64, NP>) -> OVector<f64, NR>,
    JF: Fn(&OVector<f64, NP>) -> OMatrix<f64, NR, NP>,
    DefaultAllocator: Allocator<NR, NP>
        + Allocator<NP>
        + Allocator<DimMinimum<NP, NP>>
        + Allocator<DimMinimum<NP, NP>, NP>
        + Allocator<NR>
        + Allocator<NR, NP>
        + Allocator<NP>,
{
    const MAX_ITERATIONS: usize = 1000;
    const TAU: f64 = 1E-3;
    const GRADIENT_EPSILON: f64 = 1E-12;
    const DELTA_EPSILON: f64 = 1E-12;
    const RESIDUAL_EPSILON: f64 = 1E-12;
    const RESIDUAL_REDUCTION_EPSILON: f64 = 0.0;

    // Levenberg-Marquardt optimization loop.
    let mut residual = calc_residuals(&params);
    let mut jacobian = calc_jacobian(&params);
    let mut jt_residual = jacobian.tr_mul(&residual);

    if jt_residual.max().abs() <= GRADIENT_EPSILON {
        return Ok(params);
    }

    let mut mu;
    {
        let jt_j = jacobian.tr_mul(&jacobian);
        mu = TAU
            * (0..jt_j.nrows())
                .map(|i| jt_j[(i, i)])
                .max_by(|a, b| a.total_cmp(b))
                .unwrap_or(1.0);
    }
    let mut nu = 2.0;
    let mut found = false;
    let mut params = params.clone();

    for _ in 0..MAX_ITERATIONS {
        let delta;
        {
            let mut jt_j = jacobian.tr_mul(&jacobian);
            for i in 0..jt_j.nrows() {
                jt_j[(i, i)] += mu;
            }
            delta = jt_j.lu().solve(&jt_residual);
        }
        let delta = if let Some(delta) = delta {
            delta
        } else {
            return Err("Failed to compute delta vector".into());
        };

        if delta.norm() <= DELTA_EPSILON * (params.norm() + DELTA_EPSILON) {
            found = true;
            break;
        }

        let new_params = (&params) + (&delta);
        let new_residual = calc_residuals(&new_params);
        let residual_norm_squared = residual.norm_squared();
        let new_residual_norm_squared = new_residual.norm_squared();

        let rho = (residual.norm_squared() - new_residual.norm_squared())
            / (delta.tr_mul(&(delta.scale(mu) + &jt_residual)))[0];

        if rho > 0.0 {
            let converged = residual_norm_squared.sqrt() - new_residual_norm_squared.sqrt()
                < RESIDUAL_REDUCTION_EPSILON * residual_norm_squared.sqrt();

            residual = new_residual;
            params = new_params;
            jacobian = calc_jacobian(&params);
            jt_residual = jacobian.tr_mul(&residual);

            if converged || jt_residual.max().abs() <= GRADIENT_EPSILON {
                found = true;
                break;
            }
            mu *= (1.0f64 / 3.0).max(1.0 - (2.0 * rho - 1.0).powi(3));
            nu = 2.0;
        } else {
            mu *= nu;
            nu *= 2.0;
        }

        if residual.norm() <= RESIDUAL_EPSILON {
            found = true;
            break;
        }
    }

    if !found {
        return Err("Levenberg-Marquardt failed to converge".into());
    }

    Ok(params)
}

impl Ord for RansacIterationResult {
    fn cmp(&self, other: &Self) -> Ordering {
        let mc_cmp = self.matches_count.cmp(&other.matches_count);
        if mc_cmp != Ordering::Equal {
            return mc_cmp;
        }
        // Prefer a real value.
        if self.best_error.is_finite() && !other.best_error.is_finite() {
            return Ordering::Greater;
        }
        if !self.best_error.is_finite() && other.best_error.is_finite() {
            return Ordering::Less;
        }
        if !self.best_error.is_finite() && !other.best_error.is_finite() {
            return Ordering::Equal;
        }

        // Errors should be minimized, so a higher error has a lower ordering rank.
        if self.best_error < other.best_error {
            Ordering::Greater
        } else if self.best_error == other.best_error {
            Ordering::Equal
        } else {
            Ordering::Less
        }
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

impl fmt::Display for RansacError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.msg)
    }
}

impl std::error::Error for RansacError {}

impl From<&'static str> for RansacError {
    fn from(msg: &'static str) -> RansacError {
        RansacError { msg }
    }
}
