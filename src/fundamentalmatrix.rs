use nalgebra::{Matrix3, SMatrix, Vector3};
use rand::{rngs::SmallRng, Rng, SeedableRng};
use rayon::prelude::*;
use roots::find_roots_cubic;
use std::cmp::Ordering;
use std::{fmt, sync::atomic::AtomicUsize, sync::atomic::Ordering as AtomicOrdering};

const MIN_INLIER_DISTANCE: usize = 4;
const RANSAC_K_AFFINE: usize = 1_000_000;
const RANSAC_K_PERSPECTIVE: usize = 1_000_000;
const RANSAC_N_AFFINE: usize = 4;
const RANSAC_N_PERSPECTIVE: usize = 7;
const RANSAC_T_AFFINE: f64 = 0.1;
const RANSAC_T_PERSPECTIVE: f64 = 1.0;
const RANSAC_D: usize = 10;
const RANSAC_D_EARLY_EXIT_AFFINE: usize = 1000;
const RANSAC_D_EARLY_EXIT_PERSPECTIVE: usize = 1000;
const RANSAC_CHECK_INTERVAL: usize = 50_000;
const RANSAC_RANK_EPSILON_AFFINE: f64 = 0.001;
const RANSAC_RANK_EPSILON_PERSPECTIVE: f64 = 0.001;

type Point = (usize, usize);
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

pub struct FundamentalMatrix {
    pub f: Matrix3<f64>,
    pub matches_count: usize,
    projection: ProjectionMode,
    ransac_k: usize,
    ransac_n: usize,
    ransac_t: f64,
    ransac_d_early_exit: usize,
}

impl FundamentalMatrix {
    pub fn new<PL: ProgressListener>(
        projection: ProjectionMode,
        point_matches: &Vec<Match>,
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
            matches_count: 0,
            projection,
            ransac_k,
            ransac_n,
            ransac_t,
            ransac_d_early_exit,
        };
        match fm.find_ransac(point_matches, progress_listener) {
            Ok(res) => {
                fm.f = res.f;
                fm.matches_count = res.matches_count;
                Ok(fm)
            }
            Err(err) => Err(err),
        }
    }

    fn find_ransac<PL: ProgressListener>(
        &self,
        point_matches: &Vec<Match>,
        progress_listener: Option<&PL>,
    ) -> Result<RansacIterationResult, RansacError> {
        if point_matches.len() < RANSAC_D + self.ransac_n {
            return Err(RansacError::new("Not enough matches"));
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
            Some(res) => Ok(res),
            None => Err(RansacError::new("No reliable matches found")),
        }
    }

    #[inline]
    fn dist(x: usize, y: usize) -> usize {
        x.max(y).saturating_sub(x.min(y))
    }

    #[inline]
    fn choose_inliers(&self, point_matches: &Vec<Match>) -> Vec<Match> {
        let rng = &mut SmallRng::from_rng(rand::thread_rng()).unwrap();
        let mut inliers: Vec<Match> = vec![];

        while inliers.len() < self.ransac_n {
            let next_index = rng.gen_range(0..point_matches.len());
            let next_match = point_matches[next_index];
            let close_to_existing = inliers.iter().any(|check_match| {
                FundamentalMatrix::dist(next_match.0 .0, check_match.0 .0) < MIN_INLIER_DISTANCE
                    || FundamentalMatrix::dist(next_match.0 .1, check_match.0 .1)
                        < MIN_INLIER_DISTANCE
                    || FundamentalMatrix::dist(next_match.1 .0, check_match.1 .0)
                        < MIN_INLIER_DISTANCE
                    || FundamentalMatrix::dist(next_match.1 .1, check_match.1 .1)
                        < MIN_INLIER_DISTANCE
            });
            if !close_to_existing {
                inliers.push(next_match);
            }
        }
        inliers
    }

    fn ransac_iteration(&self, point_matches: &Vec<Match>) -> Vec<RansacIterationResult> {
        let inliers = self.choose_inliers(point_matches);
        if inliers.len() < self.ransac_n {
            return vec![];
        }

        let f = match self.projection {
            ProjectionMode::Affine => FundamentalMatrix::calculate_model_affine(&inliers),
            ProjectionMode::Perspective => FundamentalMatrix::calculate_model_perspective(&inliers),
        };
        f.iter()
            .filter_map(|f| self.validate_f(f.to_owned(), &inliers, point_matches))
            .collect::<Vec<_>>()
    }

    fn validate_f(
        &self,
        f: Matrix3<f64>,
        inliers: &Vec<Match>,
        point_matches: &Vec<Match>,
    ) -> Option<RansacIterationResult> {
        if !f.iter().all(|v| v.is_finite()) {
            return None;
        }
        let inliers_pass = inliers
            .into_iter()
            .all(|m| self.fits_model(&f, &m).is_some());
        if !inliers_pass {
            return None;
        }
        let all_inliers: (usize, f64) = point_matches
            .iter()
            .filter_map(|point_match| {
                if let Some(match_error) = self.fits_model(&f, point_match) {
                    Some((1, match_error))
                } else {
                    None
                }
            })
            .fold((0, 0.0), |acc, err| (acc.0 + err.0, acc.1 + err.1));

        if all_inliers.0 < RANSAC_D + self.ransac_n {
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

    #[inline]
    fn calculate_model_affine(inliers: &[Match]) -> Vec<Matrix3<f64>> {
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
    fn calculate_model_perspective(inliers: &Vec<Match>) -> Vec<Matrix3<f64>> {
        let mut a = SMatrix::<f64, RANSAC_N_PERSPECTIVE, 9>::zeros();
        let mut x1 = SMatrix::<f64, 3, RANSAC_N_PERSPECTIVE>::zeros();
        let mut x2 = SMatrix::<f64, 3, RANSAC_N_PERSPECTIVE>::zeros();
        for i in 0..RANSAC_N_PERSPECTIVE {
            let inlier = inliers[i];
            let p1 = Vector3::new(inlier.0 .0 as f64, inlier.0 .1 as f64, 1.0);
            let p2 = Vector3::new(inlier.1 .0 as f64, inlier.1 .1 as f64, 1.0);
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

        // TODO 0.17 remove this external dependency.
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
                let f = root.to_owned() * f1 + (1.0 - root) * f2;

                let usv = f.transpose().svd(false, true);
                let s = usv.singular_values;
                if s[1].abs() < RANSAC_RANK_EPSILON_PERSPECTIVE
                    || s[2].abs() > RANSAC_RANK_EPSILON_PERSPECTIVE
                {
                    return None;
                }

                // Normalize by last element of F.
                //f.unscale_mut(f[(2, 2)]);

                // Check sign consistency.
                let v_t = usv.v_t?;
                let e1 = v_t.row(v_t.nrows() - 1);
                let e1_skewsymmetric =
                    Matrix3::new(0.0, -e1[2], e1[1], e1[2], 0.0, -e1[0], -e1[1], e1[0], 0.0);
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
