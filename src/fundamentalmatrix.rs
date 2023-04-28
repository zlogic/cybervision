use nalgebra::{Matrix3, SMatrix, Vector3};
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rayon::prelude::*;
use std::cmp::Ordering;
use std::{fmt, sync::atomic::AtomicUsize, sync::atomic::Ordering as AtomicOrdering};

const MATCH_GRID_SIZE: usize = 8;
const RANSAC_K_AFFINE: usize = 1_000_000;
const RANSAC_K_PERSPECTIVE: usize = 10_000_000;
const RANSAC_N_AFFINE: usize = 4;
const RANSAC_N_PERSPECTIVE: usize = 8;
const RANSAC_T_AFFINE: f64 = 0.1;
const RANSAC_T_PERSPECTIVE: f64 = 10.0;
const RANSAC_D: usize = 10;
const RANSAC_D_EARLY_EXIT_AFFINE: usize = 1000;
const RANSAC_D_EARLY_EXIT_PERSPECTIVE: usize = 1000;
const RANSAC_CHECK_INTERVAL: usize = 50_000;
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
            .filter_map(|bucket| bucket.choose(rng).map(|m| m.to_owned()))
            .collect();
        if inliers.len() < self.ransac_n {
            return None;
        }

        let f = match self.projection {
            ProjectionMode::Affine => FundamentalMatrix::calculate_model_affine(&inliers)?,
            ProjectionMode::Perspective => {
                FundamentalMatrix::calculate_model_perspective(&inliers)?
            }
        };
        if !f.iter().all(|v| v.is_finite()) {
            return None;
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
        let s = usv.singular_values;
        if s[1] < RANSAC_RANK_EPSILON_PERSPECTIVE {
            return None;
        }
        let vt = &usv.v_t?;

        let vtc = vt.row(vt.nrows() - 1);
        let e = vtc.dot(&mean);
        let f = Matrix3::new(0.0, 0.0, vtc[0], 0.0, 0.0, vtc[1], vtc[2], vtc[3], -e);
        Some(f / f[(2, 2)])
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

        let s = usv.singular_values;
        if s[s.len() - 1].abs() < RANSAC_RANK_EPSILON_PERSPECTIVE {
            return None;
        }

        let vtc = vt.row(vt.nrows() - 1);
        let f = Matrix3::new(
            vtc[0], vtc[1], vtc[2], vtc[3], vtc[4], vtc[5], vtc[6], vtc[7], vtc[8],
        );
        let usv = f.svd(true, true);
        let u = usv.u?;
        let s = usv.singular_values;
        if s[1] < RANSAC_RANK_EPSILON_PERSPECTIVE {
            return None;
        }
        let vt = &usv.v_t?;
        let s = Vector3::new(s[0], s[1], 0.0);
        let s = Matrix3::from_diagonal(&s);
        let f = u * s * vt;

        // Scale back to image coordinates.
        let mut f = m2.tr_mul(&f) * m1;
        // Normalize by last element of F.
        f.unscale_mut(f[(2, 2)]);

        Some(f)
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
