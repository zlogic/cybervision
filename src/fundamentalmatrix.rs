use nalgebra::{Matrix3, SMatrix, Vector3};
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rayon::prelude::*;
use std::cmp::Ordering;
use std::{fmt, sync::atomic::AtomicUsize, sync::atomic::Ordering as AtomicOrdering};

const MATCH_GRID_SIZE: usize = 8;
const RANSAC_K_AFFINE: usize = 10_000_000;
const RANSAC_K_PERSPECTIVE: usize = 10_000_000;
const RANSAC_N_AFFINE: usize = 4;
const RANSAC_N_PERSPECTIVE: usize = 8;
const RANSAC_RANK_EPSILON: f64 = 0.0;
const RANSAC_T_AFFINE: f64 = 0.1;
const RANSAC_T_PERSPECTIVE: f64 = 1.0;
const RANSAC_D: usize = 10;
const RANSAC_D_EARLY_EXIT_AFFINE: usize = 1000;
const RANSAC_D_EARLY_EXIT_PERSPECTIVE: usize = 20;
const RANSAC_CHECK_INTERVAL: usize = 500_000;

type Point = (usize, usize);
type Match = (Point, Point);

#[derive(Debug)]
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
    pub fn matches_to_buckets(
        point_matches: &Vec<Match>,
        dimensions: (u32, u32),
    ) -> Vec<Vec<Match>> {
        let width = dimensions.0;
        let height = dimensions.1;
        let mut match_buckets: Vec<Vec<Match>> = Vec::new();
        match_buckets.resize(MATCH_GRID_SIZE * MATCH_GRID_SIZE as usize, vec![]);
        point_matches.into_iter().for_each(|(p1, p2)| {
            let x_i = (MATCH_GRID_SIZE * p1.0) / (width as usize);
            let y_i = (MATCH_GRID_SIZE * p1.1) / (height as usize);
            let pos = y_i * MATCH_GRID_SIZE + x_i;
            match_buckets[pos].push((*p1, *p2));
        });
        return match_buckets
            .into_iter()
            .filter(|l| !l.is_empty())
            .collect();
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
        let matches_count: usize = match_buckets.into_iter().map(|b| b.len()).sum();
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
                    self.ransac_iteration(&match_buckets)
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
        return match best_result {
            Some(res) => Ok(res),
            None => Err(RansacError::new("No reliable matches found")),
        };
    }

    fn ransac_iteration(&self, match_buckets: &Vec<Vec<Match>>) -> Option<RansacIterationResult> {
        let rng = &mut SmallRng::from_rng(rand::thread_rng()).unwrap();
        // It's possible to select multiple points from the same bucket.
        let mut inliers = Vec::<Match>::with_capacity(self.ransac_n);
        while inliers.len() < self.ransac_n {
            let bucket = match_buckets.choose(rng)?;
            let inlier = bucket.choose(rng)?;
            if !inliers.contains(inlier) {
                inliers.push(*inlier);
            }
        }

        let f = match self.projection {
            ProjectionMode::Affine => self.calculate_model_affine(&inliers)?,
            ProjectionMode::Perspective => unimplemented!(),
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
            .into_iter()
            .map(|bucket| {
                bucket
                    .into_iter()
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
        return Some(RansacIterationResult {
            f,
            matches_count,
            best_error: inliers_error,
        });
    }

    #[inline]
    fn calculate_model_affine(&self, inliers: &Vec<Match>) -> Option<Matrix3<f64>> {
        let mut a = SMatrix::<f64, RANSAC_N_AFFINE, 4>::zeros();
        for i in 0..RANSAC_N_AFFINE {
            a[(0, i)] = inliers[i].1 .0 as f64;
            a[(1, i)] = inliers[i].1 .1 as f64;
            a[(2, i)] = inliers[i].0 .0 as f64;
            a[(3, i)] = inliers[i].0 .1 as f64;
        }
        let mean = a.row_mean();
        a.row_iter_mut().for_each(|mut r| r -= mean);
        let usv = a.svd(false, true);
        let s = usv.singular_values;
        let vt = &usv.v_t?;
        if s[RANSAC_N_AFFINE - 1].abs() < RANSAC_RANK_EPSILON {
            return None;
        }
        let vtc = vt.row(vt.nrows() - 1);
        let e = vtc.dot(&mean);
        let f = Matrix3::new(0.0, 0.0, vtc[0], 0.0, 0.0, vtc[1], vtc[2], vtc[3], -e);
        return Some(f);
    }

    #[inline]
    fn fits_model(&self, f: &Matrix3<f64>, m: &Match) -> Option<f64> {
        let p1 = Vector3::new(m.0 .0 as f64, m.0 .1 as f64, 1.0);
        let p2 = Vector3::new(m.1 .0 as f64, m.1 .1 as f64, 1.0);
        let p2t_f_p1 = p2.transpose() * f * p1;
        let f_p1 = f * p1;
        let ft_p2 = f.transpose() * p2;
        let nominator = (p2t_f_p1[0]) * (p2t_f_p1[0]);
        let denominator =
            f_p1[0] * f_p1[0] + f_p1[1] * f_p1[1] + ft_p2[0] * ft_p2[0] + ft_p2[1] * ft_p2[1];
        let err = nominator / denominator;
        if !err.is_finite() || err.abs() > self.ransac_t {
            return None;
        }
        return Some(err);
    }
}

impl Ord for RansacIterationResult {
    fn cmp(&self, other: &Self) -> Ordering {
        let mc_cmp = self.matches_count.cmp(&other.matches_count);
        if mc_cmp != Ordering::Equal {
            return mc_cmp;
        }
        // TODO: check if this is actually valid
        return self
            .best_error
            .partial_cmp(&other.best_error)
            .unwrap_or_else(|| {
                // Errors should be minimized, so a normal value is preferred (should be greater)
                // But the condition is reversed
                if self.best_error.is_finite() {
                    return Ordering::Less;
                } else if other.best_error.is_finite() {
                    return Ordering::Greater;
                }
                return Ordering::Equal;
            })
            .reverse();
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
        return write!(f, "{}", self.msg);
    }
}
