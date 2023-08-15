use std::{fmt, ops::Range, sync::atomic::AtomicUsize, sync::atomic::Ordering as AtomicOrdering};

use nalgebra::{
    DMatrix, Matrix2x3, Matrix2x6, Matrix3, Matrix3x4, Matrix3x6, Matrix6, MatrixXx1, MatrixXx4,
    Vector2, Vector3, Vector4,
};

use rand::seq::SliceRandom;
use rand::{rngs::SmallRng, SeedableRng};
use rayon::prelude::*;

use crate::correlation;

const BUNDLE_ADJUSTMENT_MAX_ITERATIONS: usize = 1000;
const OUTLIER_FILTER_STDEV_THRESHOLD: f64 = 1.0;
const OUTLIER_FILTER_SEARCH_AREA: usize = 50;
const OUTLIER_FILTER_MIN_NEIGHBORS: usize = 250;
const PERSPECTIVE_SCALE_THRESHOLD: f64 = 0.0001;
const RANSAC_N: usize = 3;
const RANSAC_K: usize = 10_000;
// TODO: this should be proportional to image size
const RANSAC_INLIERS_T: f64 = 5.0;
const RANSAC_T: f64 = 5.0;
const RANSAC_D: usize = 100;
const RANSAC_D_EARLY_EXIT: usize = 100_000;
const RANSAC_CHECK_INTERVAL: usize = 100;

#[derive(Clone, Copy)]
pub struct Point {
    pub original: (usize, usize),
    pub reconstructed: Vector3<f64>,
    pub index: usize,
}

impl Point {
    fn new(original: (usize, usize), reconstructed: Vector3<f64>, index: usize) -> Point {
        Point {
            original,
            reconstructed,
            index,
        }
    }
}

pub type Surface = Vec<Point>;

type Match = (u32, u32);

pub trait ProgressListener
where
    Self: Sync + Sized,
{
    fn report_status(&self, pos: f32);
}

#[derive(Debug, PartialEq, Copy, Clone)]
pub enum ProjectionMode {
    Affine,
    Perspective,
}

pub struct Triangulation {
    affine: Option<AffineTriangulation>,
    perspective: Option<PerspectiveTriangulation>,
}

impl Triangulation {
    pub fn new(
        projection: ProjectionMode,
        scale: (f64, f64, f64),
        bundle_adjustment: bool,
    ) -> Triangulation {
        let (affine, perspective) = match projection {
            ProjectionMode::Affine => (
                Some(AffineTriangulation {
                    surface: vec![],
                    scale,
                }),
                None,
            ),
            ProjectionMode::Perspective => (
                None,
                Some(PerspectiveTriangulation {
                    calibration: vec![],
                    projections: vec![],
                    cameras: vec![],
                    tracks: vec![],
                    image_shapes: vec![],
                    scale,
                    bundle_adjustment,
                }),
            ),
        };

        Triangulation {
            affine,
            perspective,
        }
    }

    pub fn push_calibration(&mut self, k1: &Matrix3<f64>, k2: &Matrix3<f64>) {
        // TODO: refactor this to be a bit safer (provide image metadata along with the triangulation call).
        if let Some(perspective) = &mut self.perspective {
            perspective.push_calibration(k1, k2)
        }
    }

    pub fn triangulate<PL: ProgressListener>(
        &mut self,
        correlated_points: &DMatrix<Option<Match>>,
        fundamental_matrix: &Matrix3<f64>,
        progress_listener: Option<&PL>,
    ) -> Result<(), TriangulationError> {
        if let Some(affine) = &mut self.affine {
            affine.triangulate(correlated_points)
        } else if let Some(perspective) = &mut self.perspective {
            perspective.triangulate(correlated_points, fundamental_matrix, progress_listener)
        } else {
            Err(TriangulationError::new("Triangulation not initialized"))
        }
    }

    pub fn triangulate_all<PL: ProgressListener>(
        &mut self,
        progress_listener: Option<&PL>,
    ) -> Result<Surface, TriangulationError> {
        if let Some(affine) = &self.affine {
            affine.triangulate_all()
        } else if let Some(perspective) = &mut self.perspective {
            perspective.triangulate_all(progress_listener)
        } else {
            Err(TriangulationError::new("Triangulation not initialized"))
        }
    }

    pub fn complete(&mut self) {
        self.affine = None;
        self.perspective = None;
    }
}

struct AffineTriangulation {
    surface: Surface,
    scale: (f64, f64, f64),
}

impl AffineTriangulation {
    fn triangulate(
        &mut self,
        correlated_points: &DMatrix<Option<Match>>,
    ) -> Result<(), TriangulationError> {
        if !self.surface.is_empty() {
            return Err(TriangulationError::new(
                "Triangulation of multiple affine image is not supported",
            ));
        }

        let index = 0;

        let points3d = correlated_points
            .column_iter()
            .enumerate()
            .par_bridge()
            .flat_map(|(col, out_col)| {
                out_col
                    .iter()
                    .enumerate()
                    .filter_map(|(row, matched_point)| {
                        AffineTriangulation::triangulate_point((row, col), matched_point, index)
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        self.surface = points3d;

        Ok(())
    }

    fn triangulate_all(&self) -> Result<Surface, TriangulationError> {
        // TODO: drop unused items?
        Ok(self.scale_points())
    }

    #[inline]
    fn triangulate_point(p1: (usize, usize), p2: &Option<Match>, index: usize) -> Option<Point> {
        if let Some(p2) = p2 {
            let dx = p1.1 as f64 - p2.1 as f64;
            let dy = p1.0 as f64 - p2.0 as f64;
            let distance = (dx * dx + dy * dy).sqrt();
            let point3d = Vector3::new(p1.1 as f64, p1.0 as f64, distance);

            return Some(Point::new((p1.1, p1.0), point3d, index));
        }
        None
    }

    fn scale_points(&self) -> Surface {
        let scale = &self.scale;

        let depth_scale = scale.2 * ((scale.0 + scale.1) / 2.0);
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

#[derive(Clone, Debug)]
struct Track {
    start_index: usize,
    points: Vec<Match>,
    point3d: Option<Vector3<f64>>,
}

impl Track {
    fn new(start_index: usize, point: Match) -> Track {
        Track {
            start_index,
            points: vec![point],
            point3d: None,
        }
    }

    fn add(&mut self, index: usize, point: Match) -> bool {
        if index == self.start_index + self.points.len() {
            self.points.push(point);
            true
        } else {
            false
        }
    }

    fn range(&self) -> Range<usize> {
        // TODO: convert this into an iterator?
        self.start_index..self.start_index + self.points.len()
    }

    fn get(&self, i: usize) -> Option<Match> {
        if i < self.start_index || i >= self.start_index + self.points.len() {
            None
        } else {
            Some(self.points[i - self.start_index])
        }
    }

    fn first(&self) -> Option<&Match> {
        self.points.first()
    }

    fn last(&self) -> Option<&Match> {
        self.points.last()
    }
}

#[derive(Debug, Clone)]
struct Camera {
    k: Matrix3<f64>,
    r: Vector3<f64>,
    r_matrix: Matrix3<f64>,
    t: Vector3<f64>,
}

impl Camera {
    fn from_matrix(k: &Matrix3<f64>, r: &Matrix3<f64>, t: &Vector3<f64>) -> Camera {
        // Rodrigues formula, using method from "Vector Representation of Rotations" by Carlo Tomasi.
        let a = (r - r.transpose()) / 2.0;
        // Decode skew-symmetric matrix a.
        let rho = Vector3::new(
            a[(2, 1)] - a[(1, 2)],
            a[(0, 2)] - a[(2, 0)],
            a[(1, 0)] - a[(0, 1)],
        );
        let s = rho.norm();
        let c = (r.trace() - 1.0) / 2.0;
        let r = if s.abs() < f64::EPSILON && (c - 1.0).abs() < f64::EPSILON {
            Vector3::zeros()
        } else if s.abs() < f64::EPSILON && (c + 1.0).abs() < f64::EPSILON {
            let mut v_i = 0;
            let mut v_norm = 0.0;
            let r_i = r + Matrix3::identity();
            for (v_candidate_i, v_candidate) in r_i.column_iter().enumerate() {
                let v_candidate_norm = v_candidate.norm();
                if v_candidate_norm > v_norm {
                    v_i = v_candidate_i;
                    v_norm = v_candidate_norm;
                }
            }
            let v = r_i.column(v_i);
            let u = v / v.norm();

            let r = u * std::f64::consts::PI;
            if (r.norm() - std::f64::consts::PI).abs() < f64::EPSILON
                && ((r.x.abs() < f64::EPSILON && r.y.abs() < f64::EPSILON && r.z < 0.0)
                    || (r.x.abs() < f64::EPSILON && r.y < 0.0)
                    || r.x < 0.0)
            {
                -r
            } else {
                r
            }
        } else {
            let u = rho / s;
            let theta = s.atan2(c);
            u * theta
        };

        let r_matrix = Camera::matrix_r(&r);
        Camera {
            k: k.to_owned(),
            r,
            r_matrix,
            t: t.to_owned(),
        }
    }

    fn update_params(&mut self, delta_r: &Vector3<f64>, delta_t: &Vector3<f64>) {
        self.r += delta_r;
        self.t += delta_t;
        self.r_matrix = Camera::matrix_r(&self.r);
    }

    fn matrix_r(r: &Vector3<f64>) -> Matrix3<f64> {
        let theta = r.norm();
        if theta.abs() < f64::EPSILON {
            Matrix3::identity()
        } else {
            let u = r / theta;
            Matrix3::identity() * theta.cos()
                + (1.0 - theta.cos()) * u * u.transpose()
                + u.cross_matrix() * theta.sin()
        }
    }

    fn point_in_front(&self, point3d: &Vector3<f64>) -> bool {
        // This is how OpenMVG does it, works great!
        (self.r_matrix * (point3d + self.r_matrix.tr_mul(&self.t))).z > 0.0
    }

    fn projection(&self) -> Matrix3x4<f64> {
        let mut projection = self.r_matrix.insert_column(3, 0.0);
        projection.column_mut(3).copy_from(&self.t);
        self.k * projection
    }
}

struct PerspectiveTriangulation {
    calibration: Vec<Matrix3<f64>>,
    projections: Vec<Matrix3x4<f64>>,
    cameras: Vec<Camera>,
    tracks: Vec<Track>,
    image_shapes: Vec<(usize, usize)>,
    scale: (f64, f64, f64),
    bundle_adjustment: bool,
}

impl PerspectiveTriangulation {
    fn triangulate<PL: ProgressListener>(
        &mut self,
        correlated_points: &DMatrix<Option<Match>>,
        fundamental_matrix: &Matrix3<f64>,
        progress_listener: Option<&PL>,
    ) -> Result<(), TriangulationError> {
        let index = self.projections.len().saturating_sub(1);
        self.extend_tracks(correlated_points, index);

        if self.projections.is_empty() {
            let (k1, k2) = if self.calibration.len() == 2 {
                (self.calibration[0], self.calibration[1])
            } else {
                return Err(TriangulationError::new("Missing calibration matrix"));
            };
            let p1 = k1 * Matrix3x4::identity();
            let camera1 = Camera::from_matrix(&k1, &Matrix3::identity(), &Vector3::zeros());
            self.projections.push(p1);
            let p2 = match PerspectiveTriangulation::find_projection_matrix(
                fundamental_matrix,
                &k1,
                &k2,
                correlated_points,
            ) {
                Some(p2) => p2,
                None => return Err(TriangulationError::new("Unable to find projection matrix")),
            };
            let camera2_r = p2.fixed_view::<3, 3>(0, 0);
            let camera2_t = p2.column(3);
            let camera2 = Camera::from_matrix(&k2, &camera2_r.into(), &camera2_t.into());

            let p2 = k2 * p2;
            self.projections.push(p2);
            self.cameras = vec![camera1.clone(), camera2];
            self.triangulate_tracks();
        } else {
            let k2 = if self.calibration.len() == self.projections.len() + 1 {
                self.calibration[self.projections.len()]
            } else {
                return Err(TriangulationError::new("Missing calibration matrix"));
            };
            let k2_inv = match k2.pseudo_inverse(f64::EPSILON) {
                Ok(k_inverse) => k_inverse,
                Err(_) => {
                    return Err(TriangulationError::new(
                        "Unable to invert calibration matrix",
                    ))
                }
            };
            let camera2 = match self.recover_relative_pose(&k2, &k2_inv, progress_listener) {
                Some(camera2) => camera2,
                None => return Err(TriangulationError::new("Unable to find projection matrix")),
            };
            self.cameras.push(camera2);
            self.projections = self
                .cameras
                .iter()
                .map(|camera| camera.projection())
                .collect();

            self.triangulate_tracks();
        }

        Ok(())
    }

    fn push_calibration(&mut self, k1: &Matrix3<f64>, k2: &Matrix3<f64>) {
        if self.calibration.is_empty() {
            self.calibration.push(k1.to_owned());
        }
        self.calibration.push(k2.to_owned());
    }

    fn triangulate_all<PL: ProgressListener>(
        &mut self,
        progress_listener: Option<&PL>,
    ) -> Result<Surface, TriangulationError> {
        if self.bundle_adjustment {
            self.bundle_adjustment(progress_listener)?;
        }
        //self.filter_outliers(progress_listener);

        let surface = self
            .tracks
            .iter()
            .par_bridge()
            .flat_map(|track| {
                let index = track.range().start;
                let point2d = track.first()?;
                let point2d = (point2d.1 as usize, point2d.0 as usize);

                let point3d = track.point3d?;

                let point = Point::new(point2d, point3d, index);
                Some(point)
            })
            .collect::<Vec<_>>();

        Ok(self.scale_points(surface))
    }

    #[inline]
    fn triangulate_track(track: &Track, projections: &[Matrix3x4<f64>]) -> Option<Vector4<f64>> {
        let points_projection = track
            .range()
            .flat_map(|i| {
                if i < projections.len() {
                    let point = track.get(i)?;
                    let point = (point.1 as f64, point.0 as f64);
                    Some((point, projections[i]))
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();

        if points_projection.len() < 2 {
            return None;
        }

        let mut a = MatrixXx4::zeros(points_projection.len() * 2);
        for (i, (point, projection)) in points_projection.iter().enumerate() {
            a.row_mut(i * 2)
                .copy_from(&(projection.row(2) * point.0 - projection.row(0)));
            a.row_mut(i * 2 + 1)
                .copy_from(&(projection.row(2) * point.1 - projection.row(1)));
        }

        let usv = a.svd(false, true);
        let vt = usv.v_t?;
        let point4d = vt.row(vt.nrows() - 1).transpose();

        if point4d.w.abs() < PERSPECTIVE_SCALE_THRESHOLD {
            return None;
        }

        Some(point4d)
    }

    fn triangulate_tracks(&mut self) {
        self.tracks.par_iter_mut().for_each(|track| {
            // All existing triangulated points will be overwritten.
            track.point3d = PerspectiveTriangulation::triangulate_track(track, &self.projections)
                .map(|point4d| point4d.remove_row(3).unscale(point4d.w));
        });
        self.prune_tracks();
    }

    fn find_projection_matrix(
        fundamental_matrix: &Matrix3<f64>,
        k1: &Matrix3<f64>,
        k2: &Matrix3<f64>,
        correlated_points: &DMatrix<Option<Match>>,
    ) -> Option<Matrix3x4<f64>> {
        // Create essential matrix and camera matrices.
        let essential_matrix = k2.tr_mul(fundamental_matrix) * k1;
        let svd = essential_matrix.svd(true, true);
        let essential_matrix =
            svd.u? * Matrix3::from_diagonal(&Vector3::new(1.0, 1.0, 0.0)) * svd.v_t?;

        // Create camera matrices and find one which one works best.
        let svd = essential_matrix.svd(true, true);
        let u = svd.u?;
        let vt = svd.v_t?;
        let u3 = u.column(2);
        const W: Matrix3<f64> = Matrix3::new(0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
        let mut r1 = u * (W) * vt;
        let mut r2 = u * (W.transpose()) * vt;
        r1.scale_mut(r1.determinant().signum());
        r2.scale_mut(r2.determinant().signum());

        let p1 = k1 * Matrix3x4::identity();

        // Solve cheirality and find the matrix that the most points in front of the image.
        let combinations: [(Matrix3<f64>, Vector3<f64>); 4] =
            [(r1, u3.into()), (r1, -u3), (r2, u3.into()), (r2, -u3)];
        combinations
            .into_iter()
            .map(|(r, t)| {
                let mut p2 = Matrix3x4::zeros();
                p2.fixed_view_mut::<3, 3>(0, 0).copy_from(&r);
                p2.column_mut(3).copy_from(&t);
                let p2_calibrated = k2 * p2;
                let camera2 = Camera::from_matrix(&k2, &r, &t);
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
                                    (point2.0 as u32, point2.1 as u32)
                                } else {
                                    return false;
                                };
                                let track = Track {
                                    start_index: 0,
                                    points: vec![point1, point2],
                                    point3d: None,
                                };
                                let point4d = PerspectiveTriangulation::triangulate_track(
                                    &track,
                                    &[p1, p2_calibrated],
                                );
                                let point4d = if let Some(point4d) = point4d {
                                    point4d
                                } else {
                                    return false;
                                };
                                let point3d = point4d.remove_row(3).unscale(point4d.w);
                                point3d.z > 0.0 && camera2.point_in_front(&point3d)
                            })
                            .count()
                    })
                    .sum();
                (p2, points_count)
            })
            .max_by(|r1, r2| r1.1.cmp(&r2.1))
            .map(|(p2, _)| p2)
    }

    fn recover_relative_pose<PL: ProgressListener>(
        &self,
        k: &Matrix3<f64>,
        k_inv: &Matrix3<f64>,
        progress_listener: Option<&PL>,
    ) -> Option<Camera> {
        let track_len = self.projections.len() + 1;
        let linked_track_len = if self.projections.len() > 1 { 2 } else { 1 };
        // Gaku Nakano, "A Simple Direct Solution to the Perspective-Three-Point Problem," BMVC2019

        let unlinked_tracks = self
            .tracks
            .iter()
            .filter(|track| track.range().len() == 2 && track.range().end == track_len)
            .map(|track| track.to_owned())
            .collect::<Vec<_>>();

        let linked_tracks = self
            .tracks
            .iter()
            .filter(|track| {
                let view_i = self.image_shapes.len() - 1;
                track.range().len() > linked_track_len
                    && track.range().end == track_len
                    && track.get(view_i).is_some()
                    && track.point3d.is_some()
            })
            .map(|track| track.to_owned())
            .collect::<Vec<_>>();

        let ransac_outer = RANSAC_K / RANSAC_CHECK_INTERVAL;

        let mut result = None;
        let counter = AtomicUsize::new(0);
        let reduce_best_result = |(c1, count1, error1), (c2, count2, error2)| {
            if count1 > count2 || (count1 == count2 && error1 < error2) {
                (c1, count1, error1)
            } else {
                (c2, count2, error2)
            }
        };

        for _ in 0..ransac_outer {
            let (camera, count, _error) = (0..RANSAC_CHECK_INTERVAL)
                .par_bridge()
                .filter_map(|_| {
                    if let Some(pl) = progress_listener {
                        let value =
                            counter.fetch_add(1, AtomicOrdering::Relaxed) as f32 / RANSAC_K as f32;
                        pl.report_status(value);
                    }
                    let rng = &mut SmallRng::from_rng(rand::thread_rng()).ok()?;

                    // Select points
                    let inliers = linked_tracks
                        .choose_multiple(rng, RANSAC_N)
                        .collect::<Vec<_>>();
                    if inliers.len() != RANSAC_N {
                        return None;
                    }

                    let inliers_tracks = inliers
                        .iter()
                        .map(|track| (*track).to_owned())
                        .collect::<Vec<_>>();

                    PerspectiveTriangulation::recover_pose_from_points(k_inv, inliers.as_slice())
                        .into_iter()
                        .filter_map(|(r, t)| {
                            let camera = Camera::from_matrix(k, &r, &t);
                            let projection = camera.projection();

                            let mut projections = self.projections.clone();
                            projections.push(projection);

                            let (count, _) = PerspectiveTriangulation::tracks_reprojection_error(
                                &inliers_tracks,
                                &projections,
                                true,
                            );
                            if count != RANSAC_N {
                                return None;
                            }

                            let (count, error) =
                                PerspectiveTriangulation::tracks_reprojection_error(
                                    &unlinked_tracks,
                                    &projections,
                                    false,
                                );
                            Some((camera, count, error / (count as f64)))
                        })
                        .reduce(reduce_best_result)
                })
                .reduce(
                    || {
                        (
                            Camera::from_matrix(k, &Matrix3::identity(), &Vector3::zeros()),
                            0,
                            f64::MAX,
                        )
                    },
                    reduce_best_result,
                );

            if count >= RANSAC_D {
                result = Some(camera)
            }
            if count >= RANSAC_D_EARLY_EXIT {
                break;
            }
        }

        result
    }

    fn recover_pose_from_points(
        k_inv: &Matrix3<f64>,
        inliers: &[&Track],
    ) -> Vec<(Matrix3<f64>, Vector3<f64>)> {
        let mut inliers = inliers
            .iter()
            .filter_map(|track| {
                let p2 = track.last()?;
                let p2 = (k_inv * Vector3::new(p2.1 as f64, p2.0 as f64, 1.0)).normalize();
                let point3d = track.point3d?;

                Some((p2, point3d))
            })
            .collect::<Vec<_>>();

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
            .flat_map(|xy| {
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

                if !r.norm().is_finite() || !t.norm().is_finite() {
                    // Sometimes the estimated pose contains NaNs, and cause SVD to enter an infinite loop
                    return None;
                }

                Some((r, t))
            })
            .collect()
    }

    fn tracks_reprojection_error(
        tracks: &[Track],
        projections: &[Matrix3x4<f64>],
        inliers: bool,
    ) -> (usize, f64) {
        let threshold = if inliers { RANSAC_INLIERS_T } else { RANSAC_T };
        // For inliers, check reprojection error only in the last images.
        // For normal points, ignore the first images.
        let skip: usize = if inliers || projections.len() <= 2 {
            projections.len() - 1
        } else {
            2
        };
        tracks
            .iter()
            .filter_map(|track| {
                PerspectiveTriangulation::point_reprojection_error(track, projections, skip)
                    .filter(|error| *error < threshold)
            })
            .fold((0, 0.0f64), |(count, error), match_error| {
                (count + 1, error.max(match_error))
            })
    }

    #[inline]
    fn point_reprojection_error(
        track: &Track,
        projections: &[Matrix3x4<f64>],
        skip: usize,
    ) -> Option<f64> {
        let point4d = PerspectiveTriangulation::triangulate_track(track, projections)?;
        projections
            .iter()
            .enumerate()
            .skip(skip)
            .filter_map(|(i, p)| {
                let original = track.get(i)?;
                let mut projected = p * point4d;
                projected.unscale_mut(projected.z);
                let dx = projected.x - original.1 as f64;
                let dy = projected.y - original.0 as f64;
                let error = (dx * dx + dy * dy).sqrt();
                Some(error)
            })
            .reduce(|acc, val| acc.max(val))
    }

    fn extend_tracks(&mut self, correlated_points: &DMatrix<Option<Match>>, index: usize) {
        let mut remaining_points = correlated_points.clone();

        self.tracks.iter_mut().for_each(|track| {
            let last_pos = track
                .last()
                .and_then(|last| correlated_points[(last.0 as usize, last.1 as usize)]);

            if let Some(last_pos) = last_pos {
                track.add(index + 1, last_pos);
                remaining_points[(last_pos.0 as usize, last_pos.1 as usize)] = None;
            };
        });

        let mut new_tracks = remaining_points
            .column_iter()
            .enumerate()
            .flat_map(|(col, start_col)| {
                start_col
                    .iter()
                    .enumerate()
                    .filter_map(|(row, m)| {
                        if m.is_none() {
                            return None;
                        }
                        let mut track = Track::new(index, (row as u32, col as u32));
                        track.add(index + 1, (*m)?);

                        Some(track)
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        self.tracks.append(&mut new_tracks);

        if self.image_shapes.is_empty() {
            // Assume first image has the same shape as the second one
            self.image_shapes.push(correlated_points.shape());
        }
        self.image_shapes.push(correlated_points.shape());
    }

    fn prune_tracks(&mut self) {
        for img_i in 0..self.cameras.len() {
            let camera = &self.cameras[img_i];

            // Clear points which are in the back of the camers.
            self.tracks.par_iter_mut().for_each(|track| {
                let point3d = if let Some(point3d) = track.point3d {
                    point3d
                } else {
                    return;
                };
                if !camera.point_in_front(&point3d) {
                    track.point3d = None;
                }
            });

            // Remove tracks which have ended and have an invalid configuration.
            self.tracks
                .retain(|track| track.range().end >= self.cameras.len() || track.point3d.is_some());
            self.tracks.shrink_to_fit();
        }
    }

    fn bundle_adjustment<PL: ProgressListener>(
        &mut self,
        progress_listener: Option<&PL>,
    ) -> Result<(), TriangulationError> {
        // Only send tracks/points that could be triangulated.
        self.tracks.retain(|track| track.point3d.is_some());
        self.tracks.shrink_to_fit();
        let mut ba = BundleAdjustment::new(self.cameras.clone(), self.tracks.as_mut_slice());
        self.cameras = ba.optimize(progress_listener)?;

        Ok(())
    }

    fn filter_outliers<PL: ProgressListener>(&mut self, progress_listener: Option<&PL>) {
        // TODO: replace this with something better?
        let counter = AtomicUsize::new(0);
        let points_count = (self.cameras.len() * self.tracks.len()) as f32;
        for img_i in 0..self.cameras.len() {
            let camera = &self.cameras[img_i];
            let camera_r = camera.r_matrix;
            let camera_rt_t = camera_r.tr_mul(&camera.t);
            let (nrows, ncols) = self
                .tracks
                .iter()
                .flat_map(|track| {
                    let point = track.get(img_i)?;
                    Some((point.0 as usize, point.1 as usize))
                })
                .fold((usize::MIN, usize::MIN), |acc, (row, col)| {
                    (acc.0.max(row), acc.1.max(col))
                });

            let mut point_depths: DMatrix<Option<f64>> =
                DMatrix::from_element(nrows + 1, ncols + 1, None);

            self.tracks.iter().for_each(|track| {
                let point2d = if let Some(point) = track.get(img_i) {
                    (point.0 as usize, point.1 as usize)
                } else {
                    return;
                };
                let point3d = if let Some(point3d) = track.point3d {
                    point3d
                } else {
                    return;
                };
                let depth = (camera_r * (point3d + camera_rt_t)).z;
                point_depths[point2d] = Some(depth)
            });

            self.tracks.par_iter_mut().for_each(|track| {
                if let Some(pl) = progress_listener {
                    let value = counter.fetch_add(1, AtomicOrdering::Relaxed) as f32 / points_count;
                    pl.report_status(0.9 + 0.1 * value);
                }
                let point2d = if let Some(point) = track.get(img_i) {
                    (point.0 as usize, point.1 as usize)
                } else {
                    return;
                };
                if !point_not_outlier(&point_depths, point2d.0, point2d.1) {
                    track.point3d = None;
                }
            });
        }
    }

    fn scale_points(&self, points3d: Surface) -> Surface {
        let scale = &self.scale;

        points3d
            .iter()
            .map(|point| {
                let mut point = *point;
                let point3d = &mut point.reconstructed;
                point3d.x = scale.0 * point3d.x;
                point3d.y = scale.1 * point3d.y;
                point3d.z = scale.2 * point3d.z;

                point
            })
            .collect()
    }
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

fn polish_roots(f: [f64; 6], g: [f64; 6], xy: &mut [(f64, f64)]) {
    const MAX_ITER: usize = 3;
    for _ in 0..MAX_ITER {
        let mut stable = true;

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
            stable = false;

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
        if stable {
            break;
        }
    }
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

struct BundleAdjustment<'a> {
    cameras: Vec<Camera>,
    projections: Vec<Matrix3x4<f64>>,
    tracks: &'a mut [Track],
    covariance: f64,
    mu: f64,
}

impl BundleAdjustment<'_> {
    const CAMERA_PARAMETERS: usize = 6;
    const INITIAL_MU: f64 = 1E-3;
    const GRADIENT_EPSILON: f64 = 1E-12;
    const DELTA_EPSILON: f64 = 1E-12;
    const RESIDUAL_EPSILON: f64 = 1E-12;
    const RESIDUAL_REDUCTION_EPSILON: f64 = 0.0;
    const PARALLEL_CHUNK_SIZE: usize = 10000;

    fn new<'a>(cameras: Vec<Camera>, tracks: &'a mut [Track]) -> BundleAdjustment<'a> {
        // For now, identity covariance is acceptable.
        let covariance = 1.0;
        let projections = cameras.iter().map(|camera| camera.projection()).collect();
        BundleAdjustment {
            cameras,
            projections,
            tracks,
            covariance,
            mu: BundleAdjustment::INITIAL_MU,
        }
    }

    fn jacobian_a(&self, point3d: &Vector3<f64>, view_j: usize) -> Matrix2x6<f64> {
        // See BundleAdjustmentAnalytical.webarchive for more details (using chain rule).
        if view_j == 0 {
            // Exclude first camera from bundle adjustment.
            return Matrix2x6::zeros();
        }
        let camera = &self.cameras[view_j];
        let projection = &self.projections[view_j];
        let point4d = point3d.insert_row(3, 1.0);
        let point_projected = projection * point4d;

        let d_projection_hpoint =
            Matrix2x3::new(1.0, 0.0, -point_projected.x, 0.0, 1.0, -point_projected.y)
                .unscale(point_projected.z);
        let d_hpoint_translation = &camera.k;

        let mut d_translation_camerapose = Matrix3x6::zeros();
        // Derivative of rotation matrix, using formula from
        // "A compact formula for the derivative of a 3-D rotation in exponential coordinates" by Guillermo Gallego, Anthony Yezzi
        let u = &camera.r;
        let u_skewsymmetric = u.cross_matrix();
        if u.norm() > f64::EPSILON {
            for i in 0..3 {
                let mut e_i = Vector3::zeros();
                e_i[i] = 1.0;
                let d_r_i = (u[i] * u_skewsymmetric
                    + u.cross(&((Matrix3::identity() - camera.r_matrix) * e_i))
                        .cross_matrix())
                    * camera.r_matrix
                    / u.norm_squared();

                d_translation_camerapose
                    .column_mut(i)
                    .copy_from(&(d_r_i * point3d));
            }
        } else {
            // When near zero, derivative of R is equal to u_skewsymmetric.
            d_translation_camerapose
                .fixed_view_mut::<3, 3>(0, 0)
                .copy_from(&-u_skewsymmetric);
        }
        d_translation_camerapose
            .fixed_view_mut::<3, 3>(0, 3)
            .copy_from(&Matrix3::identity());

        d_projection_hpoint * d_hpoint_translation * d_translation_camerapose
    }

    fn jacobian_b(&self, point3d: &Vector3<f64>, view_j: usize) -> Matrix2x3<f64> {
        // See BundleAdjustmentAnalytical.webarchive for more details (using chain rule).
        let camera = &self.cameras[view_j];
        let projection = &self.projections[view_j];
        let point4d = point3d.insert_row(3, 1.0);
        let point_projected = projection * point4d;

        let d_projection_hpoint =
            Matrix2x3::new(1.0, 0.0, -point_projected.x, 0.0, 1.0, -point_projected.y)
                .unscale(point_projected.z);

        let d_hpoint_translation = &camera.k;
        let d_translation_camerapose = &camera.r_matrix;

        d_projection_hpoint * d_hpoint_translation * d_translation_camerapose
    }

    #[inline]
    fn residual(&self, point_i: usize, view_j: usize) -> Vector2<f64> {
        let point3d = &self.tracks[point_i].point3d;
        let projection = &self.projections[view_j];
        let original = if let Some(original) = self.tracks[point_i].get(view_j) {
            original
        } else {
            return Vector2::zeros();
        };
        if let Some(point3d) = point3d {
            let point4d = point3d.insert_row(3, 1.0);
            let mut projected = projection * point4d;
            projected.unscale_mut(projected.z);
            let dx = original.1 as f64 - projected.x;
            let dy = original.0 as f64 - projected.y;

            Vector2::new(dx, dy)
        } else {
            Vector2::zeros()
        }
    }

    #[inline]
    fn calculate_v_inv(&self, point3d: &Option<Vector3<f64>>) -> Option<Matrix3<f64>> {
        let mut v = Matrix3::zeros();
        let point3d = (*point3d)?;
        for view_j in 0..self.cameras.len() {
            let b = self.jacobian_b(&point3d, view_j);
            v += b.transpose() * self.covariance * b;
        }
        for i in 0..3 {
            v[(i, i)] += self.mu;
        }
        match v.pseudo_inverse(f64::EPSILON) {
            Ok(v_inv) => Some(v_inv),
            Err(_) => None,
        }
    }

    fn calculate_residual_vector(&self) -> MatrixXx1<f64> {
        const PARALLEL_CHUNK_SIZE: usize = BundleAdjustment::PARALLEL_CHUNK_SIZE;
        let cameras_len = self.cameras.len();
        let tracks_len = self.tracks.len();
        let mut residuals = MatrixXx1::zeros(tracks_len * cameras_len * 2);

        for (track_i, _) in self.tracks.iter().enumerate().step_by(PARALLEL_CHUNK_SIZE) {
            let tracks_residuals = self
                .tracks
                .iter()
                .enumerate()
                .skip(track_i)
                .take(PARALLEL_CHUNK_SIZE)
                .par_bridge()
                .flat_map(|(track_i, track)| {
                    if track.point3d.is_none() {
                        return None;
                    }
                    let track_residuals = self
                        .cameras
                        .iter()
                        .enumerate()
                        .map(|(view_j, _)| self.residual(track_i, view_j))
                        .collect::<Vec<_>>();
                    Some((track_i, track_residuals))
                })
                .collect::<Vec<_>>();

            tracks_residuals
                .iter()
                .for_each(|(track_i, track_residuals)| {
                    for (view_j, residual) in track_residuals.iter().enumerate() {
                        residuals
                            .fixed_rows_mut::<2>(track_i * cameras_len * 2 + view_j * 2)
                            .copy_from(residual);
                    }
                });
        }

        residuals
    }

    fn calculate_jt_residual(&self) -> MatrixXx1<f64> {
        const PARALLEL_CHUNK_SIZE: usize = BundleAdjustment::PARALLEL_CHUNK_SIZE;
        let camera_residuals_len = self.cameras.len() * BundleAdjustment::CAMERA_PARAMETERS;
        let track_residuals_len = self.tracks.len() * 3;

        // gradient = Jt * residual
        let mut g = MatrixXx1::zeros(camera_residuals_len + track_residuals_len);
        for (track_i, _) in self.tracks.iter().enumerate().step_by(PARALLEL_CHUNK_SIZE) {
            let tracks_gradients = self
                .tracks
                .iter()
                .enumerate()
                .skip(track_i)
                .take(PARALLEL_CHUNK_SIZE)
                .par_bridge()
                .flat_map(|(track_i, track)| {
                    let point_i = if let Some(point_i) = &track.point3d {
                        point_i
                    } else {
                        return None;
                    };
                    let track_residuals = self
                        .cameras
                        .iter()
                        .enumerate()
                        .map(|(view_j, _)| {
                            let residual_i_j = self.residual(track_i, view_j);
                            let jac_a_i = self.jacobian_a(point_i, view_j);
                            let camera_jt_residual = jac_a_i.tr_mul(&residual_i_j);

                            let jac_b_i = self.jacobian_b(point_i, view_j);
                            let point_jt_residual = jac_b_i.tr_mul(&residual_i_j);

                            (camera_jt_residual, point_jt_residual)
                        })
                        .collect::<Vec<_>>();
                    Some((track_i, track_residuals))
                })
                .collect::<Vec<_>>();

            tracks_gradients
                .iter()
                .for_each(|(track_i, track_residuals)| {
                    for (view_j, (camera_residual, point_residual)) in
                        track_residuals.iter().enumerate()
                    {
                        // First 6*m rows of Jt are residuals from camera matrices.
                        let mut target_block =
                            g.fixed_rows_mut::<6>(view_j * BundleAdjustment::CAMERA_PARAMETERS);
                        target_block += camera_residual;
                        // Last 3*n rows of Jt are residuals from point coordinates.
                        let mut target_block =
                            g.fixed_rows_mut::<3>(camera_residuals_len + track_i * 3);
                        target_block += point_residual;
                    }
                });
        }

        g
    }

    fn calculate_delta_step(&self) -> Option<MatrixXx1<f64>> {
        const PARALLEL_CHUNK_SIZE: usize = BundleAdjustment::PARALLEL_CHUNK_SIZE;
        let (mut s, e) = self
            .tracks
            .iter()
            .enumerate()
            .par_bridge()
            .flat_map(|(track_i, track)| {
                let point3d = track.point3d?;
                let v_inv = self.calculate_v_inv(&track.point3d)?;

                let mut e =
                    MatrixXx1::zeros(self.cameras.len() * BundleAdjustment::CAMERA_PARAMETERS);
                let mut s = DMatrix::<f64>::zeros(
                    self.cameras.len() * BundleAdjustment::CAMERA_PARAMETERS,
                    self.cameras.len() * BundleAdjustment::CAMERA_PARAMETERS,
                );
                for view_j in 0..self.cameras.len() {
                    let jac_a_j = self.jacobian_a(&point3d, view_j);
                    let jac_b_j = self.jacobian_b(&point3d, view_j);
                    let w_ij = jac_a_j.tr_mul(&jac_b_j) * self.covariance;
                    let u_j = jac_a_j.tr_mul(&jac_a_j) * self.covariance;
                    let y_ij = w_ij * v_inv;
                    for view_k in 0..self.cameras.len() {
                        let jac_ak = self.jacobian_a(&point3d, view_k);
                        let jac_bk = self.jacobian_b(&point3d, view_k);
                        let w_ik = jac_ak.tr_mul(&jac_bk) * self.covariance;
                        let mut s_jk = s.fixed_view_mut::<6, 6>(
                            view_j * BundleAdjustment::CAMERA_PARAMETERS,
                            view_k * BundleAdjustment::CAMERA_PARAMETERS,
                        );
                        if view_j == view_k {
                            s_jk += u_j;
                        }
                        s_jk -= y_ij * (w_ik.transpose());
                    }

                    let res_ij = self.residual(track_i, view_j);
                    let residual_a_ij = jac_a_j.tr_mul(&res_ij) * self.covariance;
                    let residual_b_ij = jac_b_j.tr_mul(&res_ij) * self.covariance;

                    let mut e_j =
                        e.fixed_rows_mut::<6>(view_j * BundleAdjustment::CAMERA_PARAMETERS);
                    e_j.copy_from(&(residual_a_ij - y_ij * residual_b_ij));
                }
                Some((s, e))
            })
            .reduce(
                || {
                    (
                        DMatrix::<f64>::zeros(
                            self.cameras.len() * BundleAdjustment::CAMERA_PARAMETERS,
                            self.cameras.len() * BundleAdjustment::CAMERA_PARAMETERS,
                        ),
                        MatrixXx1::zeros(self.cameras.len() * BundleAdjustment::CAMERA_PARAMETERS),
                    )
                },
                |a, b| (a.0 + b.0, a.1 + b.1),
            );

        for view_i in 0..self.cameras.len() {
            let mut s_jk = s.fixed_view_mut::<6, 6>(
                view_i * BundleAdjustment::CAMERA_PARAMETERS,
                view_i * BundleAdjustment::CAMERA_PARAMETERS,
            );
            s_jk += Matrix6::identity() * self.mu;
        }

        let delta_a_len = self.cameras.len() * BundleAdjustment::CAMERA_PARAMETERS;
        let delta_b_len = self.tracks.len() * 3;

        let delta_a = s.lu().solve(&e)?;

        let mut delta = MatrixXx1::zeros(delta_a_len + delta_b_len);
        delta.rows_mut(0, delta_a_len).copy_from(&delta_a);
        let mut delta_b = delta.rows_mut(delta_a_len, delta_b_len);
        for (track_i, _) in self.tracks.iter().enumerate().step_by(PARALLEL_CHUNK_SIZE) {
            let tracks_delta_b = self
                .tracks
                .iter()
                .enumerate()
                .skip(track_i)
                .take(PARALLEL_CHUNK_SIZE)
                .par_bridge()
                .flat_map(|(track_i, track)| {
                    let point3d = track.point3d?;
                    let v_inv = self.calculate_v_inv(&track.point3d)?;

                    let mut delta_b_i = Vector3::zeros();
                    for view_j in 0..self.cameras.len() {
                        let jac_a_j = self.jacobian_a(&point3d, view_j);
                        let jac_b_j = self.jacobian_b(&point3d, view_j);
                        let w_ij = jac_a_j.tr_mul(&jac_b_j) * self.covariance;

                        let res_ij = self.residual(track_i, view_j);
                        let residual_b_ij = jac_b_j.tr_mul(&res_ij) * self.covariance;

                        let delta_a_j =
                            delta_a.fixed_rows::<6>(view_j * BundleAdjustment::CAMERA_PARAMETERS);

                        delta_b_i += v_inv * residual_b_ij - v_inv * w_ij.tr_mul(&delta_a_j);
                    }
                    Some((track_i, delta_b_i))
                })
                .collect::<Vec<_>>();

            tracks_delta_b.iter().for_each(|(track_i, delta_b_i)| {
                let mut delta_b = delta_b.fixed_rows_mut::<3>(track_i * 3);
                delta_b.copy_from(&delta_b_i);
            });
        }

        Some(delta)
    }

    fn update_params(&mut self, delta: &MatrixXx1<f64>) {
        for view_j in 0..self.cameras.len() {
            let camera = &mut self.cameras[view_j];
            let delta_r = delta.fixed_rows::<3>(BundleAdjustment::CAMERA_PARAMETERS * view_j);
            let delta_t = delta.fixed_rows::<3>(BundleAdjustment::CAMERA_PARAMETERS * view_j + 3);
            camera.update_params(&delta_r.into(), &delta_t.into());
        }
        self.projections = self
            .cameras
            .iter()
            .map(|camera| camera.projection())
            .collect();

        let points_source = delta.rows(
            self.cameras.len() * BundleAdjustment::CAMERA_PARAMETERS,
            self.tracks.len() * 3,
        );

        for (i, track_i) in self.tracks.iter_mut().enumerate() {
            let point_i = if let Some(point_i) = &track_i.point3d {
                point_i
            } else {
                continue;
            };

            let point_source = points_source.fixed_rows::<3>(i * 3);
            track_i.point3d = Some(point_i + Vector3::from(point_source));
        }
    }

    fn optimize<PL: ProgressListener>(
        &mut self,
        progress_listener: Option<&PL>,
    ) -> Result<Vec<Camera>, TriangulationError> {
        // Levenberg-Marquardt optimization loop.
        let mut residual = self.calculate_residual_vector();
        let mut jt_residual = self.calculate_jt_residual();

        if jt_residual.max().abs() <= BundleAdjustment::GRADIENT_EPSILON {
            return Ok(self.cameras.clone());
        }

        self.mu = BundleAdjustment::INITIAL_MU;
        let mut nu = 2.0;
        let mut found = false;
        for iter in 0..BUNDLE_ADJUSTMENT_MAX_ITERATIONS {
            if let Some(pl) = progress_listener {
                let value = iter as f32 / BUNDLE_ADJUSTMENT_MAX_ITERATIONS as f32;
                pl.report_status(value);
            }
            let delta = if let Some(delta) = self.calculate_delta_step() {
                delta
            } else {
                return Err(TriangulationError::new("Failed to compute delta vector"));
            };

            let params_norm;
            {
                let sum_cameras = self
                    .cameras
                    .iter()
                    .map(|camera| camera.r.norm_squared() + camera.t.norm_squared())
                    .sum::<f64>();
                let sum_points = self
                    .tracks
                    .iter()
                    .filter_map(|track| Some(track.point3d?.norm_squared()))
                    .sum::<f64>();
                params_norm = (sum_cameras + sum_points).sqrt();
            }

            if delta.norm()
                <= BundleAdjustment::DELTA_EPSILON * (params_norm + BundleAdjustment::DELTA_EPSILON)
            {
                found = true;
                break;
            }

            let current_cameras = self.cameras.clone();
            let current_projections = self.projections.clone();
            let current_points3d = self
                .tracks
                .iter()
                .map(|track| track.point3d)
                .collect::<Vec<_>>();

            self.update_params(&delta);

            let new_residual = self.calculate_residual_vector();
            let residual_norm_squared = residual.norm_squared();
            let new_residual_norm_squared = new_residual.norm_squared();

            let rho = (residual_norm_squared - new_residual_norm_squared)
                / (delta.tr_mul(&(delta.scale(self.mu) + &jt_residual)))[0];

            if rho > 0.0 {
                let converged = residual_norm_squared.sqrt() - new_residual_norm_squared.sqrt()
                    < BundleAdjustment::RESIDUAL_REDUCTION_EPSILON * residual_norm_squared.sqrt();

                residual = new_residual;
                jt_residual = self.calculate_jt_residual();
                if converged || jt_residual.max().abs() <= BundleAdjustment::GRADIENT_EPSILON {
                    found = true;
                    break;
                }
                self.mu *= (1.0f64 / 3.0).max(1.0 - (2.0 * rho - 1.0).powf(3.0));
                nu = 2.0;
            } else {
                self.cameras = current_cameras;
                self.projections = current_projections;
                self.tracks
                    .iter_mut()
                    .zip(current_points3d)
                    .for_each(|(track, point3d)| track.point3d = point3d);
                self.mu *= nu;
                nu *= 2.0;
            }

            if residual.norm() <= BundleAdjustment::RESIDUAL_EPSILON {
                found = true;
                break;
            }
        }

        if !found {
            return Err(TriangulationError::new(
                "Levenberg-Marquardt failed to converge",
            ));
        }

        Ok(self.cameras.clone())
    }
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
