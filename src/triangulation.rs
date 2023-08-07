use std::{fmt, ops::Range, sync::atomic::AtomicUsize, sync::atomic::Ordering as AtomicOrdering};

use nalgebra::{
    DMatrix, Matrix2x3, Matrix2x6, Matrix3, Matrix3x4, Matrix6, Matrix6x1, Matrix6x3, MatrixXx1,
    MatrixXx4, Vector2, Vector3, Vector4,
};

use rand::seq::SliceRandom;
use rand::{rngs::SmallRng, SeedableRng};
use rayon::prelude::*;

use crate::crosscorrelation;

const PERSPECTIVE_VALUE_RANGE: f64 = 100.0;
const BUNDLE_ADJUSTMENT_MAX_ITERATIONS: usize = 1000;
const OUTLIER_FILTER_STDEV_THRESHOLD: f64 = 1.0;
const OUTLIER_FILTER_SEARCH_AREA: usize = 5;
const OUTLIER_FILTER_MIN_NEIGHBORS: usize = 10;
const PERSPECTIVE_SCALE_THRESHOLD: f64 = 0.001;
const RANSAC_N: usize = 3;
const RANSAC_K: usize = 100_000;
// TODO: this should be proportional to image size
const RANSAC_INLIERS_T: f64 = 3.0;
const RANSAC_T: f64 = 10.0;
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
                    calibration_matrix: Matrix3::identity(),
                    calibration_matrix_inv: Matrix3::identity(),
                    projections: vec![],
                    cameras: vec![],
                    tracks: vec![],
                    image_shapes: vec![],
                    points3d: vec![],
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
}

impl Track {
    fn new(start_index: usize, point: Match) -> Track {
        let points = vec![point];
        Track {
            start_index,
            points,
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
    r: Vector3<f64>,
    t: Vector3<f64>,
}

impl Camera {
    fn decode(r: &Matrix3<f64>, t: &Vector3<f64>) -> Camera {
        let (r_x, r_y, r_z) = if r[(0, 2)] < 1.0 {
            if r[(0, 2)] > -1.0 {
                let r_y = r[(0, 2)].asin();
                let r_x = (-r[(1, 2)]).atan2(r[(2, 2)]);
                let r_z = (-r[(0, 1)]).atan2(r[(0, 0)]);
                (r_x, r_y, r_z)
            } else {
                let r_y = -std::f64::consts::FRAC_PI_2;
                let r_x = r[(1, 0)].atan2(r[(0, 0)]);
                let r_z = 0.0;
                (r_x, r_y, r_z)
            }
        } else {
            let r_y = std::f64::consts::FRAC_PI_2;
            let r_x = r[(1, 0)].atan2(r[(0, 0)]);
            let r_z = 0.0;
            (r_x, r_y, r_z)
        };
        let r = Vector3::new(r_x, r_y, r_z);

        Camera { r, t: t.to_owned() }
    }

    fn matrix_r(&self) -> Matrix3<f64> {
        let (r_x, r_y, r_z) = (self.r.x, self.r.y, self.r.z);
        let rotation_x = Matrix3::new(
            1.0,
            0.0,
            0.0,
            0.0,
            r_x.cos(),
            -r_x.sin(),
            0.0,
            r_x.sin(),
            r_x.cos(),
        );
        let rotation_y = Matrix3::new(
            r_y.cos(),
            0.0,
            r_y.sin(),
            0.0,
            1.0,
            0.0,
            -r_y.sin(),
            0.0,
            r_y.cos(),
        );
        let rotation_z = Matrix3::new(
            r_z.cos(),
            -r_z.sin(),
            0.0,
            r_z.sin(),
            r_z.cos(),
            0.0,
            0.0,
            0.0,
            1.0,
        );

        rotation_x * rotation_y * rotation_z
    }

    fn projection(&self) -> Matrix3x4<f64> {
        let mut projection = self.matrix_r().insert_column(3, 0.0);
        projection.column_mut(3).copy_from(&self.t);
        projection
    }
}

struct PerspectiveTriangulation {
    calibration_matrix: Matrix3<f64>,
    calibration_matrix_inv: Matrix3<f64>,
    projections: Vec<Matrix3x4<f64>>,
    cameras: Vec<Camera>,
    tracks: Vec<Track>,
    image_shapes: Vec<(usize, usize)>,
    points3d: Vec<Option<Vector3<f64>>>,
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

        // TODO 0.17 Get focal distance from EXIF metadata.
        let focal_distance =
            26.0 / 36.0 * correlated_points.nrows().max(correlated_points.ncols()) as f64;
        let k = Matrix3::new(
            focal_distance,
            0.0,
            correlated_points.ncols() as f64 / 2.0,
            0.0,
            focal_distance,
            correlated_points.nrows() as f64 / 2.0,
            0.0,
            0.0,
            1.0,
        );
        if self.projections.is_empty() {
            self.calibration_matrix = k;
            self.calibration_matrix_inv = if let Some(k_inverse) = k.lu().try_inverse() {
                k_inverse
            } else {
                return Err(TriangulationError::new(
                    "Unable to invert calibration matrix",
                ));
            };

            // For the first pair, find a temporary projection matrix to estimate 3D points.
            self.projections.push(Matrix3x4::identity());
            let p2 = match self.find_projection_matrix(fundamental_matrix, correlated_points) {
                Some(p2) => p2,
                None => return Err(TriangulationError::new("Unable to find projection matrix")),
            };
            self.projections.push(p2);
            self.triangulate_tracks();

            // First pair shoudn't be re-calibrated, as it's used to define all other structure.
            let camera_r = p2.fixed_view::<3, 3>(0, 0);
            let camera_t = p2.column(3);
            self.cameras = vec![Camera::decode(&Matrix3::identity(), &Vector3::zeros())];
            self.projections = vec![Matrix3x4::identity()];
        }
        let camera2 = match self.recover_relative_pose(progress_listener) {
            Some(camera2) => camera2,
            None => return Err(TriangulationError::new("Unable to find projection matrix")),
        };
        self.cameras.push(camera2);
        self.projections = self
            .cameras
            .iter()
            .map(|camera| camera.projection())
            .collect();

        self.points3d.clear();
        self.triangulate_tracks();

        Ok(())
    }

    fn triangulate_all<PL: ProgressListener>(
        &mut self,
        progress_listener: Option<&PL>,
    ) -> Result<Surface, TriangulationError> {
        if self.bundle_adjustment {
            self.bundle_adjustment(progress_listener)?;
        }

        //self.filter_outliers();

        let surface = self
            .tracks
            .iter()
            .enumerate()
            .par_bridge()
            .flat_map(|(i, track)| {
                let index = track.range().start;
                let point2d = track.first()?;
                let point2d = (point2d.1 as usize, point2d.0 as usize);

                let point3d = self.points3d[i]?;

                let point = Point::new(point2d, point3d, index);
                Some(point)
            })
            .collect::<Vec<_>>();

        Ok(self.scale_points(surface))
    }

    #[inline]
    fn triangulate_track(
        &self,
        track: &Track,
        projections: &[Matrix3x4<f64>],
    ) -> Option<Vector4<f64>> {
        let points_projection = track
            .range()
            .flat_map(|i| {
                if i < projections.len() {
                    let point = track.get(i)?;
                    let point = self.calibration_matrix_inv
                        * Vector3::new(point.1 as f64, point.0 as f64, 1.0);
                    let point = (point.x / point.z, point.y / point.z);
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

        Some(point4d)
    }

    fn triangulate_tracks(&mut self) {
        let mut new_points3d = self
            .tracks
            .par_iter()
            .skip(self.points3d.len())
            .map(|track: &Track| {
                let point4d = self.triangulate_track(track, &self.projections)?;

                if point4d.w.abs() < PERSPECTIVE_SCALE_THRESHOLD {
                    return None;
                }
                let w = point4d.w * point4d.z.signum() * point4d.w.signum();
                let point3d = point4d.remove_row(3).unscale(w);
                Some(point3d)
            })
            .collect::<Vec<_>>();
        self.points3d.append(&mut new_points3d)
    }

    fn find_projection_matrix(
        &self,
        fundamental_matrix: &Matrix3<f64>,
        correlated_points: &DMatrix<Option<Match>>,
    ) -> Option<Matrix3x4<f64>> {
        // Create essential matrix and camera matrices.
        let k = &self.calibration_matrix;
        let essential_matrix = k.tr_mul(fundamental_matrix) * k;
        let svd = essential_matrix.svd(true, true);
        let essential_matrix =
            svd.u? * Matrix3::from_diagonal(&Vector3::new(1.0, 1.0, 0.0)) * svd.v_t?;

        // Create camera matrices and find one where
        let svd = essential_matrix.svd(true, true);
        let u = svd.u?;
        let vt = svd.v_t?;
        let u3 = u.column(2).clone();
        let u3 = u3.unscale(u3[2]);
        const W: Matrix3<f64> = Matrix3::new(0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
        let mut r1 = u * (W) * vt;
        let mut r2 = u * (W.transpose()) * vt;
        if r1.determinant() < 0.0 {
            r1 = -r1;
        }
        if r2.determinant() < 0.0 {
            r2 = -r2;
        }

        let mut p2_1 = r1.insert_column(3, 0.0);
        let mut p2_2 = r1.insert_column(3, 0.0);
        let mut p2_3 = r2.insert_column(3, 0.0);
        let mut p2_4 = r2.insert_column(3, 0.0);

        let p1 = Matrix3x4::identity();

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
                                };
                                let point4d = self.triangulate_track(&track, &[p1, p2]);
                                let point4d = if let Some(point4d) = point4d {
                                    point4d.unscale(point4d.w)
                                } else {
                                    return false;
                                };
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

    fn recover_relative_pose<PL: ProgressListener>(
        &self,
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
            .enumerate()
            .filter(|(track_i, track)| {
                let view_i = self.image_shapes.len() - 1;
                track.range().len() > linked_track_len
                    && track.range().end == track_len
                    && track.get(view_i).is_some()
                    && self.points3d[*track_i].is_some()
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
                        .map(|(i, track)| (*i, *track))
                        .collect::<Vec<_>>();
                    if inliers.len() != RANSAC_N {
                        return None;
                    }

                    let inliers_tracks = inliers
                        .iter()
                        .map(|(_, track)| (*track).to_owned())
                        .collect::<Vec<_>>();

                    // TODO: check if points are collinear?
                    self.recover_pose_from_points(&inliers)
                        .into_iter()
                        .filter_map(|(r, t)| {
                            let camera = Camera::decode(&r, &t);
                            let projection = camera.projection();

                            let mut projections = self.projections.clone();
                            projections.push(projection);

                            let (count, _) =
                                self.tracks_reprojection_error(&inliers_tracks, &projections, true);
                            if count != RANSAC_N {
                                return None;
                            }

                            let (count, error) = self.tracks_reprojection_error(
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
                            Camera::decode(&Matrix3::identity(), &Vector3::zeros()),
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
        &self,
        inliers: &[(usize, &Track)],
    ) -> Vec<(Matrix3<f64>, Vector3<f64>)> {
        let mut inliers = inliers
            .iter()
            .filter_map(|(i, track)| {
                let p2 = track.last()?;
                let p2 = (self.calibration_matrix_inv
                    * Vector3::new(p2.1 as f64, p2.0 as f64, 1.0))
                .normalize();
                let point3d = (*(self.points3d.get(*i)?))?;

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
        &self,
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
                self.point_reprojection_error(track, projections, skip)
                    .filter(|error| *error < threshold)
            })
            .fold((0, 0.0f64), |(count, error), match_error| {
                (count + 1, error.max(match_error))
            })
    }

    #[inline]
    fn point_reprojection_error(
        &self,
        track: &Track,
        projections: &[Matrix3x4<f64>],
        skip: usize,
    ) -> Option<f64> {
        let point4d = self.triangulate_track(track, projections)?;
        projections
            .iter()
            .enumerate()
            .skip(skip)
            .filter_map(|(i, p)| {
                let original = track.get(i)?;
                let mut original = self.calibration_matrix_inv
                    * Vector3::new(original.1 as f64, original.0 as f64, 1.0);
                original.unscale_mut(original.z);
                let mut projected = p * point4d;
                projected.unscale_mut(projected.z * projected.z.signum());
                let dx = projected.x - original.x;
                let dy = projected.y - original.y;
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

    fn bundle_adjustment<PL: ProgressListener>(
        &mut self,
        progress_listener: Option<&PL>,
    ) -> Result<(), TriangulationError> {
        // Only send tracks/points that could be triangulated.
        let filtered_tracks = self
            .tracks
            .iter()
            .enumerate()
            .filter(|(i, _)| self.points3d[*i].is_some())
            .map(|(_, track)| track.to_owned())
            .collect::<Vec<_>>();
        let mut filtered_points3d = self
            .points3d
            .iter()
            .filter(|point3d| point3d.is_some())
            .map(|point3d| point3d.to_owned())
            .collect::<Vec<_>>();
        let mut ba = BundleAdjustment::new(
            self.cameras.clone(),
            &filtered_tracks,
            &mut filtered_points3d,
        );
        self.cameras = ba.optimize(progress_listener)?;
        drop(ba);
        drop(filtered_tracks);

        // Write pack updated 3D points.
        let iter_dest = self.points3d.iter_mut().filter(|point3d| point3d.is_some());
        let iter_src = filtered_points3d.into_iter();
        iter_dest.zip(iter_src).for_each(|(dest, src)| *dest = src);

        Ok(())
    }

    fn filter_outliers(&mut self) {
        // TODO: replace this with something better?
        for img_i in 0..self.cameras.len() {
            let projection = self.cameras[img_i].projection();
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

            self.tracks.iter().enumerate().for_each(|(i, track)| {
                let point2d = if let Some(point) = track.get(img_i) {
                    (point.0 as usize, point.1 as usize)
                } else {
                    return;
                };
                let point3d = if let Some(point3d) = self.points3d[i] {
                    point3d.insert_row(3, 1.0)
                } else {
                    return;
                };
                let depth = (projection * point3d).z;
                point_depths[point2d] = Some(depth)
            });

            self.points3d
                .iter_mut()
                .enumerate()
                .par_bridge()
                .for_each(|(i, point3d)| {
                    let track = &self.tracks[i];
                    let point2d = if let Some(point) = track.get(img_i) {
                        (point.0 as usize, point.1 as usize)
                    } else {
                        return;
                    };
                    if !point_not_outlier(&point_depths, point2d.0, point2d.1) {
                        *point3d = None;
                    }
                });
        }
    }

    fn scale_points(&self, points3d: Surface) -> Surface {
        let scale = &self.scale;

        let (min_x, max_x, min_y, max_y, min_z, max_z) = points3d.iter().fold(
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
        let range_adjusted = (max_x - min_x).max(max_y - min_y).max(max_z - min_z);

        points3d
            .iter()
            .map(|point| {
                let mut point = *point;
                let point3d = &mut point.reconstructed;
                point3d.x =
                    scale.0 * PERSPECTIVE_VALUE_RANGE * (point3d.x - min_x) / range_adjusted;
                point3d.y =
                    scale.1 * PERSPECTIVE_VALUE_RANGE * (point3d.y - min_y) / range_adjusted;
                point3d.z =
                    scale.2 * PERSPECTIVE_VALUE_RANGE * (point3d.z - min_z) / range_adjusted;

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
    const MAX_ITER: usize = 50;
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
    if !crosscorrelation::point_inside_bounds::<SEARCH_RADIUS>(img.shape(), row, col) {
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
    tracks: &'a [Track],
    points3d: &'a mut Vec<Option<Vector3<f64>>>,
    covariance: f64,
    mu: f64,
}

impl BundleAdjustment<'_> {
    const CAMERA_PARAMETERS: usize = 6;
    const INITIAL_MU: f64 = 1E-3;
    const JACOBIAN_H: f64 = 0.001;
    const GRADIENT_EPSILON: f64 = 1E-12;
    const DELTA_EPSILON: f64 = 1E-12;
    const RESIDUAL_EPSILON: f64 = 1E-12;
    const RESIDUAL_REDUCTION_EPSILON: f64 = 0.0;

    fn new<'a>(
        cameras: Vec<Camera>,
        tracks: &'a [Track],
        points3d: &'a mut Vec<Option<Vector3<f64>>>,
    ) -> BundleAdjustment<'a> {
        // For now, identity covariance is acceptable.
        let covariance = 1.0;
        let projections = cameras.iter().map(|camera| camera.projection()).collect();
        BundleAdjustment {
            cameras,
            projections,
            tracks,
            points3d,
            covariance,
            mu: BundleAdjustment::INITIAL_MU,
        }
    }

    fn jacobian_view(&self, camera: &Camera, point: &Vector4<f64>, param: usize) -> Vector2<f64> {
        let delta_r = match param {
            0 => Vector3::new(BundleAdjustment::JACOBIAN_H, 0.0, 0.0),
            1 => Vector3::new(0.0, BundleAdjustment::JACOBIAN_H, 0.0),
            2 => Vector3::new(0.0, 0.0, BundleAdjustment::JACOBIAN_H),
            _ => Vector3::zeros(),
        };
        let delta_t = match param {
            3 => Vector3::new(BundleAdjustment::JACOBIAN_H, 0.0, 0.0),
            4 => Vector3::new(0.0, BundleAdjustment::JACOBIAN_H, 0.0),
            5 => Vector3::new(0.0, 0.0, BundleAdjustment::JACOBIAN_H),
            _ => Vector3::zeros(),
        };
        let mut p_plus = camera.clone();
        p_plus.r += delta_r;
        p_plus.t += delta_t;
        let mut p_minus = camera.clone();
        p_minus.r -= delta_r;
        p_minus.t -= delta_t;

        // TODO: pre-compute projection matrices
        let p_plus = p_plus.projection();
        let p_minus = p_minus.projection();

        let projection_plus = p_plus * point;
        let projection_plus = projection_plus.remove_row(2).unscale(projection_plus.z);
        let projection_minus = p_minus * point;
        let projection_minus = projection_minus.remove_row(2).unscale(projection_minus.z);
        (projection_plus - projection_minus).unscale(2.0 * BundleAdjustment::JACOBIAN_H)
    }

    fn jacobian_a(&self, point3d: &Vector3<f64>, view_j: usize) -> Matrix2x6<f64> {
        // jac is a 2x6 Jacobian for point i and projection matrix parameter j.
        let mut jac = Matrix2x6::zeros();

        let camera = &self.cameras[view_j];
        let point4d = point3d.insert_row(3, 1.0);
        // Calculate Jacobian using finite differences (central difference)
        for i in 0..6 {
            jac.column_mut(i)
                .copy_from(&self.jacobian_view(camera, &point4d, i));
        }

        jac
    }

    fn jacobian_b(&self, point3d: &Vector3<f64>, view_j: usize) -> Matrix2x3<f64> {
        // Point coordinates matrix is converted to a single row with coordinates x, y and z.
        // jac is a 2x3 Jacobian for point i and projection matrix j.
        let mut jac = Matrix2x3::zeros();
        let point4d = point3d.insert_row(3, 1.0);

        let projection = self.projections[view_j];

        // Using a symbolic formula (not finite differences/central difference), check the Rust LM library for more info.
        // P is the projection matrix for image
        // Prc is the r-th row, c-th column of P
        // X is a 3D coordinate (4-component vector [x y z 1])

        // Image contains xpro and ypro (projected point coordinates), affected by projection matrix and the point coordinates.
        // xpro = (P11*x+P12*y+P13*z+P14)/(P31*x+P32*y+P33*z+P34)
        // ypro = (P21*x+P22*y+P23*z+P24)/(P31*x+P32*y+P33*z+P34)
        // To keep things sane, create some aliases
        // Poi -> i-th element of 3D point e.g. x, y, z, or w
        // Pr1 = P11*x+P12*y+P13*z+P14
        // Pr2 = P21*x+P22*y+P23*z+P24
        // Pr3 = P31*x+P32*y+P33*z+P34
        let p_r = projection * point4d;
        // dxpro/dx = (P11*(P32*y+P33*z+P34)-P31*(P12*y+P13*z+P14))/(Pr3^2) = (P11*Pr3[x=0]-P31*Pr1[x=0])/(Pr3^2)
        // dxpro/di = (P1i*Pr3[i=0]-P3i*Pr1[i=0])/(Pr3^2)
        // dypro/dx = (P21*(P32*y+P33*z+P34)-P31*(P22*y+P23*z+P24))/(Pr3^2) = (P21*Pr3[x=0]-P31*Pr2[x=0])/(Pr3^2)
        // dypro/di = (P2i*Pr3[i=0]-P3i*Pr2[i=0])/(Pr3^2)
        for coord in 0..3 {
            // Create a vector where coord = 0
            let mut vec_diff = point4d;
            vec_diff[coord] = 0.0;
            // Create projection where coord = 0
            let p_r_diff = projection * vec_diff;
            jac[(0, coord)] = (projection[(0, coord)] * p_r_diff[2]
                - projection[(2, coord)] * p_r_diff[0])
                / (p_r[2] * p_r[2]);
            jac[(1, coord)] = (projection[(1, coord)] * p_r_diff[2]
                - projection[(2, coord)] * p_r_diff[1])
                / (p_r[2] * p_r[2]);
        }

        jac
    }

    #[inline]
    fn residual(&self, point_i: usize, view_j: usize) -> Vector2<f64> {
        let point3d = &self.points3d[point_i];
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

    fn residual_a(&self, view_j: usize) -> Option<Matrix6x1<f64>> {
        let mut residual = Matrix6x1::zeros();
        for (point_i, point3d) in self.points3d.iter().enumerate() {
            let point3d = (*point3d)?;
            let a = self.jacobian_a(&point3d, view_j);
            let res = self.residual(point_i, view_j);
            residual += a.transpose() * self.covariance * res;
        }
        Some(residual)
    }

    fn residual_b(&self, point_i: usize) -> Option<Vector3<f64>> {
        let mut residual = Vector3::zeros();
        let point3d = &self.points3d[point_i];
        let point3d = (*point3d)?;
        for view_j in 0..self.cameras.len() {
            let b = self.jacobian_b(&point3d, view_j);
            let res = self.residual(point_i, view_j);
            residual += b.transpose() * self.covariance * res;
        }
        Some(residual)
    }

    #[inline]
    fn calculate_u(&self, view_j: usize) -> Option<Matrix6<f64>> {
        let mut u = Matrix6::zeros();
        for point3d in self.points3d.iter() {
            let point3d = (*point3d)?;
            let a = self.jacobian_a(&point3d, view_j);
            u += a.transpose() * self.covariance * a;
        }
        for i in 0..BundleAdjustment::CAMERA_PARAMETERS {
            u[(i, i)] += self.mu;
        }
        Some(u)
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

    #[inline]
    fn calculate_w(&self, point3d: &Option<Vector3<f64>>, view_j: usize) -> Option<Matrix6x3<f64>> {
        let point3d = (*point3d)?;
        let a = self.jacobian_a(&point3d, view_j);
        let b = self.jacobian_b(&point3d, view_j);
        Some(a.transpose() * self.covariance * b)
    }

    #[inline]
    fn calculate_y(&self, point3d: &Option<Vector3<f64>>, view_j: usize) -> Option<Matrix6x3<f64>> {
        let v_inv = self.calculate_v_inv(point3d)?;
        let w = self.calculate_w(point3d, view_j)?;
        Some(w * v_inv)
    }

    fn calculate_s(&self) -> DMatrix<f64> {
        let mut s = DMatrix::<f64>::zeros(
            self.cameras.len() * BundleAdjustment::CAMERA_PARAMETERS,
            self.cameras.len() * BundleAdjustment::CAMERA_PARAMETERS,
        );
        // Divide blocks for parallelization.
        let s_blocks = (0..self.cameras.len())
            .flat_map(|j| (0..self.cameras.len()).map(|k| (j, k)).collect::<Vec<_>>())
            .collect::<Vec<_>>();

        let s_blocks = s_blocks
            .into_par_iter()
            .map(|(view_j, view_k)| {
                let mut s_jk = Matrix6::zeros();
                if view_j == view_k {
                    let u = self.calculate_u(view_j)?;
                    s_jk += u;
                }
                for point3d in self.points3d.iter() {
                    let y_ij = self.calculate_y(point3d, view_j)?;
                    let w_ik = self.calculate_w(point3d, view_k)?;
                    let y_ij_w = y_ij * w_ik.transpose();
                    s_jk -= y_ij_w;
                }
                Some((view_j, view_k, s_jk))
            })
            .collect::<Vec<_>>();

        s_blocks.iter().for_each(|block| {
            let (j, k, s_jk) = if let Some((j, k, s_jk)) = block {
                (j, k, s_jk)
            } else {
                return;
            };
            s.fixed_view_mut::<6, 6>(
                j * BundleAdjustment::CAMERA_PARAMETERS,
                k * BundleAdjustment::CAMERA_PARAMETERS,
            )
            .copy_from(s_jk);
        });

        s
    }

    fn calculate_e(&self) -> MatrixXx1<f64> {
        let mut e = MatrixXx1::zeros(self.cameras.len() * BundleAdjustment::CAMERA_PARAMETERS);

        let e_blocks = (0..self.cameras.len())
            .par_bridge()
            .map(|view_j| {
                let mut e_j = self.residual_a(view_j)?;

                for (i, point3d) in self.points3d.iter().enumerate() {
                    let y_ij = self.calculate_y(point3d, view_j)?;
                    let res = self.residual_b(i)?;
                    e_j -= y_ij * res;
                }
                Some((view_j, e_j))
            })
            .collect::<Vec<_>>();

        e_blocks.iter().for_each(|block| {
            let (j, e_j) = if let Some((j, e_j)) = block {
                (j, e_j)
            } else {
                return;
            };
            e.fixed_view_mut::<6, 1>(j * BundleAdjustment::CAMERA_PARAMETERS, 0)
                .copy_from(e_j);
        });

        e
    }

    fn calculate_delta_b(&self, delta_a: &MatrixXx1<f64>) -> MatrixXx1<f64> {
        let mut delta_b = MatrixXx1::zeros(self.points3d.len() * 3);

        let delta_b_blocks = self
            .points3d
            .iter()
            .enumerate()
            .par_bridge()
            .map(|(point_i, point3d)| {
                let mut residual_b = self.residual_b(point_i)?;

                for view_j in 0..self.cameras.len() {
                    let w_ij = self.calculate_w(point3d, view_j)?;
                    let delta_a_j =
                        delta_a.fixed_view::<6, 1>(view_j * BundleAdjustment::CAMERA_PARAMETERS, 0);
                    residual_b -= w_ij.tr_mul(&delta_a_j);
                }

                let v_inv = self.calculate_v_inv(point3d)?;
                let delta_b_i = v_inv * residual_b;
                Some((point_i, delta_b_i))
            })
            .collect::<Vec<_>>();

        delta_b_blocks.iter().for_each(|block| {
            let (i, delta_b_i) = if let Some((i, delta_b_i)) = block {
                (i, delta_b_i)
            } else {
                return;
            };
            delta_b
                .fixed_view_mut::<3, 1>(i * 3, 0)
                .copy_from(delta_b_i);
        });

        delta_b
    }

    fn calculate_residual_vector(&self) -> MatrixXx1<f64> {
        let mut residuals = MatrixXx1::zeros(self.points3d.len() * self.cameras.len() * 2);

        // TODO: run this in parallel
        for (i, point_i) in self.points3d.iter().enumerate() {
            if point_i.is_none() {
                continue;
            }
            for view_j in 0..self.cameras.len() {
                let residual_b_i = self.residual(i, view_j);
                residuals
                    .fixed_rows_mut::<2>(i * self.cameras.len() * 2 + view_j)
                    .copy_from(&residual_b_i);
            }
        }

        residuals
    }

    fn calculate_jt_residual(&self) -> MatrixXx1<f64> {
        let mut g = MatrixXx1::zeros(
            self.cameras.len() * BundleAdjustment::CAMERA_PARAMETERS + self.points3d.len() * 3,
        );

        // TODO: run this in parallel
        // gradient = Jt * residual
        // First 6*m rows of Jt are residuals from camera matrices.
        for (i, point_i) in self.points3d.iter().enumerate() {
            let point_i = if let Some(point_i) = point_i {
                point_i
            } else {
                continue;
            };
            for view_j in 0..self.cameras.len() {
                let jac_a_i = self.jacobian_a(point_i, view_j);
                let residual_a_i = self.residual(i, view_j);
                let block = jac_a_i.tr_mul(&residual_a_i);
                let mut target_block =
                    g.fixed_rows_mut::<6>(view_j * BundleAdjustment::CAMERA_PARAMETERS);
                target_block += block;
            }
        }

        // Last 3*n rows of Jt are residuals from point coordinates.
        let mut points_target = g.rows_mut(
            self.cameras.len() * BundleAdjustment::CAMERA_PARAMETERS,
            self.points3d.len() * 3,
        );
        for (i, point_i) in self.points3d.iter().enumerate() {
            let point_i = if let Some(point_i) = point_i {
                point_i
            } else {
                continue;
            };
            for view_j in 0..self.cameras.len() {
                let jac_b_i = self.jacobian_b(point_i, view_j);
                let residual_b_i = self.residual(i, view_j);
                let block = jac_b_i.tr_mul(&residual_b_i);
                let mut target_block = points_target.fixed_rows_mut::<3>(i * 3);
                target_block += block;
            }
        }

        g
    }

    fn calculate_delta_step(&self) -> Option<MatrixXx1<f64>> {
        let s = self.calculate_s();
        let e = self.calculate_e();
        let delta_a = s.lu().solve(&e)?;
        let delta_b = self.calculate_delta_b(&delta_a);

        let mut delta = MatrixXx1::zeros(delta_a.len() + delta_b.len());
        delta.rows_mut(0, delta_a.len()).copy_from(&delta_a);
        delta
            .rows_mut(delta_a.len(), delta_b.len())
            .copy_from(&delta_b);

        Some(delta)
    }

    fn update_params(&mut self, delta: &MatrixXx1<f64>) {
        for view_j in 0..self.cameras.len() {
            let camera = &mut self.cameras[view_j];
            camera.r += delta.fixed_rows::<3>(BundleAdjustment::CAMERA_PARAMETERS * view_j);
            camera.t += delta.fixed_rows::<3>(BundleAdjustment::CAMERA_PARAMETERS * view_j + 3);
        }
        self.projections = self
            .cameras
            .iter()
            .map(|camera| camera.projection())
            .collect();

        let points_source = delta.rows(
            self.cameras.len() * BundleAdjustment::CAMERA_PARAMETERS,
            self.points3d.len() * 3,
        );

        for (i, point_i) in self.points3d.iter_mut().enumerate() {
            let point_i = if let Some(point_i) = point_i {
                point_i
            } else {
                continue;
            };

            let point_source = points_source.fixed_rows::<3>(i * 3);
            *point_i += Vector3::from(point_source);
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
            let delta = self.calculate_delta_step();
            let delta = if let Some(delta) = delta {
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
                    .points3d
                    .iter()
                    .filter_map(|p| Some((*p)?.norm_squared()))
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
            let mut current_points3d = self.points3d.clone();

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
                self.points3d.clear();
                self.points3d.append(&mut current_points3d);
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
