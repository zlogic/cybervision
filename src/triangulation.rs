use std::{fmt, sync::atomic::AtomicUsize, sync::atomic::Ordering as AtomicOrdering};

use nalgebra::{
    DMatrix, Matrix2x3, Matrix2x6, Matrix3, Matrix3x4, Matrix3x6, Matrix6, MatrixXx1, MatrixXx4,
    Vector2, Vector3, Vector4,
};

use rand::seq::SliceRandom;
use rand::{rngs::SmallRng, SeedableRng};
use rayon::prelude::*;

const BUNDLE_ADJUSTMENT_MAX_ITERATIONS: usize = 1000;
const EXTEND_TRACKS_SEARCH_RADIUS: usize = 7;
const MERGE_TRACKS_MAX_DISTANCE: usize = 100;
const MERGE_TRACKS_SEARCH_RADIUS: usize = 7;
const PERSPECTIVE_SCALE_THRESHOLD: f64 = 0.0001;
const RANSAC_N: usize = 3;
const RANSAC_K: usize = 100_000;
// TODO: this should be proportional to image size
const RANSAC_INLIERS_T: f64 = 25.0;
const RANSAC_T: f64 = 50.0;
const RANSAC_D: usize = 100;
const RANSAC_D_EARLY_EXIT: usize = 100_000;
const RANSAC_CHECK_INTERVAL: usize = 1000;
// Lower this value to get more points (especially on far distance).
const MIN_ANGLE_BETWEEN_RAYS: f64 = (2.0 / 180.0) * std::f64::consts::PI;

pub struct Surface {
    tracks: Vec<Track>,
    cameras: Vec<Camera>,
    projections: Vec<Matrix3x4<f64>>,
}

impl Surface {
    pub fn iter_tracks(&self) -> std::slice::Iter<Track> {
        self.tracks.iter()
    }

    pub fn tracks_len(&self) -> usize {
        self.tracks.len()
    }

    #[inline]
    pub fn get_point(&self, track_i: usize) -> Option<Vector3<f64>> {
        self.tracks[track_i].point3d
    }

    #[inline]
    pub fn get_camera_points(&self, track_i: usize) -> &[Option<Match>] {
        self.tracks[track_i].points.as_slice()
    }

    #[inline]
    pub fn camera_center(&self, camera_i: usize) -> Vector3<f64> {
        self.cameras[camera_i].center
    }

    #[inline]
    pub fn point_depth(&self, camera_i: usize, track_i: usize) -> Option<f64> {
        let track = &self.tracks[track_i];
        if self.cameras.is_empty() {
            return Some(track.point3d?.z);
        }
        let camera = &self.cameras[camera_i];
        track.get(camera_i)?;
        Some(camera.point_depth(&track.point3d?))
    }

    #[inline]
    pub fn point_in_camera(&self, camera_i: usize, point3d: &Vector3<f64>) -> Vector3<f64> {
        let camera = &self.cameras[camera_i];
        camera.r_matrix * (point3d + camera.r_matrix.tr_mul(&camera.t))
    }

    pub fn cameras_len(&self) -> usize {
        self.cameras.len()
    }
}

type Match = (u32, u32);
type CorrelatedPoints = DMatrix<Option<(u32, u32, f32)>>;

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
        images_count: usize,
        projection: ProjectionMode,
        bundle_adjustment: bool,
    ) -> Triangulation {
        let surface = Surface {
            tracks: vec![],
            cameras: vec![],
            projections: vec![],
        };
        let (affine, perspective) = match projection {
            ProjectionMode::Affine => (Some(AffineTriangulation { surface }), None),
            ProjectionMode::Perspective => (
                None,
                Some(PerspectiveTriangulation {
                    images_count,
                    calibration: vec![None; images_count],
                    projections: vec![None; images_count],
                    cameras: vec![None; images_count],
                    tracks: vec![],
                    image_shapes: vec![None; images_count],
                    best_initial_p2: None,
                    best_initial_score: None,
                    best_initial_pair: None,
                    remaining_images: (0..images_count).collect(),
                    retained_images: vec![],
                    bundle_adjustment,
                }),
            ),
        };

        Triangulation {
            affine,
            perspective,
        }
    }

    pub fn set_image_data(
        &mut self,
        image_index: usize,
        k: &Matrix3<f64>,
        image_shape: (usize, usize),
    ) {
        if let Some(perspective) = &mut self.perspective {
            perspective.set_image_data(image_index, k, image_shape)
        }
    }

    pub fn triangulate<PL: ProgressListener>(
        &mut self,
        image1_index: usize,
        image2_index: usize,
        correlated_points: &CorrelatedPoints,
        fundamental_matrix: &Matrix3<f64>,
        progress_listener: Option<&PL>,
    ) -> Result<(), TriangulationError> {
        if let Some(affine) = &mut self.affine {
            affine.triangulate(correlated_points)
        } else if let Some(perspective) = &mut self.perspective {
            perspective.add_image_pair(
                image1_index,
                image2_index,
                correlated_points,
                fundamental_matrix,
                progress_listener,
            )
        } else {
            Err(TriangulationError::new("Triangulation not initialized"))
        }
    }

    pub fn recover_next_camera<PL: ProgressListener>(
        &mut self,
        progress_listener: Option<&PL>,
    ) -> Result<Option<usize>, TriangulationError> {
        if self.affine.is_some() {
            Ok(None)
        } else if let Some(perspective) = &mut self.perspective {
            perspective.recover_next_camera(progress_listener)
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

    pub fn retained_images(&self) -> Result<Vec<usize>, TriangulationError> {
        if let Some(affine) = &self.affine {
            Ok(affine.retained_images())
        } else if let Some(perspective) = &self.perspective {
            Ok(perspective.retained_images())
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
}

impl AffineTriangulation {
    fn triangulate(
        &mut self,
        correlated_points: &CorrelatedPoints,
    ) -> Result<(), TriangulationError> {
        if !self.surface.tracks.is_empty() {
            return Err(TriangulationError::new(
                "Triangulation of multiple affine image is not supported",
            ));
        }

        let points3d = correlated_points
            .column_iter()
            .enumerate()
            .par_bridge()
            .flat_map(|(col, out_col)| {
                out_col
                    .iter()
                    .enumerate()
                    .filter_map(|(row, matched_point)| {
                        let point2 = matched_point.map(|p| (p.0, p.1));
                        AffineTriangulation::triangulate_point((row, col), &point2)
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        self.surface.tracks = points3d;

        Ok(())
    }

    fn triangulate_all(&self) -> Result<Surface, TriangulationError> {
        // TODO: drop unused items?
        let surface = Surface {
            tracks: self.surface.tracks.clone(),
            cameras: self.surface.cameras.clone(),
            projections: self.surface.projections.clone(),
        };
        Ok(surface)
    }

    pub fn retained_images(&self) -> Vec<usize> {
        self.surface
            .cameras
            .iter()
            .enumerate()
            .map(|(index, _)| index)
            .collect::<Vec<_>>()
    }

    #[inline]
    fn triangulate_point(p1: (usize, usize), p2: &Option<Match>) -> Option<Track> {
        if let Some(p2) = p2 {
            let dx = p1.1 as f64 - p2.1 as f64;
            let dy = p1.0 as f64 - p2.0 as f64;
            let distance = (dx * dx + dy * dy).sqrt();
            let point3d = Vector3::new(p1.1 as f64, p1.0 as f64, distance);
            let point1 = (p1.0 as u32, p1.1 as u32);
            let point2 = (p2.0, p2.1);

            let track = Track {
                points: vec![Some(point1), Some(point2)],
                point3d: Some(point3d),
            };

            Some(track)
        } else {
            None
        }
    }
}

#[derive(Clone, Debug)]
pub struct Track {
    points: Vec<Option<Match>>,
    point3d: Option<Vector3<f64>>,
}

impl Track {
    fn new(images_count: usize) -> Track {
        Track {
            points: vec![None; images_count],
            point3d: None,
        }
    }

    fn can_merge(&self, other: &Track) -> bool {
        for i in 0..self.points.len() {
            let p1 = if let Some(point) = self.points[i] {
                point
            } else {
                continue;
            };

            let p2 = if let Some(point) = other.points[i] {
                point
            } else {
                continue;
            };
            let drow = p1.0.max(p2.0) as usize - p1.0.min(p2.0) as usize;
            let dcol = p1.1.max(p2.1) as usize - p1.1.min(p2.1) as usize;
            let distance = drow * drow + dcol * dcol;
            if distance > MERGE_TRACKS_MAX_DISTANCE {
                return false;
            }
        }
        true
    }

    fn add(&mut self, index: usize, point: Match) {
        if self.points[index].is_some() {
            return;
        };
        self.points[index] = Some(point);
    }

    fn remove_projections(&mut self, remap_projections: &[Option<usize>]) {
        let mut remaining_images_count = 0;
        for (i, remap_projection) in remap_projections.iter().enumerate() {
            if let Some(new_index) = remap_projection {
                self.points[*new_index] = self.points[i];
                remaining_images_count += 1;
            }
        }
        self.points.truncate(remaining_images_count);
    }

    #[inline]
    pub fn get(&self, i: usize) -> Option<Match> {
        self.points[i]
    }

    #[inline]
    pub fn points(&self) -> &[Option<Match>] {
        self.points.as_slice()
    }

    #[inline]
    pub fn get_point3d(&self) -> Option<Vector3<f64>> {
        self.point3d
    }
}

#[derive(Debug, Clone)]
struct Camera {
    k: Matrix3<f64>,
    r: Vector3<f64>,
    r_matrix: Matrix3<f64>,
    t: Vector3<f64>,
    center: Vector3<f64>,
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
        let center = Camera::center(&r_matrix, t);
        Camera {
            k: k.to_owned(),
            r,
            r_matrix,
            t: t.to_owned(),
            center,
        }
    }

    fn update_params(&mut self, delta_r: &Vector3<f64>, delta_t: &Vector3<f64>) {
        self.r += delta_r;
        self.t += delta_t;
        self.r_matrix = Camera::matrix_r(&self.r);
        self.center = Camera::center(&self.r_matrix, &self.t)
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

    fn center(r_matrix: &Matrix3<f64>, t: &Vector3<f64>) -> Vector3<f64> {
        -r_matrix.tr_mul(t)
    }

    #[inline]
    fn point_depth(&self, point3d: &Vector3<f64>) -> f64 {
        // This is how OpenMVG does it, works great!
        (self.r_matrix * (point3d + self.r_matrix.tr_mul(&self.t))).z
    }

    #[inline]
    fn point_in_front(&self, point3d: &Vector3<f64>) -> bool {
        self.point_depth(point3d) > 0.0
    }

    fn projection(&self) -> Matrix3x4<f64> {
        let mut projection = self.r_matrix.insert_column(3, 0.0);
        projection.column_mut(3).copy_from(&self.t);
        self.k * projection
    }
}

struct PerspectiveTriangulation {
    images_count: usize,
    calibration: Vec<Option<Matrix3<f64>>>,
    projections: Vec<Option<Matrix3x4<f64>>>,
    cameras: Vec<Option<Camera>>,
    tracks: Vec<Track>,
    image_shapes: Vec<Option<(usize, usize)>>,
    best_initial_p2: Option<Matrix3x4<f64>>,
    best_initial_score: Option<f64>,
    best_initial_pair: Option<(usize, usize)>,
    remaining_images: Vec<usize>,
    retained_images: Vec<usize>,
    bundle_adjustment: bool,
}

impl PerspectiveTriangulation {
    fn add_image_pair<PL: ProgressListener>(
        &mut self,
        image1_index: usize,
        image2_index: usize,
        correlated_points: &CorrelatedPoints,
        fundamental_matrix: &Matrix3<f64>,
        progress_listener: Option<&PL>,
    ) -> Result<(), TriangulationError> {
        self.extend_tracks(
            image1_index,
            image2_index,
            correlated_points,
            progress_listener,
        );

        // Get the relative pose for the image pair.
        let k1 = if let Some(calibration) = self.calibration[image1_index] {
            calibration
        } else {
            return Err(TriangulationError::new("Missing calibration matrix"));
        };
        let k2 = if let Some(calibration) = self.calibration[image2_index] {
            calibration
        } else {
            return Err(TriangulationError::new("Missing calibration matrix"));
        };
        let short_tracks = self
            .tracks
            .par_iter()
            .filter_map(|track| {
                let point1 = track.get(image1_index)?;
                let point2 = track.get(image2_index)?;
                let mut short_track = Track::new(2);
                short_track.add(0, point1);
                short_track.add(1, point2);
                Some(short_track)
            })
            .collect::<Vec<_>>();
        let (p2, score) = match PerspectiveTriangulation::find_projection_matrix(
            fundamental_matrix,
            &k1,
            &k2,
            short_tracks.as_slice(),
        ) {
            Some(res) => res,
            None => return Err(TriangulationError::new("Unable to find projection matrix")),
        };

        if self
            .best_initial_score
            .map_or(true, |current_score| score > current_score)
        {
            self.best_initial_p2 = Some(p2);
            self.best_initial_pair = Some((image1_index, image2_index));
            self.best_initial_score = Some(score);
        }

        Ok(())
    }

    fn set_image_data(&mut self, img_index: usize, k: &Matrix3<f64>, image_shape: (usize, usize)) {
        self.calibration[img_index] = Some(k.to_owned());
        self.image_shapes[img_index] = Some(image_shape);
    }

    fn recover_next_camera<PL: ProgressListener>(
        &mut self,
        progress_listener: Option<&PL>,
    ) -> Result<Option<usize>, TriangulationError> {
        if let Some(initial_images) = self.best_initial_pair {
            // Use projection matrix and camera configurations from the best initial pair..
            let k1 = if let Some(calibration) = self.calibration[initial_images.0] {
                calibration
            } else {
                return Err(TriangulationError::new("Missing calibration matrix"));
            };
            let k2 = if let Some(calibration) = self.calibration[initial_images.1] {
                calibration
            } else {
                return Err(TriangulationError::new("Missing calibration matrix"));
            };
            let p1 = k1 * Matrix3x4::identity();
            let camera1 = Camera::from_matrix(&k1, &Matrix3::identity(), &Vector3::zeros());
            self.projections[initial_images.0] = Some(p1);
            self.cameras[initial_images.0] = Some(camera1);
            let p2 = if let Some(p2) = self.best_initial_p2 {
                p2
            } else {
                return Err(TriangulationError::new(
                    "Missing projection matrix for initial image pair",
                ));
            };
            let camera2_r = p2.fixed_view::<3, 3>(0, 0);
            let camera2_t = p2.column(3);
            let camera2 = Camera::from_matrix(&k2, &camera2_r.into(), &camera2_t.into());

            let p2 = k2 * p2;
            self.projections[initial_images.1] = Some(p2);
            self.cameras[initial_images.1] = Some(camera2);
            self.triangulate_tracks();
            self.remaining_images
                .retain(|i| *i != initial_images.0 && *i != initial_images.1);

            self.best_initial_pair = None;

            return Ok(Some(initial_images.1));
        }

        // Find image with the most matches with current 3D points.
        let matches_count = self
            .tracks
            .par_iter()
            .flat_map(|track| {
                if track.get_point3d().is_none() {
                    return None;
                }
                let mut count_projections = vec![0usize; self.images_count];
                let unknown_cameras = self
                    .remaining_images
                    .iter()
                    .any(|camera_i| track.get(*camera_i).is_some());
                if !unknown_cameras {
                    return None;
                }
                self.remaining_images.iter().for_each(|camera_i| {
                    if track.get(*camera_i).is_some() {
                        count_projections[*camera_i] = 1
                    }
                });
                Some(count_projections)
            })
            .reduce(
                || vec![0usize; self.images_count],
                |mut a, b| {
                    for i in 0..self.images_count {
                        a[i] += b[i];
                    }
                    a
                },
            );
        let best_candidate = self
            .remaining_images
            .iter()
            .max_by_key(|i| matches_count[**i]);
        let best_candidate = if let Some(best_candidate) = best_candidate {
            best_candidate.to_owned()
        } else {
            return Ok(None);
        };
        self.remaining_images.retain(|i| *i != best_candidate);

        self.merge_tracks(best_candidate, progress_listener);

        let k2 = if let Some(calibration) = self.calibration[best_candidate] {
            calibration
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
        let camera2 = match self.recover_pose(best_candidate, &k2, &k2_inv, progress_listener) {
            Some(camera2) => camera2,
            None => return Err(TriangulationError::new("Unable to find projection matrix")),
        };
        let projection2 = camera2.projection();
        self.cameras[best_candidate] = Some(camera2);
        self.projections[best_candidate] = Some(projection2);

        self.triangulate_tracks();
        Ok(Some(best_candidate))
    }

    fn triangulate_all<PL: ProgressListener>(
        &mut self,
        progress_listener: Option<&PL>,
    ) -> Result<Surface, TriangulationError> {
        self.retained_images = self
            .cameras
            .iter()
            .enumerate()
            .filter_map(|(index, camera)| if camera.is_some() { Some(index) } else { None })
            .collect::<Vec<_>>();
        self.prune_projections();

        // At this point, all cameras and projections should be valid.
        let cameras = self
            .cameras
            .iter()
            .map(|camera| {
                if let Some(camera) = camera {
                    camera.to_owned()
                } else {
                    Camera {
                        k: Matrix3::from_element(f64::NAN),
                        r: Vector3::from_element(f64::NAN),
                        r_matrix: Matrix3::from_element(f64::NAN),
                        t: Vector3::from_element(f64::NAN),
                        center: Vector3::from_element(f64::NAN),
                    }
                }
            })
            .collect::<Vec<_>>();

        self.filter_outliers(cameras.as_slice());
        if self.bundle_adjustment {
            self.bundle_adjustment(cameras.as_slice(), progress_listener)?;
        }

        let surface_projections = self
            .cameras
            .iter()
            .map(|camera| {
                if let Some(camera) = camera {
                    camera.projection()
                } else {
                    Matrix3x4::from_element(f64::NAN)
                }
            })
            .collect::<Vec<_>>();
        let surface_tracks = self
            .tracks
            .iter()
            .par_bridge()
            .map(|track| track.to_owned())
            .collect::<Vec<_>>();

        let surface = Surface {
            tracks: surface_tracks,
            cameras,
            projections: surface_projections,
        };

        Ok(surface)
    }

    pub fn retained_images(&self) -> Vec<usize> {
        self.retained_images.to_owned()
    }

    #[inline]
    fn triangulate_track(
        track: &Track,
        projections: &[Option<Matrix3x4<f64>>],
    ) -> Option<Vector4<f64>> {
        let points_projection = projections
            .iter()
            .enumerate()
            .flat_map(|(i, projection)| {
                let projection = (*projection)?;
                let point = track.get(i)?;
                let point = (point.1 as f64, point.0 as f64);
                Some((point, projection))
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
    }

    fn prune_projections(&mut self) {
        let mut remap_projections = vec![None; self.images_count];
        let mut projection_index = 0;
        for (i, remap_projection) in remap_projections.iter_mut().enumerate() {
            *remap_projection = if self.projections[i].is_some() {
                let remap_index = projection_index;
                projection_index += 1;
                Some(remap_index)
            } else {
                None
            };
        }
        for (i, remap_projection) in remap_projections.iter().enumerate() {
            if let Some(new_index) = remap_projection {
                self.cameras[*new_index] = self.cameras[i].to_owned();
                self.projections[*new_index] = self.projections[i];
            }
        }
        self.projections.truncate(projection_index);
        self.cameras.truncate(projection_index);
        self.tracks
            .iter_mut()
            .par_bridge()
            .for_each(|track| track.remove_projections(remap_projections.as_slice()));
    }

    fn find_projection_matrix(
        fundamental_matrix: &Matrix3<f64>,
        k1: &Matrix3<f64>,
        k2: &Matrix3<f64>,
        tracks: &[Track],
    ) -> Option<(Matrix3x4<f64>, f64)> {
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
        let (p2, count) = combinations
            .into_iter()
            .map(|(r, t)| {
                let mut p2 = Matrix3x4::zeros();
                p2.fixed_view_mut::<3, 3>(0, 0).copy_from(&r);
                p2.column_mut(3).copy_from(&t);
                let p2_calibrated = k2 * p2;
                let camera2 = Camera::from_matrix(k2, &r, &t);
                let projections = &[Some(p1), Some(p2_calibrated)];
                let points_count: usize = tracks
                    .par_iter()
                    .filter(|track| {
                        let point4d =
                            PerspectiveTriangulation::triangulate_track(track, projections);
                        let point4d = if let Some(point4d) = point4d {
                            point4d
                        } else {
                            return false;
                        };
                        let point3d = point4d.remove_row(3).unscale(point4d.w);
                        point3d.z > 0.0 && camera2.point_in_front(&point3d)
                    })
                    .count();
                (p2, points_count)
            })
            .max_by(|r1, r2| r1.1.cmp(&r2.1))?;
        Some((p2, count as f64))
    }

    fn min_ray_angle_cos(cameras: &[Camera], track: &Track) -> Option<f64> {
        let rays = track
            .points()
            .iter()
            .enumerate()
            .filter_map(|(camera_i, point2d)| {
                if point2d.is_none() {
                    return None;
                };
                let camera_center = cameras[camera_i].center;
                let point3d = track.point3d?;
                let ray = point3d - camera_center;
                if ray.norm() < f64::EPSILON {
                    return None;
                }
                Some(ray.normalize())
            })
            .collect::<Vec<_>>();

        // This fuction returns the minimum cosine between angle rays.
        // Cosine values decrease when the angle increases (cos(0)==1).
        // So to get the maximum angle, the minimum cosine should be used.
        let mut min_angle_cos: Option<f64> = None;
        for r_i in 0..rays.len() {
            let ray_i = rays[r_i];
            for ray_j in rays.iter().skip(r_i + 1) {
                let angle = ray_i.dot(ray_j).abs();
                min_angle_cos = if let Some(min_angle_cos) = min_angle_cos {
                    Some(min_angle_cos.min(angle))
                } else {
                    Some(angle)
                }
            }
        }
        min_angle_cos
    }

    fn recover_pose<PL: ProgressListener>(
        &self,
        image_index: usize,
        k: &Matrix3<f64>,
        k_inv: &Matrix3<f64>,
        progress_listener: Option<&PL>,
    ) -> Option<Camera> {
        // Gaku Nakano, "A Simple Direct Solution to the Perspective-Three-Point Problem," BMVC2019

        let inlier_projections = vec![image_index];
        let validate_projections = self
            .projections
            .iter()
            .enumerate()
            .filter_map(|(i, projection)| {
                if projection.is_some() || i == image_index {
                    Some(i)
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();

        let linked_tracks = self
            .tracks
            .par_iter()
            .filter(|track| track.get(image_index).is_some() && track.point3d.is_some())
            .map(|track| track.to_owned())
            .collect::<Vec<_>>();

        if linked_tracks.len() < RANSAC_N {
            return None;
        }

        let ransac_outer = RANSAC_K / RANSAC_CHECK_INTERVAL;

        let mut best_result = (
            Camera::from_matrix(k, &Matrix3::identity(), &Vector3::zeros()),
            0,
            f64::MAX,
        );
        let counter = AtomicUsize::new(0);
        let reduce_best_result = |(c1, count1, error1), (c2, count2, error2)| {
            if count1 > count2 || (count1 == count2 && error1 < error2) {
                (c1, count1, error1)
            } else {
                (c2, count2, error2)
            }
        };

        for _ in 0..ransac_outer {
            let (camera, count, error) = (0..RANSAC_CHECK_INTERVAL)
                .par_bridge()
                .filter_map(|_| {
                    if let Some(pl) = progress_listener {
                        let value =
                            counter.fetch_add(1, AtomicOrdering::Relaxed) as f32 / RANSAC_K as f32;
                        pl.report_status(0.02 + 0.98 * value);
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

                    PerspectiveTriangulation::recover_pose_from_points(
                        image_index,
                        k_inv,
                        inliers.as_slice(),
                    )
                    .into_iter()
                    .filter_map(|(r, t)| {
                        let camera = Camera::from_matrix(k, &r, &t);
                        let projection = camera.projection();

                        let mut projections = self.projections.clone();
                        projections[image_index] = Some(projection);

                        let (count, _) = PerspectiveTriangulation::tracks_reprojection_error(
                            &inliers_tracks,
                            &projections,
                            inlier_projections.as_slice(),
                            true,
                        );
                        if count != RANSAC_N {
                            return None;
                        }

                        let (count, error) = PerspectiveTriangulation::tracks_reprojection_error(
                            &linked_tracks,
                            &projections,
                            validate_projections.as_slice(),
                            false,
                        );
                        Some((camera, count, error / (count as f64)))
                    })
                    .reduce(reduce_best_result)
                })
                .reduce(|| best_result.clone(), reduce_best_result);

            best_result = (camera, count, error);
            if count >= RANSAC_D_EARLY_EXIT {
                break;
            }
        }

        let count = best_result.1;
        if count > RANSAC_D {
            Some(best_result.0)
        } else {
            None
        }
    }

    fn recover_pose_from_points(
        image_index: usize,
        k_inv: &Matrix3<f64>,
        inliers: &[&Track],
    ) -> Vec<(Matrix3<f64>, Vector3<f64>)> {
        let mut inliers = inliers
            .iter()
            .filter_map(|track| {
                let p2 = track.get(image_index)?;
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
        projections: &[Option<Matrix3x4<f64>>],
        include_projections: &[usize],
        inliers: bool,
    ) -> (usize, f64) {
        let threshold = if inliers { RANSAC_INLIERS_T } else { RANSAC_T };
        tracks
            .iter()
            .filter_map(|track| {
                PerspectiveTriangulation::point_reprojection_error(
                    track,
                    projections,
                    include_projections,
                )
                .filter(|error| *error < threshold)
            })
            .fold((0, 0.0f64), |(count, error), match_error| {
                (count + 1, error.max(match_error))
            })
    }

    #[inline]
    fn point_reprojection_error(
        track: &Track,
        projections: &[Option<Matrix3x4<f64>>],
        include_projections: &[usize],
    ) -> Option<f64> {
        let point4d = PerspectiveTriangulation::triangulate_track(track, projections)?;
        include_projections
            .iter()
            .filter_map(|i| {
                let projection = (projections[*i])?;
                let original = track.get(*i)?;
                let mut projected = projection * point4d;
                projected.unscale_mut(projected.z);
                let dx = projected.x - original.1 as f64;
                let dy = projected.y - original.0 as f64;
                let error = (dx * dx + dy * dy).sqrt();
                Some(error)
            })
            .reduce(|acc, val| acc.max(val))
    }

    fn extend_tracks<PL: ProgressListener>(
        &mut self,
        image1_index: usize,
        image2_index: usize,
        correlated_points: &CorrelatedPoints,
        progress_listener: Option<&PL>,
    ) {
        let mut remaining_points = correlated_points.clone();
        let counter = AtomicUsize::new(0);
        let total_iterations = self.tracks.len();

        self.tracks.iter_mut().for_each(|track| {
            if let Some(pl) = progress_listener {
                let value =
                    counter.fetch_add(1, AtomicOrdering::Relaxed) as f32 / total_iterations as f32;
                pl.report_status(value * 0.98);
            }
            let point1 = if let Some(point) = track.get(image1_index) {
                point
            } else {
                return;
            };
            let row_start = (point1.0 as usize).saturating_sub(EXTEND_TRACKS_SEARCH_RADIUS);
            let col_start = (point1.1 as usize).saturating_sub(EXTEND_TRACKS_SEARCH_RADIUS);
            let row_end =
                (point1.0 as usize + EXTEND_TRACKS_SEARCH_RADIUS).min(correlated_points.nrows());
            let col_end =
                (point1.1 as usize + EXTEND_TRACKS_SEARCH_RADIUS).min(correlated_points.ncols());
            let mut min_distance = None;
            let mut best_match = None;
            for row in row_start..row_end {
                for col in col_start..col_end {
                    let next_point = if let Some(point) = correlated_points[(row, col)] {
                        point
                    } else {
                        continue;
                    };
                    let drow = row.max(point1.0 as usize) - row.min(point1.0 as usize);
                    let dcol = col.max(point1.1 as usize) - col.min(point1.1 as usize);
                    let distance = drow * drow + dcol * dcol;
                    if min_distance.map_or(true, |min_distance| distance < min_distance) {
                        min_distance = Some(distance);
                        best_match = Some(next_point);
                    }
                }
            }

            if let Some(best_match) = best_match {
                let track_point = (best_match.0, best_match.1);
                track.add(image2_index, track_point);
                remaining_points[(best_match.0 as usize, best_match.1 as usize)] = None;
            };
        });

        let counter = AtomicUsize::new(0);
        let total_iterations = remaining_points.ncols();
        let mut new_tracks = remaining_points
            .column_iter()
            .enumerate()
            .par_bridge()
            .flat_map(|(col, start_col)| {
                if let Some(pl) = progress_listener {
                    let value = counter.fetch_add(1, AtomicOrdering::Relaxed) as f32
                        / total_iterations as f32;
                    pl.report_status(0.98 + value * 0.02);
                }
                start_col
                    .iter()
                    .enumerate()
                    .filter_map(|(row, m)| {
                        let track_point2 = m.map(|m| (m.0, m.1))?;
                        let mut track = Track::new(self.images_count);
                        track.add(image1_index, (row as u32, col as u32));
                        track.add(image2_index, track_point2);

                        Some(track)
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        self.tracks.append(&mut new_tracks);
    }

    fn merge_tracks<PL: ProgressListener>(
        &mut self,
        image_i: usize,
        progress_listener: Option<&PL>,
    ) {
        let shape = if let Some(shape) = self.image_shapes[image_i] {
            shape
        } else {
            return;
        };
        let tracks_count = self.tracks.len();
        let mut tracks_index = DMatrix::<Option<usize>>::from_element(shape.0, shape.1, None);
        for track_i in 0..self.tracks.len() {
            let point = if let Some(point) = self.tracks[track_i].get(image_i) {
                point
            } else {
                continue;
            };
            if let Some(pl) = progress_listener {
                let value = track_i as f32 / tracks_count as f32;
                pl.report_status(value * 0.02);
            }
            let row_start = (point.0 as usize).saturating_sub(MERGE_TRACKS_SEARCH_RADIUS);
            let col_start = (point.1 as usize).saturating_sub(MERGE_TRACKS_SEARCH_RADIUS);
            let row_end = (point.0 as usize + MERGE_TRACKS_SEARCH_RADIUS).min(tracks_index.nrows());
            let col_end = (point.1 as usize + MERGE_TRACKS_SEARCH_RADIUS).min(tracks_index.ncols());
            let mut min_distance = None;
            let mut best_match = None;
            for row in row_start..row_end {
                for col in col_start..col_end {
                    let potential_match = if let Some(track_j) = tracks_index[(row, col)] {
                        track_j
                    } else {
                        continue;
                    };
                    if potential_match == track_i
                        || !self.tracks[track_i].can_merge(&self.tracks[potential_match])
                    {
                        continue;
                    }
                    let drow = row.max(point.0 as usize) - row.min(point.0 as usize);
                    let dcol = col.max(point.1 as usize) - col.min(point.1 as usize);
                    let distance = drow * drow + dcol * dcol;
                    if min_distance.map_or(true, |min_distance| distance < min_distance) {
                        min_distance = Some(distance);
                        best_match = Some(potential_match);
                    }
                }
            }
            if let Some(best_match) = best_match {
                let src_track = self.tracks[track_i].to_owned();
                let dst_track = &mut self.tracks[best_match];
                for (i, point) in src_track.points().iter().enumerate() {
                    if let Some(point) = point {
                        dst_track.add(i, *point);
                    }
                }
                self.tracks[track_i].points.clear();
            } else {
                tracks_index[(point.0 as usize, point.1 as usize)] = Some(track_i);
            }
        }
        self.tracks.retain(|track| !track.points.is_empty());
    }

    fn bundle_adjustment<PL: ProgressListener>(
        &mut self,
        cameras: &[Camera],
        progress_listener: Option<&PL>,
    ) -> Result<(), TriangulationError> {
        // Only send tracks/points that could be triangulated.
        self.tracks.retain(|track| track.point3d.is_some());
        self.tracks.shrink_to_fit();
        let mut ba = BundleAdjustment::new(cameras, self.tracks.as_mut_slice());
        let refined_cameras = ba.optimize(progress_listener)?;
        self.cameras = refined_cameras
            .iter()
            .map(|camera| Some(camera.to_owned()))
            .collect::<Vec<_>>();

        Ok(())
    }

    fn filter_outliers(&mut self, cameras: &[Camera]) {
        // For an angle to be larger than a threshold, its cosine needs to be smaller than the
        // threshold.
        let angle_cos_threshold = MIN_ANGLE_BETWEEN_RAYS.cos();
        self.tracks.par_iter_mut().for_each(|track| {
            let point3d = if let Some(point3d) = track.point3d {
                point3d
            } else {
                return;
            };
            // Clear points which are in the back of the camers.
            if track
                .points()
                .iter()
                .enumerate()
                .any(|(camera_i, point2d)| {
                    point2d.is_some() && !cameras[camera_i].point_in_front(&point3d)
                })
            {
                track.point3d = None;
                return;
            }
            // Clear points where the maximum angle between rays is too low.
            if let Some(min_angle_cos) = PerspectiveTriangulation::min_ray_angle_cos(cameras, track)
            {
                if min_angle_cos > angle_cos_threshold {
                    track.point3d = None;
                }
            } else {
                track.point3d = None;
            }
        });
        self.tracks.retain(|track| track.point3d.is_some());
        self.tracks.shrink_to_fit();
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
    const MAX_ITER: usize = 5;
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

    fn new<'a>(cameras: &[Camera], tracks: &'a mut [Track]) -> BundleAdjustment<'a> {
        // For now, identity covariance is acceptable.
        let covariance = 1.0;
        let cameras = cameras.to_vec();
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
        let camera = &self.cameras[view_j];
        let projection = &self.projections[view_j];
        let point4d = point3d.insert_row(3, 1.0);
        let point_projected = projection * point4d;

        let (u, v, w) = (point_projected.x, point_projected.y, point_projected.z);
        let d_projection_hpoint =
            Matrix2x3::new(1.0 / w, 0.0, -u / (w * w), 0.0, 1.0 / w, -v / (w * w));
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

        let (u, v, w) = (point_projected.x, point_projected.y, point_projected.z);
        let d_projection_hpoint =
            Matrix2x3::new(1.0 / w, 0.0, -u / (w * w), 0.0, 1.0 / w, -v / (w * w));

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
            let dx = projected.x - original.1 as f64;
            let dy = projected.y - original.0 as f64;

            Vector2::new(dx, dy)
        } else {
            Vector2::zeros()
        }
    }

    #[inline]
    fn calculate_v_inv(&self, point3d: &Option<Vector3<f64>>) -> Option<Matrix3<f64>> {
        let mut v = Matrix3::identity() * self.mu;
        let point3d = (*point3d)?;
        for view_j in 0..self.cameras.len() {
            let b = self.jacobian_b(&point3d, view_j);
            v += b.tr_mul(&b) * self.covariance;
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
                    track.point3d?;
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
                    let point_i = &track.point3d?;
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
                        let jac_a_k = self.jacobian_a(&point3d, view_k);
                        let jac_b_k = self.jacobian_b(&point3d, view_k);
                        let w_ik = jac_a_k.tr_mul(&jac_b_k) * self.covariance;
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
                    e_j += residual_a_ij - y_ij * residual_b_ij;
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
                delta_b.copy_from(delta_b_i);
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
            return Ok(self.cameras.to_vec());
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

            let rho_denominator = delta
                .row_iter()
                .enumerate()
                .map(|(i, delta_i)| delta_i[0] * (delta_i[0] * self.mu + jt_residual[i]))
                .sum::<f64>();
            let rho = (residual_norm_squared - new_residual_norm_squared) / rho_denominator;

            drop(delta);
            if rho > 0.0 {
                drop(current_points3d);
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

        Ok(self.cameras.to_vec())
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
