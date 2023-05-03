use std::{fmt, sync::atomic::AtomicUsize, sync::atomic::Ordering as AtomicOrdering};

use levenberg_marquardt::{LeastSquaresProblem, LevenbergMarquardt};
use nalgebra::{
    DMatrix, Dyn, Matrix3, Matrix3x4, MatrixXx4, OMatrix, OVector, Owned, SMatrix, Vector3, Vector4,
};

use rand::seq::SliceRandom;
use rand::{rngs::SmallRng, SeedableRng};
use rayon::prelude::*;

use crate::correlation;

const PERSPECTIVE_VALUE_RANGE: f64 = 100.0;
const OUTLIER_FILTER_STDEV_THRESHOLD: f64 = 1.0;
const OUTLIER_FILTER_SEARCH_AREA: usize = 5;
const OUTLIER_FILTER_MIN_NEIGHBORS: usize = 10;
const RANSAC_N: usize = 6;
const RANSAC_K: usize = 10_000_000;
// TODO: this should pe proportional to image size
const RANSAC_INLIERS_T: f64 = 1.0;
const RANSAC_T: f64 = 3.0;
const RANSAC_D: usize = 100;
const RANSAC_D_EARLY_EXIT: usize = 10_000;
const RANSAC_CHECK_INTERVAL: usize = 10_000;

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

type Track = Vec<Option<Match>>;

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
    pub fn new(projection: ProjectionMode, scale: (f64, f64, f64)) -> Triangulation {
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
                    projection: vec![],
                    tracks: vec![],
                    scale,
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
    pub fn triangulate_all(&mut self) -> Result<Surface, TriangulationError> {
        if let Some(affine) = &self.affine {
            affine.triangulate_all()
        } else if let Some(perspective) = &mut self.perspective {
            perspective.triangulate_all()
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

pub struct PerspectiveTriangulation {
    projection: Vec<Matrix3x4<f64>>,
    tracks: Vec<Track>,
    scale: (f64, f64, f64),
}

impl PerspectiveTriangulation {
    fn triangulate<PL: ProgressListener>(
        &mut self,
        correlated_points: &DMatrix<Option<Match>>,
        fundamental_matrix: &Matrix3<f64>,
        progress_listener: Option<&PL>,
    ) -> Result<(), TriangulationError> {
        let index = self.projection.len().saturating_sub(1);
        self.extend_tracks(correlated_points, index);

        if self.projection.is_empty() {
            self.projection.push(Matrix3x4::identity());
            let p2 = match PerspectiveTriangulation::f_to_projection_matrix(&fundamental_matrix) {
                Some(p2) => p2,
                None => return Err(TriangulationError::new("Unable to find projection matrix")),
            };
            self.projection.push(p2);
        } else {
            let p2 = match self.next_projection_matrix(progress_listener) {
                Some(p2) => p2,
                None => return Err(TriangulationError::new("Unable to find projection matrix")),
            };
            self.projection.push(p2);
        }

        //let points = filter_outliers(correlated_points.shape(), &points);
        //self.points3d.push(points);
        //self.track_points(correlated_points);

        Ok(())
    }

    fn triangulate_all(&mut self) -> Result<Surface, TriangulationError> {
        let points3d = self.bundle_adjustment()?;
        let points3d = self.triangulate_tracks(&self.tracks);
        // TODO: drop unused items?

        Ok(self.scale_points(points3d))
    }

    #[inline]
    fn triangulate_track(track: &Track, projection: &[Matrix3x4<f64>]) -> Option<Vector4<f64>> {
        let track_projection = track
            .iter()
            .enumerate()
            .flat_map(|(i, point)| {
                if point.is_some() && i < projection.len() {
                    Some(&projection[i])
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();

        if track_projection.is_empty() {
            return None;
        }

        let points = track
            .iter()
            .flat_map(|point| {
                if let Some(point) = point {
                    Some((point.1 as f64, point.0 as f64))
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();

        if points.is_empty() {
            return None;
        }

        let len = track_projection.len().min(points.len());
        let mut a = MatrixXx4::zeros(len * 2);
        for i in 0..len {
            a.row_mut(i * 2).copy_from(
                &(track_projection[i].row(2) * points[i].0 - track_projection[i].row(0)),
            );
            a.row_mut(i * 2 + 1).copy_from(
                &(track_projection[i].row(2) * points[i].1 - track_projection[i].row(1)),
            );
        }

        let usv = a.svd(false, true);
        let vt = usv.v_t.unwrap();
        let point4d = vt.row(vt.nrows() - 1).transpose();

        Some(point4d)
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

    fn next_projection_matrix<PL: ProgressListener>(
        &self,
        progress_listener: Option<&PL>,
    ) -> Option<Matrix3x4<f64>> {
        let unlinked_tracks = self
            .tracks
            .iter()
            .filter(|track| track.len() >= 1 && track[track.len() - 1].is_some())
            .map(|track| track.to_owned())
            .collect::<Vec<_>>();

        let linked_tracks = unlinked_tracks
            .iter()
            .filter(|track| {
                track.len() >= 2
                    && track[track.len() - 1].is_some()
                    && track[track.len() - 2].is_some()
            })
            .map(|track| track.to_owned())
            .collect::<Vec<_>>();

        let projection = &self.projection;

        let ransac_outer = RANSAC_K / RANSAC_CHECK_INTERVAL;

        let mut result = None;
        let counter = AtomicUsize::new(0);
        for _ in 0..ransac_outer {
            let (projection, count, _error) = (0..RANSAC_CHECK_INTERVAL)
                .into_iter()
                .par_bridge()
                .filter_map(|_| {
                    if let Some(pl) = progress_listener {
                        let value =
                            counter.fetch_add(1, AtomicOrdering::Relaxed) as f32 / RANSAC_K as f32;
                        pl.report_status(value);
                    }
                    let rng = &mut SmallRng::from_rng(rand::thread_rng()).unwrap();

                    // Select points
                    let inliers = linked_tracks
                        .choose_multiple(rng, RANSAC_N)
                        .map(|track| track.to_owned())
                        .collect::<Vec<_>>();
                    if inliers.len() != RANSAC_N {
                        return None;
                    }

                    // TODO: check if points are collinear?
                    let p = self.find_projection_matrix(&inliers)?;

                    let mut projection = projection.clone();
                    projection.push(p);

                    let (count, _) = PerspectiveTriangulation::tracks_reprojection_error(
                        &inliers,
                        &projection,
                        true,
                    );
                    if count < RANSAC_N {
                        // Inliers cannot be reliably reprojected.
                        return None;
                    }

                    let (count, error) = PerspectiveTriangulation::tracks_reprojection_error(
                        &unlinked_tracks,
                        &projection,
                        false,
                    );

                    Some((p, count, error))
                })
                .reduce(
                    || (Matrix3x4::identity(), 0, f64::MAX),
                    |(projection1, count1, error1), (projection2, count2, error2)| {
                        if count1 > count2 || (count1 == count2 && error1 < error2) {
                            (projection1, count1, error1)
                        } else {
                            (projection2, count2, error2)
                        }
                    },
                );

            if count >= RANSAC_D {
                result = Some(projection)
            }
            if count >= RANSAC_D_EARLY_EXIT {
                break;
            }
        }

        result
    }

    fn find_projection_matrix(&self, inliers: &Vec<Track>) -> Option<Matrix3x4<f64>> {
        const SVD_ROWS: usize = RANSAC_N * 2;
        let points4d = inliers
            .iter()
            .filter_map(|inlier| {
                let point4d =
                    PerspectiveTriangulation::triangulate_track(inlier, &self.projection)?;
                let projection = inlier.last()?.to_owned()?;
                Some((point4d, projection))
            })
            .collect::<Vec<_>>();

        if points4d.len() < RANSAC_N {
            return None;
        }

        let mut a = SMatrix::<f64, SVD_ROWS, 12>::zeros();
        for i in 0..RANSAC_N {
            let inlier = points4d[i];

            let point4d = inlier.0;
            let projection = inlier.1;
            let x = projection.1 as f64;
            let y = projection.0 as f64;
            let w = 1.0;
            a[(i * 2, 0)] = w * point4d.x;
            a[(i * 2, 1)] = w * point4d.y;
            a[(i * 2, 2)] = w * point4d.z;
            a[(i * 2, 3)] = w * point4d.w;
            a[(i * 2, 8)] = -x * point4d.x;
            a[(i * 2, 9)] = -x * point4d.y;
            a[(i * 2, 10)] = -x * point4d.z;
            a[(i * 2, 11)] = -x * point4d.w;
            a[(i * 2 + 1, 4)] = w * point4d.x;
            a[(i * 2 + 1, 5)] = w * point4d.y;
            a[(i * 2 + 1, 6)] = w * point4d.z;
            a[(i * 2 + 1, 7)] = w * point4d.w;
            a[(i * 2 + 1, 8)] = -y * point4d.x;
            a[(i * 2 + 1, 9)] = -y * point4d.y;
            a[(i * 2 + 1, 10)] = -y * point4d.z;
            a[(i * 2 + 1, 11)] = -y * point4d.w;
        }
        let usv = a.svd(false, true);
        let vt = &usv.v_t?;
        let vtc = vt.row(vt.nrows() - 1);
        let p = Matrix3x4::new(
            vtc[0], vtc[1], vtc[2], vtc[3], vtc[4], vtc[5], vtc[6], vtc[7], vtc[8], vtc[9],
            vtc[10], vtc[11],
        );

        Some(p)
    }

    fn tracks_reprojection_error(
        tracks: &[Track],
        projection: &[Matrix3x4<f64>],
        inliers: bool,
    ) -> (usize, f64) {
        let threshold = if inliers { RANSAC_INLIERS_T } else { RANSAC_T };
        // For inliers, check reprojection error only in the last images.
        // For normal points, ignore the first images.
        let skip = if inliers { projection.len() - 1 } else { 2 };
        tracks
            .iter()
            .filter_map(|track| {
                PerspectiveTriangulation::point_reprojection_error(track, projection, skip)
                    .filter(|error| *error < threshold)
            })
            .fold((0, 0.0f64), |(count, error), match_error| {
                (count + 1, error.max(match_error))
            })
    }

    #[inline]
    fn point_reprojection_error(
        track: &Track,
        projection: &[Matrix3x4<f64>],
        skip: usize,
    ) -> Option<f64> {
        let point4d = PerspectiveTriangulation::triangulate_track(track, &projection)?;
        projection
            .iter()
            .enumerate()
            .skip(skip)
            .filter_map(|(i, p)| {
                let original = track[i]?;
                let mut projected = p * point4d;
                projected.unscale_mut(projected.z * projected.z.signum());
                let dx = projected.x - original.1 as f64;
                let dy = projected.y - original.0 as f64;
                let error = (dx * dx + dy * dy).sqrt();
                Some(error)
            })
            .reduce(|acc, val| acc.max(val))
    }

    fn triangulate_tracks(&self, tracks: &[Track]) -> Surface {
        tracks
            .par_iter()
            .flat_map(|track| {
                let point4d = PerspectiveTriangulation::triangulate_track(track, &self.projection)?;

                let (point2d, index) = track
                    .iter()
                    .enumerate()
                    .flat_map(|(i, point)| {
                        point.map(|point| ((point.1 as usize, point.0 as usize), i))
                    })
                    .next()?;

                let w = point4d.w * point4d.z.signum() * point4d.w.signum();
                let point3d = Vector3::new(point4d.x / w, point4d.y / w, point4d.z / w);

                let point = Point::new(point2d, point3d, index);
                Some(point)
            })
            .collect::<Vec<_>>()
    }

    fn extend_tracks(&mut self, correlated_points: &DMatrix<Option<Match>>, index: usize) {
        let mut remaining_points = correlated_points.clone();

        self.tracks.iter_mut().for_each(|track| {
            let last_pos = track
                .last()
                .map(|last| {
                    last.map(|last| correlated_points[(last.0 as usize, last.1 as usize)])
                        .flatten()
                })
                .flatten();

            track.push(last_pos);
            if let Some(last_pos) = last_pos {
                remaining_points[(last_pos.0 as usize, last_pos.1 as usize)] = None
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
                        let mut track = vec![None; index];
                        track.push(Some((row as u32, col as u32)));
                        track.push(*m);

                        Some(track)
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        self.tracks.append(&mut new_tracks);
    }

    fn bundle_adjustment(&mut self) -> Result<(), TriangulationError> {
        const MAX_JACOBIAN: usize = 500_000_000;
        let full_jacobian = std::mem::size_of::<f64>()
            * (12 * self.projection.len() + 3 * self.tracks.len())
            * (2 * self.projection.len() * self.tracks.len());

        let full_jacobian = std::mem::size_of::<f64>()
            * (12 * self.projection.len())
            * (2 * self.projection.len() * self.tracks.len());
        let tracks_stride = full_jacobian / MAX_JACOBIAN + 1;

        for track_offset in 0..tracks_stride {
            let tracks = (0..self.tracks.len())
                .into_iter()
                .skip(track_offset)
                .step_by(tracks_stride)
                .map(|i| self.tracks[i].to_owned())
                .collect::<Vec<_>>();
            println!(
                "Processing offset {:?} out of {:?} {} {}",
                track_offset,
                tracks_stride,
                tracks.len(),
                self.tracks.len()
            );
            let problem = ReprojectionErrorMinimization::new(
                &self.projection.as_slice(),
                &tracks,
                false,
                true,
            );
            let (result, report) = LevenbergMarquardt::new().minimize(problem);
            if !report.termination.was_successful() {
                return Err(TriangulationError::from_string(format!(
                    "Failed bundle adjustment: {:?}",
                    report.termination
                )));
            }
            self.projection = result.extract_projection();
        }

        Ok(())
        /*
        let tracks_stride = self.tracks.len() / 10 + 1;
        let mut surface = Vec::new();
        for track_offset in 0..tracks_stride {
            let tracks = (0..self.tracks.len())
                .into_iter()
                .skip(track_offset)
                .step_by(tracks_stride)
                .map(|i| self.tracks[i].to_owned())
                .collect::<Vec<_>>();
            println!(
                "Processing offset {:?} out of {:?} {}",
                track_offset,
                tracks_stride,
                tracks.len()
            );
            let problem = ReprojectionErrorMinimization::new(
                self.projection.as_slice(),
                &tracks,
                true,
                false,
            );
            let (result, report) = LevenbergMarquardt::new().minimize(problem);
            if !report.termination.was_successful() {
                println!("Failed bundle adjustment: {:?}", report.termination);
                continue;
                /*
                return Err(TriangulationError::from_string(format!(
                    "Failed bundle adjustment: {:?}",
                    report.termination
                )));
                */
            }
            let mut points3d = PerspectiveTriangulation::triangulate_tracks(&self, &tracks);

            result
                .extract_points3d()
                .iter()
                .enumerate()
                .for_each(|(i, point3d)| {
                    if let Some(point3d) = point3d {
                        points3d[i].reconstructed = *point3d;
                    }
                });

            surface.append(&mut points3d);
        }
        Ok(surface)
        */
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
        let range_xy = (max_x - min_x).min(max_y - min_y);

        points3d
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
                            let point1 = Some((*row as u32, col as u32));
                            let point2 = correlated_points[(*row, col)];
                            let point2 = if let Some(point2) = point2 {
                                Some((point2.0 as u32, point2.1 as u32))
                            } else {
                                return false;
                            };
                            let point4d = PerspectiveTriangulation::triangulate_track(
                                &vec![point1, point2],
                                &[p1, p2],
                            );
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

struct ReprojectionErrorMinimization<'a> {
    /// Optimization parameters vector.
    /// First 12 items are columns of P2; followed by 3D coordinates of triangulated points.
    params: OVector<f64, Dyn>,
    projection: Vec<Matrix3x4<f64>>,
    tracks: &'a [Track],
    points_offset: usize,
    optimize_points: bool,
    optimize_projection: bool,
}

impl ReprojectionErrorMinimization<'_> {
    pub fn new<'a>(
        projection: &[Matrix3x4<f64>],
        tracks: &'a [Track],
        optimize_points: bool,
        optimize_projection: bool,
    ) -> ReprojectionErrorMinimization<'a> {
        let points_offset = if optimize_projection {
            projection.len() * 12
        } else {
            0
        };
        let parameters_len = if optimize_points {
            points_offset + tracks.len() * 3
        } else {
            points_offset
        };
        let mut params = OVector::<f64, Dyn>::zeros(parameters_len);
        if optimize_projection {
            for (i, p) in projection.iter().enumerate() {
                for col in 0..4 {
                    for row in 0..3 {
                        params[i * 12 + col * 3 + row] = p[(row, col)];
                    }
                }
            }
        }
        if optimize_points {
            for tr_i in 0..tracks.len() {
                let track = &tracks[tr_i];
                let point4d = if let Some(point4d) =
                    PerspectiveTriangulation::triangulate_track(track, projection)
                {
                    point4d.unscale(point4d.w)
                } else {
                    Vector4::zeros()
                };
                for m_c in 0..3 {
                    params[points_offset + tr_i * 3 + m_c] = point4d[m_c];
                }
            }
        }
        ReprojectionErrorMinimization {
            params,
            projection: projection.into(),
            tracks,
            points_offset,
            optimize_points,
            optimize_projection,
        }
    }

    fn extract_projection(&self) -> Vec<Matrix3x4<f64>> {
        if self.optimize_projection {
            (0..self.projection.len())
                .into_iter()
                .map(|i| {
                    let mut p = Matrix3x4::zeros();
                    for col in 0..4 {
                        for row in 0..3 {
                            p[(row, col)] = self.params[i * 12 + col * 3 + row];
                        }
                    }
                    p
                })
                .collect::<Vec<_>>()
        } else {
            self.projection.clone()
        }
    }

    fn extract_points3d(&self) -> Vec<Option<Vector3<f64>>> {
        let points_offset = self.points_offset;

        self.tracks
            .iter()
            .enumerate()
            .par_bridge()
            .map(|(tr_i, _track)| {
                if self.optimize_points {
                    let mut point3d = Vector3::zeros();
                    for m_c in 0..3 {
                        point3d[m_c] = self.params[points_offset + tr_i * 3 + m_c];
                    }
                    Some(point3d)
                } else {
                    // TODO: triangulate points here?
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
        let points_offset = self.points_offset;
        let projection = self.extract_projection();
        // Residuals contain point reprojection errors.
        let mut residuals = OVector::<f64, Dyn>::zeros(self.tracks.len() * 2 * projection.len());
        for tr_i in 0..self.tracks.len() {
            let track = &self.tracks[tr_i];
            let point4d = if self.optimize_points {
                let mut point4d = Vector4::zeros();
                for m_c in 0..3 {
                    point4d[m_c] = self.params[points_offset + tr_i * 3 + m_c];
                }
                point4d[3] = 1.0;
                Some(point4d)
            } else {
                PerspectiveTriangulation::triangulate_track(track, &projection)
            };
            projection.iter().enumerate().for_each(|(i, p)| {
                let offset = tr_i * projection.len() * 2 + i * 2;
                let original = if let Some(original) = track[i] {
                    original
                } else {
                    return;
                };
                let point4d = if let Some(point4d) = point4d {
                    point4d
                } else {
                    return;
                };
                let mut projected = p * point4d;
                projected.unscale_mut(projected.z * projected.z.signum());
                let dx = original.1 as f64 - projected.x;
                let dy = original.0 as f64 - projected.y;
                residuals[offset + 0] = dx;
                residuals[offset + 1] = dy;
            });
        }
        Some(residuals)
    }

    fn jacobian(&self) -> Option<OMatrix<f64, Dyn, Dyn>> {
        let points_offset = self.points_offset;
        let parameters_len = if self.optimize_points {
            points_offset + self.tracks.len() * 3
        } else {
            points_offset
        };
        // TODO: use a sparse LM method?
        let projection = self.extract_projection();
        let residuals_count = 2 * projection.len();
        let mut jac =
            OMatrix::<f64, Dyn, Dyn>::zeros(self.tracks.len() * residuals_count, parameters_len);
        // Write a row for each residual (reprojection error).
        // Using a symbolic formula (not finite differences/central difference), check the Rust LM library for more info.
        for tr_i in 0..self.tracks.len() {
            // Read data from parameters
            let p3d = if self.optimize_points {
                Vector4::new(
                    self.params[points_offset + tr_i * 3 + 0],
                    self.params[points_offset + tr_i * 3 + 1],
                    self.params[points_offset + tr_i * 3 + 2],
                    1.0,
                )
            } else {
                let track = &self.tracks[tr_i];
                if let Some(point4d) =
                    PerspectiveTriangulation::triangulate_track(track, &projection)
                {
                    point4d.unscale(point4d.w)
                } else {
                    continue;
                }
            };
            for (p_i, p) in projection.iter().enumerate() {
                let rx = tr_i * residuals_count + p_i * 2 + 0;
                let ry = tr_i * residuals_count + p_i * 2 + 1;
                // P is the projection matrix for image
                // Prc is the r-th row, c-th column of P
                // X is a 3D coordinate (4-component vector [x y z 1])

                // Image contains rx and ry (residuals for x and y), affected by projection matrix and the point coordinates.
                // rx = -(P11*x+P12*y+P13*z+P14)/(P31*x+P32*y+P33*z+P34)
                // ry = -(P21*x+P22*y+P23*z+P24)/(P31*x+P32*y+P33*z+P34)
                // To keep things sane, create some aliases
                // Poi -> i-th element of 3D point e.g. x, y, z, or w
                // Pr1 = P11*x+P12*y+P13*z+P14
                // Pr2 = P21*x+P22*y+P23*z+P24
                // Pr3 = P31*x+P32*y+P33*z+P34
                let p_r = p * p3d;
                if self.optimize_projection {
                    // drx/dP1i = -Poi/(P31*x+P32*y+P33*z+P34) = -Poi/Pr3
                    for p_col in 0..4 {
                        jac[(rx, p_i * 12 + p_col * 3 + 0)] = -p3d[p_col] / p_r[2];
                    }
                    // dry/dP2i = -Poi/(P31*x+P32*y+P33*z+P34) = -Poi/Pr3
                    for p_col in 0..4 {
                        jac[(ry, p_i * 12 + p_col * 3 + 1)] = -p3d[p_col] / p_r[2];
                    }
                    // drx/dP3i = Poi*(P11*x+P12*y+P13*z+P14)/((P31*x+P32*y+P33*z+P34)^2) = Poi*Pr1/(Pr3^2)
                    for p_col in 0..4 {
                        jac[(rx, p_i * 12 + p_col * 3 + 2)] =
                            p3d[p_col] * p_r[0] / (p_r[2] * p_r[2]);
                    }
                    // dry/dP3i = Poi*(P21*x+P22*y+P23*z+P24)/((P31*x+P32*y+P33*z+P34)^2) = Poi*Pr2/(Pr3^2)
                    for p_col in 0..4 {
                        jac[(ry, p_i * 12 + p_col * 3 + 2)] =
                            p3d[p_col] * p_r[1] / (p_r[2] * p_r[2]);
                    }
                }
                if self.optimize_points {
                    // drx/dx = -(P11*(P32*y+P33*z+P34)-P31*(P12*y+P13*z+P14))/(Pr3^2) = -(P11*Pr3[x=0]-P31*Pr1[x=0])/(Pr3^2)
                    // drx/di = -(P1i*Pr3[i=0]-P3i*Pr1[i=0])/(Pr3^2)
                    // dry/dx = -(P21*(P32*y+P33*z+P34)-P31*(P22*y+P23*z+P24))/(Pr3^2) = -(P21*Pr3[x=0]-P31*Pr2[x=0])/(Pr3^2)
                    // dry/di = -(P2i*Pr3[i=0]-P3i*Pr2[i=0])/(Pr3^2)
                    for coord in 0..3 {
                        // Create a vector where coord = 0
                        let mut vec_diff = p3d;
                        vec_diff[coord] = 0.0;
                        // Create projection where coord = 0
                        let p_r_diff = p * vec_diff;
                        jac[(rx, points_offset + tr_i * 3 + coord)] =
                            -(p[(0, coord)] * p_r_diff[2] - p[(2, coord)] * p_r_diff[0])
                                / (p_r[2] * p_r[2]);
                        jac[(ry, points_offset + tr_i * 3 + coord)] =
                            -(p[(1, coord)] * p_r_diff[2] - p[(2, coord)] * p_r_diff[1])
                                / (p_r[2] * p_r[2]);
                    }
                }
            }
        }

        Some(jac)
    }
}

#[derive(Debug)]
pub struct TriangulationError {
    msg: String,
}

impl TriangulationError {
    fn new(msg: &'static str) -> TriangulationError {
        TriangulationError {
            msg: msg.to_owned(),
        }
    }

    fn from_string(msg: String) -> TriangulationError {
        TriangulationError { msg }
    }
}

impl std::error::Error for TriangulationError {}

impl fmt::Display for TriangulationError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.msg)
    }
}
