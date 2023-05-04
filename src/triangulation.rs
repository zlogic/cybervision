use std::{fmt, sync::atomic::AtomicUsize, sync::atomic::Ordering as AtomicOrdering};

use nalgebra::{
    DMatrix, Matrix2x3, Matrix3, Matrix3x4, MatrixXx1, MatrixXx4, SMatrix, Vector2, Vector3,
    Vector4,
};

use rand::seq::SliceRandom;
use rand::{rngs::SmallRng, SeedableRng};
use rayon::prelude::*;

use crate::correlation;

const PERSPECTIVE_VALUE_RANGE: f64 = 100.0;
const BUNDLE_ADJUSTMENT_MAX_ITERATIONS: usize = 1000;
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

    fn triangulate_all<PL: ProgressListener>(
        &mut self,
        progress_listener: Option<&PL>,
    ) -> Result<Surface, TriangulationError> {
        let points3d = self.bundle_adjustment(progress_listener)?;
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

    fn bundle_adjustment<PL: ProgressListener>(
        &mut self,
        progress_listener: Option<&PL>,
    ) -> Result<Surface, TriangulationError> {
        let mut ba: BundleAdjustment = BundleAdjustment::new(self.projection.clone(), &self.tracks);
        let points3d = ba.optimize(progress_listener)?;
        drop(ba);

        let surface = self
            .tracks
            .iter()
            .enumerate()
            .par_bridge()
            .flat_map(|(i, track)| {
                let (point2d, index) = track
                    .iter()
                    .enumerate()
                    .flat_map(|(i, point)| {
                        point.map(|point| ((point.1 as usize, point.0 as usize), i))
                    })
                    .next()?;

                let point3d = points3d[i]?;

                let point = Point::new(point2d, point3d, index);
                Some(point)
            })
            .collect::<Vec<_>>();

        Ok(surface)
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

struct BundleAdjustment<'a> {
    projection: Vec<Matrix3x4<f64>>,
    tracks: &'a [Track],
    points3d: Vec<Option<Vector3<f64>>>,
    covariance: f64,
    mu: f64,
}

type Matrix12x12<T> =
    nalgebra::Matrix<T, nalgebra::U12, nalgebra::U12, nalgebra::ArrayStorage<T, 12, 12>>;
type Matrix12x3<T> =
    nalgebra::Matrix<T, nalgebra::U12, nalgebra::U3, nalgebra::ArrayStorage<T, 12, 3>>;
type Matrix12x1<T> =
    nalgebra::Matrix<T, nalgebra::U12, nalgebra::U1, nalgebra::ArrayStorage<T, 12, 1>>;
type Matrix2x12<T> =
    nalgebra::Matrix<T, nalgebra::U2, nalgebra::U12, nalgebra::ArrayStorage<T, 2, 12>>;

impl BundleAdjustment<'_> {
    const PROJECTION_PARAMETERS: usize = 12;
    const INITIAL_MU: f64 = 1E-2;
    const GRADIENT_EPSILON: f64 = 1E-12;
    const DELTA_EPSILON: f64 = 1E-12;
    const RESIDUAL_EPSILON: f64 = 1E-12;
    const RESIDUAL_REDUCTION_EPSILON: f64 = 0.0;

    fn new<'a>(projection: Vec<Matrix3x4<f64>>, tracks: &'a [Track]) -> BundleAdjustment {
        let points3d = BundleAdjustment::triangulate_tracks(&projection, tracks);
        // For now, identity covariance is acceptable.
        let covariance = 1.0;
        BundleAdjustment {
            projection,
            tracks,
            points3d,
            covariance,
            mu: BundleAdjustment::INITIAL_MU,
        }
    }

    fn triangulate_tracks(
        projection: &Vec<Matrix3x4<f64>>,
        tracks: &[Track],
    ) -> Vec<Option<Vector3<f64>>> {
        tracks
            .par_iter()
            .map(|track| {
                let point4d = PerspectiveTriangulation::triangulate_track(track, projection)?;

                let w = point4d.w * point4d.z.signum() * point4d.w.signum();
                let point3d = point4d.remove_row(3).unscale(w);

                Some(point3d)
            })
            .collect::<Vec<_>>()
    }

    fn jacobian_a(&self, point3d: &Vector3<f64>, projection: &Matrix3x4<f64>) -> Matrix2x12<f64> {
        // Projection matrix is unrolled to a single row with 12 elements.
        // jac is a 2x12 Jacobian for point i and projection matrix j.
        let mut jac = Matrix2x12::zeros();
        let point4d = point3d.insert_row(3, 1.0);

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
        // dxpro/dP1i = Poi/(P31*x+P32*y+P33*z+P34) = Poi/Pr3
        for p_col in 0..4 {
            jac[(0, p_col * 3 + 0)] = point4d[p_col] / p_r[2];
        }
        // dypro/dP2i = Poi/(P31*x+P32*y+P33*z+P34) = Poi/Pr3
        for p_col in 0..4 {
            jac[(1, p_col * 3 + 1)] = point4d[p_col] / p_r[2];
        }
        // dxpro/dP3i = Poi*(P11*x+P12*y+P13*z+P14)/((P31*x+P32*y+P33*z+P34)^2) = Poi*Pr1/(Pr3^2)
        for p_col in 0..4 {
            jac[(0, p_col * 3 + 2)] = -point4d[p_col] * p_r[0] / (p_r[2] * p_r[2]);
        }
        // dypro/dP3i = Poi*(P21*x+P22*y+P23*z+P24)/((P31*x+P32*y+P33*z+P34)^2) = Poi*Pr2/(Pr3^2)
        for p_col in 0..4 {
            jac[(1, p_col * 3 + 2)] = -point4d[p_col] * p_r[1] / (p_r[2] * p_r[2]);
        }

        jac
    }

    fn jacobian_b(&self, point3d: &Vector3<f64>, projection: &Matrix3x4<f64>) -> Matrix2x3<f64> {
        // Point coordinates matrix is converted to a single row with coordinates x, y and z.
        // jac is a 2x3 Jacobian for point i and projection matrix j.
        let mut jac = Matrix2x3::zeros();
        let point4d = point3d.insert_row(3, 1.0);

        // See jacobian_a for more details.
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
    fn residual(&self, point_i: usize, projection_j: usize) -> Vector2<f64> {
        let point3d = &self.points3d[point_i];
        let projection = &self.projection[projection_j];
        let original = if let Some(original) = self.tracks[point_i][projection_j] {
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

    fn residual_a(&self, projection_j: usize) -> Option<Matrix12x1<f64>> {
        let mut residual = Matrix12x1::zeros();
        let projection = &self.projection[projection_j];
        for (point_i, point3d) in self.points3d.iter().enumerate() {
            let point3d = (*point3d)?;
            let a = self.jacobian_a(&point3d, &projection);
            let res = self.residual(point_i, projection_j);
            residual += a.transpose() * self.covariance * res;
        }
        Some(residual)
    }

    fn residual_b(&self, point_i: usize) -> Option<Vector3<f64>> {
        let mut residual = Vector3::zeros();
        let point3d = &self.points3d[point_i];
        for (projection_j, projection) in self.projection.iter().enumerate() {
            let point3d = (*point3d)?;
            let b = self.jacobian_b(&point3d, &projection);
            let res = self.residual(point_i, projection_j);
            residual += b.transpose() * self.covariance * res;
        }
        Some(residual)
    }

    #[inline]
    fn calculate_u(&self, projection: &Matrix3x4<f64>) -> Option<Matrix12x12<f64>> {
        let mut u = Matrix12x12::zeros();
        for point3d in &self.points3d {
            let point3d = (*point3d)?;
            let a = self.jacobian_a(&point3d, &projection);
            u += a.transpose() * self.covariance * a;
        }
        for i in 0..BundleAdjustment::PROJECTION_PARAMETERS {
            u[(i, i)] += self.mu;
        }
        Some(u)
    }

    #[inline]
    fn calculate_v_inv(&self, point3d: &Option<Vector3<f64>>) -> Option<Matrix3<f64>> {
        let mut v = Matrix3::zeros();
        let point3d = (*point3d)?;
        for projection in &self.projection {
            let b = self.jacobian_b(&point3d, &projection);
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
    fn calculate_w(
        &self,
        point3d: &Option<Vector3<f64>>,
        projection: &Matrix3x4<f64>,
    ) -> Option<Matrix12x3<f64>> {
        let point3d = (*point3d)?;
        let a = self.jacobian_a(&point3d, &projection);
        let b = self.jacobian_b(&point3d, &projection);
        Some(a.transpose() * self.covariance * b)
    }

    #[inline]
    fn calculate_y(
        &self,
        point3d: &Option<Vector3<f64>>,
        projection: &Matrix3x4<f64>,
    ) -> Option<Matrix12x3<f64>> {
        let v_inv = self.calculate_v_inv(point3d)?;
        let w = self.calculate_w(point3d, projection)?;
        Some(w * v_inv)
    }

    fn calculate_s(&self) -> DMatrix<f64> {
        let mut s = DMatrix::<f64>::zeros(
            self.projection.len() * BundleAdjustment::PROJECTION_PARAMETERS,
            self.projection.len() * BundleAdjustment::PROJECTION_PARAMETERS,
        );
        // Divide blocks for parallelization.
        let s_blocks = (0..self.projection.len())
            .into_iter()
            .flat_map(|j| {
                (0..self.projection.len())
                    .into_iter()
                    .map(|k| (j, k))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        let s_blocks = s_blocks
            .into_par_iter()
            .map(|(j, k)| {
                let projection_j = &self.projection[j];
                let mut s_jk = Matrix12x12::zeros();
                if j == k {
                    let u = self.calculate_u(projection_j)?;
                    s_jk += u;
                }
                let projection_k = &self.projection[k];
                for point3d in self.points3d.iter() {
                    let y_ij = self.calculate_y(point3d, projection_j)?;
                    let w_ik = self.calculate_w(point3d, projection_k)?;
                    let y_ij_w = y_ij * w_ik.transpose();
                    s_jk -= y_ij_w;
                }
                Some((j, k, s_jk))
            })
            .collect::<Vec<_>>();

        s_blocks.iter().for_each(|block| {
            let (j, k, s_jk) = if let Some((j, k, s_jk)) = block {
                (j, k, s_jk)
            } else {
                return;
            };
            s.fixed_view_mut::<12, 12>(
                j * BundleAdjustment::PROJECTION_PARAMETERS,
                k * BundleAdjustment::PROJECTION_PARAMETERS,
            )
            .copy_from(s_jk);
        });

        s
    }

    fn calculate_e(&self) -> MatrixXx1<f64> {
        let mut e =
            MatrixXx1::zeros(self.projection.len() * BundleAdjustment::PROJECTION_PARAMETERS);

        let e_blocks = self
            .projection
            .iter()
            .enumerate()
            .par_bridge()
            .map(|(j, projection_j)| {
                let mut e_j = self.residual_a(j)?;

                for (i, point3d) in self.points3d.iter().enumerate() {
                    let y_ij = self.calculate_y(point3d, projection_j)?;
                    let res = self.residual_b(i)?;
                    e_j -= y_ij * res;
                }
                Some((j, e_j))
            })
            .collect::<Vec<_>>();

        e_blocks.iter().for_each(|block| {
            let (j, e_j) = if let Some((j, e_j)) = block {
                (j, e_j)
            } else {
                return;
            };
            e.fixed_view_mut::<12, 1>(j * BundleAdjustment::PROJECTION_PARAMETERS, 0)
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

                for (j, projection_j) in self.projection.iter().enumerate() {
                    let w_ij = self.calculate_w(point3d, projection_j)?;
                    let delta_a_j =
                        delta_a.fixed_view::<12, 1>(j * BundleAdjustment::PROJECTION_PARAMETERS, 0);
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
        let mut residuals = MatrixXx1::zeros(self.points3d.len() * self.projection.len() * 2);

        for (i, point_i) in self.points3d.iter().enumerate() {
            if point_i.is_none() {
                continue;
            }
            for (j, _) in self.projection.iter().enumerate() {
                let residual_b_i = self.residual(i, j);
                residuals
                    .fixed_rows_mut::<2>(i * self.projection.len() * 2 + j)
                    .copy_from(&residual_b_i);
            }
        }

        residuals
    }

    fn calculate_jt_residual(&self) -> MatrixXx1<f64> {
        let mut g = MatrixXx1::zeros(
            self.projection.len() * BundleAdjustment::PROJECTION_PARAMETERS
                + self.points3d.len() * 3,
        );

        // g = Jt * residual
        // First 12*m rows of Jt are residuals from projection matrices.
        for (i, point_i) in self.points3d.iter().enumerate() {
            let point_i = if let Some(point_i) = point_i {
                point_i
            } else {
                continue;
            };
            for (j, projection_j) in self.projection.iter().enumerate() {
                let jac_a_i = self.jacobian_a(point_i, projection_j);
                let residual_a_i = self.residual(i, j);
                let block = jac_a_i.tr_mul(&residual_a_i);
                let mut target_block =
                    g.fixed_rows_mut::<12>(j * BundleAdjustment::PROJECTION_PARAMETERS);
                target_block += block;
            }
        }

        // Last 3*n rows of Jt are residuals from point coordinates.
        let mut points_target = g.rows_mut(
            self.projection.len() * BundleAdjustment::PROJECTION_PARAMETERS,
            self.points3d.len() * 3,
        );
        for (i, point_i) in self.points3d.iter().enumerate() {
            let point_i = if let Some(point_i) = point_i {
                point_i
            } else {
                continue;
            };
            for (j, projection_j) in self.projection.iter().enumerate() {
                let jac_b_i = self.jacobian_b(point_i, projection_j);
                let residual_b_i = self.residual(i, j);
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
        for (j, projection_j) in self.projection.iter_mut().enumerate() {
            for col in 0..4 {
                for row in 0..3 {
                    projection_j[(row, col)] += delta[(
                        BundleAdjustment::PROJECTION_PARAMETERS * j + col * 3 + row,
                        0,
                    )]
                }
            }
        }

        let points_source = delta.rows(
            self.projection.len() * BundleAdjustment::PROJECTION_PARAMETERS,
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
    ) -> Result<Vec<Option<Vector3<f64>>>, TriangulationError> {
        // Levenberg-Marquardt optimization loop.
        let mut residual = self.calculate_residual_vector();
        let mut jt_residual = self.calculate_jt_residual();

        self.mu = BundleAdjustment::INITIAL_MU;
        let mut nu = 2.0;

        let mut found = false;

        for iter in 0..BUNDLE_ADJUSTMENT_MAX_ITERATIONS {
            if let Some(pl) = progress_listener {
                let value = iter as f32 / BUNDLE_ADJUSTMENT_MAX_ITERATIONS as f32;
                pl.report_status(value);
            }
            if jt_residual.max().abs() <= BundleAdjustment::GRADIENT_EPSILON {
                found = true;
                break;
            }
            let delta = self.calculate_delta_step();
            let delta = if let Some(delta) = delta {
                delta
            } else {
                return Err(TriangulationError::new("Failed to compute delta vector"));
            };

            let params_norm;
            {
                let sum_projection = self
                    .projection
                    .iter()
                    .map(|p| p.norm_squared())
                    .sum::<f64>();
                let sum_points = self
                    .points3d
                    .iter()
                    .filter_map(|p| Some((*p)?.norm_squared()))
                    .sum::<f64>();
                params_norm = (sum_projection + sum_points).sqrt();
            }

            if delta.norm()
                <= BundleAdjustment::DELTA_EPSILON * (params_norm + BundleAdjustment::DELTA_EPSILON)
            {
                found = true;
                break;
            }

            let current_projection = self.projection.clone();
            let current_points3d = self.points3d.clone();

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
                self.projection = current_projection;
                self.points3d = current_points3d;
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

        let mut result = Vec::new();
        result.append(&mut self.points3d);

        Ok(result)
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
