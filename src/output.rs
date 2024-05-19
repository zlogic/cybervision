use crate::{
    data::{Grid, Point2D},
    triangulation,
};
use core::fmt;
use std::{
    collections::HashMap,
    fs::File,
    io::{BufWriter, Write},
    path::Path,
    sync::atomic::AtomicUsize,
    sync::atomic::Ordering as AtomicOrdering,
};

use image::{RgbImage, Rgba, RgbaImage};
use nalgebra::Vector3;
use spade::{DelaunayTriangulation, HasPosition, Point2, Triangulation};

use rayon::prelude::*;

const PROJECTIONS_INDEX_GRID_SIZE: usize = 1000;
const MAX_NORMAL_COS: f64 = 0.2;

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum InterpolationMode {
    Delaunay,
    None,
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum VertexMode {
    Plain,
    Color,
    Texture,
}

pub trait ProgressListener
where
    Self: Sync + Sized,
{
    fn report_status(&self, pos: f32);
}

#[derive(Debug, Clone)]
struct Point {
    point: Point2<f64>,
    track_i: usize,
}

#[derive(Debug, PartialEq, Clone, Copy)]
struct Polygon {
    camera_i: usize,
    vertices: [usize; 3],
}

impl Polygon {
    fn new(camera_i: usize, vertices: [usize; 3]) -> Polygon {
        Polygon { camera_i, vertices }
    }
}

struct CameraGrid {
    camera_i: usize,
    min_x: f64,
    max_x: f64,
    min_y: f64,
    max_y: f64,
    grid: Grid<Vec<usize>>,
    grid_step: f64,
}

impl CameraGrid {
    fn new(camera_i: usize) -> CameraGrid {
        CameraGrid {
            camera_i,
            min_x: f64::MAX,
            max_x: f64::MIN,
            min_y: f64::MAX,
            max_y: f64::MIN,
            grid: Grid::new(0, 0, vec![]),
            grid_step: 0.0,
        }
    }

    fn from_point(camera_i: usize, point: Point2D<f64>) -> CameraGrid {
        CameraGrid {
            camera_i,
            min_x: point.x,
            max_x: point.x,
            min_y: point.y,
            max_y: point.y,
            grid: Grid::new(0, 0, vec![]),
            grid_step: 0.0,
        }
    }

    fn merge(&self, other: &CameraGrid) -> CameraGrid {
        CameraGrid {
            camera_i: self.camera_i,
            min_x: self.min_x.min(other.min_x),
            max_x: self.max_x.max(other.max_x),
            min_y: self.min_y.min(other.min_y),
            max_y: self.max_y.max(other.max_y),
            grid: Grid::new(0, 0, vec![]),
            grid_step: 0.0,
        }
    }

    fn get_step(&self) -> f64 {
        (self.max_x - self.min_x).min(self.max_y - self.min_y) / PROJECTIONS_INDEX_GRID_SIZE as f64
    }

    fn index_projections(&mut self, points: &triangulation::Surface) {
        let grid_step = self.get_step();
        let width = ((self.max_x - self.min_x) / grid_step).ceil() as usize;
        let height = ((self.max_y - self.min_y) / grid_step).ceil() as usize;
        let mut index = Grid::<Vec<usize>>::new(width, height, vec![]);
        points
            .iter_tracks()
            .enumerate()
            .for_each(|(track_i, track)| {
                let point3d = if let Some(point3d) = track.get_point3d() {
                    point3d
                } else {
                    return;
                };
                let point3d_in_camera = points.point_in_camera(self.camera_i, &point3d);
                let x = point3d_in_camera.x / point3d_in_camera.z;
                let y = point3d_in_camera.y / point3d_in_camera.z;
                if x < self.min_x || y < self.min_y || x > self.max_x || y > self.max_y {
                    return;
                }
                let x = (((x - self.min_x) / grid_step).floor() as usize).clamp(0, width - 1);
                let y = (((y - self.min_y) / grid_step).floor() as usize).clamp(0, height - 1);
                index.val_mut(x, y).push(track_i);
            });

        self.grid_step = grid_step;
        self.grid = index;
    }

    fn clear_grid(&mut self) {
        self.grid = Grid::new(0, 0, vec![]);
        self.grid_step = 0.0;
    }
}

struct PolygonIndex {
    index: HashMap<usize, Vec<[usize; 2]>>,
}

impl PolygonIndex {
    fn new() -> PolygonIndex {
        PolygonIndex {
            index: HashMap::new(),
        }
    }

    #[inline]
    fn entry(&self, polygon: &Polygon) -> (usize, [usize; 2]) {
        let v = &polygon.vertices;
        if v[0] < v[1] && v[0] < v[2] {
            (v[0], [v[1], v[2]])
        } else if v[1] < v[0] && v[1] < v[2] {
            (v[1], [v[2], v[0]])
        } else {
            (v[2], [v[0], v[1]])
        }
    }

    fn contains(&self, polygon: &Polygon) -> bool {
        let (i, remain) = self.entry(polygon);
        self.index.get(&i).map_or(false, |polygons| {
            polygons
                .iter()
                .any(|polygon| polygon[0] == remain[0] && polygon[1] == remain[1])
        })
    }

    fn add(&mut self, polygon: &Polygon) -> bool {
        let (i, remain) = self.entry(polygon);
        let mut found = false;
        self.index
            .entry(i)
            .and_modify(|polygons| {
                found = polygons
                    .iter()
                    .any(|polygon| polygon[0] == remain[0] && polygon[1] == remain[1]);
                if !found {
                    polygons.push(remain)
                }
            })
            .or_insert(vec![remain]);
        found
    }
}

struct Mesh {
    points: triangulation::Surface,
    polygons: Vec<Polygon>,
    camera_ranges: Vec<CameraGrid>,
    polygon_index: PolygonIndex,
    point_normals: Vec<Vector3<f64>>,
}

impl Mesh {
    fn create<PL: ProgressListener>(
        surface: triangulation::Surface,
        interpolation: InterpolationMode,
        progress_listener: Option<&PL>,
    ) -> Result<Mesh, OutputError> {
        let point_normals = vec![Vector3::zeros(); surface.tracks_len()];
        let mut surface = Mesh {
            points: surface,
            polygons: vec![],
            camera_ranges: vec![],
            polygon_index: PolygonIndex::new(),
            point_normals,
        };

        if surface.points.cameras_len() == 0 {
            surface.process_camera(0, interpolation, progress_listener)?;
        } else {
            for camera_i in 0..surface.points.cameras_len() {
                surface.process_camera(camera_i, interpolation, progress_listener)?;
            }
        }

        surface.camera_ranges = vec![];
        surface.polygon_index = PolygonIndex::new();

        Ok(surface)
    }

    fn camera_ranges(&self) -> Vec<CameraGrid> {
        (0..self.points.cameras_len())
            .map(|camera_i| {
                self.points
                    .iter_tracks()
                    .par_bridge()
                    .flat_map(|track| {
                        // Do not include invisible points to extend index range.
                        track.get(camera_i)?;
                        let point3d = track.get_point3d()?;
                        let point3d_in_camera = self.points.point_in_camera(camera_i, &point3d);
                        if point3d_in_camera.z <= f64::EPSILON {
                            return None;
                        }
                        Some(CameraGrid::from_point(
                            camera_i,
                            Point2D::new(
                                point3d_in_camera.x / point3d_in_camera.z,
                                point3d_in_camera.y / point3d_in_camera.z,
                            ),
                        ))
                    })
                    .reduce(|| CameraGrid::new(camera_i), |a, b| a.merge(&b))
            })
            .collect::<Vec<_>>()
    }

    #[inline]
    fn get_polygon_points(
        &self,
        polygon: &Polygon,
    ) -> Option<(Vector3<f64>, Vector3<f64>, Vector3<f64>)> {
        let point0 = self.points.get_point(polygon.vertices[0])?;
        let point1 = self.points.get_point(polygon.vertices[1])?;
        let point2 = self.points.get_point(polygon.vertices[2])?;
        Some((point0, point1, point2))
    }

    #[inline]
    fn max_angle_cos(&self, polygon: &Polygon) -> Option<f64> {
        let (point0, point1, point2) = self.get_polygon_points(polygon)?;
        let polygon_normal = (point1 - point0).cross(&(point2 - point0)).normalize();
        let point0_projections = self.points.get_camera_points(polygon.vertices[0]);
        let point1_projections = self.points.get_camera_points(polygon.vertices[1]);
        let point2_projections = self.points.get_camera_points(polygon.vertices[2]);
        let projections_count = point0_projections
            .len()
            .min(point1_projections.len())
            .min(point2_projections.len());

        let affine_projection = self.points.cameras_len() == 0;

        (0..projections_count)
            .filter_map(|camera_i| {
                if point0_projections[camera_i].is_none()
                    || point1_projections[camera_i].is_none()
                    || point2_projections[camera_i].is_none()
                {
                    return None;
                }
                let direction = if affine_projection {
                    Vector3::new(0.0, 0.0, 1.0)
                } else {
                    ((point0 + point1 + point2).unscale(3.0) - self.points.camera_center(camera_i))
                        .normalize()
                };
                let cos_angle = direction.dot(&polygon_normal);
                Some(cos_angle)
            })
            .reduce(|a, b| a.max(b))
    }

    #[inline]
    fn polygon_obstructs(&self, camera_i: usize, grid: &CameraGrid, polygon: &Polygon) -> bool {
        let (point0, point1, point2) = if let Some(points) = self.get_polygon_points(polygon) {
            points
        } else {
            return false;
        };
        let (min_x, max_x, min_y, max_y) = [point0, point1, point2]
            .iter()
            .map(|point| {
                let point3d_in_camera = self.points.point_in_camera(camera_i, point);
                let x = point3d_in_camera.x / point3d_in_camera.z;
                let y = point3d_in_camera.y / point3d_in_camera.z;
                Point2D::new(x, y)
            })
            .fold((f64::MAX, f64::MIN, f64::MAX, f64::MIN), |acc, v| {
                (
                    acc.0.min(v.x),
                    acc.1.max(v.x),
                    acc.2.min(v.y),
                    acc.3.max(v.y),
                )
            });

        let min_x = (((min_x - grid.min_x) / grid.grid_step).floor() as usize)
            .saturating_sub(1)
            .clamp(0, grid.grid.width());
        let max_x = (((max_x - grid.min_x) / grid.grid_step).ceil() as usize)
            .saturating_add(1)
            .clamp(0, grid.grid.width());
        let min_y = (((min_y - grid.min_y) / grid.grid_step).floor() as usize)
            .saturating_sub(1)
            .clamp(0, grid.grid.height());
        let max_y = (((max_y - grid.min_y) / grid.grid_step).ceil() as usize)
            .saturating_add(1)
            .clamp(0, grid.grid.height());

        // Möller–Trumbore intersection algorithm.
        let edge1 = point1 - point0;
        let edge2 = point2 - point0;
        let camera_center = self.points.camera_center(camera_i);
        for y in min_y..max_y {
            for x in min_x..max_x {
                let tracks = grid.grid.val(x, y);
                let obstruction = tracks.iter().any(|track_i| {
                    let point3d = if let Some(point3d) = self.points.get_point(*track_i) {
                        point3d
                    } else {
                        return false;
                    };
                    if point3d == point0 || point3d == point1 || point3d == point2 {
                        return false;
                    }
                    let ray_vector = point3d - camera_center;
                    let ray_cross_e2 = ray_vector.cross(&edge2);
                    let det = edge1.dot(&ray_cross_e2);
                    if det.abs() < f64::EPSILON {
                        return false;
                    }
                    let inv_det = 1.0 / det;
                    let s = camera_center - point0;
                    let u = inv_det * s.dot(&ray_cross_e2);
                    if !(0.0..=1.0).contains(&u) {
                        return false;
                    }
                    let s_cross_e1 = s.cross(&edge1);
                    let v = inv_det * ray_vector.dot(&s_cross_e1);
                    if v < 0.0 || u + v > 1.0 {
                        return false;
                    }
                    let t = inv_det * edge2.dot(&s_cross_e1);
                    t > 1.0
                });

                if obstruction {
                    return true;
                }
            }
        }

        false
    }

    fn polygon_normal(&self, polygon: &Polygon) -> Vector3<f64> {
        let (point0, point1, point2) = if let Some(points) = self.get_polygon_points(polygon) {
            points
        } else {
            return Vector3::zeros();
        };
        let edge_01 = point1 - point0;
        let edge_12 = point2 - point1;
        let edge_20 = point0 - point2;
        // Exclude the shortest edge.
        let dist_01 = edge_01.norm();
        let dist_12 = edge_12.norm();
        let dist_20 = edge_20.norm();
        if dist_01 < dist_12 && dist_01 < dist_20 {
            edge_12.cross(&edge_20)
        } else if dist_12 < dist_20 && dist_12 < dist_01 {
            edge_20.cross(&edge_01)
        } else {
            edge_01.cross(&edge_12)
        }
    }

    fn process_camera<PL: ProgressListener>(
        &mut self,
        camera_i: usize,
        interpolation: InterpolationMode,
        progress_listener: Option<&PL>,
    ) -> Result<(), OutputError> {
        if interpolation != InterpolationMode::Delaunay {
            return Ok(());
        }

        let camera_points = self
            .points
            .iter_tracks()
            .enumerate()
            .par_bridge()
            .filter_map(|(track_i, track)| {
                // Do not include invisible points to build point index.
                track.get(camera_i)?;
                let point3d = track.get_point3d()?;
                let projection = self.points.project_point(camera_i, &point3d);
                let point = Point2::new(projection.x, projection.y);
                Some(Point { track_i, point })
            })
            .collect::<Vec<_>>();
        if self.camera_ranges.is_empty() {
            self.camera_ranges = self.camera_ranges();
        }

        let triangulated_surface = DelaunayTriangulation::<Point>::bulk_load(camera_points)?;

        let cameras_len = self.points.cameras_len();
        let (percent_complete, percent_multiplier) = if cameras_len > 0 {
            (
                camera_i as f32 / cameras_len as f32,
                1.0 / cameras_len as f32,
            )
        } else {
            (0.0, 1.0)
        };

        let percent_complete = percent_complete + percent_multiplier * 0.2;
        if let Some(pl) = progress_listener {
            pl.report_status(0.9 * percent_complete);
        }

        let mut new_polygons = triangulated_surface
            .inner_faces()
            .par_bridge()
            .flat_map(|f| {
                let vertices = f.vertices();
                let v0 = vertices[0].data().track_i;
                let v1 = vertices[1].data().track_i;
                let v2 = vertices[2].data().track_i;
                let polygon = Polygon::new(camera_i, [v0, v1, v2]);

                // Discard polygons that are too steep.
                if self
                    .max_angle_cos(&polygon)
                    .map_or(true, |angle_cos| angle_cos < MAX_NORMAL_COS)
                {
                    None
                } else {
                    Some(polygon)
                }
            })
            .collect::<Vec<_>>();
        drop(triangulated_surface);

        if self.points.cameras_len() > 0 {
            let mut processed_cameras = 0usize;
            for camera_j in 0..cameras_len {
                if camera_i == camera_j {
                    continue;
                }

                let percent_multiplier =
                    0.8 * percent_multiplier / (self.points.cameras_len() - 1) as f32;
                let percent_complete =
                    percent_complete + percent_multiplier * processed_cameras as f32;

                self.camera_ranges[camera_j].index_projections(&self.points);
                let camera_range = &self.camera_ranges[camera_j];

                let percent_complete = percent_complete + percent_multiplier * 0.4;
                let percent_multiplier = percent_multiplier * 0.6;
                if let Some(pl) = progress_listener {
                    pl.report_status(0.9 * percent_complete);
                }

                let polygons_count = new_polygons.len();
                let counter = AtomicUsize::new(0);
                new_polygons = new_polygons
                    .par_iter()
                    .filter_map(|polygon| {
                        if let Some(pl) = progress_listener {
                            let value = percent_complete
                                + percent_multiplier
                                    * (counter.fetch_add(1, AtomicOrdering::Relaxed) as f32
                                        / polygons_count as f32);
                            pl.report_status(0.9 * value);
                        }
                        if self.polygon_index.contains(polygon) {
                            return None;
                        }
                        let polygon = polygon.to_owned();

                        // Discard polygons that obstruct points.
                        // TODO: cut holes in the polygon?
                        if self.polygon_obstructs(camera_i, camera_range, &polygon) {
                            None
                        } else {
                            Some(polygon)
                        }
                    })
                    .collect::<Vec<_>>();

                self.camera_ranges[camera_j].clear_grid();
                processed_cameras += 1;
            }
        }

        new_polygons.iter().for_each(|polygon| {
            if self.polygon_index.add(polygon) {
                return;
            }
            let polygon_normal = self.polygon_normal(polygon);
            self.polygons.push(*polygon);
            for point_i in polygon.vertices {
                self.point_normals[point_i] += polygon_normal;
            }
        });

        Ok(())
    }

    fn output<PL: ProgressListener>(
        &self,
        mut writer: Box<dyn MeshWriter>,
        progress_listener: Option<&PL>,
    ) -> Result<(), OutputError> {
        writer.output_header(self.points.tracks_len(), self.polygons.len())?;
        let nvertices = self.points.tracks_len() as f32;
        self.points
            .iter_tracks()
            .enumerate()
            .try_for_each(|(i, v)| {
                if let Some(pl) = progress_listener {
                    pl.report_status(0.90 + 0.02 * (i as f32 / nvertices));
                }
                let normal = self.point_normals[i].normalize();
                writer.output_vertex(v, &normal)
            })?;
        self.point_normals
            .iter()
            .enumerate()
            .try_for_each(|(i, normal)| {
                if let Some(pl) = progress_listener {
                    pl.report_status(0.92 + 0.02 * (i as f32 / nvertices));
                }
                writer.output_vertex_normal(&normal.normalize())
            })?;
        self.points
            .iter_tracks()
            .enumerate()
            .try_for_each(|(i, v)| {
                if let Some(pl) = progress_listener {
                    pl.report_status(0.94 + 0.02 * (i as f32 / nvertices));
                }
                writer.output_vertex_uv(v)
            })?;
        let nfaces = self.polygons.len() as f32;
        self.polygons.iter().enumerate().try_for_each(|(i, f)| {
            if let Some(pl) = progress_listener {
                pl.report_status(0.96 + 0.04 * (i as f32 / nfaces));
            }

            let point0 = self.points.get_camera_points(f.vertices[0]);
            let point1 = self.points.get_camera_points(f.vertices[1]);
            let point2 = self.points.get_camera_points(f.vertices[2]);
            let polygon_tracks = [point0, point1, point2];
            writer.output_face(f, polygon_tracks)
        })?;
        writer.complete()
    }
}

pub fn output<PL: ProgressListener>(
    surface: triangulation::Surface,
    out_scale: (f64, f64, f64),
    images: Vec<RgbImage>,
    path: &str,
    interpolation: InterpolationMode,
    vertex_mode: VertexMode,
    progress_listener: Option<&PL>,
) -> Result<(), OutputError> {
    let output_normals = interpolation != InterpolationMode::None;
    let writer: Box<dyn MeshWriter> = if path.to_lowercase().ends_with(".obj") {
        Box::new(ObjWriter::new(
            path,
            images,
            output_normals,
            vertex_mode,
            out_scale,
        )?)
    } else if path.to_lowercase().ends_with(".ply") {
        Box::new(PlyWriter::new(
            path,
            images,
            output_normals,
            vertex_mode,
            out_scale,
        )?)
    } else {
        Box::new(ImageWriter::new(
            path,
            images,
            &surface,
            out_scale.2.signum(),
        )?)
    };

    let mesh = Mesh::create(surface, interpolation, progress_listener)?;
    mesh.output(writer, progress_listener)
}

type Track = [Option<Point2D<u32>>];

trait MeshWriter {
    fn output_header(&mut self, _nvertices: usize, _nfaces: usize) -> Result<(), std::io::Error> {
        Ok(())
    }

    fn output_vertex(
        &mut self,
        _point: &triangulation::Track,
        _normal: &Vector3<f64>,
    ) -> Result<(), OutputError> {
        Ok(())
    }

    fn output_vertex_normal(&mut self, _normal: &Vector3<f64>) -> Result<(), OutputError> {
        Ok(())
    }

    fn output_vertex_uv(&mut self, _point: &triangulation::Track) -> Result<(), OutputError> {
        Ok(())
    }

    fn output_face(&mut self, _p: &Polygon, _tracks: [&Track; 3]) -> Result<(), OutputError> {
        Ok(())
    }

    fn complete(&mut self) -> Result<(), OutputError>;
}

const WRITE_BUFFER_SIZE: usize = 1024 * 1024;

struct PlyWriter {
    writer: BufWriter<File>,
    buffer: Vec<u8>,
    output_normals: bool,
    vertex_mode: VertexMode,
    out_scale: (f64, f64, f64),
    images: Vec<RgbImage>,
}

impl PlyWriter {
    fn new(
        path: &str,
        images: Vec<RgbImage>,
        output_normals: bool,
        vertex_mode: VertexMode,
        out_scale: (f64, f64, f64),
    ) -> Result<PlyWriter, OutputError> {
        let writer = BufWriter::new(File::create(path)?);
        let buffer = Vec::with_capacity(WRITE_BUFFER_SIZE);

        Ok(PlyWriter {
            writer,
            buffer,
            output_normals,
            vertex_mode,
            out_scale,
            images,
        })
    }

    fn check_flush_buffer(&mut self) -> Result<(), std::io::Error> {
        let buffer = &mut self.buffer;
        let w = &mut self.writer;
        if buffer.len() >= WRITE_BUFFER_SIZE {
            w.write_all(buffer)?;
            buffer.clear();
        }
        Ok(())
    }
}

impl MeshWriter for PlyWriter {
    fn output_header(&mut self, nvertices: usize, nfaces: usize) -> Result<(), std::io::Error> {
        self.check_flush_buffer()?;
        let w = &mut self.buffer;
        writeln!(w, "ply")?;
        writeln!(w, "format binary_big_endian 1.0")?;
        writeln!(w, "comment Cybervision 3D surface")?;
        writeln!(w, "element vertex {}", nvertices)?;
        writeln!(w, "property double x")?;
        writeln!(w, "property double y")?;
        writeln!(w, "property double z")?;
        if self.output_normals {
            writeln!(w, "property double nx")?;
            writeln!(w, "property double ny")?;
            writeln!(w, "property double nz")?;
        }

        match self.vertex_mode {
            VertexMode::Plain => {}
            VertexMode::Texture => {}
            VertexMode::Color => {
                writeln!(w, "property uchar red")?;
                writeln!(w, "property uchar green")?;
                writeln!(w, "property uchar blue")?;
            }
        }
        writeln!(w, "element face {}", nfaces)?;
        writeln!(w, "property list uchar int vertex_indices")?;
        writeln!(w, "end_header")
    }

    fn output_vertex(
        &mut self,
        track: &triangulation::Track,
        normal: &Vector3<f64>,
    ) -> Result<(), OutputError> {
        let color = match self.vertex_mode {
            VertexMode::Plain | VertexMode::Texture => None,
            VertexMode::Color => {
                if let Some((image_i, point2d)) = track
                    .points()
                    .iter()
                    .enumerate()
                    .find_map(|(i, p)| Some((i, (*p)?)))
                {
                    let img = &self.images[image_i];
                    img.get_pixel_checked(point2d.x, point2d.y)
                        .map(|pixel| pixel.0)
                } else {
                    return Err("Track has no images".into());
                }
            }
        };
        self.check_flush_buffer()?;
        let w = &mut self.buffer;

        let p = if let Some(point3d) = track.get_point3d() {
            point3d
        } else {
            return Err("Point has no 3D coordinates".into());
        };
        let (x, y, z) = (
            p.x * self.out_scale.0,
            -p.y * self.out_scale.1,
            p.z * self.out_scale.2,
        );
        w.write_all(&x.to_be_bytes())?;
        w.write_all(&y.to_be_bytes())?;
        w.write_all(&z.to_be_bytes())?;
        if self.output_normals {
            let nx = normal.x;
            let ny = normal.y;
            let nz = normal.z;
            w.write_all(&nx.to_be_bytes())?;
            w.write_all(&ny.to_be_bytes())?;
            w.write_all(&nz.to_be_bytes())?;
        }
        if let Some(color) = color {
            w.write_all(&color)?;
        }
        Ok(())
    }

    fn output_face(&mut self, polygon: &Polygon, _tracks: [&Track; 3]) -> Result<(), OutputError> {
        let indices = polygon.vertices;

        self.check_flush_buffer()?;
        let w = &mut self.buffer;
        const NUM_POINTS: [u8; 1] = 3u8.to_be_bytes();
        w.write_all(&NUM_POINTS)?;
        w.write_all(&(indices[2] as u32).to_be_bytes())?;
        w.write_all(&(indices[1] as u32).to_be_bytes())?;
        w.write_all(&(indices[0] as u32).to_be_bytes())?;
        Ok(())
    }

    fn complete(&mut self) -> Result<(), OutputError> {
        let buffer = &mut self.buffer;
        let w = &mut self.writer;
        w.write_all(buffer)?;
        buffer.clear();
        Ok(())
    }
}

struct ObjWriter {
    uv_index: Vec<usize>,
    writer: BufWriter<File>,
    current_image: Option<usize>,
    buffer: Vec<u8>,
    output_normals: bool,
    vertex_mode: VertexMode,
    out_scale: (f64, f64, f64),
    images: Vec<RgbImage>,
    path: String,
}

impl ObjWriter {
    fn new(
        path: &str,
        images: Vec<RgbImage>,
        output_normals: bool,
        vertex_mode: VertexMode,
        out_scale: (f64, f64, f64),
    ) -> Result<ObjWriter, OutputError> {
        let writer = BufWriter::new(File::create(path)?);
        let buffer = Vec::with_capacity(WRITE_BUFFER_SIZE);
        Ok(ObjWriter {
            uv_index: vec![0],
            writer,
            current_image: None,
            buffer,
            output_normals,
            vertex_mode,
            out_scale,
            images,
            path: path.to_string(),
        })
    }

    fn check_flush_buffer(&mut self) -> Result<(), std::io::Error> {
        let buffer = &mut self.buffer;
        let w = &mut self.writer;
        if buffer.len() >= WRITE_BUFFER_SIZE {
            w.write_all(buffer)?;
            buffer.clear();
        }
        Ok(())
    }

    fn get_output_filename(&self) -> Option<String> {
        Path::new(&self.path)
            .file_stem()
            .and_then(|n| n.to_str().map(|n| n.to_string()))
    }

    fn get_uv_index(&self, camera_i: usize, point_i: usize, track: &Track) -> usize {
        let camera_offset = self.uv_index[point_i];
        camera_offset
            + track
                .iter()
                .take(camera_i)
                .filter(|point2d| point2d.is_some())
                .count()
    }

    fn switch_material(&mut self, img_i: usize) -> Result<(), std::io::Error> {
        self.check_flush_buffer()?;
        let w = &mut self.buffer;

        match self.vertex_mode {
            VertexMode::Plain | VertexMode::Color => {}
            VertexMode::Texture => {
                writeln!(w, "usemtl Textured{}", img_i)?;
            }
        }
        Ok(())
    }

    fn write_materials(&mut self) -> Result<(), OutputError> {
        let out_filename = match self.vertex_mode {
            VertexMode::Plain | VertexMode::Color => return Ok(()),
            VertexMode::Texture => self.get_output_filename().unwrap(),
        };

        let destination_path = Path::new(&self.path).parent().unwrap();
        let mut w = BufWriter::new(File::create(
            destination_path.join(format!("{}.mtl", out_filename)),
        )?);

        for (img_i, img) in self.images.iter().enumerate() {
            let image_filename = format!("{}-{}.png", out_filename, img_i);

            writeln!(w, "newmtl Textured{}", img_i)?;
            writeln!(w, "Ka 0.2 0.2 0.2")?;
            writeln!(w, "Kd 0.8 0.8 0.8")?;
            writeln!(w, "Ks 1.0 1.0 1.0")?;
            writeln!(w, "illum 2")?;
            writeln!(w, "Ns 0.000500")?;
            writeln!(w, "map_Ka {}", image_filename)?;
            writeln!(w, "map_Kd {}", image_filename)?;
            writeln!(w)?;

            img.save(destination_path.join(image_filename))?;
        }

        Ok(())
    }
}

impl MeshWriter for ObjWriter {
    fn output_header(&mut self, _nvertices: usize, _nfaces: usize) -> Result<(), std::io::Error> {
        self.check_flush_buffer()?;
        let out_filename = self.get_output_filename().unwrap();
        let w = &mut self.buffer;

        match self.vertex_mode {
            VertexMode::Plain | VertexMode::Color => {}
            VertexMode::Texture => {
                writeln!(w, "mtllib {}.mtl", out_filename)?;
            }
        }
        Ok(())
    }

    fn output_vertex(
        &mut self,
        track: &triangulation::Track,
        _normal: &Vector3<f64>,
    ) -> Result<(), OutputError> {
        self.check_flush_buffer()?;
        let w = &mut self.buffer;

        let color = match self.vertex_mode {
            VertexMode::Plain | VertexMode::Texture => None,
            VertexMode::Color => {
                if let Some((image_i, point2d)) = track
                    .points()
                    .iter()
                    .enumerate()
                    .find_map(|(i, p)| Some((i, (*p)?)))
                {
                    let img = &self.images[image_i];
                    img.get_pixel_checked(point2d.x, point2d.y)
                        .map(|pixel| pixel.0)
                } else {
                    return Err("Track has no images".into());
                }
            }
        };

        let p = if let Some(point3d) = track.get_point3d() {
            point3d
        } else {
            return Err("Point has no 3D coordinates".into());
        };
        let (x, y, z) = (
            p.x * self.out_scale.0,
            -p.y * self.out_scale.1,
            p.z * self.out_scale.2,
        );
        write!(w, "v {} {} {}", x, y, z)?;
        if let Some(color) = color {
            write!(
                w,
                " {} {} {}",
                color[0] as f64 / 255.0,
                color[1] as f64 / 255.0,
                color[2] as f64 / 255.0,
            )?
        }
        writeln!(w)?;

        Ok(())
    }

    fn output_vertex_normal(&mut self, normal: &Vector3<f64>) -> Result<(), OutputError> {
        self.check_flush_buffer()?;
        let w = &mut self.buffer;

        if self.output_normals {
            writeln!(w, "vn {} {} {}", normal.x, normal.y, normal.z)?;
        }

        Ok(())
    }

    fn output_vertex_uv(&mut self, track: &triangulation::Track) -> Result<(), OutputError> {
        self.check_flush_buffer()?;
        match self.vertex_mode {
            VertexMode::Plain | VertexMode::Color => {}
            VertexMode::Texture => {
                let w = &mut self.buffer;
                let mut projections_count = 0;
                for (image_i, p) in track.points().iter().enumerate() {
                    let point2d = if let Some(point2d) = p {
                        point2d
                    } else {
                        continue;
                    };
                    let img = &self.images[image_i];
                    projections_count += 1;
                    writeln!(
                        w,
                        "vt {} {}",
                        point2d.x as f64 / img.width() as f64,
                        1.0f64 - point2d.y as f64 / img.height() as f64,
                    )?
                }
                if projections_count == 0 {
                    return Err("Track has no images".into());
                }
                // Tracks are output in an ordered way, so uv_index will use the same ordering as tracks.
                let last_index = self.uv_index.last().map_or(0, |last_index| *last_index);
                self.uv_index.push(last_index + projections_count);
            }
        }
        Ok(())
    }

    fn output_face(&mut self, polygon: &Polygon, tracks: [&Track; 3]) -> Result<(), OutputError> {
        let indices = polygon.vertices;

        // Polygons are sorted by their image IDs, so if the index is increased, this is a new image.
        if Some(polygon.camera_i) != self.current_image {
            self.switch_material(polygon.camera_i)?;
            self.current_image = Some(polygon.camera_i);
        }

        self.check_flush_buffer()?;
        write!(self.buffer, "f")?;
        for i in [2, 1, 0] {
            let index = indices[i] + 1;
            match self.vertex_mode {
                VertexMode::Plain | VertexMode::Color => {
                    if self.output_normals {
                        write!(self.buffer, " {}//{}", index, index)?;
                    } else {
                        write!(self.buffer, " {}", index)?;
                    }
                }
                VertexMode::Texture => {
                    let uv_index =
                        self.get_uv_index(polygon.camera_i, polygon.vertices[i], tracks[i]) + 1;
                    if self.output_normals {
                        write!(self.buffer, " {}/{}/{}", index, uv_index, index)?;
                    } else {
                        write!(self.buffer, " {}/{}", index, uv_index)?;
                    }
                }
            }
        }
        writeln!(self.buffer)?;
        Ok(())
    }

    fn complete(&mut self) -> Result<(), OutputError> {
        let buffer = &mut self.buffer;
        let w = &mut self.writer;
        w.write_all(buffer)?;
        buffer.clear();
        self.write_materials()?;
        Ok(())
    }
}

struct ImageWriter {
    output_map: Grid<Option<f64>>,
    point_projections: Vec<Option<(Point2D<u32>, f64)>>,
    path: String,
    scale: f64,
    img1_width: u32,
    img1_height: u32,
}

impl ImageWriter {
    fn new(
        path: &str,
        images: Vec<RgbImage>,
        surface: &triangulation::Surface,
        scale: f64,
    ) -> Result<ImageWriter, std::io::Error> {
        let point_projections = surface
            .iter_tracks()
            .enumerate()
            .map(|(track_i, track)| {
                let point = track.get(0)?;
                let point_depth = surface.point_depth(0, track_i)?;
                Some((point, point_depth))
            })
            .collect::<Vec<_>>();
        let img1 = &images[0];
        let output_map = Grid::new(img1.width() as usize, img1.height() as usize, None);
        Ok(ImageWriter {
            output_map,
            point_projections,
            path: path.to_owned(),
            scale,
            img1_width: img1.width(),
            img1_height: img1.height(),
        })
    }

    #[inline]
    fn barycentric_interpolation(&self, polygon: &Polygon, pos: Point2D<usize>) -> Option<f64> {
        let convert_projection = |i: usize| {
            self.point_projections[polygon.vertices[i]]
                .map(|(point, _depth)| Point2D::new(point.x as f64, point.y as f64))
        };

        let convert_depth =
            |i: usize| self.point_projections[polygon.vertices[i]].map(|(_point, depth)| depth);
        let polygon_projection = [
            convert_projection(0)?,
            convert_projection(1)?,
            convert_projection(2)?,
        ];
        let polygon_depths = [convert_depth(0)?, convert_depth(1)?, convert_depth(2)?];

        let lambda = barycentric_interpolation(
            polygon_projection,
            Point2D::new(pos.x as f64, pos.y as f64),
        )?;
        let value = lambda[0] * polygon_depths[0]
            + lambda[1] * polygon_depths[1]
            + lambda[2] * polygon_depths[2];

        Some(value)
    }
}

impl MeshWriter for ImageWriter {
    fn output_vertex(
        &mut self,
        track: &triangulation::Track,
        _normal: &Vector3<f64>,
    ) -> Result<(), OutputError> {
        // TODO: project all polygons into first image
        let point2d = if let Some(point2d) = track.get(0) {
            point2d
        } else {
            return Err("Track is absent from first image".into());
        };
        let point3d = if let Some(point3d) = track.get_point3d() {
            point3d
        } else {
            return Err("Point has no 3D coordinates".into());
        };
        let (x, y) = (point2d.x as usize, point2d.y as usize);
        if x < self.output_map.width() && y < self.output_map.height() {
            *self.output_map.val_mut(x, y) = Some(point3d.z * self.scale);
        }
        Ok(())
    }

    fn output_face(&mut self, polygon: &Polygon, _tracks: [&Track; 3]) -> Result<(), OutputError> {
        let vertices = polygon.vertices;
        let (min_x, max_x, min_y, max_y) = vertices.iter().fold(
            (self.output_map.width(), 0, self.output_map.height(), 0),
            |acc, v| {
                let p = if let Some(point) = self.point_projections[*v] {
                    Point2D::new(point.0.x as usize, point.0.y as usize)
                } else {
                    return acc;
                };
                (
                    acc.0.min(p.x),
                    acc.1.max(p.x),
                    acc.2.min(p.y),
                    acc.3.max(p.y),
                )
            },
        );

        for y in min_y..=max_y {
            for x in min_x..=max_x {
                if self.output_map.val(x, y).is_some() {
                    continue;
                }
                let value = if let Some(value) =
                    self.barycentric_interpolation(polygon, Point2D::new(x, y))
                {
                    value * self.scale
                } else {
                    continue;
                };

                if x < self.output_map.width() && y < self.output_map.height() {
                    *self.output_map.val_mut(x, y) = Some(value);
                }
            }
        }
        Ok(())
    }

    fn complete(&mut self) -> Result<(), OutputError> {
        let (min_depth, max_depth) = self.output_map.iter().fold((f64::MAX, f64::MIN), |acc, v| {
            if let Some(v) = v.2 {
                (acc.0.min(*v), acc.1.max(*v))
            } else {
                acc
            }
        });
        let mut output_image =
            RgbaImage::from_pixel(self.img1_width, self.img1_height, Rgba::from([0, 0, 0, 0]));
        output_image
            .enumerate_pixels_mut()
            .par_bridge()
            .for_each(|(x, y, value)| {
                let depth = if let Some(depth) = self.output_map.val(x as usize, y as usize) {
                    depth
                } else {
                    return;
                };
                *value = map_depth((depth - min_depth) / (max_depth - min_depth))
            });
        output_image.save(&self.path)?;
        Ok(())
    }
}

#[inline]
fn barycentric_interpolation(
    projections: [Point2D<f64>; 3],
    pos: Point2D<f64>,
) -> Option<[f64; 3]> {
    let v0 = projections[0];
    let v1 = projections[1];
    let v2 = projections[2];

    let (x0, x1, x2) = (v0.x, v1.x, v2.x);
    let (y0, y1, y2) = (v0.y, v1.y, v2.y);
    let det = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2);
    if det.abs() < f64::EPSILON {
        return None;
    }
    let lambda0 = ((y1 - y2) * (pos.x - x2) + (x2 - x1) * (pos.y - y2)) / det;
    let lambda1 = ((y2 - y0) * (pos.x - x2) + (x0 - x2) * (pos.y - y2)) / det;
    let lambda2 = 1.0 - lambda0 - lambda1;

    if lambda0 < -f64::EPSILON
        || lambda1 < -f64::EPSILON
        || lambda2 < -f64::EPSILON
        || lambda0 > 1.0 + f64::EPSILON
        || lambda1 > 1.0 + f64::EPSILON
        || lambda2 > 1.0 + f64::EPSILON
    {
        return None;
    }
    Some([lambda0, lambda1, lambda2])
}

#[inline]
fn map_depth(value: f64) -> Rgba<u8> {
    // viridis from https://bids.github.io/colormap/
    const COLORMAP_R: [u8; 256] = [
        0xfd, 0xfb, 0xf8, 0xf6, 0xf4, 0xf1, 0xef, 0xec, 0xea, 0xe7, 0xe5, 0xe2, 0xdf, 0xdd, 0xda,
        0xd8, 0xd5, 0xd2, 0xd0, 0xcd, 0xca, 0xc8, 0xc5, 0xc2, 0xc0, 0xbd, 0xba, 0xb8, 0xb5, 0xb2,
        0xb0, 0xad, 0xaa, 0xa8, 0xa5, 0xa2, 0xa0, 0x9d, 0x9b, 0x98, 0x95, 0x93, 0x90, 0x8e, 0x8b,
        0x89, 0x86, 0x84, 0x81, 0x7f, 0x7c, 0x7a, 0x77, 0x75, 0x73, 0x70, 0x6e, 0x6c, 0x69, 0x67,
        0x65, 0x63, 0x60, 0x5e, 0x5c, 0x5a, 0x58, 0x56, 0x54, 0x52, 0x50, 0x4e, 0x4c, 0x4a, 0x48,
        0x46, 0x44, 0x42, 0x40, 0x3f, 0x3d, 0x3b, 0x3a, 0x38, 0x37, 0x35, 0x34, 0x32, 0x31, 0x2f,
        0x2e, 0x2d, 0x2c, 0x2a, 0x29, 0x28, 0x27, 0x26, 0x25, 0x25, 0x24, 0x23, 0x22, 0x22, 0x21,
        0x21, 0x20, 0x20, 0x1f, 0x1f, 0x1f, 0x1f, 0x1f, 0x1f, 0x1e, 0x1e, 0x1e, 0x1f, 0x1f, 0x1f,
        0x1f, 0x1f, 0x1f, 0x1f, 0x20, 0x20, 0x20, 0x21, 0x21, 0x21, 0x21, 0x22, 0x22, 0x22, 0x23,
        0x23, 0x23, 0x24, 0x24, 0x25, 0x25, 0x25, 0x26, 0x26, 0x26, 0x27, 0x27, 0x27, 0x28, 0x28,
        0x29, 0x29, 0x29, 0x2a, 0x2a, 0x2a, 0x2b, 0x2b, 0x2c, 0x2c, 0x2c, 0x2d, 0x2d, 0x2e, 0x2e,
        0x2e, 0x2f, 0x2f, 0x30, 0x30, 0x31, 0x31, 0x31, 0x32, 0x32, 0x33, 0x33, 0x34, 0x34, 0x35,
        0x35, 0x36, 0x36, 0x37, 0x37, 0x38, 0x38, 0x39, 0x39, 0x3a, 0x3a, 0x3b, 0x3b, 0x3c, 0x3c,
        0x3d, 0x3d, 0x3e, 0x3e, 0x3e, 0x3f, 0x3f, 0x40, 0x40, 0x41, 0x41, 0x42, 0x42, 0x42, 0x43,
        0x43, 0x44, 0x44, 0x44, 0x45, 0x45, 0x45, 0x46, 0x46, 0x46, 0x46, 0x47, 0x47, 0x47, 0x47,
        0x47, 0x48, 0x48, 0x48, 0x48, 0x48, 0x48, 0x48, 0x48, 0x48, 0x48, 0x48, 0x48, 0x48, 0x48,
        0x48, 0x48, 0x48, 0x47, 0x47, 0x47, 0x47, 0x47, 0x46, 0x46, 0x46, 0x46, 0x45, 0x45, 0x44,
        0x44,
    ];
    const COLORMAP_G: [u8; 256] = [
        0xe7, 0xe7, 0xe6, 0xe6, 0xe6, 0xe5, 0xe5, 0xe5, 0xe5, 0xe4, 0xe4, 0xe4, 0xe3, 0xe3, 0xe3,
        0xe2, 0xe2, 0xe2, 0xe1, 0xe1, 0xe1, 0xe0, 0xe0, 0xdf, 0xdf, 0xdf, 0xde, 0xde, 0xde, 0xdd,
        0xdd, 0xdc, 0xdc, 0xdb, 0xdb, 0xda, 0xda, 0xd9, 0xd9, 0xd8, 0xd8, 0xd7, 0xd7, 0xd6, 0xd6,
        0xd5, 0xd5, 0xd4, 0xd3, 0xd3, 0xd2, 0xd1, 0xd1, 0xd0, 0xd0, 0xcf, 0xce, 0xcd, 0xcd, 0xcc,
        0xcb, 0xcb, 0xca, 0xc9, 0xc8, 0xc8, 0xc7, 0xc6, 0xc5, 0xc5, 0xc4, 0xc3, 0xc2, 0xc1, 0xc1,
        0xc0, 0xbf, 0xbe, 0xbd, 0xbc, 0xbc, 0xbb, 0xba, 0xb9, 0xb8, 0xb7, 0xb6, 0xb6, 0xb5, 0xb4,
        0xb3, 0xb2, 0xb1, 0xb0, 0xaf, 0xae, 0xad, 0xad, 0xac, 0xab, 0xaa, 0xa9, 0xa8, 0xa7, 0xa6,
        0xa5, 0xa4, 0xa3, 0xa2, 0xa1, 0xa1, 0xa0, 0x9f, 0x9e, 0x9d, 0x9c, 0x9b, 0x9a, 0x99, 0x98,
        0x97, 0x96, 0x95, 0x94, 0x93, 0x92, 0x92, 0x91, 0x90, 0x8f, 0x8e, 0x8d, 0x8c, 0x8b, 0x8a,
        0x89, 0x88, 0x87, 0x86, 0x85, 0x84, 0x83, 0x82, 0x82, 0x81, 0x80, 0x7f, 0x7e, 0x7d, 0x7c,
        0x7b, 0x7a, 0x79, 0x78, 0x77, 0x76, 0x75, 0x74, 0x73, 0x72, 0x71, 0x71, 0x70, 0x6f, 0x6e,
        0x6d, 0x6c, 0x6b, 0x6a, 0x69, 0x68, 0x67, 0x66, 0x65, 0x64, 0x63, 0x62, 0x61, 0x60, 0x5f,
        0x5e, 0x5d, 0x5c, 0x5b, 0x5a, 0x59, 0x58, 0x56, 0x55, 0x54, 0x53, 0x52, 0x51, 0x50, 0x4f,
        0x4e, 0x4d, 0x4c, 0x4a, 0x49, 0x48, 0x47, 0x46, 0x45, 0x44, 0x42, 0x41, 0x40, 0x3f, 0x3e,
        0x3d, 0x3b, 0x3a, 0x39, 0x38, 0x37, 0x35, 0x34, 0x33, 0x32, 0x30, 0x2f, 0x2e, 0x2d, 0x2c,
        0x2a, 0x29, 0x28, 0x26, 0x25, 0x24, 0x23, 0x21, 0x20, 0x1f, 0x1d, 0x1c, 0x1b, 0x1a, 0x18,
        0x17, 0x16, 0x14, 0x13, 0x11, 0x10, 0x0e, 0x0d, 0x0b, 0x0a, 0x08, 0x07, 0x05, 0x04, 0x02,
        0x01,
    ];
    const COLORMAP_B: [u8; 256] = [
        0x25, 0x23, 0x21, 0x20, 0x1e, 0x1d, 0x1c, 0x1b, 0x1a, 0x19, 0x19, 0x18, 0x18, 0x18, 0x19,
        0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1f, 0x20, 0x21, 0x23, 0x25, 0x26, 0x28, 0x29, 0x2b, 0x2d,
        0x2f, 0x30, 0x32, 0x34, 0x36, 0x37, 0x39, 0x3b, 0x3c, 0x3e, 0x40, 0x41, 0x43, 0x45, 0x46,
        0x48, 0x49, 0x4b, 0x4d, 0x4e, 0x50, 0x51, 0x53, 0x54, 0x56, 0x57, 0x58, 0x5a, 0x5b, 0x5c,
        0x5e, 0x5f, 0x60, 0x62, 0x63, 0x64, 0x65, 0x67, 0x68, 0x69, 0x6a, 0x6b, 0x6c, 0x6d, 0x6e,
        0x6f, 0x70, 0x71, 0x72, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x79, 0x7a, 0x7b, 0x7c,
        0x7c, 0x7d, 0x7e, 0x7f, 0x7f, 0x80, 0x81, 0x81, 0x82, 0x82, 0x83, 0x83, 0x84, 0x85, 0x85,
        0x85, 0x86, 0x86, 0x87, 0x87, 0x88, 0x88, 0x88, 0x89, 0x89, 0x89, 0x8a, 0x8a, 0x8a, 0x8b,
        0x8b, 0x8b, 0x8b, 0x8c, 0x8c, 0x8c, 0x8c, 0x8c, 0x8d, 0x8d, 0x8d, 0x8d, 0x8d, 0x8d, 0x8d,
        0x8e, 0x8e, 0x8e, 0x8e, 0x8e, 0x8e, 0x8e, 0x8e, 0x8e, 0x8e, 0x8e, 0x8e, 0x8e, 0x8e, 0x8e,
        0x8e, 0x8e, 0x8e, 0x8e, 0x8e, 0x8e, 0x8e, 0x8e, 0x8e, 0x8e, 0x8e, 0x8e, 0x8e, 0x8e, 0x8e,
        0x8e, 0x8e, 0x8e, 0x8e, 0x8e, 0x8e, 0x8e, 0x8e, 0x8e, 0x8e, 0x8d, 0x8d, 0x8d, 0x8d, 0x8d,
        0x8d, 0x8d, 0x8d, 0x8d, 0x8c, 0x8c, 0x8c, 0x8c, 0x8c, 0x8c, 0x8b, 0x8b, 0x8b, 0x8b, 0x8a,
        0x8a, 0x8a, 0x8a, 0x89, 0x89, 0x89, 0x88, 0x88, 0x88, 0x87, 0x87, 0x86, 0x86, 0x85, 0x85,
        0x84, 0x84, 0x83, 0x83, 0x82, 0x81, 0x81, 0x80, 0x7f, 0x7e, 0x7e, 0x7d, 0x7c, 0x7b, 0x7a,
        0x7a, 0x79, 0x78, 0x77, 0x76, 0x75, 0x74, 0x73, 0x71, 0x70, 0x6f, 0x6e, 0x6d, 0x6c, 0x6a,
        0x69, 0x68, 0x67, 0x65, 0x64, 0x63, 0x61, 0x60, 0x5e, 0x5d, 0x5c, 0x5a, 0x59, 0x57, 0x56,
        0x54,
    ];

    Rgba::from([
        map_color(&COLORMAP_R, value),
        map_color(&COLORMAP_G, value),
        map_color(&COLORMAP_B, value),
        u8::MAX,
    ])
}

#[inline]
fn map_color(colormap: &[u8; 256], value: f64) -> u8 {
    if value >= 1.0 {
        return colormap[colormap.len() - 1];
    }
    let step = 1.0 / (colormap.len() - 1) as f64;
    let box_index = ((value / step).floor() as usize).clamp(0, colormap.len() - 2);
    let ratio = (value - step * box_index as f64) / step;
    let c1 = colormap[box_index] as f64;
    let c2 = colormap[box_index + 1] as f64;
    (c2 * ratio + c1 * (1.0 - ratio)).round() as u8
}

impl HasPosition for Point {
    fn position(&self) -> Point2<f64> {
        self.point
    }

    type Scalar = f64;
}

#[derive(Debug)]
pub enum OutputError {
    Internal(&'static str),
    Triangulation(spade::InsertionError),
    Io(std::io::Error),
    Image(image::ImageError),
}

impl fmt::Display for OutputError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            OutputError::Internal(msg) => f.write_str(msg),
            OutputError::Triangulation(ref err) => err.fmt(f),
            OutputError::Io(ref err) => err.fmt(f),
            OutputError::Image(ref err) => err.fmt(f),
        }
    }
}

impl std::error::Error for OutputError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match *self {
            OutputError::Internal(_msg) => None,
            OutputError::Triangulation(ref err) => err.source(),
            OutputError::Io(ref err) => err.source(),
            OutputError::Image(ref err) => err.source(),
        }
    }
}

impl From<&'static str> for OutputError {
    fn from(msg: &'static str) -> OutputError {
        OutputError::Internal(msg)
    }
}

impl From<spade::InsertionError> for OutputError {
    fn from(e: spade::InsertionError) -> OutputError {
        OutputError::Triangulation(e)
    }
}

impl From<std::io::Error> for OutputError {
    fn from(e: std::io::Error) -> OutputError {
        OutputError::Io(e)
    }
}

impl From<image::ImageError> for OutputError {
    fn from(e: image::ImageError) -> OutputError {
        OutputError::Image(e)
    }
}
