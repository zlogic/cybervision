use crate::{
    data::{Grid, Point2D},
    triangulation,
};
use core::fmt;
use std::{
    cmp::Ordering,
    fs::File,
    io::{BufWriter, Write},
    path::Path,
    sync::atomic::{AtomicUsize, Ordering as AtomicOrdering},
};

use image::{RgbImage, Rgba, RgbaImage};
use nalgebra::Vector3;
use spade::{DelaunayTriangulation, HasPosition, Point2, Triangulation};

use rayon::prelude::*;

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

#[derive(Clone)]
struct Point {
    point: Point2<f64>,
    track_i: usize,
}

#[derive(Eq, Clone, Copy)]
struct Polygon {
    camera_i: usize,
    vertices: [usize; 3],
}

impl Polygon {
    fn new(camera_i: usize, vertices: [usize; 3]) -> Polygon {
        let v = &vertices;
        // Sort vertices to allow comparison.
        let vertices = if v[0] < v[1] && v[0] < v[2] {
            [v[0], v[1], v[2]]
        } else if v[1] < v[0] && v[1] < v[2] {
            [v[1], v[2], v[0]]
        } else {
            [v[2], v[0], v[1]]
        };
        Polygon { camera_i, vertices }
    }

    #[inline]
    fn get_points(
        &self,
        surface: &triangulation::Surface,
    ) -> Option<(Vector3<f64>, Vector3<f64>, Vector3<f64>)> {
        let point0 = surface.get_point(self.vertices[0])?;
        let point1 = surface.get_point(self.vertices[1])?;
        let point2 = surface.get_point(self.vertices[2])?;
        Some((point0, point1, point2))
    }
}

impl Ord for Polygon {
    fn cmp(&self, other: &Self) -> Ordering {
        let vs = &self.vertices;
        let vo = &other.vertices;

        vs[0]
            .cmp(&vo[0])
            .then(vs[1].cmp(&vo[1]))
            .then(vs[2].cmp(&vo[2]))
    }
}

impl PartialOrd for Polygon {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for Polygon {
    fn eq(&self, other: &Self) -> bool {
        let vs = &self.vertices;
        let vo = &other.vertices;
        vs[0] == vo[0] && vs[1] == vo[1] && vs[2] == vo[2]
    }
}

struct ProjectedPolygon {
    projections: [Point2D<f64>; 3],
    values: [f64; 3],
    max_x: usize,
    max_y: usize,
}

impl ProjectedPolygon {
    fn new(mut points: [Vector3<f64>; 3], max_x: usize, max_y: usize) -> ProjectedPolygon {
        points.sort_by(|a, b| a.y.total_cmp(&b.y));
        let projections = [
            Point2D::new(points[0].x, points[0].y),
            Point2D::new(points[1].x, points[1].y),
            Point2D::new(points[2].x, points[2].y),
        ];
        let values = [points[0].z, points[1].z, points[2].z];
        ProjectedPolygon {
            projections,
            values,
            max_x,
            max_y,
        }
    }

    fn iter(&self) -> ProjectedPolygonIterator {
        let min_y = self.projections[0].y.floor().clamp(0.0, self.max_y as f64) as usize;
        let max_y = (self.projections[2].y + 1.0)
            .ceil()
            .clamp(0.0, self.max_y as f64) as usize;

        ProjectedPolygonIterator {
            polygon: self,
            max_x: self.max_x,
            max_y,
            x: 0,
            y: min_y,
            scanline_y: None,
            start_x: 0.0,
            end_x: 0.0,
            start_value: 0.0,
            end_value: 0.0,
            scanline_max_x: 0,
        }
    }
}

struct ProjectedPolygonIterator<'a> {
    polygon: &'a ProjectedPolygon,
    max_x: usize,
    max_y: usize,
    x: usize,
    y: usize,
    scanline_y: Option<usize>,
    start_x: f64,
    end_x: f64,
    start_value: f64,
    end_value: f64,
    scanline_max_x: usize,
}

impl<'a> ProjectedPolygonIterator<'a> {
    fn update_scanline(&mut self, y: usize) -> bool {
        const EPSILON: f64 = f64::EPSILON;
        let scanline_ready = if let Some(scanline_y) = self.scanline_y {
            y == scanline_y
        } else {
            false
        };
        if scanline_ready {
            return true;
        }
        let y_usize = y;
        let y = y as f64;
        let polygon = self.polygon;
        let (a, b, c) = (
            polygon.projections[0],
            polygon.projections[1],
            polygon.projections[2],
        );
        if y < a.y || y > c.y {
            return false;
        }

        let (start_x, start_value) = if y < b.y || ((b.y - c.y) / (b.x - c.x)).abs() < EPSILON {
            let coeff = (y - a.y) / (b.y - a.y);
            let start_x = a.x * (1.0 - coeff) + b.x * coeff;
            let start_value = polygon.values[0] * (1.0 - coeff) + polygon.values[1] * coeff;
            (start_x, start_value)
        } else {
            let coeff = (y - b.y) / (c.y - b.y);
            let start_x = b.x * (1.0 - coeff) + c.x * coeff;
            let start_value = polygon.values[1] * (1.0 - coeff) + polygon.values[2] * coeff;
            (start_x, start_value)
        };

        let coeff = (y - a.y) / (c.y - a.y);
        let end_x = a.x * (1.0 - coeff) + c.x * coeff;
        let end_value = polygon.values[0] * (1.0 - coeff) + polygon.values[2] * coeff;

        self.scanline_y = Some(y_usize);
        if start_x < end_x {
            self.start_x = start_x;
            self.end_x = end_x;
            self.start_value = start_value;
            self.end_value = end_value;
        } else {
            self.start_x = end_x;
            self.end_x = start_x;
            self.start_value = end_value;
            self.end_value = start_value;
        }

        self.x = self.start_x.floor().clamp(0.0, self.max_x as f64) as usize;
        self.scanline_max_x = (self.end_x + 1.0).ceil().clamp(0.0, self.max_x as f64) as usize;

        true
    }

    fn scanline_value(&self, x: f64) -> Option<f64> {
        let x_c = (x - self.start_x) / (self.end_x - self.start_x);
        if (0.0..=1.0).contains(&x_c) {
            Some(self.start_value * (1.0 - x_c) + x_c * self.end_value)
        } else {
            None
        }
    }
}

impl<'a> Iterator for ProjectedPolygonIterator<'a> {
    type Item = (Point2D<usize>, f64);

    fn next(&mut self) -> Option<Self::Item> {
        for y in self.y..self.max_y {
            self.y = y;
            if !self.update_scanline(y) {
                continue;
            }

            for x in self.x..self.scanline_max_x {
                if let Some(value) = self.scanline_value(x as f64) {
                    self.x = x + 1;
                    return Some((Point2D::new(x, y), value));
                }
            }
        }
        None
    }
}

struct DepthBuffer {
    points_projection: Grid<Option<f64>>,
    camera_j: usize,
}

impl DepthBuffer {
    fn new(surface: &triangulation::Surface, camera_j: usize) -> DepthBuffer {
        let camera_points = surface
            .iter_tracks()
            .filter_map(|track| {
                track.get(camera_j)?;
                let point3d = track.get_point3d()?;
                let point_depth = surface.point_depth(camera_j, &point3d);
                let projection = surface.project_point(camera_j, &point3d);

                Some(Vector3::new(projection.x, projection.y, point_depth))
            })
            .collect::<Vec<_>>();

        let (max_x, max_y) = if let Some(stats) = camera_points
            .iter()
            .map(|point| (point.x, point.y))
            .reduce(|a, b| (a.0.max(b.0), a.1.max(b.1)))
        {
            stats
        } else {
            return DepthBuffer {
                points_projection: Grid::new(0, 0, None),
                camera_j,
            };
        };

        let mut points_projection =
            Grid::new(max_x.ceil() as usize + 1, max_y.ceil() as usize + 1, None);
        camera_points.into_iter().for_each(|point| {
            let point_x = point.x.round() as usize;
            let point_y = point.y.round() as usize;
            let current_value = points_projection.val_mut(point_x, point_y);
            let better_value = if let Some(val) = current_value {
                point.z < *val
            } else {
                true
            };
            if better_value {
                *current_value = Some(point.z);
            }
        });

        DepthBuffer {
            points_projection,
            camera_j,
        }
    }

    fn polygon_obstructs(&self, surface: &triangulation::Surface, polygon: &Polygon) -> bool {
        let camera_j = self.camera_j;
        let project_polygon_point = |point3d: Vector3<f64>| {
            let projection = surface.project_point(camera_j, &point3d);
            let point_depth = surface.point_depth(camera_j, &point3d);
            Vector3::new(projection.x, projection.y, point_depth)
        };
        let project_polygon = |(p0, p1, p2)| {
            [
                project_polygon_point(p0),
                project_polygon_point(p1),
                project_polygon_point(p2),
            ]
        };
        let polygon_points = if let Some(points) = polygon.get_points(surface) {
            points
        } else {
            return false;
        };

        let projected_polygon = ProjectedPolygon::new(
            project_polygon(polygon_points),
            self.points_projection.width(),
            self.points_projection.height(),
        );
        projected_polygon.iter().any(|(point, depth)| {
            // TODO: cut holes in the polygon?
            // TODO: check if polygon obstructs other polygons?
            self.points_projection
                .val(point.x, point.y)
                .map(|point_depth| depth < point_depth)
                .unwrap_or(false)
        })
    }
}

struct Mesh {
    interpolation: InterpolationMode,
    points: triangulation::Surface,
    polygons: Vec<Polygon>,
}

impl Mesh {
    fn create<PL: ProgressListener>(
        surface: triangulation::Surface,
        configuration: MeshConfiguration,
        progress_listener: Option<&PL>,
    ) -> Result<Mesh, OutputError> {
        let mut surface = Mesh {
            interpolation: configuration.interpolation,
            points: surface,
            polygons: vec![],
        };

        if surface.points.cameras_len() == 0 {
            surface.process_camera(0, progress_listener)?;
        } else {
            for camera_i in 0..surface.points.cameras_len() {
                surface.process_camera(camera_i, progress_listener)?;
            }
        }

        // Some writers need polygons to ge grouped by camera index.
        surface.polygons.sort_by_key(|polygon| polygon.camera_i);

        Ok(surface)
    }

    fn process_camera<PL: ProgressListener>(
        &mut self,
        camera_i: usize,
        progress_listener: Option<&PL>,
    ) -> Result<(), OutputError> {
        if self.interpolation != InterpolationMode::Delaunay {
            return Ok(());
        }

        let affine_projection = self.points.cameras_len() == 0;

        let camera_points = self
            .points
            .iter_tracks()
            .enumerate()
            .filter_map(|(track_i, track)| {
                // Do not include invisible points to build point index.
                let point_in_image = track.get(camera_i)?;
                let point3d = track.get_point3d()?;
                let point = if affine_projection {
                    Point2::new(point_in_image.x as f64, point_in_image.y as f64)
                } else {
                    let projection = self.points.project_point(camera_i, &point3d);
                    Point2::new(projection.x, projection.y)
                };
                Some(Point { track_i, point })
            })
            .collect::<Vec<_>>();

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
            .map(|f| {
                let vertices = f.vertices();
                let v0 = vertices[0].data().track_i;
                let v1 = vertices[1].data().track_i;
                let v2 = vertices[2].data().track_i;
                let polygon = Polygon::new(camera_i, [v0, v1, v2]);

                Some(polygon)
            })
            .collect::<Vec<_>>();
        drop(triangulated_surface);

        if self.points.cameras_len() > 0 && !affine_projection {
            let mut processed_cameras = 0usize;
            for camera_j in 0..cameras_len {
                if camera_i == camera_j {
                    continue;
                }

                let percent_multiplier =
                    0.8 * percent_multiplier / (self.points.cameras_len() - 1) as f32;
                let percent_complete =
                    percent_complete + percent_multiplier * processed_cameras as f32;

                let percent_complete = percent_complete + percent_multiplier * 0.4;
                let percent_multiplier = percent_multiplier * 0.6;
                if let Some(pl) = progress_listener {
                    pl.report_status(0.9 * percent_complete);
                }

                let depth_buffer = DepthBuffer::new(&self.points, camera_j);
                if let Some(pl) = progress_listener {
                    let camera_percent = 0.05;
                    let value = percent_complete + percent_multiplier * camera_percent;

                    pl.report_status(0.9 * value);
                };

                let counter = AtomicUsize::new(0);
                let polygons_count = new_polygons.len() as f32;
                new_polygons.par_iter_mut().for_each(|check_polygon| {
                    if let Some(pl) = progress_listener {
                        let percent =
                            counter.fetch_add(1, AtomicOrdering::Relaxed) as f32 / polygons_count;
                        let camera_percent = 0.05 + percent * 0.95;
                        let value = percent_complete + percent_multiplier * camera_percent;

                        pl.report_status(0.9 * value);
                    }

                    let polygon = if let Some(polygon) = check_polygon {
                        polygon
                    } else {
                        return;
                    };
                    if depth_buffer.polygon_obstructs(&self.points, polygon) {
                        *check_polygon = None;
                    }
                });

                new_polygons.retain(|polygon| polygon.is_some());
                processed_cameras += 1;
            }
        }

        new_polygons.into_iter().for_each(|polygon| {
            if let Some(polygon) = polygon {
                self.polygons.push(polygon);
            }
        });
        self.polygons.sort_unstable();
        self.polygons.dedup();

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
                writer.output_vertex(v)
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

pub struct MeshConfiguration {
    pub interpolation: InterpolationMode,
    pub vertex_mode: VertexMode,
}

pub fn output<PL: ProgressListener>(
    surface: triangulation::Surface,
    out_scale: (f64, f64, f64),
    project_to_image: usize,
    images: Vec<RgbImage>,
    path: &str,
    configuration: MeshConfiguration,
    progress_listener: Option<&PL>,
) -> Result<(), OutputError> {
    let writer: Box<dyn MeshWriter> = if path.to_lowercase().ends_with(".obj") {
        Box::new(ObjWriter::new(
            path,
            images,
            configuration.vertex_mode,
            out_scale,
        )?)
    } else if path.to_lowercase().ends_with(".ply") {
        Box::new(PlyWriter::new(
            path,
            images,
            configuration.vertex_mode,
            out_scale,
        )?)
    } else {
        Box::new(ImageWriter::new(
            path,
            &surface,
            project_to_image,
            out_scale.2.signum(),
        )?)
    };

    let mesh = Mesh::create(surface, configuration, progress_listener)?;
    mesh.output(writer, progress_listener)
}

type Track = [Option<Point2D<u32>>];

trait MeshWriter {
    fn output_header(&mut self, _nvertices: usize, _nfaces: usize) -> Result<(), std::io::Error> {
        Ok(())
    }

    fn output_vertex(&mut self, _point: &triangulation::Track) -> Result<(), OutputError> {
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
    vertex_mode: VertexMode,
    out_scale: (f64, f64, f64),
    images: Vec<RgbImage>,
}

impl PlyWriter {
    fn new(
        path: &str,
        images: Vec<RgbImage>,
        vertex_mode: VertexMode,
        out_scale: (f64, f64, f64),
    ) -> Result<PlyWriter, OutputError> {
        let writer = BufWriter::new(File::create(path)?);
        let buffer = Vec::with_capacity(WRITE_BUFFER_SIZE);

        Ok(PlyWriter {
            writer,
            buffer,
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

    fn output_vertex(&mut self, track: &triangulation::Track) -> Result<(), OutputError> {
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
    vertex_mode: VertexMode,
    out_scale: (f64, f64, f64),
    images: Vec<RgbImage>,
    path: String,
}

impl ObjWriter {
    fn new(
        path: &str,
        images: Vec<RgbImage>,
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

    fn output_vertex(&mut self, track: &triangulation::Track) -> Result<(), OutputError> {
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
                    write!(self.buffer, " {}", index)?;
                }
                VertexMode::Texture => {
                    let uv_index =
                        self.get_uv_index(polygon.camera_i, polygon.vertices[i], tracks[i]) + 1;
                    write!(self.buffer, " {}/{}", index, uv_index)?;
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
    point_projections: Vec<Option<Vector3<f64>>>,
    path: String,
}

impl ImageWriter {
    fn new(
        path: &str,
        surface: &triangulation::Surface,
        project_to_image: usize,
        scale: f64,
    ) -> Result<ImageWriter, OutputError> {
        // Project all points onto the first image.
        let camera_points = surface
            .iter_tracks()
            .map(|track| {
                let point3d = track.get_point3d()?;
                let point_depth = surface.point_depth(project_to_image, &point3d);
                let projection = surface.project_point(project_to_image, &point3d);

                Some(Vector3::new(projection.x, projection.y, point_depth))
            })
            .collect::<Vec<_>>();

        let (min_x, max_x, min_y, max_y) = camera_points
            .iter()
            .filter_map(|point| {
                let point = (*point)?;
                Some((point.x, point.x, point.y, point.y))
            })
            .reduce(|a, b| (a.0.min(b.0), a.1.max(b.1), a.2.min(b.2), a.3.max(b.3)))
            .ok_or("No point projections found")?;

        let width = (max_x.ceil() - min_x.floor()) as usize + 1;
        let height = (max_y.ceil() - min_y.floor()) as usize + 1;
        let mut output_map = Grid::new(width, height, None);

        let point_projections = camera_points
            .into_iter()
            .map(|projection| {
                let projection = projection?;
                let point_depth = projection.z * scale;
                let projection =
                    Vector3::new(projection.x - min_x, projection.y - min_y, point_depth);
                let dst_x = (projection.x.round() as usize).clamp(0, width - 1);
                let dst_y = (projection.y.round() as usize).clamp(0, height - 1);
                let current_value = output_map.val_mut(dst_x, dst_y);
                let better_value = if let Some(val) = current_value {
                    point_depth > *val
                } else {
                    true
                };
                if better_value {
                    *current_value = Some(point_depth);
                }
                Some(projection)
            })
            .collect::<Vec<_>>();
        Ok(ImageWriter {
            output_map,
            point_projections,
            path: path.to_owned(),
        })
    }
}

impl MeshWriter for ImageWriter {
    fn output_vertex(&mut self, _track: &triangulation::Track) -> Result<(), OutputError> {
        // Points are output when processing surface.
        // The output_map already contains projected vertices.
        Ok(())
    }

    fn output_face(&mut self, polygon: &Polygon, _tracks: [&Track; 3]) -> Result<(), OutputError> {
        let polygon_projection = match (
            self.point_projections[polygon.vertices[0]],
            self.point_projections[polygon.vertices[1]],
            self.point_projections[polygon.vertices[2]],
        ) {
            (Some(p0), Some(p1), Some(p2)) => [p0, p1, p2],
            _ => return Ok(()),
        };

        let polygon_projection = ProjectedPolygon::new(
            polygon_projection,
            self.output_map.width() - 1,
            self.output_map.height() - 1,
        );
        polygon_projection.iter().for_each(|(point, new_value)| {
            let current_value = self.output_map.val_mut(point.x, point.y);
            let better_value = if let Some(val) = current_value {
                new_value > *val
            } else {
                true
            };
            if better_value {
                *current_value = Some(new_value);
            }
        });
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
        let mut output_image = RgbaImage::from_pixel(
            self.output_map.width() as u32,
            self.output_map.height() as u32,
            Rgba::from([0, 0, 0, 0]),
        );
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
