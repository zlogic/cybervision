use core::fmt;
use std::{
    error,
    fs::File,
    io::{BufWriter, Write},
    path::Path,
    sync::atomic::AtomicUsize,
    sync::atomic::Ordering as AtomicOrdering,
};

use image::{RgbImage, Rgba, RgbaImage};
use nalgebra::{DMatrix, Vector3};
use spade::{DelaunayTriangulation, HasPosition, Point2, Triangulation};

use rayon::prelude::*;

use crate::triangulation;

const PROJECTIONS_INDEX_GRID_SIZE: usize = 1000;

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

#[derive(Debug)]
struct Point {
    point: Point2<f64>,
    track_i: usize,
}

#[derive(Debug, PartialEq, Clone, Copy)]
struct Polygon {
    vertices: [usize; 3],
}

impl Polygon {
    fn new(v0: &Point, v1: &Point, v2: &Point) -> Polygon {
        let vertices = [v0.track_i, v1.track_i, v2.track_i];
        Polygon { vertices }
    }
}

struct CameraGrid {
    camera_i: usize,
    min_row: f64,
    max_row: f64,
    min_col: f64,
    max_col: f64,
    grid: DMatrix<Vec<Vector3<f64>>>,
    grid_step: f64,
}

impl CameraGrid {
    fn new(camera_i: usize) -> CameraGrid {
        CameraGrid {
            camera_i,
            min_row: f64::MAX,
            max_row: f64::MIN,
            min_col: f64::MAX,
            max_col: f64::MIN,
            grid: DMatrix::from_element(0, 0, vec![]),
            grid_step: 0.0,
        }
    }

    fn from_point(camera_i: usize, point: (f64, f64)) -> CameraGrid {
        CameraGrid {
            camera_i,
            min_row: point.0,
            max_row: point.0,
            min_col: point.1,
            max_col: point.1,
            grid: DMatrix::from_element(0, 0, vec![]),
            grid_step: 0.0,
        }
    }

    fn merge(&self, other: &CameraGrid) -> CameraGrid {
        CameraGrid {
            camera_i: self.camera_i,
            min_row: self.min_row.min(other.min_row),
            max_row: self.max_row.max(other.max_row),
            min_col: self.min_col.min(other.min_col),
            max_col: self.max_col.max(other.max_col),
            grid: DMatrix::from_element(0, 0, vec![]),
            grid_step: 0.0,
        }
    }

    fn get_step(&self) -> f64 {
        (self.max_row - self.min_row).min(self.max_col - self.min_col)
            / PROJECTIONS_INDEX_GRID_SIZE as f64
    }

    fn index_projections(&mut self, points: &triangulation::Surface) {
        let grid_step = self.get_step();
        let nrows = ((self.max_row - self.min_row) / grid_step).ceil() as usize;
        let ncols = ((self.max_col - self.min_col) / grid_step).ceil() as usize;
        let mut index = DMatrix::<Vec<Vector3<f64>>>::from_element(nrows, ncols, vec![]);
        points
            .iter_tracks()
            .enumerate()
            .for_each(|(track_i, _track)| {
                let point = if let Some(point) = points.project_point(self.camera_i, track_i) {
                    point
                } else {
                    return;
                };
                let point_depth =
                    if let Some(point_depth) = points.point_depth(self.camera_i, track_i) {
                        point_depth
                    } else {
                        return;
                    };
                let row =
                    (((point.y - self.min_row) / grid_step).floor() as usize).clamp(0, nrows - 1);
                let col =
                    (((point.x - self.min_col) / grid_step).floor() as usize).clamp(0, ncols - 1);
                let point_in_camera = Vector3::new(point.x, point.y, point_depth);
                index[(row, col)].push(point_in_camera);
            });

        self.grid_step = grid_step;
        self.grid = index;
    }

    fn clear_grid(&mut self) {
        self.grid = DMatrix::from_element(0, 0, vec![]);
        self.grid_step = 0.0;
    }
}

struct Mesh {
    points: triangulation::Surface,
    polygons: Vec<Polygon>,
    camera_ranges: Vec<CameraGrid>,
}

impl Mesh {
    fn create<PL: ProgressListener>(
        surface: triangulation::Surface,
        interpolation: InterpolationMode,
        progress_listener: Option<&PL>,
    ) -> Result<Mesh, Box<dyn error::Error>> {
        let mut surface = Mesh {
            points: surface,
            polygons: vec![],
            camera_ranges: vec![],
        };

        if surface.points.cameras_len() == 0 {
            surface.process_camera(0, interpolation, progress_listener)?;
        } else {
            for camera_i in 0..surface.points.cameras_len() {
                surface.process_camera(camera_i, interpolation, progress_listener)?;
            }
        }

        Ok(surface)
    }

    fn camera_ranges(&self) -> Vec<CameraGrid> {
        (0..self.points.cameras_len())
            .map(|camera_i| {
                self.points
                    .iter_tracks()
                    .enumerate()
                    .par_bridge()
                    .flat_map(|(track_i, _track)| {
                        let point = self.points.project_point(camera_i, track_i)?;
                        Some(CameraGrid::from_point(camera_i, (point.y, point.x)))
                    })
                    .reduce(|| CameraGrid::new(camera_i), |a, b| a.merge(&b))
            })
            .collect::<Vec<_>>()
    }

    #[inline]
    fn polygon_obstructs(&self, camera_i: usize, grid: &CameraGrid, polygon: &Polygon) -> bool {
        let project_polygon_point = |i: usize| {
            let point = self.points.project_point(camera_i, polygon.vertices[i])?;
            Some((point.y, point.x))
        };
        let polygon_projection = || {
            Some([
                project_polygon_point(0)?,
                project_polygon_point(1)?,
                project_polygon_point(2)?,
            ])
        };
        let polygon_projection = if let Some(projection) = polygon_projection() {
            projection
        } else {
            return false;
        };
        let get_polygon_depth = |i: usize| self.points.point_depth(camera_i, polygon.vertices[i]);
        let polygon_depths = || {
            Some([
                get_polygon_depth(0)?,
                get_polygon_depth(1)?,
                get_polygon_depth(2)?,
            ])
        };
        let polygon_depths = if let Some(depths) = polygon_depths() {
            depths
        } else {
            return false;
        };

        let (min_row, max_row, min_col, max_col) = polygon_projection
            .iter()
            .map(|point| (point.0, point.1))
            .fold((f64::MAX, f64::MIN, f64::MAX, f64::MIN), |acc, v| {
                (
                    acc.0.min(v.0),
                    acc.1.max(v.0),
                    acc.2.min(v.1),
                    acc.3.max(v.1),
                )
            });

        let min_row = (((min_row - grid.min_row) / grid.grid_step).floor() as usize)
            .saturating_sub(1)
            .clamp(0, grid.grid.nrows());
        let max_row = (((max_row - grid.min_row) / grid.grid_step).ceil() as usize)
            .saturating_add(1)
            .clamp(0, grid.grid.nrows());
        let min_col = (((min_col - grid.min_col) / grid.grid_step).floor() as usize)
            .saturating_sub(1)
            .clamp(0, grid.grid.ncols());
        let max_col = (((max_col - grid.min_col) / grid.grid_step).ceil() as usize)
            .saturating_add(1)
            .clamp(0, grid.grid.ncols());

        for row in min_row..max_row {
            for col in min_col..max_col {
                let points = &grid.grid[(row, col)];
                let obstruction = points.iter().any(|point| {
                    let point2d = (point.y, point.x);
                    let barycentric_coordinates =
                        barycentric_interpolation(&polygon_projection, &point2d);
                    if let Some(lambda) = barycentric_coordinates {
                        let polygon_depth = lambda[0] * polygon_depths[0]
                            + lambda[1] * polygon_depths[1]
                            + lambda[2] * polygon_depths[2];
                        polygon_depth < point.z
                    } else {
                        false
                    }
                });

                if obstruction {
                    return true;
                }
            }
        }

        false
    }

    fn process_camera<PL: ProgressListener>(
        &mut self,
        camera_i: usize,
        interpolation: InterpolationMode,
        progress_listener: Option<&PL>,
    ) -> Result<(), Box<dyn error::Error>> {
        if interpolation != InterpolationMode::Delaunay {
            return Ok(());
        }

        let camera_points = self
            .points
            .iter_tracks()
            .enumerate()
            .par_bridge()
            .filter_map(|(track_i, _track)| {
                let projection = self.points.project_point(camera_i, track_i)?;
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
            .map(|f| {
                let vertices = f.vertices();
                let v0 = vertices[0].data();
                let v1 = vertices[1].data();
                let v2 = vertices[2].data();
                Polygon::new(v0, v1, v2)
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
                    percent_multiplier / (self.points.cameras_len() - 1) as f32;
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

        self.polygons.append(&mut new_polygons);

        Ok(())
    }

    fn output<PL: ProgressListener>(
        &self,
        mut writer: Box<dyn MeshWriter>,
        progress_listener: Option<&PL>,
    ) -> Result<(), Box<dyn error::Error>> {
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
                    pl.report_status(0.92 + 0.02 * (i as f32 / nvertices));
                }
                writer.output_vertex_uv(v)
            })?;
        let nfaces = self.polygons.len() as f32;
        self.polygons.iter().enumerate().try_for_each(|(i, f)| {
            if let Some(pl) = progress_listener {
                pl.report_status(0.94 + 0.06 * (i as f32 / nfaces));
            }
            writer.output_face(f)
        })?;
        writer.complete()
    }
}

pub fn output<PL: ProgressListener>(
    surface: triangulation::Surface,
    images: Vec<RgbImage>,
    path: &str,
    interpolation: InterpolationMode,
    vertex_mode: VertexMode,
    progress_listener: Option<&PL>,
) -> Result<(), Box<dyn error::Error>> {
    let writer: Box<dyn MeshWriter> = if path.to_lowercase().ends_with(".obj") {
        Box::new(ObjWriter::new(path, images, vertex_mode)?)
    } else if path.to_lowercase().ends_with(".ply") {
        Box::new(PlyWriter::new(path, images, vertex_mode)?)
    } else {
        Box::new(ImageWriter::new(path, images, &surface)?)
    };

    let mesh = Mesh::create(surface, interpolation, progress_listener)?;
    mesh.output(writer, progress_listener)
}

trait MeshWriter {
    fn output_header(&mut self, _nvertices: usize, _nfaces: usize) -> Result<(), std::io::Error> {
        Ok(())
    }
    fn output_vertex(
        &mut self,
        _point: &triangulation::Track,
    ) -> Result<(), Box<dyn error::Error>> {
        Ok(())
    }
    fn output_vertex_uv(
        &mut self,
        _point: &triangulation::Track,
    ) -> Result<(), Box<dyn error::Error>> {
        Ok(())
    }
    fn output_face(&mut self, _p: &Polygon) -> Result<(), Box<dyn error::Error>> {
        Ok(())
    }
    fn complete(&mut self) -> Result<(), Box<dyn error::Error>>;
}

const WRITE_BUFFER_SIZE: usize = 1024 * 1024;

struct PlyWriter {
    writer: BufWriter<File>,
    buffer: Vec<u8>,
    vertex_mode: VertexMode,
    images: Vec<RgbImage>,
}

impl PlyWriter {
    fn new(
        path: &str,
        images: Vec<RgbImage>,
        vertex_mode: VertexMode,
    ) -> Result<PlyWriter, Box<dyn error::Error>> {
        let writer = BufWriter::new(File::create(path)?);
        let buffer = Vec::with_capacity(WRITE_BUFFER_SIZE);

        Ok(PlyWriter {
            writer,
            buffer,
            vertex_mode,
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

    fn output_vertex(&mut self, track: &triangulation::Track) -> Result<(), Box<dyn error::Error>> {
        let color = match self.vertex_mode {
            VertexMode::Plain | VertexMode::Texture => None,
            VertexMode::Color => {
                let first_image = track.range().start;
                if let Some(point2d) = track.get(first_image) {
                    let img = &self.images[first_image];
                    img.get_pixel_checked(point2d.1 as u32, point2d.0 as u32)
                        .map(|pixel| pixel.0)
                } else {
                    return Err(OutputError::new("Track has no image").into());
                }
            }
        };
        self.check_flush_buffer()?;
        let w = &mut self.buffer;

        let p = if let Some(point3d) = track.get_point3d() {
            point3d
        } else {
            return Err(OutputError::new("Point has no 3D coordinates").into());
        };
        let (x, y, z) = (p.x, -p.y, p.z);
        w.write_all(&x.to_be_bytes())?;
        w.write_all(&y.to_be_bytes())?;
        w.write_all(&z.to_be_bytes())?;
        if let Some(color) = color {
            w.write_all(&color)?;
        }
        Ok(())
    }

    fn output_face(&mut self, polygon: &Polygon) -> Result<(), Box<dyn error::Error>> {
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

    fn complete(&mut self) -> Result<(), Box<dyn error::Error>> {
        let buffer = &mut self.buffer;
        let w = &mut self.writer;
        w.write_all(buffer)?;
        buffer.clear();
        Ok(())
    }
}

struct ObjWriter {
    writer: BufWriter<File>,
    buffer: Vec<u8>,
    vertex_mode: VertexMode,
    images: Vec<RgbImage>,
    path: String,
}

impl ObjWriter {
    fn new(
        path: &str,
        images: Vec<RgbImage>,
        vertex_mode: VertexMode,
    ) -> Result<ObjWriter, Box<dyn error::Error>> {
        let writer = BufWriter::new(File::create(path)?);
        let buffer = Vec::with_capacity(WRITE_BUFFER_SIZE);
        Ok(ObjWriter {
            writer,
            buffer,
            vertex_mode,
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
            .file_name()
            .and_then(|n| n.to_str().map(|n| n.to_string()))
    }
    fn write_material(&mut self) -> Result<(), Box<dyn error::Error>> {
        let texture_filename = match self.vertex_mode {
            VertexMode::Plain | VertexMode::Color => return Ok(()),
            VertexMode::Texture => self.get_output_filename().unwrap(),
        };

        let mut w = BufWriter::new(File::create(self.path.to_string() + ".mtl")?);

        let image_filename = texture_filename + ".png";
        writeln!(w, "newmtl Textured")?;
        writeln!(w, "Ka 1.0 1.0 1.0")?;
        writeln!(w, "Kd 1.0 1.0 1.0")?;
        writeln!(w, "Ks 0.0 0.0 0.0")?;
        writeln!(w, "illum 1")?;
        writeln!(w, "Ns 0.000000")?;
        writeln!(w, "map_Ka {}", image_filename)?;
        writeln!(w, "map_Kd {}", image_filename)?;

        // TODO: support multiple textures
        let img1 = &self.images[0];
        img1.save(self.path.to_string() + ".png")?;

        Ok(())
    }
}

impl MeshWriter for ObjWriter {
    fn output_header(&mut self, _nvertices: usize, _nfaces: usize) -> Result<(), std::io::Error> {
        self.check_flush_buffer()?;
        let material_filename = self.get_output_filename().unwrap();
        let w = &mut self.buffer;

        match self.vertex_mode {
            VertexMode::Plain | VertexMode::Color => {}
            VertexMode::Texture => {
                writeln!(w, "mtllib {}", material_filename + ".mtl")?;
                writeln!(w, "usemtl Textured")?;
            }
        }
        Ok(())
    }

    fn output_vertex(&mut self, track: &triangulation::Track) -> Result<(), Box<dyn error::Error>> {
        self.check_flush_buffer()?;
        let w = &mut self.buffer;

        let color = match self.vertex_mode {
            VertexMode::Plain | VertexMode::Texture => None,
            VertexMode::Color => {
                let first_image = track.range().start;
                if let Some(point2d) = track.get(first_image) {
                    let img = &self.images[first_image];
                    img.get_pixel_checked(point2d.1 as u32, point2d.0 as u32)
                        .map(|pixel| pixel.0)
                } else {
                    return Err(OutputError::new("Track has no image").into());
                }
            }
        };

        let p = if let Some(point3d) = track.get_point3d() {
            point3d
        } else {
            return Err(OutputError::new("Point has no 3D coordinates").into());
        };
        if let Some(color) = color {
            writeln!(
                w,
                "v {} {} {} {} {} {}",
                p.x,
                -p.y,
                p.z,
                color[0] as f64 / 255.0,
                color[1] as f64 / 255.0,
                color[2] as f64 / 255.0,
            )?
        } else {
            writeln!(w, "v {} {} {}", p.x, -p.y, p.z)?
        }

        Ok(())
    }

    fn output_vertex_uv(
        &mut self,
        track: &triangulation::Track,
    ) -> Result<(), Box<dyn error::Error>> {
        self.check_flush_buffer()?;
        match self.vertex_mode {
            VertexMode::Plain | VertexMode::Color => {}
            VertexMode::Texture => {
                let w = &mut self.buffer;
                let first_image = track.range().start;
                if let Some(point2d) = track.get(first_image) {
                    let img = &self.images[first_image];
                    writeln!(
                        w,
                        "vt {} {}",
                        point2d.1 as f64 / img.width() as f64,
                        1.0f64 - point2d.0 as f64 / img.height() as f64,
                    )?;
                } else {
                    return Err(OutputError::new("Track has no image").into());
                }
            }
        }
        Ok(())
    }

    fn output_face(&mut self, polygon: &Polygon) -> Result<(), Box<dyn error::Error>> {
        let indices = polygon.vertices;

        self.check_flush_buffer()?;
        let w = &mut self.buffer;
        match self.vertex_mode {
            VertexMode::Plain | VertexMode::Color => {
                writeln!(
                    w,
                    "f {} {} {}",
                    indices[2] + 1,
                    indices[1] + 1,
                    indices[0] + 1
                )?;
            }
            VertexMode::Texture => {
                let w = &mut self.buffer;

                writeln!(
                    w,
                    "f {}/{} {}/{} {}/{}",
                    indices[2] + 1,
                    indices[2] + 1,
                    indices[1] + 1,
                    indices[1] + 1,
                    indices[0] + 1,
                    indices[0] + 1,
                )?;
            }
        }
        Ok(())
    }

    fn complete(&mut self) -> Result<(), Box<dyn error::Error>> {
        let buffer = &mut self.buffer;
        let w = &mut self.writer;
        w.write_all(buffer)?;
        buffer.clear();
        self.write_material()?;
        Ok(())
    }
}

struct ImageWriter {
    output_map: DMatrix<Option<f64>>,
    point_projections: Vec<Option<(u32, u32, f64)>>,
    path: String,
    img1_width: u32,
    img1_height: u32,
}

impl ImageWriter {
    fn new(
        path: &str,
        images: Vec<RgbImage>,
        surface: &triangulation::Surface,
    ) -> Result<ImageWriter, std::io::Error> {
        let point_projections = surface
            .iter_tracks()
            .enumerate()
            .map(|(track_i, track)| {
                let point = track.get(0)?;
                let point_depth = surface.point_depth(0, track_i)?;
                Some((point.0, point.1, point_depth))
            })
            .collect::<Vec<_>>();
        let img1 = &images[0];
        let output_map = DMatrix::from_element(img1.height() as usize, img1.width() as usize, None);
        Ok(ImageWriter {
            output_map,
            point_projections,
            path: path.to_owned(),
            img1_width: img1.width(),
            img1_height: img1.height(),
        })
    }

    #[inline]
    fn barycentric_interpolation(&self, polygon: &Polygon, pos: (usize, usize)) -> Option<f64> {
        let convert_projection = |i: usize| {
            self.point_projections[polygon.vertices[i]]
                .map(|(row, col, _depth)| (row as f64, col as f64))
        };

        let convert_depth =
            |i: usize| self.point_projections[polygon.vertices[i]].map(|(_row, _col, depth)| depth);
        let polygon_projection = [
            convert_projection(0)?,
            convert_projection(1)?,
            convert_projection(2)?,
        ];
        let polygon_depths = [convert_depth(0)?, convert_depth(1)?, convert_depth(2)?];

        let lambda = barycentric_interpolation(&polygon_projection, &(pos.0 as f64, pos.1 as f64))?;
        let value = lambda[0] * polygon_depths[0]
            + lambda[1] * polygon_depths[1]
            + lambda[2] * polygon_depths[2];

        Some(value)
    }
}

impl MeshWriter for ImageWriter {
    fn output_vertex(&mut self, track: &triangulation::Track) -> Result<(), Box<dyn error::Error>> {
        // TODO: project all polygons into first image
        let point2d = if let Some(point2d) = track.get(0) {
            point2d
        } else {
            return Err(OutputError::new("Track is absent from first image").into());
        };
        let point3d = if let Some(point3d) = track.get_point3d() {
            point3d
        } else {
            return Err(OutputError::new("Point has no 3D coordinates").into());
        };
        let (row, col) = (point2d.0 as usize, point2d.1 as usize);
        if row < self.output_map.nrows() && col < self.output_map.ncols() {
            self.output_map[(row, col)] = Some(point3d.z);
        }
        Ok(())
    }

    fn output_face(&mut self, polygon: &Polygon) -> Result<(), Box<dyn error::Error>> {
        let vertices = polygon.vertices;
        let (min_row, max_row, min_col, max_col) = vertices.iter().fold(
            (self.output_map.nrows(), 0, self.output_map.ncols(), 0),
            |acc, v| {
                let (row, col) = if let Some(point) = self.point_projections[*v] {
                    (point.0 as usize, point.1 as usize)
                } else {
                    return acc;
                };
                (
                    acc.0.min(row),
                    acc.1.max(row),
                    acc.2.min(col),
                    acc.3.max(col),
                )
            },
        );

        for row in min_row..=max_row {
            for col in min_col..=max_col {
                if self.output_map[(row, col)].is_some() {
                    continue;
                }
                let value = if let Some(value) = self.barycentric_interpolation(polygon, (row, col))
                {
                    value
                } else {
                    continue;
                };

                if row < self.output_map.nrows() && col < self.output_map.ncols() {
                    self.output_map[(row, col)] = Some(value);
                }
            }
        }
        Ok(())
    }

    fn complete(&mut self) -> Result<(), Box<dyn error::Error>> {
        let (min_depth, max_depth) = self.output_map.iter().fold((f64::MAX, f64::MIN), |acc, v| {
            if let Some(v) = v {
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
                let depth = if let Some(depth) = self.output_map[(y as usize, x as usize)] {
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
fn barycentric_interpolation(projections: &[(f64, f64); 3], pos: &(f64, f64)) -> Option<[f64; 3]> {
    let v0 = projections[0];
    let v1 = projections[1];
    let v2 = projections[2];

    let (row0, row1, row2) = (v0.0, v1.0, v2.0);
    let (col0, col1, col2) = (v0.1, v1.1, v2.1);
    let (row, col) = (pos.0, pos.1);
    let det = (row1 - row2) * (col0 - col2) + (col2 - col1) * (row0 - row2);
    if det.abs() < f64::EPSILON {
        return None;
    }
    let lambda0 = ((row1 - row2) * (col - col2) + (col2 - col1) * (row - row2)) / det;
    let lambda1 = ((row2 - row0) * (col - col2) + (col0 - col2) * (row - row2)) / det;
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
pub struct OutputError {
    msg: &'static str,
}

impl OutputError {
    fn new(msg: &'static str) -> OutputError {
        OutputError { msg }
    }
}

impl std::error::Error for OutputError {}

impl fmt::Display for OutputError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.msg)
    }
}
