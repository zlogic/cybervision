use std::{
    error,
    fs::File,
    io::{BufWriter, Write},
    path::Path,
};

use image::{RgbImage, Rgba, RgbaImage};
use nalgebra::DMatrix;
use rayon::prelude::*;
use spade::{
    handles::{FaceHandle, InnerTag},
    DelaunayTriangulation, HasPosition, Point2, Triangulation,
};

use crate::triangulation;

#[derive(Debug, PartialEq)]
pub enum InterpolationMode {
    Delaunay,
    None,
}

#[derive(Debug, PartialEq)]
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

type TriangulatedSurfaceFace<'a> = FaceHandle<'a, InnerTag, triangulation::Point, (), (), ()>;

pub fn output<PL: ProgressListener>(
    points: triangulation::Surface,
    img1: RgbImage,
    path: &str,
    interpolation: InterpolationMode,
    vertex_mode: VertexMode,
    progress_listener: Option<&PL>,
) -> Result<(), Box<dyn error::Error>> {
    let img1_shape = (img1.height() as usize, img1.width() as usize);
    let mut writer: Box<dyn MeshWriter> = if path.to_lowercase().ends_with(".obj") {
        Box::new(ObjWriter::new(path, img1, vertex_mode)?)
    } else if path.to_lowercase().ends_with(".ply") {
        Box::new(PlyWriter::new(path, img1, vertex_mode)?)
    } else {
        Box::new(ImageWriter::new(path, img1, &points)?)
    };

    if interpolation == InterpolationMode::Delaunay {
        let triangulated_surface: DelaunayTriangulation<triangulation::Point> =
            Triangulation::bulk_load(points)?;

        if let Some(pl) = progress_listener {
            pl.report_status(0.2);
        }

        writer.output_header(
            triangulated_surface.num_vertices(),
            triangulated_surface.num_inner_faces(),
        )?;
        let nvertices = triangulated_surface.num_vertices() as f32;
        triangulated_surface
            .vertices()
            .enumerate()
            .try_for_each(|(i, v)| {
                if let Some(pl) = progress_listener {
                    pl.report_status(0.2 + 0.2 * (i as f32 / nvertices));
                }
                writer.output_vertex(v.data())
            })?;
        triangulated_surface
            .vertices()
            .enumerate()
            .try_for_each(|(i, v)| {
                if let Some(pl) = progress_listener {
                    pl.report_status(0.4 + 0.2 * (i as f32 / nvertices));
                }
                writer.output_vertex_uv(v.data().original)
            })?;
        let nfaces = triangulated_surface.num_inner_faces() as f32;
        triangulated_surface
            .inner_faces()
            .enumerate()
            .try_for_each(|(i, f)| {
                if let Some(pl) = progress_listener {
                    pl.report_status(0.6 + 0.3 * (i as f32 / nfaces));
                }
                writer.output_face(f)
            })?;
        return writer.complete();
    }

    let surface_vertices_uninterpolated = triangulate_without_interpolation(img1_shape, &points);
    let npoints = points.len() as f32;
    writer.output_header(points.len(), surface_vertices_uninterpolated.len())?;
    points.iter().enumerate().try_for_each(|(i, p)| {
        if let Some(pl) = progress_listener {
            pl.report_status(0.2 + 0.2 * (i as f32 / npoints));
        }
        writer.output_vertex(p)
    })?;
    points.iter().enumerate().try_for_each(|(i, p)| {
        if let Some(pl) = progress_listener {
            pl.report_status(0.4 + 0.2 * (i as f32 / npoints));
        }
        writer.output_vertex_uv(p.original)
    })?;
    surface_vertices_uninterpolated
        .iter()
        .enumerate()
        .try_for_each(|(i, p)| {
            if let Some(pl) = progress_listener {
                pl.report_status(0.6 + 0.3 * (i as f32 / npoints));
            }
            writer.output_face_uninterpolated(p)
        })?;
    return writer.complete();
}

trait MeshWriter {
    fn output_header(&mut self, _nvertices: usize, _nfaces: usize) -> Result<(), std::io::Error> {
        Ok(())
    }
    fn output_vertex(
        &mut self,
        _point: &triangulation::Point,
    ) -> Result<(), Box<dyn error::Error>> {
        Ok(())
    }
    fn output_vertex_uv(&mut self, _point: (usize, usize)) -> Result<(), Box<dyn error::Error>> {
        Ok(())
    }
    fn output_face(&mut self, _f: TriangulatedSurfaceFace) -> Result<(), Box<dyn error::Error>> {
        Ok(())
    }
    fn output_face_uninterpolated(
        &mut self,
        _indices: &[usize; 3],
    ) -> Result<(), Box<dyn error::Error>> {
        Ok(())
    }
    fn complete(&mut self) -> Result<(), Box<dyn error::Error>>;
}

const WRITE_BUFFER_SIZE: usize = 1024 * 1024;

struct PlyWriter {
    writer: BufWriter<File>,
    buffer: Vec<u8>,
    vertex_mode: VertexMode,
    img1: RgbImage,
}

impl PlyWriter {
    fn new(
        path: &str,
        img1: RgbImage,
        vertex_mode: VertexMode,
    ) -> Result<PlyWriter, Box<dyn error::Error>> {
        let writer = BufWriter::new(File::create(path)?);
        let buffer = Vec::with_capacity(WRITE_BUFFER_SIZE);

        Ok(PlyWriter {
            writer,
            buffer,
            vertex_mode,
            img1,
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
        writeln!(w, "property float x")?;
        writeln!(w, "property float y")?;
        writeln!(w, "property float z")?;

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

    fn output_vertex(&mut self, p: &triangulation::Point) -> Result<(), Box<dyn error::Error>> {
        let color = match self.vertex_mode {
            VertexMode::Plain | VertexMode::Texture => None,
            VertexMode::Color => self
                .img1
                .get_pixel_checked(p.original.0 as u32, p.original.1 as u32)
                .map(|pixel| pixel.0),
        };
        self.check_flush_buffer()?;
        let w = &mut self.buffer;

        let p = p.reconstructed;
        let (x, y, z) = (p.x as f32, -p.y as f32, p.z as f32);
        w.write_all(&x.to_be_bytes())?;
        w.write_all(&y.to_be_bytes())?;
        w.write_all(&z.to_be_bytes())?;
        if let Some(color) = color {
            w.write_all(&color)?;
        }
        Ok(())
    }

    fn output_face(&mut self, f: TriangulatedSurfaceFace) -> Result<(), Box<dyn error::Error>> {
        let indices = f.vertices().map(|v| v.index());
        self.output_face_uninterpolated(&indices)
    }

    fn output_face_uninterpolated(
        &mut self,
        indices: &[usize; 3],
    ) -> Result<(), Box<dyn error::Error>> {
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
    img1: RgbImage,
    path: String,
}

impl ObjWriter {
    fn new(
        path: &str,
        img1: RgbImage,
        vertex_mode: VertexMode,
    ) -> Result<ObjWriter, Box<dyn error::Error>> {
        let writer = BufWriter::new(File::create(path)?);
        let buffer = Vec::with_capacity(WRITE_BUFFER_SIZE);
        Ok(ObjWriter {
            writer,
            buffer,
            vertex_mode,
            img1,
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

        self.img1.save(self.path.to_string() + ".png")?;

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

    fn output_vertex(&mut self, p: &triangulation::Point) -> Result<(), Box<dyn error::Error>> {
        self.check_flush_buffer()?;
        let w = &mut self.buffer;

        let color = match self.vertex_mode {
            VertexMode::Plain | VertexMode::Texture => None,
            VertexMode::Color => self
                .img1
                .get_pixel_checked(p.original.0 as u32, p.original.1 as u32)
                .map(|pixel| pixel.0),
        };

        let p = p.reconstructed;
        if let Some(color) = color {
            writeln!(
                w,
                "v {} {} {} {} {} {}",
                p.x,
                -p.y,
                p.z,
                color[0] as f32 / 255.0,
                color[1] as f32 / 255.0,
                color[2] as f32 / 255.0,
            )?
        } else {
            writeln!(w, "v {} {} {}", p.x, -p.y, p.z)?
        }

        Ok(())
    }

    fn output_vertex_uv(&mut self, p: (usize, usize)) -> Result<(), Box<dyn error::Error>> {
        self.check_flush_buffer()?;
        match self.vertex_mode {
            VertexMode::Plain | VertexMode::Color => {}
            VertexMode::Texture => {
                let w = &mut self.buffer;
                writeln!(
                    w,
                    "vt {} {}",
                    p.0 as f32 / self.img1.width() as f32,
                    1.0f32 - p.1 as f32 / self.img1.height() as f32,
                )?;
            }
        }
        Ok(())
    }

    fn output_face(&mut self, f: TriangulatedSurfaceFace) -> Result<(), Box<dyn error::Error>> {
        let indices = f.vertices().map(|v| v.index());
        self.output_face_uninterpolated(&indices)
    }

    #[inline]
    fn output_face_uninterpolated(
        &mut self,
        indices: &[usize; 3],
    ) -> Result<(), Box<dyn error::Error>> {
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
    output_image: RgbaImage,
    path: String,
    min_depth: f64,
    max_depth: f64,
}

impl ImageWriter {
    fn new(
        path: &str,
        img1: RgbImage,
        points: &triangulation::Surface,
    ) -> Result<ImageWriter, std::io::Error> {
        let (min_depth, max_depth) = points.iter().fold((f64::MAX, f64::MIN), |acc, p| {
            let z = p.reconstructed.z;
            (acc.0.min(z), acc.1.max(z))
        });
        Ok(ImageWriter {
            output_image: RgbaImage::from_pixel(
                img1.width(),
                img1.height(),
                Rgba::from([0, 0, 0, 0]),
            ),
            path: path.to_owned(),
            min_depth,
            max_depth,
        })
    }
}

impl MeshWriter for ImageWriter {
    fn output_vertex(&mut self, p: &triangulation::Point) -> Result<(), Box<dyn error::Error>> {
        let (x, y) = (p.original.0 as u32, p.original.1 as u32);
        let value = (p.reconstructed.z - self.min_depth) / (self.max_depth - self.min_depth);
        self.output_image.put_pixel(x, y, map_depth(value));
        Ok(())
    }

    fn output_face(&mut self, f: TriangulatedSurfaceFace) -> Result<(), Box<dyn error::Error>> {
        let vertices = f.vertices();
        let (min_x, max_x) = vertices
            .iter()
            .fold((self.output_image.width(), 0), |acc, f| {
                let point = f.data().original;
                (acc.0.min(point.0 as u32), acc.1.max(point.0 as u32))
            });
        let (min_y, max_y) = vertices
            .iter()
            .fold((self.output_image.height(), 0), |acc, f| {
                let point = f.data().original;
                (acc.0.min(point.1 as u32), acc.1.max(point.1 as u32))
            });

        for x in min_x..=max_x {
            for y in min_y..=max_y {
                if self.output_image.get_pixel(x, y).0[3] != 0 {
                    continue;
                }
                let lambda = f.barycentric_interpolation(Point2 {
                    x: x as f64,
                    y: y as f64,
                });
                if lambda
                    .iter()
                    .any(|lambda| *lambda > 1.0 + f64::EPSILON || *lambda < 0.0 - f64::EPSILON)
                {
                    continue;
                }
                let value: f64 = vertices
                    .iter()
                    .zip(lambda)
                    .map(|(vertex, lambda)| vertex.data().reconstructed.z * lambda)
                    .sum();

                let value = (value - self.min_depth) / (self.max_depth - self.min_depth);
                self.output_image.put_pixel(x, y, map_depth(value));
            }
        }
        Ok(())
    }

    fn complete(&mut self) -> Result<(), Box<dyn error::Error>> {
        self.output_image.save(&self.path)?;
        Ok(())
    }
}

fn triangulate_without_interpolation(
    shape: (usize, usize),
    surface: &triangulation::Surface,
) -> Vec<[usize; 3]> {
    let mut indices: DMatrix<Option<usize>> = DMatrix::from_element(shape.0, shape.1, None);
    surface.iter().enumerate().for_each(|(i, point)| {
        indices[(point.original.1, point.original.0)] = Some(i);
    });
    surface
        .par_iter()
        .flat_map(|p| {
            let row = p.original.1;
            let col = p.original.0;
            let forward_neighbor = [(row, col), (row, col + 1), (row + 1, col)];
            let back_neighbor = [(row, col), (row, col - 1), (row - 1, col)];

            let mut has_forward_neighbor = true;
            let mut has_back_neighbor = true;
            let mut forward_neighbor_indices = [0; 3];
            let mut back_neighbor_indices = [0; 3];
            for i in 0..3 {
                let forward_index = indices
                    .get(forward_neighbor[i])
                    .map(|v| v.to_owned())
                    .unwrap_or_default();
                let back_index = indices
                    .get(back_neighbor[i])
                    .map(|v| v.to_owned())
                    .unwrap_or_default();
                has_forward_neighbor &= forward_index.is_some();
                has_back_neighbor &= back_index.is_some();
                forward_neighbor_indices[i] = forward_index.unwrap_or(usize::MAX);
                back_neighbor_indices[i] = back_index.unwrap_or(usize::MAX);
            }

            if has_forward_neighbor && has_back_neighbor {
                vec![forward_neighbor_indices, back_neighbor_indices]
            } else if has_forward_neighbor {
                vec![forward_neighbor_indices]
            } else if has_back_neighbor {
                vec![back_neighbor_indices]
            } else {
                vec![]
            }
        })
        .collect()
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

impl HasPosition for triangulation::Point {
    fn position(&self) -> Point2<f64> {
        Point2::new(self.original.0 as f64, self.original.1 as f64)
    }

    type Scalar = f64;
}
