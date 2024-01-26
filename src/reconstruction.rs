use crate::correlation;
use crate::correlation::PointCorrelations;
use crate::data::{Grid, Point2D};
use crate::fundamentalmatrix;
use crate::fundamentalmatrix::FundamentalMatrix;
use crate::orb;
use crate::output;
use crate::pointmatching;
use crate::triangulation;
use crate::Args;

use image::{imageops::FilterType, GenericImageView, GrayImage, RgbImage};
use indicatif::{ProgressBar, ProgressState, ProgressStyle};
use nalgebra::Matrix3;
use std::error;
use std::fmt::Write;
use std::fs::File;
use std::io::BufReader;
use std::str::FromStr;
use std::time::SystemTime;

const TIFFTAG_META_PHENOM: exif::Tag = exif::Tag(exif::Context::Tiff, 34683);
const TIFFTAG_META_QUANTA: exif::Tag = exif::Tag(exif::Context::Tiff, 34682);

struct SourceImage {
    img: GrayImage,
    scale: (f32, f32),
    focal_length: Option<u32>,
    tilt_angle: Option<f32>,
    filename: String,
}

#[derive(Debug)]
struct ImageMeta {
    scale: (f32, f32),
    tilt_angle: Option<f32>,
    databar_height: u32,
    focal_length: Option<u32>,
}

impl SourceImage {
    fn load(path: &str) -> Result<SourceImage, image::ImageError> {
        let metadata = SourceImage::get_metadata(path);
        let img = image::open(path)?.into_luma8();
        let img = img.view(0, 0, img.width(), img.height() - metadata.databar_height);

        Ok(SourceImage {
            img: img.to_image(),
            scale: metadata.scale,
            focal_length: metadata.focal_length,
            tilt_angle: metadata.tilt_angle,
            filename: path.to_string(),
        })
    }

    fn load_rgb(path: &str) -> Result<RgbImage, image::ImageError> {
        let metadata = SourceImage::get_metadata(path);
        let img = image::open(path)?.into_rgb8();
        Ok(img
            .view(0, 0, img.width(), img.height() - metadata.databar_height)
            .to_image())
    }

    fn get_metadata(path: &str) -> ImageMeta {
        let default_metadata = ImageMeta {
            scale: (1.0, 1.0),
            tilt_angle: None,
            databar_height: 0,
            focal_length: None,
        };
        match SourceImage::get_metadata_exif(path) {
            Ok(metadata) => metadata,
            Err(_) => default_metadata,
        }
    }

    fn tag_value<F: FromStr>(line: &str) -> Option<F> {
        line.split_once('=')
            .and_then(|(_, value)| value.parse::<F>().ok())
    }

    fn get_metadata_exif(path: &str) -> Result<ImageMeta, exif::Error> {
        let mut reader = BufReader::new(File::open(path)?);
        let exif = exif::Reader::new().read_from_container(&mut reader)?;
        let sem_metadata = exif
            .get_field(TIFFTAG_META_PHENOM, exif::In::PRIMARY)
            .or(exif.get_field(TIFFTAG_META_QUANTA, exif::In::PRIMARY));

        let mut result_metadata = ImageMeta {
            scale: (1.0, 1.0),
            tilt_angle: None,
            databar_height: 0,
            focal_length: None,
        };
        let sem_metadata = if let Some(metadata) = sem_metadata {
            match metadata.value {
                exif::Value::Ascii(ref data) => {
                    Some(data.iter().fold(String::new(), |acc, block| {
                        acc + std::str::from_utf8(block.as_slice())
                            .ok()
                            .unwrap_or_default()
                    }))
                }
                _ => None,
            }
        } else {
            None
        };
        if let Some(data) = sem_metadata {
            const SEPARATORS: [char; 2] = ['\r', '\n'];
            let mut section = "";
            let mut scale_width = None;
            let mut scale_height = None;
            for line in data.split_terminator(&SEPARATORS[..]) {
                if line.starts_with('[') && line.ends_with(']') {
                    section = line;
                    continue;
                }
                if section.eq("[Scan]") {
                    if line.starts_with("PixelWidth") {
                        scale_width = scale_width.or(SourceImage::tag_value(line));
                    } else if line.starts_with("PixelHeight") {
                        scale_height = scale_height.or(SourceImage::tag_value(line));
                    }
                } else if section.eq("[Stage]") {
                    if line.starts_with("StageT=") {
                        // TODO: use rotation (see "Real scale (Tomasi) stuff.pdf")
                        result_metadata.tilt_angle = SourceImage::tag_value(line)
                    }
                } else if section.eq("[PrivateFei]") && line.starts_with("DatabarHeight=") {
                    if let Some(databar_height) = SourceImage::tag_value(line) {
                        result_metadata.databar_height = databar_height;
                    }
                }
            }
            result_metadata.scale = (scale_width.unwrap_or(1.0), scale_height.unwrap_or(1.0));
        }

        if let Some(focal_length) =
            exif.get_field(exif::Tag::FocalLengthIn35mmFilm, exif::In::PRIMARY)
        {
            result_metadata.focal_length = focal_length.value.get_uint(0);
        }
        Ok(result_metadata)
    }

    fn resize(&self, scale: f32) -> Grid<u8> {
        let img_resized = image::imageops::resize(
            &self.img,
            (self.img.width() as f32 * scale) as u32,
            (self.img.height() as f32 * scale) as u32,
            FilterType::Lanczos3,
        );
        let mut img = Grid::<u8>::new(
            img_resized.width() as usize,
            img_resized.height() as usize,
            0,
        );
        for (x, y, val) in img_resized.enumerate_pixels() {
            *img.val_mut(x as usize, y as usize) = val[0];
        }
        img
    }

    fn calibration_matrix(&self, focal_length: Option<u32>) -> Matrix3<f64> {
        // Scale focal length: f_img/f_35mm == width / 36mm (because 35mm film is 36mm wide).
        let max_width = self.img.width().max(self.img.height()) as f64;
        let focal_length =
            focal_length.or(self.focal_length).unwrap_or(1) as f64 / 36.0 * max_width;
        Matrix3::new(
            focal_length,
            0.0,
            self.img.width() as f64 / 2.0,
            0.0,
            focal_length,
            self.img.height() as f64 / 2.0,
            0.0,
            0.0,
            1.0,
        )
    }
}

fn max_image_size(filenames: &[String]) -> usize {
    let mut max_size = 0;
    for path in filenames.iter() {
        let (width, height) = match image::image_dimensions(path) {
            Ok(dimensions) => dimensions,
            Err(err) => {
                eprintln!("Failed to get size for image {}: {}", path, err);
                continue;
            }
        };
        max_size = max_size.max(width as usize * height as usize);
    }
    max_size
}

struct ImageReconstruction {
    hardware_mode: correlation::HardwareMode,
    interpolation_mode: output::InterpolationMode,
    projection_mode: fundamentalmatrix::ProjectionMode,
    vertex_mode: output::VertexMode,
    triangulation: triangulation::Triangulation,
    focal_length: Option<u32>,
    img_filenames: Vec<String>,
}

pub fn reconstruct(args: &Args) -> Result<(), Box<dyn error::Error>> {
    let start_time = SystemTime::now();

    let projection_mode = match args.projection {
        crate::ProjectionMode::Parallel => fundamentalmatrix::ProjectionMode::Affine,
        crate::ProjectionMode::Perspective => fundamentalmatrix::ProjectionMode::Perspective,
    };

    let hardware_mode = match args.mode {
        crate::HardwareMode::Gpu => correlation::HardwareMode::Gpu,
        crate::HardwareMode::GpuLowPower => correlation::HardwareMode::GpuLowPower,
        crate::HardwareMode::Cpu => correlation::HardwareMode::Cpu,
    };

    let interpolation_mode = match args.interpolation {
        crate::InterpolationMode::Delaunay => output::InterpolationMode::Delaunay,
        crate::InterpolationMode::None => output::InterpolationMode::None,
    };

    let vertex_mode = match args.mesh {
        crate::Mesh::Plain => output::VertexMode::Plain,
        crate::Mesh::VertexColors => output::VertexMode::Color,
        crate::Mesh::TextureCoordinates => output::VertexMode::Texture,
    };

    // Most 3D viewers don't display coordinates below 0, reset to default 1.0 - instead of image metadata
    //let out_scale = img1.scale;
    let out_scale = (1.0, 1.0, args.scale as f64);
    let out_scale = match args.projection {
        crate::ProjectionMode::Parallel => (
            out_scale.0,
            out_scale.1,
            out_scale.2 * ((out_scale.0 + out_scale.1) / 2.0),
        ),
        crate::ProjectionMode::Perspective => out_scale,
    };
    let focal_length = args.focal_length;

    let triangulation_projection = match args.projection {
        crate::ProjectionMode::Parallel => triangulation::ProjectionMode::Affine,
        crate::ProjectionMode::Perspective => triangulation::ProjectionMode::Perspective,
    };
    let bundle_adjustment = !args.no_bundle_adjustment;
    let triangulation = triangulation::Triangulation::new(
        args.img_src.len(),
        triangulation_projection,
        bundle_adjustment,
    );
    let img_filenames = args.img_src.to_owned();

    let max_image_size = max_image_size(img_filenames.as_slice());

    let mut reconstruction_task = ImageReconstruction {
        hardware_mode,
        interpolation_mode,
        projection_mode,
        vertex_mode,
        triangulation,
        focal_length,
        img_filenames,
    };

    let images_count = reconstruction_task.img_filenames.len();
    for img_i in 0..images_count - 1 {
        let img1_filename = reconstruction_task.img_filenames[img_i].to_owned();
        for img_j in img_i + 1..images_count {
            let img2_filename = reconstruction_task.img_filenames[img_j].to_owned();
            match reconstruction_task.reconstruct(img_i, img_j) {
                Ok(_) => {}
                Err(err) => {
                    eprintln!(
                        "Failed to match images {} and {} ({})",
                        img1_filename, img2_filename, err
                    );
                }
            }
        }
    }

    match reconstruction_task.recover_poses() {
        Ok(_) => {}
        Err(err) => {
            eprintln!("Failed to recover image poses: {}", err)
        }
    }

    let surface = reconstruction_task.complete_triangulation()?;
    let img_filenames = reconstruction_task.img_filenames.to_owned();
    reconstruction_task.output_surface(
        surface,
        out_scale,
        img_filenames.as_slice(),
        &args.img_out,
    )?;

    if let Ok(t) = start_time.elapsed() {
        println!("Completed reconstruction in {:.3} seconds", t.as_secs_f32());
    }

    Ok(())
}

type CorrelatedPoints = Grid<Option<(Point2D<u32>, f32)>>;

impl ImageReconstruction {
    fn reconstruct(
        &mut self,
        img1_index: usize,
        img2_index: usize,
    ) -> Result<(), Box<dyn error::Error>> {
        let img1_filename = &self.img_filenames[img1_index];
        let img2_filename = &self.img_filenames[img2_index];
        println!("Processing images {} and {}", img1_filename, img2_filename);
        let img1 = SourceImage::load(img1_filename)?;
        let img2 = SourceImage::load(img2_filename)?;
        println!(
            "Image {} has scale width {:?}, height {:?}",
            img1_filename, img1.scale.0, img1.scale.1
        );
        if let Some(focal_length) = img1.focal_length {
            println!(
                "Image {} has focal length {} equivalent to 35mm film",
                img1.filename, focal_length
            );
        } else if self.projection_mode == fundamentalmatrix::ProjectionMode::Perspective {
            println!("Couldn't extract focal length from image {}", img1.filename);
        }
        println!(
            "Image {} has scale width {:?}, height {:?}",
            img2_filename, img2.scale.0, img1.scale.1
        );
        if let Some(focal_length) = img1.focal_length {
            println!(
                "Image {} has focal length {} equivalent to 35mm film",
                img2.filename, focal_length
            );
        } else if self.projection_mode == fundamentalmatrix::ProjectionMode::Perspective {
            println!("Couldn't extract focal length from image {}", img2.filename);
        }
        let tilt_angle = img1
            .tilt_angle
            .and_then(|a1| img2.tilt_angle.map(|a2| a2 - a1));
        if let Some(tilt_angle) = tilt_angle {
            println!("Relative tilt angle is {}", tilt_angle);
        }

        self.triangulation.set_image_data(
            img1_index,
            &img1.calibration_matrix(self.focal_length),
            (img1.img.width() as usize, img1.img.height() as usize),
        );
        self.triangulation.set_image_data(
            img2_index,
            &img2.calibration_matrix(self.focal_length),
            (img2.img.width() as usize, img2.img.height() as usize),
        );

        let point_matches = self.match_keypoints(&img1, &img2);

        let fm = match self.find_fundamental_matrix(
            img1.img.dimensions(),
            img2.img.dimensions(),
            point_matches,
        ) {
            Ok(f) => f,
            Err(err) => {
                eprintln!("Failed to complete RANSAC task: {}", err);
                return Err(err.into());
            }
        };
        println!("Kept {} matches", fm.matches_count);

        let correlated_points = match self.correlate_points(&img1, &img2, fm.f) {
            Ok(correlated_points) => correlated_points,
            Err(err) => {
                eprintln!("Failed to complete points correlation: {}", err);
                return Err(err);
            }
        };

        match self.triangulate_surface(img1_index, img2_index, fm.f, correlated_points) {
            Ok(()) => Ok(()),
            Err(err) => {
                eprintln!("Failed to add image pair: {}", err);
                Err(err.into())
            }
        }
    }

    fn match_keypoints(
        &self,
        img1: &SourceImage,
        img2: &SourceImage,
    ) -> Vec<(Point2D<usize>, Point2D<usize>)> {
        let start_time = SystemTime::now();

        let scale_steps = orb::optimal_scale_steps(img1.img.dimensions());
        let total_percent: f32 = (0..=scale_steps)
            .map(|step| 1.0 / ((1 << (scale_steps - step)) as f32).powi(2))
            .sum::<f32>()
            * 2.0;

        let mut total_percent_complete = 0.0;

        let pb = new_progress_bar(false);
        let mut keypoints1 = vec![];
        let mut keypoints2 = vec![];
        for i in 0..=scale_steps {
            let scale = 1.0 / (1 << (scale_steps - i)) as f32;

            let img1_scaled = img1.resize(scale);
            let img2_scaled = img2.resize(scale);

            let img1_pb = CorrelationProgressBar {
                total_percent_complete,
                total_percent,
                pb: &pb,
                scale,
            };
            let mut new_keypoints1 = orb::extract_points(&img1_scaled, Some(&img1_pb))
                .iter()
                .map(|(p, descriptor)| {
                    (
                        Point2D::new((p.x as f32 / scale) as usize, (p.y as f32 / scale) as usize),
                        descriptor.to_owned(),
                    )
                })
                .collect::<Vec<_>>();
            keypoints1.append(&mut new_keypoints1);
            total_percent_complete += scale * scale / total_percent;

            let img2_pb = CorrelationProgressBar {
                total_percent_complete,
                total_percent,
                pb: &pb,
                scale,
            };
            let mut new_keypoints2 = orb::extract_points(&img2_scaled, Some(&img2_pb))
                .iter()
                .map(|(p, descriptor)| {
                    (
                        Point2D::new((p.x as f32 / scale) as usize, (p.y as f32 / scale) as usize),
                        descriptor.to_owned(),
                    )
                })
                .collect::<Vec<_>>();
            keypoints2.append(&mut new_keypoints2);
            total_percent_complete += scale * scale / total_percent;
        }
        pb.finish_and_clear();
        if let Ok(t) = start_time.elapsed() {
            println!("Extracted feature points in {:.3} seconds", t.as_secs_f32(),);
        }
        println!(
            "Image {} has {} feature points",
            img1.filename,
            keypoints1.len()
        );
        println!(
            "Image {} has {} feature points",
            img2.filename,
            keypoints2.len()
        );

        let start_time = SystemTime::now();
        let projection_mode = match self.projection_mode {
            fundamentalmatrix::ProjectionMode::Affine => pointmatching::ProjectionMode::Affine,
            fundamentalmatrix::ProjectionMode::Perspective => {
                pointmatching::ProjectionMode::Perspective
            }
        };
        let pb = new_progress_bar(false);
        let matcher = pointmatching::KeypointMatching::new(
            &keypoints1,
            &keypoints2,
            projection_mode,
            Some(&pb),
        );
        pb.finish_and_clear();
        drop(keypoints1);
        drop(keypoints2);

        let point_matches = matcher.matches;

        if let Ok(t) = start_time.elapsed() {
            println!("Matched keypoints in {:.3} seconds", t.as_secs_f32());
        }
        println!("Found {} matches", point_matches.len());
        point_matches
    }

    fn find_fundamental_matrix(
        &self,
        img1_dimensions: (u32, u32),
        img2_dimensions: (u32, u32),
        point_matches: Vec<(Point2D<usize>, Point2D<usize>)>,
    ) -> Result<FundamentalMatrix, fundamentalmatrix::RansacError> {
        let start_time = SystemTime::now();
        let pb = new_progress_bar(true);

        let max_dimension = img1_dimensions
            .0
            .max(img1_dimensions.1)
            .max(img2_dimensions.0)
            .max(img2_dimensions.1) as f64;

        let result = FundamentalMatrix::new(
            self.projection_mode,
            max_dimension,
            &point_matches,
            Some(&pb),
        );
        pb.finish_and_clear();

        if let Ok(t) = start_time.elapsed() {
            println!("Completed RANSAC fitting in {:.3} seconds", t.as_secs_f32());
        }

        result
    }

    fn correlate_points(
        &self,
        img1: &SourceImage,
        img2: &SourceImage,
        f: Matrix3<f64>,
    ) -> Result<CorrelatedPoints, Box<dyn error::Error>> {
        let mut point_correlations;

        let start_time = SystemTime::now();
        let scale_steps = PointCorrelations::optimal_scale_steps(img1.img.dimensions());
        let total_percent: f32 = (0..=scale_steps)
            .map(|step| 1.0 / ((1 << (scale_steps - step)) as f32).powi(2))
            .sum::<f32>();

        let pb = new_progress_bar(false);
        let projection_mode = match self.projection_mode {
            fundamentalmatrix::ProjectionMode::Affine => correlation::ProjectionMode::Affine,
            fundamentalmatrix::ProjectionMode::Perspective => {
                correlation::ProjectionMode::Perspective
            }
        };

        let mut total_percent_complete = 0.0;
        point_correlations = PointCorrelations::new(
            img1.img.dimensions(),
            img2.img.dimensions(),
            f,
            projection_mode,
            self.hardware_mode,
        );
        println!(
            "Selected hardware: {}",
            point_correlations.get_selected_hardware()
        );
        for i in 0..=scale_steps {
            let scale = 1.0 / (1 << (scale_steps - i)) as f32;
            let img1 = img1.resize(scale);
            let img2 = img2.resize(scale);

            let pb = CorrelationProgressBar {
                total_percent_complete,
                total_percent,
                pb: &pb,
                scale,
            };

            point_correlations.correlate_images(img1, img2, scale, Some(&pb))?;
            total_percent_complete += scale * scale / total_percent;
        }
        pb.finish_and_clear();
        if let Ok(t) = start_time.elapsed() {
            println!(
                "Completed surface generation in {:.3} seconds",
                t.as_secs_f32()
            );
        }

        point_correlations.complete()?;
        Ok(point_correlations.correlated_points)
    }

    fn recover_poses(&mut self) -> Result<(), triangulation::TriangulationError> {
        loop {
            let start_time = SystemTime::now();
            let pb = new_progress_bar(true);
            let result = self.triangulation.recover_next_camera(Some(&pb));
            pb.finish_and_clear();

            match result {
                Ok(Some(image)) => {
                    println!("Recovered pose for image {}", self.img_filenames[image])
                }
                Ok(None) => break,
                Err(err) => {
                    eprintln!("Failed to recover pose for next image: {}", err);
                }
            }

            if let Ok(t) = start_time.elapsed() {
                println!("Recovered pose in {:.3} seconds", t.as_secs_f32());
            }
        }

        Ok(())
    }
    fn triangulate_surface(
        &mut self,
        img1_index: usize,
        img2_index: usize,
        f: Matrix3<f64>,
        correlated_points: CorrelatedPoints,
    ) -> Result<(), triangulation::TriangulationError> {
        let start_time = SystemTime::now();

        let pb = new_progress_bar(true);
        let result = self.triangulation.triangulate(
            img1_index,
            img2_index,
            &correlated_points,
            &f,
            Some(&pb),
        );
        pb.finish_and_clear();

        if let Ok(t) = start_time.elapsed() {
            println!("Added image pair in {:.3} seconds", t.as_secs_f32());
        }

        result
    }

    fn complete_triangulation(&mut self) -> Result<triangulation::Surface, Box<dyn error::Error>> {
        let start_time = SystemTime::now();

        let pb = new_progress_bar(false);

        let surface = self.triangulation.triangulate_all(Some(&pb))?;
        let retained_images = self
            .triangulation
            .retained_images()?
            .iter()
            .map(|src_img_i| self.img_filenames[*src_img_i].to_owned())
            .collect::<Vec<_>>();
        self.triangulation.complete();

        pb.finish_and_clear();

        self.img_filenames = retained_images;

        if let Ok(t) = start_time.elapsed() {
            println!(
                "Completed triangulation post-processing in {:.3} seconds",
                t.as_secs_f32()
            );
        }

        Ok(surface)
    }

    fn output_surface(
        &mut self,
        surface: triangulation::Surface,
        out_scale: (f64, f64, f64),
        texture_filenames: &[String],
        output_filename: &str,
    ) -> Result<(), Box<dyn error::Error>> {
        let start_time = SystemTime::now();
        let pb = new_progress_bar(false);
        let images = texture_filenames
            .iter()
            .map(|img_filename| SourceImage::load_rgb(img_filename).unwrap())
            .collect();

        let result = output::output(
            surface,
            out_scale,
            images,
            output_filename,
            self.interpolation_mode,
            self.vertex_mode,
            Some(&pb),
        );
        pb.finish_and_clear();
        match result {
            Ok(_) => {}
            Err(err) => {
                eprintln!("Failed to save image: {}", err);
                return Err(err);
            }
        }

        if let Ok(t) = start_time.elapsed() {
            println!("Saved result in {:.3} seconds", t.as_secs_f32());
        }

        Ok(())
    }
}

fn new_progress_bar(show_message: bool) -> ProgressBar {
    let template = if show_message {
        "{bar:40} {percent_decimal}% (eta: {eta}{msg})"
    } else {
        "{bar:40} {percent_decimal}% (eta: {eta})"
    };
    let pb_style = ProgressStyle::default_bar()
        .template(template)
        .unwrap()
        .with_key("percent_decimal", |s: &ProgressState, w: &mut dyn Write| {
            write!(w, "{:.2}", s.fraction() * 100.0).unwrap()
        });
    ProgressBar::new(10000).with_style(pb_style)
}

impl fundamentalmatrix::ProgressListener for ProgressBar {
    fn report_status(&self, pos: f32) {
        self.set_position((pos * 10000.0) as u64);
    }
    fn report_matches(&self, matches_count: usize) {
        if matches_count > 0 {
            self.set_message(format!(", {} matches", matches_count));
        }
    }
}

struct CorrelationProgressBar<'p> {
    total_percent_complete: f32,
    total_percent: f32,
    scale: f32,
    pb: &'p ProgressBar,
}

impl orb::ProgressListener for CorrelationProgressBar<'_> {
    fn report_status(&self, pos: f32) {
        let percent_complete =
            self.total_percent_complete + pos * self.scale * self.scale / self.total_percent;
        self.pb.set_position((percent_complete * 10000.0) as u64);
    }
}

impl pointmatching::ProgressListener for ProgressBar {
    fn report_status(&self, pos: f32) {
        self.set_position((pos * 10000.0) as u64);
    }
}

impl correlation::ProgressListener for CorrelationProgressBar<'_> {
    fn report_status(&self, pos: f32) {
        let percent_complete =
            self.total_percent_complete + pos * self.scale * self.scale / self.total_percent;
        self.pb.set_position((percent_complete * 10000.0) as u64);
    }
}

impl triangulation::ProgressListener for ProgressBar {
    fn report_status(&self, pos: f32) {
        self.set_position((pos * 10000.0) as u64);
    }
}

impl output::ProgressListener for ProgressBar {
    fn report_status(&self, pos: f32) {
        self.set_position((pos * 10000.0) as u64);
    }
}
