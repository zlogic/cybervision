use crate::crosscorrelation;
use crate::crosscorrelation::PointCorrelations;
use crate::fundamentalmatrix;
use crate::fundamentalmatrix::FundamentalMatrix;
use crate::orb;
use crate::output;
use crate::pointmatching;
use crate::triangulation;
use crate::Cli;

use image::imageops::FilterType;
use image::GenericImageView;
use image::GrayImage;
use image::RgbImage;
use indicatif::ProgressState;
use indicatif::{ProgressBar, ProgressStyle};
use nalgebra::DMatrix;
use nalgebra::Matrix3;
use std::error;
use std::fmt;
use std::fmt::Write;
use std::fs::File;
use std::io::BufReader;
use std::str::FromStr;
use std::time::SystemTime;

const TIFFTAG_META_PHENOM: tiff::tags::Tag = tiff::tags::Tag::Unknown(34683);
const TIFFTAG_META_QUANTA: tiff::tags::Tag = tiff::tags::Tag::Unknown(34682);

struct SourceImage {
    img: GrayImage,
    scale: (f32, f32),
    tilt_angle: Option<f32>,
    filename: String,
}

#[derive(Debug)]
struct ImageMeta {
    scale: (f32, f32),
    tilt_angle: Option<f32>,
    databar_height: u32,
}

impl SourceImage {
    fn load(path: &str) -> Result<SourceImage, image::ImageError> {
        let metadata = SourceImage::get_metadata(path);
        let img = image::open(path)?.into_luma8();
        let img = img.view(0, 0, img.width(), img.height() - metadata.databar_height);

        Ok(SourceImage {
            img: img.to_image(),
            scale: metadata.scale,
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
        };
        match image::ImageFormat::from_path(path) {
            Ok(image::ImageFormat::Tiff) => {
                SourceImage::get_metadata_tiff(path).unwrap_or(default_metadata)
            }
            _ => default_metadata,
        }
    }

    fn tag_value<F: FromStr>(line: &str) -> Option<F> {
        line.split_once('=')
            .and_then(|(_, value)| value.parse::<F>().ok())
    }

    fn get_metadata_tiff(path: &str) -> Result<ImageMeta, tiff::TiffError> {
        let reader = BufReader::new(File::open(path)?);
        let mut decoder = tiff::decoder::Decoder::new(reader)?;
        let metadata = decoder
            .get_tag_ascii_string(TIFFTAG_META_PHENOM)
            .or(decoder.get_tag_ascii_string(TIFFTAG_META_QUANTA));

        match metadata {
            Ok(data) => {
                let separators = ['\r', '\n'];
                let mut section = "";
                let mut scale_width = None;
                let mut scale_height = None;
                let mut tilt_angle = None;
                let mut databar_height = None;
                for line in data.split_terminator(&separators[..]) {
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
                            // or allow to specify a custom depth scale (e.g. a negative one)
                            tilt_angle = tilt_angle.or(SourceImage::tag_value(line))
                        }
                    } else if section.eq("[PrivateFei]") && line.starts_with("DatabarHeight=") {
                        databar_height = databar_height.or(SourceImage::tag_value(line))
                    }
                }
                let metadata = ImageMeta {
                    scale: (scale_width.unwrap_or(1.0), scale_height.unwrap_or(1.0)),
                    tilt_angle,
                    databar_height: databar_height.unwrap_or(0),
                };
                Ok(metadata)
            }
            Err(e) => Err(e),
        }
    }

    fn resize(&self, scale: f32) -> DMatrix<u8> {
        let img_resized = image::imageops::resize(
            &self.img,
            (self.img.width() as f32 * scale) as u32,
            (self.img.height() as f32 * scale) as u32,
            FilterType::Lanczos3,
        );
        let mut img =
            DMatrix::<u8>::zeros(img_resized.height() as usize, img_resized.width() as usize);
        for (x, y, val) in img_resized.enumerate_pixels() {
            img[(y as usize, x as usize)] = val[0];
        }
        img
    }
}

struct ImageReconstruction {
    hardware_mode: crosscorrelation::HardwareMode,
    interpolation_mode: output::InterpolationMode,
    projection_mode: fundamentalmatrix::ProjectionMode,
    vertex_mode: output::VertexMode,
    triangulation: triangulation::Triangulation,
}

pub fn reconstruct(args: &Cli) -> Result<(), Box<dyn error::Error>> {
    let start_time = SystemTime::now();

    let projection_mode = match args.projection {
        crate::ProjectionMode::Parallel => fundamentalmatrix::ProjectionMode::Affine,
        crate::ProjectionMode::Perspective => fundamentalmatrix::ProjectionMode::Perspective,
    };

    let hardware_mode = match args.mode {
        crate::HardwareMode::Gpu => crosscorrelation::HardwareMode::Gpu,
        crate::HardwareMode::GpuLowPower => crosscorrelation::HardwareMode::GpuLowPower,
        crate::HardwareMode::Cpu => crosscorrelation::HardwareMode::Cpu,
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

    let triangulation_projection = match args.projection {
        crate::ProjectionMode::Parallel => triangulation::ProjectionMode::Affine,
        crate::ProjectionMode::Perspective => triangulation::ProjectionMode::Perspective,
    };
    let bundle_adjustment = !args.no_bundle_adjustment;
    let triangulation =
        triangulation::Triangulation::new(triangulation_projection, out_scale, bundle_adjustment);

    let mut reconstruction_task = ImageReconstruction {
        hardware_mode,
        interpolation_mode,
        projection_mode,
        vertex_mode,
        triangulation,
    };

    if args.img_src.len() > 2 && interpolation_mode != output::InterpolationMode::None {
        return Err(ReconstructionError::new(
            "Interpolation should be none when reconstructing from more than 2 images",
        )
        .into());
    }

    for i in 0..args.img_src.len() - 1 {
        let img1_filename = &args.img_src[i];
        let img2_filename = &args.img_src[i + 1];
        reconstruction_task.reconstruct(img1_filename, img2_filename)?;
    }

    let surface = reconstruction_task.complete_triangulation()?;
    let img_filenames = &args.img_src[0..args.img_src.len() - 1];
    reconstruction_task.output_surface(surface, img_filenames, &args.img_out)?;

    if let Ok(t) = start_time.elapsed() {
        println!("Completed reconstruction in {:.3} seconds", t.as_secs_f32());
    }

    Ok(())
}

type CorrelatedPoints = DMatrix<Option<(u32, u32)>>;

impl ImageReconstruction {
    fn reconstruct(
        &mut self,
        img1_filename: &str,
        img2_filename: &str,
    ) -> Result<(), Box<dyn error::Error>> {
        println!("Processing images {} and {}", img1_filename, img2_filename);
        let img1 = SourceImage::load(img1_filename)?;
        let img2 = SourceImage::load(img2_filename)?;
        println!(
            "Image {} has scale width {:?}, height {:?}",
            img1_filename, img1.scale.0, img1.scale.1
        );
        println!(
            "Image {} has scale width {:?}, height {:?}",
            img2_filename, img2.scale.0, img1.scale.1
        );
        let tilt_angle = img1
            .tilt_angle
            .and_then(|a1| img2.tilt_angle.map(|a2| a2 - a1));
        if let Some(tilt_angle) = tilt_angle {
            println!("Relative tilt angle is {}", tilt_angle);
        }

        let point_matches = self.match_keypoints(&img1, &img2);

        let fm = match self.find_fundamental_matrix(point_matches) {
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

        match self.triangulate_surface(fm.f, correlated_points) {
            Ok(()) => Ok(()),
            Err(err) => {
                eprintln!("Failed to triangulate surface: {}", err);
                Err(err.into())
            }
        }
    }

    fn match_keypoints(
        &self,
        img1: &SourceImage,
        img2: &SourceImage,
    ) -> Vec<((usize, usize), (usize, usize))> {
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
                .map(|((row, col), descriptor)| {
                    (
                        (
                            (*col as f32 / scale) as usize,
                            (*row as f32 / scale) as usize,
                        ),
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
                .map(|((row, col), descriptor)| {
                    (
                        (
                            (*col as f32 / scale) as usize,
                            (*row as f32 / scale) as usize,
                        ),
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
        point_matches: Vec<((usize, usize), (usize, usize))>,
    ) -> Result<FundamentalMatrix, fundamentalmatrix::RansacError> {
        let start_time = SystemTime::now();
        let pb = new_progress_bar(true);

        let result = FundamentalMatrix::new(self.projection_mode, &point_matches, Some(&pb));
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
            fundamentalmatrix::ProjectionMode::Affine => crosscorrelation::ProjectionMode::Affine,
            fundamentalmatrix::ProjectionMode::Perspective => {
                crosscorrelation::ProjectionMode::Perspective
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
        let mut reverse_point_correlations = PointCorrelations::new(
            img2.img.dimensions(),
            img1.img.dimensions(),
            f.transpose(),
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

            point_correlations.correlate_images(img1, img2, scale, Some(&pb));
            total_percent_complete += scale * scale / total_percent;
        }
        pb.finish_and_clear();
        if let Ok(t) = start_time.elapsed() {
            println!(
                "Completed surface generation in {:.3} seconds",
                t.as_secs_f32()
            );
        }

        reverse_point_correlations.complete()?;
        point_correlations.complete()?;
        Ok(point_correlations.correlated_points)
    }

    fn triangulate_surface(
        &mut self,
        f: Matrix3<f64>,
        correlated_points: DMatrix<Option<(u32, u32)>>,
    ) -> Result<(), triangulation::TriangulationError> {
        let start_time = SystemTime::now();

        let pb = new_progress_bar(true);
        let result = self
            .triangulation
            .triangulate(&correlated_points, &f, Some(&pb));
        pb.finish_and_clear();

        if let Ok(t) = start_time.elapsed() {
            println!(
                "Completed point triangulation in {:.3} seconds",
                t.as_secs_f32()
            );
        }

        result
    }

    fn complete_triangulation(&mut self) -> Result<triangulation::Surface, Box<dyn error::Error>> {
        let start_time = SystemTime::now();

        let pb = new_progress_bar(false);

        let surface = self.triangulation.triangulate_all(Some(&pb))?;
        self.triangulation.complete();

        pb.finish_and_clear();

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

impl crosscorrelation::ProgressListener for CorrelationProgressBar<'_> {
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

#[derive(Debug)]
struct ReconstructionError {
    msg: &'static str,
}

impl ReconstructionError {
    fn new(msg: &'static str) -> ReconstructionError {
        ReconstructionError { msg }
    }
}

impl std::error::Error for ReconstructionError {}

impl fmt::Display for ReconstructionError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.msg)
    }
}
