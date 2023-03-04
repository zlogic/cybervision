use crate::correlation;
use crate::crosscorrelation;
use crate::crosscorrelation::PointCorrelations;
use crate::fast::Fast;
use crate::fundamentalmatrix;
use crate::fundamentalmatrix::FundamentalMatrix;
use crate::output;
use crate::triangulation;
use crate::triangulation::Surface;
use crate::Cli;

use image::imageops::FilterType;
use image::GenericImageView;
use image::GrayImage;
use indicatif::ProgressState;
use indicatif::{ProgressBar, ProgressStyle};
use nalgebra::DMatrix;
use rayon::prelude::*;
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
}

#[derive(Debug)]
struct ImageMeta {
    scale: (f32, f32),
    tilt_angle: Option<f32>,
    databar_height: u32,
}

impl SourceImage {
    fn load(path: &String) -> Result<SourceImage, image::ImageError> {
        let metadata = SourceImage::get_metadata(&path);
        let img = image::open(&path)?.into_luma8();
        let img = img.view(0, 0, img.width(), img.height() - metadata.databar_height);

        Ok(SourceImage {
            img: img.to_image(),
            scale: metadata.scale,
            tilt_angle: metadata.tilt_angle,
        })
    }

    fn get_metadata(path: &String) -> ImageMeta {
        let default_metadata = ImageMeta {
            scale: (1.0, 1.0),
            tilt_angle: None,
            databar_height: 0,
        };
        match image::ImageFormat::from_path(&path) {
            Ok(format) => match format {
                image::ImageFormat::Tiff => {
                    SourceImage::get_metadata_tiff(&path).unwrap_or(default_metadata)
                }
                _ => default_metadata,
            },
            _ => default_metadata,
        }
    }

    fn tag_value<F: FromStr>(line: &str) -> Option<F> {
        line.split_once("=")
            .map(|(_, value)| value.parse::<F>().ok())
            .flatten()
    }

    fn get_metadata_tiff(path: &String) -> Result<ImageMeta, tiff::TiffError> {
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
                    if line.starts_with("[") && line.ends_with("]") {
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
                    } else if section.eq("[PrivateFei]") {
                        if line.starts_with("DatabarHeight=") {
                            databar_height = databar_height.or(SourceImage::tag_value(line))
                        }
                    }
                }
                let metadata = ImageMeta {
                    scale: (scale_width.unwrap_or(1.0), scale_height.unwrap_or(1.0)),
                    tilt_angle: tilt_angle,
                    databar_height: databar_height.unwrap_or(0),
                };
                return Ok(metadata);
            }
            Err(e) => return Err(e),
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
        return img;
    }
}

type Point = (usize, usize);

pub fn reconstruct(args: &Cli) {
    let img1 = SourceImage::load(&args.img1).unwrap();
    let img2 = SourceImage::load(&args.img2).unwrap();
    println!(
        "Image {} has scale width {:?}, height {:?}",
        args.img1, img1.scale.0, img1.scale.1
    );
    println!(
        "Image {} has scale width {:?}, height {:?}",
        args.img2, img2.scale.0, img1.scale.1
    );
    let tilt_angle = img1
        .tilt_angle
        .map(|a1| img2.tilt_angle.map(|a2| a2 - a1))
        .flatten();
    if tilt_angle.is_some() {
        println!("Relative tilt angle is {}", tilt_angle.unwrap());
    }
    let start_time = SystemTime::now();
    let keypoint_scale: f32;
    let img1_scaled: DMatrix<u8>;
    let img2_scaled: DMatrix<u8>;
    let keypoints1: Fast;
    let keypoints2: Fast;
    {
        let start_time = SystemTime::now();
        keypoint_scale = Fast::optimal_keypoint_scale(img1.img.dimensions());
        img1_scaled = img1.resize(keypoint_scale);
        img2_scaled = img2.resize(keypoint_scale);

        keypoints1 = Fast::new(&img1_scaled);
        keypoints2 = Fast::new(&img2_scaled);

        match start_time.elapsed() {
            Ok(t) => println!("Extracted feature points in {:.3} seconds", t.as_secs_f32(),),
            Err(_) => {}
        }
        println!(
            "Image {} has {} feature points",
            args.img1,
            keypoints1.keypoints().len()
        );
        println!(
            "Image {} has {} feature points",
            args.img2,
            keypoints2.keypoints().len()
        );
    }

    let point_matches: Vec<(Point, Point)>;
    {
        let start_time = SystemTime::now();

        let pb = new_progress_bar(false);
        let matcher = correlation::KeypointMatching::new(
            &img1_scaled,
            &img2_scaled,
            keypoints1.keypoints(),
            keypoints2.keypoints(),
            Some(&pb),
        );
        pb.finish_and_clear();
        point_matches = matcher.matches;
        drop(keypoints1);
        drop(keypoints2);
        drop(img1_scaled);
        drop(img2_scaled);
        match start_time.elapsed() {
            Ok(t) => println!("Matched keypoints in {:.3} seconds", t.as_secs_f32(),),
            Err(_) => {}
        }
        println!("Found {} matches", point_matches.len());
    }

    let fm: FundamentalMatrix;
    {
        let start_time = SystemTime::now();
        let point_matches = point_matches
            .into_par_iter()
            .map(|(p1, p2)| {
                (
                    (
                        (p1.0 as f32 * keypoint_scale) as usize,
                        (p1.1 as f32 * keypoint_scale) as usize,
                    ),
                    (
                        (p2.0 as f32 * keypoint_scale) as usize,
                        (p2.1 as f32 * keypoint_scale) as usize,
                    ),
                )
            })
            .collect();
        let match_buckets =
            FundamentalMatrix::matches_to_buckets(&point_matches, img1.img.dimensions());
        drop(point_matches);
        let pb = new_progress_bar(true);
        let projection_mode = match args.projection {
            crate::ProjectionMode::Parallel => fundamentalmatrix::ProjectionMode::Affine,
            crate::ProjectionMode::Perspective => fundamentalmatrix::ProjectionMode::Perspective,
        };
        let f = FundamentalMatrix::new(projection_mode, &match_buckets, Some(&pb));
        pb.finish_and_clear();
        match start_time.elapsed() {
            Ok(t) => println!("Completed RANSAC fitting in {:.3} seconds", t.as_secs_f32(),),
            Err(_) => {}
        }
        match f {
            Ok(f) => fm = f,
            Err(e) => {
                eprintln!("Failed to complete RANSAC task: {}", e);
                return;
            }
        }

        println!("Kept {} matches", fm.matches_count);
    }

    let mut point_correlations: PointCorrelations;
    {
        let start_time = SystemTime::now();
        let scale_steps = PointCorrelations::optimal_scale_steps(img1.img.dimensions());
        let total_percent: f32 = (0..scale_steps + 1)
            .map(|step| 1.0 / ((1 << (scale_steps - step)) as f32).powi(2))
            .sum();

        let pb = new_progress_bar(false);
        let projection_mode = match args.projection {
            crate::ProjectionMode::Parallel => crosscorrelation::ProjectionMode::Affine,
            crate::ProjectionMode::Perspective => crosscorrelation::ProjectionMode::Perspective,
        };
        let hardware_mode = match args.mode {
            crate::HardwareMode::GPU => crosscorrelation::HardwareMode::GPU,
            crate::HardwareMode::CPU => crosscorrelation::HardwareMode::CPU,
        };

        let mut total_percent_complete = 0.0;
        point_correlations = PointCorrelations::new(
            img1.img.dimensions(),
            img2.img.dimensions(),
            fm.f,
            projection_mode,
            hardware_mode,
        );
        println!(
            "Selected hardware: {}",
            point_correlations.get_selected_hardware()
        );
        for i in 0..scale_steps + 1 {
            let scale = 1.0 / (1 << (scale_steps - i)) as f32;
            let img1 = img1.resize(scale);
            let img2 = img2.resize(scale);

            let pb = CrossCorrelationProgressBar {
                total_percent_complete,
                total_percent,
                pb: &pb,
                scale,
            };

            point_correlations.correlate_images(img1, img2, scale, Some(&pb));
            total_percent_complete += scale * scale / total_percent;
        }
        match point_correlations.complete() {
            Ok(_) => {}
            Err(err) => {
                eprintln!("Failed to complete points correlation: {}", err)
            }
        }
        pb.finish_and_clear();

        match start_time.elapsed() {
            Ok(t) => println!(
                "Completed surface generation in {:.3} seconds",
                t.as_secs_f32()
            ),
            Err(_) => {}
        }
    }

    // Most 3D viewers don't display coordinates below 0, reset to default 1.0 - instead of image metadata
    //let out_scale = img1.scale;
    let out_scale = (1.0, 1.0, args.scale);
    let surface: triangulation::Surface;
    {
        let start_time = SystemTime::now();

        let projection_mode = match args.projection {
            crate::ProjectionMode::Parallel => triangulation::ProjectionMode::Affine,
            crate::ProjectionMode::Perspective => triangulation::ProjectionMode::Perspective,
        };
        surface = Surface::new(
            &point_correlations.correlated_points,
            projection_mode,
            out_scale,
        );
        drop(point_correlations);

        match start_time.elapsed() {
            Ok(t) => println!(
                "Completed point triangulation in {:.3} seconds",
                t.as_secs_f32()
            ),
            Err(_) => {}
        }
    }

    {
        let start_time = SystemTime::now();

        let pb = new_progress_bar(false);
        let interpolation_mode = match args.interpolation {
            crate::InterpolationMode::Delaunay => output::InterpolationMode::Delaunay,
            crate::InterpolationMode::None => output::InterpolationMode::None,
        };

        let result = output::output(surface.points, &args.img_out, interpolation_mode, Some(&pb));
        pb.finish_and_clear();
        match result {
            Ok(_) => {}
            Err(e) => {
                eprintln!("Failed to save image: {}", e);
                return;
            }
        }

        match start_time.elapsed() {
            Ok(t) => println!("Saved result in {:.3} seconds", t.as_secs_f32()),
            Err(_) => {}
        }
    }

    match start_time.elapsed() {
        Ok(t) => println!("Completed reconstruction in {:.3} seconds", t.as_secs_f32(),),
        Err(_) => {}
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
    return ProgressBar::new(10000).with_style(pb_style);
}

impl correlation::ProgressListener for ProgressBar {
    fn report_status(&self, pos: f32) {
        self.set_position((pos * 10000.0) as u64);
    }
}

impl fundamentalmatrix::ProgressListener for ProgressBar {
    fn report_status(&self, pos: f32) {
        self.set_position((pos * 10000.0) as u64);
    }
    fn report_matches(&self, matches_count: usize) {
        self.set_message(format!(", {} matches", matches_count));
    }
}

struct CrossCorrelationProgressBar<'p> {
    total_percent_complete: f32,
    total_percent: f32,
    scale: f32,
    pb: &'p ProgressBar,
}

impl crosscorrelation::ProgressListener for CrossCorrelationProgressBar<'_> {
    fn report_status(&self, pos: f32) {
        let percent_complete =
            self.total_percent_complete + pos * self.scale * self.scale / self.total_percent;
        self.pb.set_position((percent_complete * 10000.0) as u64);
    }
}

impl output::ProgressListener for ProgressBar {
    fn report_status(&self, pos: f32) {
        self.set_position((pos * 10000.0) as u64);
    }
}
