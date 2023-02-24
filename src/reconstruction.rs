use crate::correlation::Correlator;
use crate::fast::FastExtractor;
use crate::Cli;

use image::imageops::FilterType;
use image::GenericImageView;
use image::GrayImage;
use indicatif::{ProgressBar, ProgressStyle};
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

    fn resize(&self, scale: f32) -> GrayImage {
        return image::imageops::resize(
            &self.img,
            (self.img.width() as f32 * scale) as u32,
            (self.img.height() as f32 * scale) as u32,
            FilterType::Lanczos3,
        );
    }
}

type Point = (u32, u32);

pub fn reconstruct(args: &Cli) {
    let img1 = SourceImage::load(&args.img1).unwrap();
    let img2 = SourceImage::load(&args.img2).unwrap();
    println!(
        "Image {} has scale width {:e}, height {:e}",
        args.img1, img1.scale.0, img1.scale.1
    );
    println!(
        "Image {} has scale width {:e}, height {:e}",
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
    let img1_scaled: GrayImage;
    let img2_scaled: GrayImage;
    let points1: Vec<Point>;
    let points2: Vec<Point>;
    {
        let start_time = SystemTime::now();
        let fast = FastExtractor::new();
        keypoint_scale = fast.optimal_keypoint_scale(&img1.img);
        img1_scaled = img1.resize(keypoint_scale);
        img2_scaled = img2.resize(keypoint_scale);

        points1 = fast.find_points(&img1_scaled);
        points2 = fast.find_points(&img2_scaled);

        match start_time.elapsed() {
            Ok(t) => println!("Extracted feature points in {:.3} seconds", t.as_secs_f32(),),
            Err(_) => {}
        }
        println!("Image {} has {} feature points", args.img1, points1.len());
        println!("Image {} has {} feature points", args.img2, points2.len());
    }

    let pb_style = ProgressStyle::default_bar()
        .template("{wide_bar} {percent}/100% (eta: {eta})")
        .unwrap();

    let point_matches: Vec<(Point, Point)>;
    {
        let start_time = SystemTime::now();
        let correlator = Correlator::new();

        let pb = ProgressBar::new(100).with_style(pb_style);
        let cb = |counter| pb.set_position((counter * 100.0) as u64);
        point_matches = correlator.match_points(&img1_scaled, &img2_scaled, &points1, &points2, cb);
        pb.finish_and_clear();
        match start_time.elapsed() {
            Ok(t) => println!("Matched keypoints in {:.3} seconds", t.as_secs_f32(),),
            Err(_) => {}
        }
        println!("Found {} matches", point_matches.len());
        drop(points1);
        drop(points2);
    }

    // Most 3D viewers don't display coordinates below 0, reset to default 1.0
    //let scale = (1.0, 1.0);

    match start_time.elapsed() {
        Ok(t) => println!("Completed reconstruction in {:.3} seconds", t.as_secs_f32(),),
        Err(_) => {}
    }
}
