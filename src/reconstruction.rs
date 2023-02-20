use image::GenericImageView;

use crate::Cli;
use std::fs::File;
use std::io::BufReader;
use std::str::FromStr;

const TIFFTAG_META_PHENOM: tiff::tags::Tag = tiff::tags::Tag::Unknown(34683);
const TIFFTAG_META_QUANTA: tiff::tags::Tag = tiff::tags::Tag::Unknown(34682);

struct Image {
    img: Vec<u8>,
    dimensions: (usize, usize),
    scale: (f32, f32),
    tilt_angle: Option<f32>,
}

struct ImageMeta {
    scale: (f32, f32),
    tilt_angle: Option<f32>,
    databar_height: u32,
}

impl Image {
    fn load(path: &String) -> Result<Image, image::ImageError> {
        let metadata = Image::get_metadata(&path);
        let img_full = image::open(&path)?.into_luma8();
        let img = img_full.view(
            0,
            0,
            img_full.width(),
            img_full.height() - metadata.databar_height,
        );

        let mut bytes = Vec::with_capacity((img.width() * img.height()) as usize);
        bytes.resize(bytes.capacity(), 0);
        for pixel in img.pixels() {
            bytes[(pixel.1 * img.width() + pixel.0) as usize] = pixel.2[0];
        }
        Ok(Image {
            img: bytes,
            dimensions: (img.width() as usize, img.height() as usize),
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
                    Image::get_metadata_tiff(&path).unwrap_or(default_metadata)
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
                            scale_width = scale_width.or(Image::tag_value(line));
                        } else if line.starts_with("PixelHeight") {
                            scale_height = scale_height.or(Image::tag_value(line));
                        }
                    } else if section.eq("[Stage]") {
                        if line.starts_with("StageT=") {
                            // TODO: use rotation (see "Real scale (Tomasi) stuff.pdf")
                            // or allow to specify a custom depth scale (e.g. a negative one)
                            tilt_angle = tilt_angle.or(Image::tag_value(line))
                        }
                    } else if section.eq("[PrivateFei]") {
                        if line.starts_with("DatabarHeight=") {
                            databar_height = databar_height.or(Image::tag_value(line))
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
}

pub fn reconstruct(args: &Cli) {
    let img1 = Image::load(&args.img1).unwrap();
    let img2 = Image::load(&args.img2).unwrap();
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
    // Most 3D viewers don't display coordinates below 0, reset to default 1.0
}
