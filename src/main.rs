use std::{env, fmt::Display, process::exit};

mod correlation;
mod data;
mod fundamentalmatrix;
mod orb;
mod output;
mod pointmatching;
mod reconstruction;
mod triangulation;

#[derive(Debug)]
pub enum HardwareMode {
    Gpu,
    GpuLowPower,
    Cpu,
}

#[derive(Debug)]
pub enum InterpolationMode {
    Delaunay,
    None,
}

#[derive(Debug)]
pub enum ProjectionMode {
    Parallel,
    Perspective,
}

#[derive(Debug)]
pub enum Mesh {
    Plain,
    VertexColors,
    TextureCoordinates,
}

#[derive(Debug)]
pub struct Args {
    scale: f32,
    focal_length: Option<u32>,
    mode: HardwareMode,
    interpolation: InterpolationMode,
    no_bundle_adjustment: bool,
    projection: ProjectionMode,
    mesh: Mesh,
    img_src: Vec<String>,
    img_out: String,
}

const USAGE_INSTRUCTIONS: &str = "Usage: cybervision [OPTIONS] <IMG_SRC>... <IMG_OUT>\n\n\
Arguments:\
\n  <IMG_SRC>...  Source image(s)\
\n  <IMG_OUT>     Output image\n\n\
Options:\
\n      --scale=<SCALE>                  Depth scale [default: -1]\
\n      --focal-length=<FOCAL_LENGTH>    Focal length in 35mm equivalent\
\n      --mode=<MODE>                    Hardware mode [default: gpu] [possible values: gpu, gpu-low-power, cpu]\
\n      --interpolation=<INTERPOLATION>  Interpolation mode [default: delaunay] [possible values: delaunay, none]\
\n      --no-bundle-adjustment           Skip bundle adjustment [if unspecified, bundle adjustment will be applied]\
\n      --projection=<PROJECTION>        Projection mode [default: perspective] [possible values: parallel, perspective]\
\n      --mesh=<MESH>                    Mesh options [default: vertex-colors] [possible values: plain, vertex-colors, texture-coordinates]\
\n      --help                           Print help";
impl Args {
    fn parse() -> Args {
        let mut args = Args {
            scale: -1.0,
            focal_length: None,
            mode: HardwareMode::Gpu,
            interpolation: InterpolationMode::Delaunay,
            no_bundle_adjustment: false,
            projection: ProjectionMode::Perspective,
            mesh: Mesh::VertexColors,
            img_src: vec![],
            img_out: "".to_string(),
        };
        let fail_with_error = |name: &str, value: &str, err: &dyn Display| {
            eprintln!(
                "Argument {} has an unsupported value {}: {}",
                name, value, err
            );
            println!("{}", USAGE_INSTRUCTIONS);
            exit(2)
        };
        let mut filenames = vec![];
        for arg in env::args().skip(1) {
            if arg.starts_with("--") && filenames.is_empty() {
                // Option flags.
                if arg == "--no-bundle-adjustment" {
                    args.no_bundle_adjustment = true;
                    continue;
                }
                if arg == "--help" {
                    println!("{}", USAGE_INSTRUCTIONS);
                    exit(0);
                }
                let (name, value) = if let Some(arg) = arg.split_once('=') {
                    arg
                } else {
                    eprintln!("Option flag {} has no value", arg);
                    println!("{}", USAGE_INSTRUCTIONS);
                    exit(2);
                };
                if name == "--scale" {
                    match value.parse() {
                        Ok(scale) => args.scale = scale,
                        Err(err) => fail_with_error(name, value, &err),
                    };
                } else if name == "--focal-length" {
                    match value.parse() {
                        Ok(focal_length) => args.focal_length = Some(focal_length),
                        Err(err) => fail_with_error(name, value, &err),
                    };
                } else if name == "--mode" {
                    match value {
                        "gpu" => args.mode = HardwareMode::Gpu,
                        "gpu-low-power" => args.mode = HardwareMode::GpuLowPower,
                        "cpu" => args.mode = HardwareMode::Cpu,
                        _ => {
                            eprintln!("Unsupported hardware mode {}", value);
                            println!("{}", USAGE_INSTRUCTIONS);
                            exit(2);
                        }
                    };
                } else if name == "--interpolation" {
                    match value {
                        "delaunay" => args.interpolation = InterpolationMode::Delaunay,
                        "none" => args.interpolation = InterpolationMode::None,
                        _ => {
                            eprintln!("Unsupported interpolation {}", value);
                            println!("{}", USAGE_INSTRUCTIONS);
                            exit(2);
                        }
                    };
                } else if name == "--projection" {
                    match value {
                        "perspective" => args.projection = ProjectionMode::Perspective,
                        "parallel" => args.projection = ProjectionMode::Parallel,
                        _ => {
                            eprintln!("Unsupported projection {}", value);
                            println!("{}", USAGE_INSTRUCTIONS);
                            exit(2);
                        }
                    };
                } else if name == "--mesh" {
                    match value {
                        "plain" => args.mesh = Mesh::Plain,
                        "vertex-colors" => args.mesh = Mesh::VertexColors,
                        "texture-coordinates" => args.mesh = Mesh::TextureCoordinates,
                        _ => {
                            eprintln!("Unsupported mesh vertex output mode {}", value);
                            println!("{}", USAGE_INSTRUCTIONS);
                            exit(2);
                        }
                    };
                } else {
                    eprintln!("Unsupported argument {}", arg);
                }
            } else {
                filenames.push(arg);
            }
        }

        args.img_out = if let Some(img_out) = filenames.pop() {
            img_out
        } else {
            eprintln!("No filenames provided");
            println!("{}", USAGE_INSTRUCTIONS);
            exit(2);
        };
        if filenames.len() < 2 {
            eprintln!("Not enough source images (need at least 2 to create a stereopair), but only {} were specified: {:?}", filenames.len(), filenames);
            println!("{}", USAGE_INSTRUCTIONS);
            exit(2);
        }
        args.img_src = filenames;

        args
    }
}

fn main() {
    println!(
        "Cybervision version {}",
        option_env!("CARGO_PKG_VERSION").unwrap_or("unknown")
    );
    let args = Args::parse();

    if let Err(err) = reconstruction::reconstruct(&args) {
        println!("Reconstruction failed, root cause is {}", err);
        exit(1);
    };
}
