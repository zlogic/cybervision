use std::process::exit;

use clap::Parser;
mod crosscorrelation;
mod fundamentalmatrix;
mod orb;
mod output;
mod pointmatching;
mod reconstruction;
mod triangulation;

#[derive(clap::ValueEnum, Clone)]
pub enum HardwareMode {
    Gpu,
    GpuLowPower,
    Cpu,
}

#[derive(clap::ValueEnum, Clone)]
pub enum InterpolationMode {
    Delaunay,
    None,
}

#[derive(clap::ValueEnum, Clone)]
pub enum ProjectionMode {
    Parallel,
    Perspective,
}

#[derive(clap::ValueEnum, Clone)]
pub enum Mesh {
    Plain,
    VertexColors,
    TextureCoordinates,
}

/// Cybervision commandline arguments
#[derive(Parser)]
#[command(author, version, about, long_about = None)]
pub struct Cli {
    /// Depth scale
    #[arg(long, default_value_t = -1.0)]
    scale: f32,

    /// Focal length in 35mm equivalent
    #[arg(long)]
    focal_length: Option<u32>,

    /// Hardware mode
    #[arg(long, value_enum, default_value_t = HardwareMode::Gpu)]
    mode: HardwareMode,

    /// Interpolation mode
    #[arg(long, value_enum, default_value_t = InterpolationMode::Delaunay)]
    interpolation: InterpolationMode,

    /// Bundle adjustment
    #[arg(long, default_value_t = false)]
    no_bundle_adjustment: bool,

    /// Hardware mode
    #[arg(long, value_enum, default_value_t = ProjectionMode::Parallel)]
    projection: ProjectionMode,

    /// Mesh options
    #[arg(long, value_enum, default_value_t = Mesh::Plain)]
    mesh: Mesh,

    /// Source image(s)
    #[arg(required = true, index = 1)]
    img_src: Vec<String>,

    /// Output image
    #[arg(required = true, index = 2)]
    img_out: String,
}

fn main() {
    let args = Cli::parse();

    if let Err(err) = reconstruction::reconstruct(&args) {
        println!("Reconstruction failed, root cause is {}", err);
        exit(-1);
    };
}
