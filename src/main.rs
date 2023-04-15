use std::process::exit;

use clap::Parser;
mod correlation;
mod crosscorrelation;
mod fast;
mod fundamentalmatrix;
mod output;
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

    /// Hardware mode
    #[arg(long, value_enum, default_value_t = HardwareMode::Gpu)]
    mode: HardwareMode,

    /// Interpolation mode
    #[arg(long, value_enum, default_value_t = InterpolationMode::Delaunay)]
    interpolation: InterpolationMode,

    /// Hardware mode
    #[arg(long, value_enum, default_value_t = ProjectionMode::Parallel)]
    projection: ProjectionMode,

    #[arg(long, value_enum, default_value_t = Mesh::Plain)]
    mesh: Mesh,

    /// Source images
    img_src: Vec<String>,

    /// Output image
    img_out: String,
}

fn main() {
    let args = Cli::parse();

    if let Err(err) = reconstruction::reconstruct(&args) {
        println!("Reconstruction failed, root cause is {}", err);
        exit(-1);
    };
}
