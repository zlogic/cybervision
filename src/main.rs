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

/// Cybervision commandline arguments
#[derive(Parser)]
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

    /// Image 1
    img1: String,

    /// Image 2
    img2: String,

    /// Output image
    img_out: String,
}

fn main() {
    let args = Cli::parse();

    reconstruction::reconstruct(&args);
}
