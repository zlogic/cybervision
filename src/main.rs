use clap::Parser;

#[derive(clap::ValueEnum, Clone)]
enum HardwareMode {
    GPU,
    CPU,
}

#[derive(clap::ValueEnum, Clone)]
enum InterpolationMode {
    Delaunay,
    None,
}

#[derive(clap::ValueEnum, Clone)]
enum ProjectionMode {
    Parallel,
    Perspective,
}

/// Cybervision commandline arguments
#[derive(Parser)]
struct Cli {
    /// Image scale
    #[arg(long, default_value_t = 1.0)]
    scale: f32,

    /// Hardware mode
    #[arg(long, value_enum, default_value_t = HardwareMode::GPU)]
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

    /// Image 3
    img3: String,
}

fn main() {
    let args = Cli::parse();

    println!("Hello {}!", args.scale)
}
