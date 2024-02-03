use nalgebra::Matrix3;
use std::{error, fmt};

use crate::data::Grid;

use super::{CorrelationDirection, ProjectionMode};

pub struct GpuContext {}

impl GpuContext {
    pub fn new(
        _: (usize, usize),
        _: (usize, usize),
        _: ProjectionMode,
        _: Matrix3<f64>,
        _: bool,
    ) -> Result<GpuContext, Box<dyn error::Error>> {
        Err(GpuError::new("Compiled without GPU support").into())
    }

    pub fn get_device_name(&self) -> &'static str {
        "undefined"
    }

    pub fn cross_check_filter(&mut self, _: f32, _: CorrelationDirection) {}

    pub fn complete_process(
        &mut self,
    ) -> Result<Grid<Option<super::Match>>, Box<dyn error::Error>> {
        Err(GpuError::new("Compiled without GPU support").into())
    }

    pub fn correlate_images<PL: super::ProgressListener>(
        &mut self,
        _: &Grid<u8>,
        _: &Grid<u8>,
        _: f32,
        _: bool,
        _: Option<&PL>,
        _: CorrelationDirection,
    ) -> Result<(), Box<dyn error::Error>> {
        Err(GpuError::new("Compiled without GPU support").into())
    }
}

#[derive(Debug)]
pub struct GpuError {
    msg: &'static str,
}

impl GpuError {
    fn new(msg: &'static str) -> GpuError {
        GpuError { msg }
    }
}

impl std::error::Error for GpuError {}

impl fmt::Display for GpuError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.msg)
    }
}
