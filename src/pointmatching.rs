use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};

type Point = (usize, usize);
type Keypoint = (Point, [u32; 8]);

const THRESHOLD_AFFINE: u32 = 32;
const THRESHOLD_PERSPECTIVE: u32 = 32;
const MAX_MATCHES: usize = 10_000;

#[derive(Debug, Clone, Copy)]
pub enum ProjectionMode {
    Affine,
    Perspective,
}

pub struct KeypointMatching {
    pub matches: Vec<(Point, Point)>,
}

pub trait ProgressListener
where
    Self: Sync + Sized,
{
    fn report_status(&self, pos: f32);
}

impl KeypointMatching {
    pub fn new<PL: ProgressListener>(
        points1: &Vec<Keypoint>,
        points2: &Vec<Keypoint>,
        projection_mode: ProjectionMode,
        progress_listener: Option<&PL>,
    ) -> KeypointMatching {
        let threshold = match projection_mode {
            ProjectionMode::Affine => THRESHOLD_AFFINE,
            ProjectionMode::Perspective => THRESHOLD_PERSPECTIVE,
        };
        let matches =
            KeypointMatching::match_points::<PL>(points1, points2, threshold, progress_listener);
        KeypointMatching { matches }
    }

    fn match_points<PL: ProgressListener>(
        points1: &Vec<Keypoint>,
        points2: &Vec<Keypoint>,
        threshold: u32,
        progress_listener: Option<&PL>,
    ) -> Vec<(Point, Point)> {
        let counter = AtomicUsize::new(0);
        let mut point_matches = points1
            .into_par_iter()
            .flat_map(|p1| {
                if let Some(pl) = progress_listener {
                    let value =
                        counter.fetch_add(1, Ordering::Relaxed) as f32 / points1.len() as f32;
                    pl.report_status(value);
                }
                let brief1 = p1.1;
                points2
                    .iter()
                    .filter_map(|p2| {
                        let brief2 = p2.1;
                        let distance: u32 =
                            (0..8).map(|i| (brief1[i] ^ (brief2[i])).count_ones()).sum();
                        if distance < threshold {
                            Some(((p1.0, p2.0), distance))
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        point_matches.sort_by(|(_, distance1), (_, distance2)| distance1.cmp(distance2));

        point_matches
            .iter()
            .take(MAX_MATCHES)
            .map(|(p, _)| *p)
            .collect()
    }
}
