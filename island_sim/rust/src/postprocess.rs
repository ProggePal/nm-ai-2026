/// Post-processing: static cell overrides, probability floor, normalization.
/// Matches Python src/prediction/postprocess.py.

use crate::types::*;

const PROB_FLOOR: f64 = 0.01;
const STATIC_CONFIDENCE: f64 = 0.98;

/// Apply postprocessing to a flat prediction array (H*W*6).
/// Returns a new processed array.
pub fn postprocess(prediction: &[f64], initial_state: &WorldState, height: usize, width: usize) -> Vec<f64> {
    let mut result = prediction.to_vec();

    // Static cell overrides
    for y in 0..height {
        for x in 0..width {
            let terrain = initial_state.grid[y][x];
            let base = (y * width + x) * NUM_CLASSES;

            if terrain == OCEAN {
                make_static_distribution(&mut result[base..base + NUM_CLASSES], 0); // CLASS_EMPTY
            } else if terrain == MOUNTAIN {
                make_static_distribution(&mut result[base..base + NUM_CLASSES], 5); // CLASS_MOUNTAIN
            }
        }
    }

    // Apply probability floor
    for val in result.iter_mut() {
        if *val < PROB_FLOOR {
            *val = PROB_FLOOR;
        }
    }

    // Renormalize each cell to sum to 1.0
    for y in 0..height {
        for x in 0..width {
            let base = (y * width + x) * NUM_CLASSES;
            let sum: f64 = result[base..base + NUM_CLASSES].iter().sum();
            if sum > 1e-10 {
                for c in 0..NUM_CLASSES {
                    result[base + c] /= sum;
                }
            }
        }
    }

    result
}

fn make_static_distribution(dist: &mut [f64], dominant_class: usize) {
    let other = (1.0 - STATIC_CONFIDENCE) / (NUM_CLASSES - 1) as f64;
    for c in 0..NUM_CLASSES {
        dist[c] = if c == dominant_class { STATIC_CONFIDENCE } else { other };
    }
}
