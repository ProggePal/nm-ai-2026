/// Post-processing: smart probability floor + static overrides.
/// Matches Python src/prediction/postprocess.py.

use crate::types::*;

// Two-tier probability floor: near settlements vs remote
const PROB_FLOOR: f64 = 0.005;
const PROB_FLOOR_REMOTE: f64 = 0.001;
const SETTLEMENT_PROXIMITY: i32 = 6;

const CLASS_EMPTY: usize = 0;
const CLASS_SETTLEMENT: usize = 1;
const CLASS_PORT: usize = 2;
const CLASS_RUIN: usize = 3;
const CLASS_FOREST: usize = 4;
const CLASS_MOUNTAIN: usize = 5;

/// Check if cell (x, y) is adjacent to ocean.
fn is_coastal(grid: &[Vec<i32>], x: usize, y: usize, height: usize, width: usize) -> bool {
    for dy in -1i32..=1 {
        for dx in -1i32..=1 {
            let ny = y as i32 + dy;
            let nx = x as i32 + dx;
            if ny >= 0 && ny < height as i32 && nx >= 0 && nx < width as i32 {
                if grid[ny as usize][nx as usize] == OCEAN {
                    return true;
                }
            }
        }
    }
    false
}

/// Apply postprocessing to a flat prediction array (H*W*6).
pub fn postprocess(prediction: &[f64], initial_state: &WorldState, height: usize, width: usize) -> Vec<f64> {
    let mut result = prediction.to_vec();

    // Precompute near-settlement mask
    let mut near_settlement = vec![vec![false; width]; height];
    for s in &initial_state.settlements {
        if !s.alive { continue; }
        for dy in -SETTLEMENT_PROXIMITY..=SETTLEMENT_PROXIMITY {
            for dx in -SETTLEMENT_PROXIMITY..=SETTLEMENT_PROXIMITY {
                let ny = s.y + dy;
                let nx = s.x + dx;
                if ny >= 0 && ny < height as i32 && nx >= 0 && nx < width as i32 {
                    near_settlement[ny as usize][nx as usize] = true;
                }
            }
        }
    }

    for y in 0..height {
        for x in 0..width {
            let terrain = initial_state.grid[y][x];
            let base = (y * width + x) * NUM_CLASSES;

            if terrain == OCEAN {
                make_exact_static(&mut result[base..base + NUM_CLASSES], CLASS_EMPTY);
                continue;
            }
            if terrain == MOUNTAIN {
                make_exact_static(&mut result[base..base + NUM_CLASSES], CLASS_MOUNTAIN);
                continue;
            }

            // Dynamic cell: zero out impossible classes
            result[base + CLASS_MOUNTAIN] = 0.0;

            let coastal = is_coastal(&initial_state.grid, x, y, height, width);
            if !coastal {
                result[base + CLASS_PORT] = 0.0;
            }

            let is_near = near_settlement[y][x];

            // Apply floor to plausible classes
            for cls in 0..NUM_CLASSES {
                if cls == CLASS_MOUNTAIN { continue; }
                if cls == CLASS_PORT && !coastal { continue; }

                // Settlement, Port, Ruin use remote floor when far from settlements
                let floor = if (cls == CLASS_SETTLEMENT || cls == CLASS_PORT || cls == CLASS_RUIN) && !is_near {
                    PROB_FLOOR_REMOTE
                } else {
                    PROB_FLOOR
                };

                if result[base + cls] < floor {
                    result[base + cls] = floor;
                }
            }

            // Normalize
            let sum: f64 = result[base..base + NUM_CLASSES].iter().sum();
            if sum > 1e-10 {
                for cls in 0..NUM_CLASSES {
                    result[base + cls] /= sum;
                }
            }
        }
    }

    result
}

fn make_exact_static(dist: &mut [f64], dominant_class: usize) {
    for c in 0..NUM_CLASSES {
        dist[c] = if c == dominant_class { 1.0 } else { 0.0 };
    }
}
