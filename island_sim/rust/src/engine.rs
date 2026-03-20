/// Main simulation engine — runs 50 years of the 5-phase lifecycle.

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::params::SimParams;
use crate::types::*;
use crate::phases::{growth, conflict, trade, winter, environment};

const NUM_YEARS: usize = 50;

/// Run one full simulation.
pub fn simulate(initial_state: &WorldState, params: &SimParams, sim_seed: u64) -> WorldState {
    let mut state = initial_state.clone();
    let mut rng = ChaCha8Rng::seed_from_u64(sim_seed);

    for _ in 0..NUM_YEARS {
        growth::run_growth(&mut state, params, &mut rng);
        conflict::run_conflict(&mut state, params, &mut rng);
        trade::run_trade(&mut state, params);
        winter::run_winter(&mut state, params, &mut rng);
        environment::run_environment(&mut state, params, &mut rng);
    }

    state
}

/// Run Monte Carlo simulations and produce a (H, W, 6) class count array.
/// Returns a flat Vec<f64> of length H * W * 6 (row-major: [y][x][class]).
pub fn generate_prediction(
    initial_state: &WorldState,
    params: &SimParams,
    num_runs: usize,
    base_seed: u64,
) -> Vec<f64> {
    let h = initial_state.height as usize;
    let w = initial_state.width as usize;
    let mut counts = vec![0.0_f64; h * w * NUM_CLASSES];

    for run_idx in 0..num_runs {
        let sim_seed = base_seed + run_idx as u64;
        let final_state = simulate(initial_state, params, sim_seed);

        for y in 0..h {
            for x in 0..w {
                let terrain = final_state.grid[y][x];
                let class_idx = terrain_to_class(terrain);
                counts[(y * w + x) * NUM_CLASSES + class_idx] += 1.0;
            }
        }
    }

    // Convert to probabilities
    let divisor = num_runs as f64;
    for val in counts.iter_mut() {
        *val /= divisor;
    }

    counts
}
