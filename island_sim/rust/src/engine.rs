/// Main simulation engine — runs 50 years of the 5-phase lifecycle.

use std::collections::HashSet;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::cmaes_inference::ViewportSettlementStats;
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

/// Run Monte Carlo simulations and produce both class probabilities and
/// per-viewport settlement statistics averaged across runs.
///
/// `viewports` is a slice of (x, y, w, h) tuples defining viewport bounds.
/// Returns (flat_probs, per_viewport_stats).
pub fn generate_prediction_with_stats(
    initial_state: &WorldState,
    params: &SimParams,
    num_runs: usize,
    base_seed: u64,
    viewports: &[(usize, usize, usize, usize)],
) -> (Vec<f64>, Vec<ViewportSettlementStats>) {
    let height = initial_state.height as usize;
    let width = initial_state.width as usize;
    let num_viewports = viewports.len();
    let mut counts = vec![0.0_f64; height * width * NUM_CLASSES];

    // Accumulators for per-viewport stats across runs
    let mut vp_num_settlements = vec![0.0_f64; num_viewports];
    let mut vp_num_ports = vec![0.0_f64; num_viewports];
    let mut vp_sum_population = vec![0.0_f64; num_viewports];
    let mut vp_sum_food = vec![0.0_f64; num_viewports];
    let mut vp_sum_wealth = vec![0.0_f64; num_viewports];
    let mut vp_num_owners: Vec<Vec<HashSet<i32>>> = (0..num_viewports)
        .map(|_| Vec::new())
        .collect();

    for run_idx in 0..num_runs {
        let sim_seed = base_seed + run_idx as u64;
        let final_state = simulate(initial_state, params, sim_seed);

        // Accumulate terrain class counts
        for y in 0..height {
            for x in 0..width {
                let terrain = final_state.grid[y][x];
                let class_idx = terrain_to_class(terrain);
                counts[(y * width + x) * NUM_CLASSES + class_idx] += 1.0;
            }
        }

        // Accumulate per-viewport settlement stats
        for (vp_idx, &(vp_x, vp_y, vp_w, vp_h)) in viewports.iter().enumerate() {
            let gx_end = (vp_x + vp_w).min(width);
            let gy_end = (vp_y + vp_h).min(height);

            let mut run_count = 0.0_f64;
            let mut run_ports = 0.0_f64;
            let mut run_pop = 0.0_f64;
            let mut run_food = 0.0_f64;
            let mut run_wealth = 0.0_f64;
            let mut run_owners = HashSet::new();

            for settlement in &final_state.settlements {
                if !settlement.alive {
                    continue;
                }
                let sx = settlement.x as usize;
                let sy = settlement.y as usize;
                if sx >= vp_x && sx < gx_end && sy >= vp_y && sy < gy_end {
                    run_count += 1.0;
                    if settlement.has_port {
                        run_ports += 1.0;
                    }
                    run_pop += settlement.population;
                    run_food += settlement.food;
                    run_wealth += settlement.wealth;
                    run_owners.insert(settlement.owner_id);
                }
            }

            vp_num_settlements[vp_idx] += run_count;
            vp_num_ports[vp_idx] += run_ports;
            // For means: accumulate sums, divide by count later
            vp_sum_population[vp_idx] += if run_count > 0.0 { run_pop / run_count } else { 0.0 };
            vp_sum_food[vp_idx] += if run_count > 0.0 { run_food / run_count } else { 0.0 };
            vp_sum_wealth[vp_idx] += if run_count > 0.0 { run_wealth / run_count } else { 0.0 };
            vp_num_owners[vp_idx].push(run_owners);
        }
    }

    // Convert terrain counts to probabilities
    let divisor = num_runs as f64;
    for val in counts.iter_mut() {
        *val /= divisor;
    }

    // Average stats across runs
    let stats: Vec<ViewportSettlementStats> = (0..num_viewports)
        .map(|vp_idx| {
            let avg_num_owners: f64 = vp_num_owners[vp_idx]
                .iter()
                .map(|owners| owners.len() as f64)
                .sum::<f64>() / divisor;

            ViewportSettlementStats {
                num_settlements: vp_num_settlements[vp_idx] / divisor,
                num_ports: vp_num_ports[vp_idx] / divisor,
                mean_population: vp_sum_population[vp_idx] / divisor,
                mean_food: vp_sum_food[vp_idx] / divisor,
                mean_wealth: vp_sum_wealth[vp_idx] / divisor,
                num_owners: avg_num_owners,
            }
        })
        .collect();

    (counts, stats)
}
