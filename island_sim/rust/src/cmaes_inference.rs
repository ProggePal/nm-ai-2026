/// CMA-ES parameter inference — NLL loss against observed terrain.
///
/// This is DIFFERENT from grid search scoring:
/// - Grid search: generate_prediction → postprocess → entropy-weighted KL divergence
/// - CMA-ES: generate_prediction → Laplace smoothing (no postprocess) → NLL
///
/// Must match `src/inference/loss.py` exactly.

use cmaes::{CMAESOptions, DVector};

use crate::engine;
use crate::params::{SimParams, bounds_lower, bounds_upper, default_params};
use crate::types::{WorldState, NUM_CLASSES};

/// Per-viewport settlement statistics observed from the API.
pub struct ViewportSettlementStats {
    pub num_settlements: f64,
    pub num_ports: f64,
    pub mean_population: f64,
    pub mean_food: f64,
    pub mean_wealth: f64,
    pub num_owners: f64,
}

/// A viewport observation — terrain codes observed at a specific location.
pub struct ObservationData {
    pub viewport_x: usize,
    pub viewport_y: usize,
    pub viewport_w: usize,
    pub viewport_h: usize,
    /// Pre-computed class indices for each cell, shape viewport_h * viewport_w (row-major).
    pub terrain_classes: Vec<usize>,
    /// Optional settlement stats from the API observation.
    pub observed_stats: Option<ViewportSettlementStats>,
}

/// Groups a seed's initial state with all observations for that seed.
pub struct SeedState {
    pub initial_state: WorldState,
    pub observations: Vec<ObservationData>,
    /// Original seed index (from Python), used for deterministic base_seed computation.
    pub seed_index: usize,
}

/// Fixed base seed for MC runs (matches Python compute_loss default).
const MC_BASE_SEED: u64 = 42;

/// Compute normalized MSE between observed and simulated settlement stats.
/// Each dimension is divided by a normalization constant to put them on similar scales.
fn compute_stat_mse(obs: &ViewportSettlementStats, sim: &ViewportSettlementStats) -> f64 {
    let diffs = [
        (obs.num_settlements - sim.num_settlements) / 30.0,
        (obs.num_ports - sim.num_ports) / 5.0,
        (obs.mean_population - sim.mean_population) / 2.0,
        (obs.mean_food - sim.mean_food) / 0.5,
        (obs.mean_wealth - sim.mean_wealth) / 2.0,
        (obs.num_owners - sim.num_owners) / 5.0,
    ];
    diffs.iter().map(|d| d * d).sum::<f64>() / diffs.len() as f64
}

/// Compute NLL + stat loss for one seed: run MC prediction with stats,
/// apply Laplace smoothing, score observations against the predicted distribution.
///
/// Returns (total_nll, total_cells, total_stat_loss, num_stat_viewports).
fn evaluate_seed_nll(
    initial_state: &WorldState,
    params: &SimParams,
    observations: &[ObservationData],
    mc_runs: usize,
    base_seed: u64,
) -> (f64, usize, f64, usize) {
    let height = initial_state.height as usize;
    let width = initial_state.width as usize;

    // Collect viewports for stat computation
    let viewports: Vec<(usize, usize, usize, usize)> = observations
        .iter()
        .map(|obs| (obs.viewport_x, obs.viewport_y, obs.viewport_w, obs.viewport_h))
        .collect();

    // Run MC prediction with settlement stats
    let (prediction, sim_stats) = engine::generate_prediction_with_stats(
        initial_state, params, mc_runs, base_seed, &viewports,
    );

    // Laplace smoothing: (prob * mc_runs + 0.5) / (mc_runs + 0.5 * NUM_CLASSES)
    let mc_runs_f = mc_runs as f64;
    let num_classes_f = NUM_CLASSES as f64;
    let smoothing_denom = mc_runs_f + 0.5 * num_classes_f;
    let mut smoothed = prediction;
    for val in smoothed.iter_mut() {
        *val = (*val * mc_runs_f + 0.5) / smoothing_denom;
    }

    let mut total_nll = 0.0;
    let mut total_cells: usize = 0;
    let mut total_stat_loss = 0.0;
    let mut num_stat_viewports: usize = 0;

    for (idx, obs) in observations.iter().enumerate() {
        let gy_start = obs.viewport_y;
        let gx_start = obs.viewport_x;
        let gy_end = (gy_start + obs.viewport_h).min(height);
        let gx_end = (gx_start + obs.viewport_w).min(width);
        let vh = gy_end - gy_start;
        let vw = gx_end - gx_start;

        for vy in 0..vh {
            for vx in 0..vw {
                let obs_class = obs.terrain_classes[vy * obs.viewport_w + vx];
                let grid_y = gy_start + vy;
                let grid_x = gx_start + vx;
                let prob = smoothed[(grid_y * width + grid_x) * NUM_CLASSES + obs_class];
                let prob = prob.max(1e-10);
                total_nll -= prob.ln();
            }
        }
        total_cells += vh * vw;

        // Stat MSE for viewports with observed stats
        if let Some(ref obs_stats) = obs.observed_stats {
            total_stat_loss += compute_stat_mse(obs_stats, &sim_stats[idx]);
            num_stat_viewports += 1;
        }
    }

    (total_nll, total_cells, total_stat_loss, num_stat_viewports)
}

/// Run CMA-ES inference to find best-fit parameters.
///
/// Uses parallel population evaluation via rayon (through the cmaes crate).
/// Each individual evaluation runs MC simulations for all seeds sequentially.
///
/// Combined objective: avg_nll + stat_weight * avg_stat_loss
///
/// Returns (best_params_vec, best_loss).
pub fn run_cmaes_inference(
    seeds: &[SeedState],
    mc_runs: usize,
    max_evals: usize,
    population_size: Option<usize>,
    sigma0: f64,
    warm_start: Option<&[f64]>,
    cmaes_seed: u64,
    stat_weight: f64,
) -> (Vec<f64>, f64) {
    let lower = bounds_lower();
    let upper = bounds_upper();
    let ndim = lower.len();

    // Normalize warm-start (or defaults) to [0, 1] space
    let defaults = default_params();
    let start = warm_start.unwrap_or(&defaults);
    let x0: Vec<f64> = (0..ndim)
        .map(|idx| {
            let range = upper[idx] - lower[idx];
            if range > 0.0 {
                ((start[idx] - lower[idx]) / range).clamp(0.0, 1.0)
            } else {
                0.5
            }
        })
        .collect();

    // Clone bounds for the closure (original copies used after for denormalization)
    let lower_cl = lower.clone();
    let upper_cl = upper.clone();

    // Objective function: denormalize, evaluate NLL + stat loss across all seeds
    let objective = move |x: &DVector<f64>| -> f64 {
        // Denormalize and clamp to bounds (soft boundary handling)
        let params_vec: Vec<f64> = (0..ndim)
            .map(|idx| {
                let xi = x[idx].clamp(0.0, 1.0);
                lower_cl[idx] + xi * (upper_cl[idx] - lower_cl[idx])
            })
            .collect();
        let params = SimParams::from_array(&params_vec);

        let mut total_nll = 0.0;
        let mut total_cells: usize = 0;
        let mut total_stat_loss = 0.0;
        let mut total_stat_viewports: usize = 0;

        for seed_state in seeds.iter() {
            let base_seed = MC_BASE_SEED + seed_state.seed_index as u64 * 10000;
            let (nll, cells, stat_loss, stat_vps) = evaluate_seed_nll(
                &seed_state.initial_state,
                &params,
                &seed_state.observations,
                mc_runs,
                base_seed,
            );
            total_nll += nll;
            total_cells += cells;
            total_stat_loss += stat_loss;
            total_stat_viewports += stat_vps;
        }

        let avg_nll = total_nll / total_cells.max(1) as f64;
        let avg_stat_loss = if total_stat_viewports > 0 {
            total_stat_loss / total_stat_viewports as f64
        } else {
            0.0
        };

        avg_nll + stat_weight * avg_stat_loss
    };

    // Build CMA-ES optimizer
    let mut opts = CMAESOptions::new(x0, sigma0)
        .max_function_evals(max_evals)
        .seed(cmaes_seed);

    if let Some(pop_size) = population_size {
        opts = opts.population_size(pop_size);
    }

    let mut cmaes_state = opts.build(objective)
        .unwrap_or_else(|err| panic!("CMA-ES build failed: {:?}", err));
    let result = cmaes_state.run_parallel();

    // Extract best solution and denormalize back to parameter space
    let best = result.overall_best.expect("CMA-ES produced no solution");

    let best_params: Vec<f64> = (0..ndim)
        .map(|idx| {
            let xi = best.point[idx].clamp(0.0, 1.0);
            lower[idx] + xi * (upper[idx] - lower[idx])
        })
        .collect();

    eprintln!(
        "CMA-ES finished: {} evals, best loss = {:.6}",
        cmaes_state.function_evals(),
        best.value,
    );

    (best_params, best.value)
}

