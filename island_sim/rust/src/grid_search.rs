/// Fully Rust grid search with Rayon parallelism.
/// Eliminates Python multiprocessing overhead entirely.

use rayon::prelude::*;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand::Rng;

use crate::engine;
use crate::params::SimParams;
use crate::scoring::score_prediction;
use crate::types::*;
use crate::postprocess;

/// A seed's ground truth data needed for evaluation.
pub struct SeedData {
    pub initial_state: WorldState,
    pub ground_truth: Vec<f64>, // flat H*W*6
    pub height: usize,
    pub width: usize,
}

/// Result from evaluating one candidate.
#[derive(Clone)]
pub struct CandidateResult {
    pub params: Vec<f64>,
    pub mean_score: f64,
    pub per_round_scores: Vec<f64>, // one per round (average of seeds in that round)
}

/// Run the full grid search.
///
/// - `seeds`: all seed data grouped by round (Vec of rounds, each a Vec of seeds)
/// - `num_candidates`: how many random param sets to try
/// - `mc_runs`: Monte Carlo runs per seed per candidate
/// - `top_n`: how many top results to return
///
/// Returns the top N candidates sorted by mean score (best first).
pub fn run_grid_search(
    rounds: &[Vec<SeedData>],
    num_candidates: usize,
    mc_runs: usize,
    top_n: usize,
    seed: u64,
) -> Vec<CandidateResult> {
    let lower = bounds_lower();
    let upper = bounds_upper();
    let ndim = lower.len();

    // Generate all candidates upfront
    let mut candidates: Vec<Vec<f64>> = Vec::with_capacity(num_candidates);

    // Candidate 0: default params
    candidates.push(SimParams::from_array(&default_params()).to_vec());

    // Latin Hypercube Sampling for the rest
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let lhs_count = num_candidates - 1;

    // Build LHS matrix
    let mut lhs = vec![vec![0.0_f64; ndim]; lhs_count];
    for dim in 0..ndim {
        let mut perm: Vec<usize> = (0..lhs_count).collect();
        // Fisher-Yates shuffle
        for idx in (1..lhs_count).rev() {
            let j = rng.gen_range(0..=idx);
            perm.swap(idx, j);
        }
        for idx in 0..lhs_count {
            let lo = perm[idx] as f64 / lhs_count as f64;
            let hi = (perm[idx] + 1) as f64 / lhs_count as f64;
            lhs[idx][dim] = lo + rng.gen::<f64>() * (hi - lo);
        }
    }

    // Scale LHS to parameter bounds
    for idx in 0..lhs_count {
        let mut params = vec![0.0; ndim];
        for dim in 0..ndim {
            params[dim] = lower[dim] + lhs[idx][dim] * (upper[dim] - lower[dim]);
        }
        candidates.push(params);
    }

    // Also add perturbations of default
    let default = default_params();
    let num_local = (lhs_count / 4).min(num_candidates / 5);
    for _ in 0..num_local {
        let mut params = vec![0.0; ndim];
        for dim in 0..ndim {
            let noise: f64 = rng.gen::<f64>() * 2.0 - 1.0; // [-1, 1]
            params[dim] = (default[dim] + noise * 0.1 * (upper[dim] - lower[dim]))
                .max(lower[dim])
                .min(upper[dim]);
        }
        candidates.push(params);
    }

    let total = candidates.len();
    eprintln!("Generated {} candidates, evaluating with {} MC runs across {} rounds ({} seeds)...",
        total, mc_runs, rounds.len(),
        rounds.iter().map(|r| r.len()).sum::<usize>());

    // Evaluate all candidates in parallel using Rayon
    let results: Vec<CandidateResult> = candidates
        .par_iter()
        .enumerate()
        .map(|(idx, param_vec)| {
            let params = SimParams::from_array(param_vec);
            let mut all_scores = Vec::new();
            let mut per_round = Vec::new();

            for (round_idx, round_seeds) in rounds.iter().enumerate() {
                let mut round_scores = Vec::new();

                for seed_data in round_seeds {
                    let score = evaluate_single_seed(
                        &seed_data.initial_state,
                        &params,
                        &seed_data.ground_truth,
                        seed_data.height,
                        seed_data.width,
                        mc_runs,
                        (round_idx * 100000 + idx) as u64,
                    );
                    round_scores.push(score);
                    all_scores.push(score);
                }

                per_round.push(
                    round_scores.iter().sum::<f64>() / round_scores.len() as f64
                );
            }

            let mean = all_scores.iter().sum::<f64>() / all_scores.len() as f64;

            // Progress logging (every 100 candidates)
            if idx % 100 == 0 && idx > 0 {
                eprintln!("  [{}/{}] mean={:.1}", idx, total, mean);
            }

            CandidateResult {
                params: param_vec.clone(),
                mean_score: mean,
                per_round_scores: per_round,
            }
        })
        .collect();

    // Sort by mean score descending
    let mut sorted = results;
    sorted.sort_by(|a, b| b.mean_score.partial_cmp(&a.mean_score).unwrap());
    sorted.truncate(top_n);
    sorted
}

/// Evaluate one candidate against one seed.
fn evaluate_single_seed(
    initial_state: &WorldState,
    params: &SimParams,
    ground_truth: &[f64],
    height: usize,
    width: usize,
    mc_runs: usize,
    base_seed: u64,
) -> f64 {
    // Run MC prediction
    let prediction = engine::generate_prediction(initial_state, params, mc_runs, base_seed);

    // Apply postprocessing (floor + static overrides + normalize)
    let processed = postprocess::postprocess(&prediction, initial_state, height, width);

    // Score against ground truth
    score_prediction(ground_truth, &processed, height, width)
}

// --- Parameter bounds (must match Python SimParams exactly) ---

fn default_params() -> Vec<f64> {
    vec![
        0.4, 0.25, 0.08, 0.2, 0.15, 0.6, 6.0, 2.5, 1.5, 2.0, 3.0, 0.35,  // growth
        2.0, 5.0, 0.3, 0.5, 0.3, 0.25, 0.3, 0.15,                           // conflict
        4.0, 0.15, 0.1, 0.1,                                                   // trade
        0.5, 0.3, 0.4, 0.4, 0.3, 3.0, 0.5,                                   // winter (+survival_bonus)
        0.1, 4.0, 2.5, 0.2, 0.05,                                              // environment
        1.5, 0.3,                                                               // collapse triggers
        0.5,                                                                     // raid probability
        0.05,                                                                    // wealth production
    ]
}

fn bounds_lower() -> Vec<f64> {
    vec![
        0.05, 0.02, 0.0, 0.1, 0.01, 0.2, 1.0, 1.0, 0.5, 2.0, 1.0, 0.1,
        1.0, 2.0, 0.05, 0.1, 0.05, 0.05, 0.05, 0.05,
        1.0, 0.02, 0.02, 0.01,
        0.1, 0.01, 0.05, 0.05, 0.0, 1.0, 0.1,
        0.01, 1.0, 1.0, 0.05, 0.01,
        1.0, 0.1,                                                               // collapse triggers
        0.1,                                                                     // raid probability
        0.01,                                                                    // wealth production
    ]
}

fn bounds_upper() -> Vec<f64> {
    vec![
        1.5, 1.0, 0.5, 1.5, 0.3, 3.0, 10.0, 8.0, 5.0, 12.0, 6.0, 0.6,
        5.0, 10.0, 1.0, 1.5, 0.7, 0.6, 0.7, 0.5,
        8.0, 0.6, 0.5, 0.4,
        2.0, 0.8, 1.5, 0.9, 1.0, 6.0, 0.8,
        0.5, 6.0, 8.0, 0.5, 0.3,
        3.0, 0.8,                                                               // collapse triggers
        1.0,                                                                     // raid probability
        0.3,                                                                     // wealth production
    ]
}

impl SimParams {
    fn to_vec(&self) -> Vec<f64> {
        vec![
            self.base_food_production, self.forest_food_bonus, self.plains_food_bonus,
            self.food_consumption_rate, self.pop_growth_rate, self.pop_growth_food_threshold,
            self.carrying_capacity_per_food, self.port_development_threshold,
            self.longship_build_threshold, self.expansion_threshold, self.expansion_range,
            self.expansion_pop_transfer,
            self.raid_range_base, self.raid_range_longship, self.raid_desperation_threshold,
            self.raid_strength_factor, self.raid_loot_fraction, self.raid_damage_factor,
            self.conquest_threshold, self.raid_kill_threshold,
            self.trade_range, self.trade_food_gain, self.trade_wealth_gain,
            self.tech_diffusion_rate,
            self.winter_severity_mean, self.winter_severity_variance,
            self.collapse_food_threshold, self.collapse_probability,
            self.survival_bonus,
            self.dispersal_range, self.dispersal_fraction,
            self.forest_reclaim_probability, self.ruin_rebuild_range,
            self.ruin_rebuild_threshold, self.ruin_rebuild_fraction,
            self.ruin_to_plains_probability,
            self.harsh_winter_collapse_factor, self.raid_collapse_threshold,
            self.raid_probability, self.wealth_production_rate,
        ]
    }
}
