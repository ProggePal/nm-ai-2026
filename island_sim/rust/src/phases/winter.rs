use rand::Rng;
use rand_distr::{Normal, Distribution};
use crate::params::SimParams;
use crate::types::*;
use crate::world::*;

pub fn run_winter<R: Rng>(state: &mut WorldState, params: &SimParams, rng: &mut R) {
    let normal = Normal::new(params.winter_severity_mean, params.winter_severity_variance.max(0.001)).unwrap();
    let severity = normal.sample(rng).max(0.0);

    let alive_indices: Vec<usize> = state.settlements.iter()
        .enumerate()
        .filter(|(_, s)| s.alive)
        .map(|(i, _)| i)
        .collect();

    // Apply winter to all
    for &idx in &alive_indices {
        apply_winter(state, idx, severity, params);
    }

    // Check collapses
    for &idx in &alive_indices {
        if !state.settlements[idx].alive { continue; }
        check_collapse(state, idx, params, rng, severity);
    }
}

fn apply_winter(state: &mut WorldState, idx: usize, severity: f64, params: &SimParams) {
    let s = &mut state.settlements[idx];
    let food_loss = severity * (1.0 + s.population * 0.2);
    s.food -= food_loss;
    if s.food < 0.0 { s.food = 0.0; }

    if severity > params.harsh_winter_collapse_factor * params.winter_severity_mean {
        let pop_loss = (severity - params.winter_severity_mean) * 0.1;
        s.population -= pop_loss;
        if s.population < 0.1 { s.population = 0.1; }
    }
}

fn check_collapse<R: Rng>(state: &mut WorldState, idx: usize, params: &SimParams, rng: &mut R, severity: f64) {
    let s = &state.settlements[idx];
    let mut collapse_prob = 0.0;

    // Trigger 1: Starvation (original mechanic)
    if s.food < params.collapse_food_threshold {
        let food_ratio = s.food / params.collapse_food_threshold.max(0.01);
        collapse_prob += params.collapse_probability * (1.0 - food_ratio);
    }

    // Trigger 2: Harsh winter — severity exceeds threshold
    let harsh_threshold = params.harsh_winter_collapse_factor * params.winter_severity_mean.max(0.01);
    if severity > harsh_threshold {
        let excess = (severity - harsh_threshold) / params.winter_severity_mean.max(0.01);
        collapse_prob += params.collapse_probability * excess.min(1.0);
    }

    // Trigger 3: Sustained raids — accumulated raid damage this turn
    if s.raid_damage_taken > params.raid_collapse_threshold {
        let excess = (s.raid_damage_taken - params.raid_collapse_threshold) / params.raid_collapse_threshold.max(0.01);
        collapse_prob += params.collapse_probability * excess.min(1.0);
    }

    if collapse_prob <= 0.0 { return; }

    // Small settlements are more vulnerable
    if s.population < 0.5 {
        collapse_prob *= 1.5;
    }
    // Large established settlements resist collapse
    if s.population > 1.0 {
        let survival_factor = 1.0 / (1.0 + (s.population - 1.0) * params.survival_bonus);
        collapse_prob *= survival_factor;
    }
    collapse_prob = collapse_prob.min(0.95);

    if rng.gen::<f64>() < collapse_prob {
        collapse(state, idx, params);
    }
}

fn collapse(state: &mut WorldState, idx: usize, params: &SimParams) {
    let sx = state.settlements[idx].x;
    let sy = state.settlements[idx].y;
    let owner = state.settlements[idx].owner_id;
    let pop = state.settlements[idx].population;

    // Find nearby friendly settlements
    let friendly: Vec<usize> = state.settlements.iter()
        .enumerate()
        .filter(|(i, s)| {
            s.alive && *i != idx && s.owner_id == owner
                && distance(sx, sy, s.x, s.y) <= params.dispersal_range
        })
        .map(|(i, _)| i)
        .collect();

    if !friendly.is_empty() {
        let dispersed = pop * params.dispersal_fraction;
        let share = dispersed / friendly.len() as f64;
        for &fi in &friendly {
            state.settlements[fi].population += share;
        }
    }

    kill_settlement(state, idx);
}
