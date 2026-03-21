use rand::Rng;
use crate::params::SimParams;
use crate::types::*;
use crate::world::*;


const MAX_ALIVE_SETTLEMENTS: usize = 150;

pub fn run_growth<R: Rng>(state: &mut WorldState, params: &SimParams, rng: &mut R) {
    let alive_indices: Vec<usize> = state.settlements.iter()
        .enumerate()
        .filter(|(_, s)| s.alive)
        .map(|(i, _)| i)
        .collect();

    for &idx in &alive_indices {
        produce_and_consume_food(state, idx, params);
        produce_wealth(state, idx, params);
        grow_population(state, idx, params);
        develop_port(state, idx, params);
        build_longship(state, idx, params);
    }
    // Expansion in separate pass (adds new settlements)
    for &idx in &alive_indices {
        try_expand(state, idx, params, rng);
    }
}

fn compute_food_production(state: &WorldState, idx: usize, params: &SimParams) -> f64 {
    let s = &state.settlements[idx];
    let counts = get_adjacent_terrain_counts(state, s.x, s.y);
    let mut production = params.base_food_production;
    production += *counts.get(&FOREST).unwrap_or(&0) as f64 * params.forest_food_bonus;
    production += *counts.get(&PLAINS).unwrap_or(&0) as f64 * params.plains_food_bonus;
    production
}

fn produce_and_consume_food(state: &mut WorldState, idx: usize, params: &SimParams) {
    let production = compute_food_production(state, idx, params);
    let s = &mut state.settlements[idx];
    let consumption = s.population * params.food_consumption_rate;
    s.food += production - consumption;
    if s.food < 0.0 { s.food = 0.0; }
}

fn produce_wealth(state: &mut WorldState, idx: usize, params: &SimParams) {
    let s = &mut state.settlements[idx];
    if s.food > params.pop_growth_food_threshold {
        s.wealth += params.wealth_production_rate * s.population;
    }
}

fn grow_population(state: &mut WorldState, idx: usize, params: &SimParams) {
    let food_production = compute_food_production(state, idx, params);
    let carrying_cap = (food_production * params.carrying_capacity_per_food).max(0.5);

    let s = &mut state.settlements[idx];
    if s.food >= params.pop_growth_food_threshold {
        let growth_factor = (1.0 - s.population / carrying_cap).max(0.0);
        s.population += params.pop_growth_rate * s.population * growth_factor;
    } else if s.food <= 0.0 {
        s.population *= 0.9;
    }
    s.population = s.population.max(0.1);
    s.defense += 0.02 * s.population;
    if s.defense > s.population { s.defense = s.population; }
}

fn develop_port(state: &mut WorldState, idx: usize, params: &SimParams) {
    let s = &state.settlements[idx];
    if s.has_port { return; }
    if s.population >= params.port_development_threshold && is_coastal(state, s.x, s.y) {
        let x = s.x;
        let y = s.y;
        state.settlements[idx].has_port = true;
        state.grid[y as usize][x as usize] = PORT;
    }
}

fn build_longship(state: &mut WorldState, idx: usize, params: &SimParams) {
    let s = &state.settlements[idx];
    if s.has_longship { return; }
    if s.has_port && s.wealth >= params.longship_build_threshold {
        state.settlements[idx].has_longship = true;
        state.settlements[idx].wealth -= params.longship_build_threshold * 0.5;
    }
}

fn try_expand<R: Rng>(state: &mut WorldState, idx: usize, params: &SimParams, rng: &mut R) {
    if state.settlements[idx].population < params.expansion_threshold {
        return;
    }

    let alive_count = state.settlements.iter().filter(|s| s.alive).count();
    if alive_count >= MAX_ALIVE_SETTLEMENTS {
        return;
    }

    let sx = state.settlements[idx].x;
    let sy = state.settlements[idx].y;
    let radius = params.expansion_range as i32;
    let candidates = find_buildable_cells(state, sx, sy, radius);
    if candidates.is_empty() { return; }

    let choice = rng.gen_range(0..candidates.len());
    let (nx, ny) = candidates[choice];

    let transfer_pop = state.settlements[idx].population * params.expansion_pop_transfer;
    state.settlements[idx].population -= transfer_pop;

    let parent_has_port = state.settlements[idx].has_port;
    let parent_tech = state.settlements[idx].tech_level;
    let owner = state.settlements[idx].owner_id;
    let is_port = is_coastal(state, nx, ny) && parent_has_port;

    place_settlement(
        state, nx, ny, owner,
        transfer_pop * 0.7,
        0.3,   // food
        0.0,   // wealth
        0.2,   // defense
        parent_tech * 0.3,
        is_port,
    );
}
