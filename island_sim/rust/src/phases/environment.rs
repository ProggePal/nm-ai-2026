use rand::Rng;
use crate::params::SimParams;
use crate::types::*;
use crate::world::*;


pub fn run_environment<R: Rng>(state: &mut WorldState, params: &SimParams, rng: &mut R) {
    // Find all ruin positions
    let mut ruin_positions: Vec<(i32, i32)> = Vec::new();
    for y in 0..state.height {
        for x in 0..state.width {
            if state.grid[y as usize][x as usize] == RUIN {
                ruin_positions.push((x, y));
            }
        }
    }

    // Shuffle
    use rand::seq::SliceRandom;
    ruin_positions.shuffle(rng);

    for (x, y) in ruin_positions {
        if state.grid[y as usize][x as usize] != RUIN { continue; }

        if try_rebuild(state, x, y, params, rng) {
            continue;
        }
        try_nature_reclaim(state, x, y, params, rng);
    }
}

fn try_rebuild<R: Rng>(
    state: &mut WorldState, x: i32, y: i32, params: &SimParams, rng: &mut R,
) -> bool {
    // Find nearby thriving settlements
    let thriving: Vec<usize> = state.settlements.iter()
        .enumerate()
        .filter(|(_, s)| {
            s.alive && s.population >= params.ruin_rebuild_threshold
                && distance(x, y, s.x, s.y) <= params.ruin_rebuild_range
        })
        .map(|(i, _)| i)
        .collect();

    if thriving.is_empty() { return false; }

    // Probabilistic rebuild
    let rebuild_prob = (0.1 * thriving.len() as f64).min(0.3);
    if rng.gen::<f64>() > rebuild_prob { return false; }

    // Pick random patron
    let patron_idx = thriving[rng.gen_range(0..thriving.len())];
    let transfer_pop = state.settlements[patron_idx].population * params.ruin_rebuild_fraction;
    let patron_tech = state.settlements[patron_idx].tech_level;
    let patron_has_port = state.settlements[patron_idx].has_port;
    let owner = state.settlements[patron_idx].owner_id;

    state.settlements[patron_idx].population -= transfer_pop;

    let has_port = is_coastal(state, x, y) && patron_has_port;
    place_settlement(
        state, x, y, owner,
        transfer_pop,
        0.3,   // food
        0.0,   // wealth
        0.2,   // defense
        patron_tech * 0.3,
        has_port,
    );
    true
}

fn try_nature_reclaim<R: Rng>(
    state: &mut WorldState, x: i32, y: i32, params: &SimParams, rng: &mut R,
) {
    let counts = get_adjacent_terrain_counts(state, x, y);

    if *counts.get(&FOREST).unwrap_or(&0) > 0 {
        if rng.gen::<f64>() < params.forest_reclaim_probability {
            state.grid[y as usize][x as usize] = FOREST;
            return;
        }
    }

    if rng.gen::<f64>() < params.ruin_to_plains_probability {
        state.grid[y as usize][x as usize] = PLAINS;
    }
}
