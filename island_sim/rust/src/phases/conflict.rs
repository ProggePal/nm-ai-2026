use rand::Rng;
use rand::seq::SliceRandom;
use crate::params::SimParams;
use crate::types::*;
use crate::world::*;
use std::collections::HashMap;

pub fn run_conflict<R: Rng>(state: &mut WorldState, params: &SimParams, rng: &mut R) {
    // Reset per-turn tracking
    state.war_pairs.clear();
    for s in state.settlements.iter_mut() {
        if s.alive {
            s.raid_damage_taken = 0.0;
        }
    }

    let alive_indices: Vec<usize> = state.settlements.iter()
        .enumerate()
        .filter(|(_, s)| s.alive)
        .map(|(i, _)| i)
        .collect();

    // Build spatial lookup: (x, y) -> settlement index
    let mut cell_map: HashMap<(i32, i32), usize> = HashMap::new();
    for &idx in &alive_indices {
        let s = &state.settlements[idx];
        cell_map.insert((s.x, s.y), idx);
    }

    // Shuffle order
    let mut order = alive_indices.clone();
    order.shuffle(rng);

    for attacker_idx in order {
        if !state.settlements[attacker_idx].alive { continue; }
        try_raid(state, attacker_idx, params, rng, &mut cell_map);
    }
}

fn try_raid<R: Rng>(
    state: &mut WorldState,
    attacker_idx: usize,
    params: &SimParams,
    rng: &mut R,
    cell_map: &mut HashMap<(i32, i32), usize>,
) {
    let attacker = &state.settlements[attacker_idx];
    let raid_range = if attacker.has_longship { params.raid_range_longship } else { params.raid_range_base };
    let raid_range_int = raid_range as i32;
    let ax = attacker.x;
    let ay = attacker.y;
    let attacker_owner = attacker.owner_id;
    let attacker_food = attacker.food;

    // Find targets
    let mut targets: Vec<usize> = Vec::new();
    for dy in -raid_range_int..=raid_range_int {
        for dx in -raid_range_int..=raid_range_int {
            if dx == 0 && dy == 0 { continue; }
            let nx = ax + dx;
            let ny = ay + dy;
            if let Some(&target_idx) = cell_map.get(&(nx, ny)) {
                let target = &state.settlements[target_idx];
                if target.alive && target.owner_id != attacker_owner {
                    targets.push(target_idx);
                }
            }
        }
    }

    if targets.is_empty() { return; }

    // Desperate settlements always raid; others roll against raid_probability
    let is_desperate = attacker_food < params.raid_desperation_threshold;
    if !is_desperate && rng.gen::<f64>() >= params.raid_probability {
        return;
    }

    // Sort by defense (pick weakest)
    targets.sort_by(|&a, &b| {
        state.settlements[a].defense.partial_cmp(&state.settlements[b].defense).unwrap()
    });
    let target_idx = targets[0];

    // Record war between these factions
    let target_owner = state.settlements[target_idx].owner_id;
    let war_key = (attacker_owner.min(target_owner), attacker_owner.max(target_owner));
    state.war_pairs.insert(war_key);

    let desperation_bonus = if is_desperate { 1.5 } else { 1.0 };
    let attack_strength = state.settlements[attacker_idx].population * params.raid_strength_factor * desperation_bonus;

    let target = &state.settlements[target_idx];
    if attack_strength <= target.defense * target.population {
        // Raid fails
        let dmg = 0.1 * params.raid_damage_factor * state.settlements[attacker_idx].population;
        state.settlements[attacker_idx].population -= dmg;
        state.settlements[attacker_idx].population = state.settlements[attacker_idx].population.max(0.1);
        return;
    }

    // Raid succeeds — loot
    let loot_food = state.settlements[target_idx].food * params.raid_loot_fraction;
    let loot_wealth = state.settlements[target_idx].wealth * params.raid_loot_fraction;
    state.settlements[target_idx].food -= loot_food;
    state.settlements[target_idx].wealth -= loot_wealth;
    state.settlements[attacker_idx].food += loot_food;
    state.settlements[attacker_idx].wealth += loot_wealth;

    // Damage defender and track accumulated raid damage
    let pop_dmg = params.raid_damage_factor * state.settlements[target_idx].population;
    let def_dmg = params.raid_damage_factor * state.settlements[target_idx].defense;
    state.settlements[target_idx].population -= pop_dmg;
    state.settlements[target_idx].defense -= def_dmg;
    state.settlements[target_idx].population = state.settlements[target_idx].population.max(0.1);
    state.settlements[target_idx].defense = state.settlements[target_idx].defense.max(0.0);
    state.settlements[target_idx].raid_damage_taken += pop_dmg;

    // Conquest or destruction
    if state.settlements[target_idx].population < params.raid_kill_threshold {
        let tx = state.settlements[target_idx].x;
        let ty = state.settlements[target_idx].y;
        kill_settlement(state, target_idx);
        cell_map.remove(&(tx, ty));
    } else if state.settlements[target_idx].population < params.conquest_threshold {
        let owner = state.settlements[attacker_idx].owner_id;
        state.settlements[target_idx].owner_id = owner;
        state.settlements[target_idx].defense = 0.2;
    }
}
