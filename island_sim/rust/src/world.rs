/// Grid utility functions.

use crate::types::*;
use std::collections::HashMap;

/// Chebyshev distance between two points.
pub fn distance(x1: i32, y1: i32, x2: i32, y2: i32) -> f64 {
    (x1 - x2).abs().max((y1 - y2).abs()) as f64
}

/// Check if position is adjacent to ocean.
pub fn is_coastal(state: &WorldState, x: i32, y: i32) -> bool {
    for dy in -1..=1 {
        for dx in -1..=1 {
            if dx == 0 && dy == 0 { continue; }
            let nx = x + dx;
            let ny = y + dy;
            if nx >= 0 && nx < state.width && ny >= 0 && ny < state.height {
                if state.grid[ny as usize][nx as usize] == OCEAN {
                    return true;
                }
            }
        }
    }
    false
}

/// Count terrain types in the 8 cells adjacent to (x, y).
pub fn get_adjacent_terrain_counts(state: &WorldState, x: i32, y: i32) -> HashMap<i32, i32> {
    let mut counts = HashMap::new();
    for dy in -1..=1 {
        for dx in -1..=1 {
            if dx == 0 && dy == 0 { continue; }
            let nx = x + dx;
            let ny = y + dy;
            if nx >= 0 && nx < state.width && ny >= 0 && ny < state.height {
                let terrain = state.grid[ny as usize][nx as usize];
                *counts.entry(terrain).or_insert(0) += 1;
            }
        }
    }
    counts
}

/// Find all buildable cells within Chebyshev radius that don't have a settlement.
pub fn find_buildable_cells(state: &WorldState, x: i32, y: i32, radius: i32) -> Vec<(i32, i32)> {
    let mut result = Vec::new();
    for dy in -radius..=radius {
        for dx in -radius..=radius {
            if dx == 0 && dy == 0 { continue; }
            let nx = x + dx;
            let ny = y + dy;
            if nx < 0 || nx >= state.width || ny < 0 || ny >= state.height { continue; }
            let terrain = state.grid[ny as usize][nx as usize];
            if !is_buildable(terrain) { continue; }
            // Check no alive settlement at this position
            let occupied = state.settlements.iter().any(|s| s.alive && s.x == nx && s.y == ny);
            if !occupied {
                result.push((nx, ny));
            }
        }
    }
    result
}

/// Place a new settlement at (x, y) and update the grid.
pub fn place_settlement(
    state: &mut WorldState,
    x: i32, y: i32,
    owner_id: i32,
    population: f64,
    food: f64,
    wealth: f64,
    defense: f64,
    tech_level: f64,
    has_port: bool,
) {
    let settlement = Settlement {
        x, y, population, food, wealth, defense, tech_level,
        has_port, has_longship: false, owner_id, alive: true,
    };
    state.settlements.push(settlement);
    state.grid[y as usize][x as usize] = if has_port { PORT } else { SETTLEMENT };
}

/// Kill a settlement and convert its cell to ruin.
pub fn kill_settlement(state: &mut WorldState, idx: usize) {
    let s = &mut state.settlements[idx];
    s.alive = false;
    state.grid[s.y as usize][s.x as usize] = RUIN;
}
