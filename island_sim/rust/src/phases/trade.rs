use crate::params::SimParams;
use crate::types::*;
use crate::world::distance;

pub fn run_trade(state: &mut WorldState, params: &SimParams) {
    // Find all alive port indices
    let port_indices: Vec<usize> = state.settlements.iter()
        .enumerate()
        .filter(|(_, s)| s.alive && s.has_port)
        .map(|(i, _)| i)
        .collect();

    for i in 0..port_indices.len() {
        for j in (i + 1)..port_indices.len() {
            let idx_a = port_indices[i];
            let idx_b = port_indices[j];

            let dist = distance(
                state.settlements[idx_a].x, state.settlements[idx_a].y,
                state.settlements[idx_b].x, state.settlements[idx_b].y,
            );
            if dist > params.trade_range { continue; }

            // Trade
            state.settlements[idx_a].food += params.trade_food_gain;
            state.settlements[idx_b].food += params.trade_food_gain;
            state.settlements[idx_a].wealth += params.trade_wealth_gain;
            state.settlements[idx_b].wealth += params.trade_wealth_gain;

            // Tech diffusion
            let tech_a = state.settlements[idx_a].tech_level;
            let tech_b = state.settlements[idx_b].tech_level;
            if tech_a > tech_b {
                state.settlements[idx_b].tech_level += (tech_a - tech_b) * params.tech_diffusion_rate;
            } else if tech_b > tech_a {
                state.settlements[idx_a].tech_level += (tech_b - tech_a) * params.tech_diffusion_rate;
            }
        }
    }
}
