"""Loss function for parameter inference — NLL of observed terrain under model."""

from __future__ import annotations

import numpy as np

from ..constants import TERRAIN_TO_CLASS, NUM_CLASSES
from ..simulator.engine import simulate
from ..simulator.params import SimParams
from ..types import Observation, WorldState


def compute_loss(
    observations: list[Observation],
    initial_states: list[WorldState],
    params: SimParams,
    num_runs: int = 20,
    base_seed: int = 42,
) -> float:
    """Compute negative log-likelihood of observations under the model.

    For each observation, we run `num_runs` simulations with the candidate params
    and build an empirical distribution of terrain classes per cell. The loss is
    the negative log-likelihood of the observed terrain under this distribution.

    Also includes MSE on settlement stats (population, food, wealth, defense)
    where available.
    """
    total_nll = 0.0
    total_cells = 0
    total_settlement_mse = 0.0
    total_settlement_count = 0

    for obs in observations:
        state = initial_states[obs.seed_index]

        # Run N simulations to build empirical distribution
        class_counts = np.zeros(
            (obs.viewport_h, obs.viewport_w, NUM_CLASSES), dtype=np.float64
        )

        # Collect settlement stats across runs for comparison
        settlement_stats_runs: dict[tuple[int, int], list[dict]] = {}

        for run_idx in range(num_runs):
            sim_seed = base_seed + obs.seed_index * 10000 + run_idx
            final_state = simulate(state, params, sim_seed)

            # Extract viewport region
            for vy in range(obs.viewport_h):
                for vx in range(obs.viewport_w):
                    gx = obs.viewport_x + vx
                    gy = obs.viewport_y + vy
                    if 0 <= gx < final_state.width and 0 <= gy < final_state.height:
                        terrain = int(final_state.grid[gy, gx])
                        class_idx = TERRAIN_TO_CLASS.get(terrain, 0)
                        class_counts[vy, vx, class_idx] += 1

            # Collect settlement stats
            for settlement in final_state.settlements:
                if not settlement.alive:
                    continue
                # Check if within viewport
                sx = settlement.x - obs.viewport_x
                sy = settlement.y - obs.viewport_y
                if 0 <= sx < obs.viewport_w and 0 <= sy < obs.viewport_h:
                    key = (settlement.x, settlement.y)
                    if key not in settlement_stats_runs:
                        settlement_stats_runs[key] = []
                    settlement_stats_runs[key].append({
                        "population": settlement.population,
                        "food": settlement.food,
                        "wealth": settlement.wealth,
                        "defense": settlement.defense,
                    })

        # Convert counts to probabilities with Laplace smoothing
        class_probs = (class_counts + 0.5) / (num_runs + 0.5 * NUM_CLASSES)

        # Compute NLL for observed terrain
        for vy in range(obs.viewport_h):
            for vx in range(obs.viewport_w):
                observed_terrain = int(obs.grid[vy, vx])
                observed_class = TERRAIN_TO_CLASS.get(observed_terrain, 0)
                prob = class_probs[vy, vx, observed_class]
                total_nll -= np.log(max(prob, 1e-10))
                total_cells += 1

        # Compute settlement stats MSE
        for obs_settlement in obs.settlements:
            if not obs_settlement.alive:
                continue
            key = (obs_settlement.x, obs_settlement.y)
            if key in settlement_stats_runs and settlement_stats_runs[key]:
                runs = settlement_stats_runs[key]
                mean_pop = np.mean([r["population"] for r in runs])
                mean_food = np.mean([r["food"] for r in runs])
                mean_wealth = np.mean([r["wealth"] for r in runs])
                mean_defense = np.mean([r["defense"] for r in runs])

                mse = (
                    (obs_settlement.population - mean_pop) ** 2
                    + (obs_settlement.food - mean_food) ** 2
                    + (obs_settlement.wealth - mean_wealth) ** 2
                    + (obs_settlement.defense - mean_defense) ** 2
                ) / 4.0
                total_settlement_mse += mse
                total_settlement_count += 1

    # Combine NLL and settlement MSE
    avg_nll = total_nll / max(total_cells, 1)
    avg_settlement_mse = (
        total_settlement_mse / max(total_settlement_count, 1)
    )

    # Weight settlement MSE relative to NLL
    # NLL is typically 0.5-3.0, MSE can be much larger
    settlement_weight = 0.1
    return avg_nll + settlement_weight * avg_settlement_mse
