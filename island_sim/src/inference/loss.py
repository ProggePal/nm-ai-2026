"""Loss function for parameter inference — NLL of observed terrain under model."""

from __future__ import annotations

import numpy as np

from ..constants import TERRAIN_TO_CLASS, NUM_CLASSES
from ..prediction.monte_carlo import generate_prediction
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

    Uses generate_prediction (Rust-accelerated) to build empirical distributions,
    then scores observed terrain against those distributions.
    """
    total_nll = 0.0
    total_cells = 0

    # Group observations by seed to avoid redundant MC runs
    obs_by_seed: dict[int, list[Observation]] = {}
    for obs in observations:
        obs_by_seed.setdefault(obs.seed_index, []).append(obs)

    # Cache predictions per seed
    pred_cache: dict[int, np.ndarray] = {}

    for seed_idx, seed_obs in obs_by_seed.items():
        state = initial_states[seed_idx]

        # Run MC once per seed (Rust-accelerated), not per observation
        if seed_idx not in pred_cache:
            pred_cache[seed_idx] = generate_prediction(
                state, params, num_runs=num_runs,
                base_seed=base_seed + seed_idx * 10000,
            )

        pred = pred_cache[seed_idx]  # shape (H, W, 6)

        # Apply Laplace smoothing
        smoothed = (pred * num_runs + 0.5) / (num_runs + 0.5 * NUM_CLASSES)

        for obs in seed_obs:
            # Score each observed cell against the model's prediction
            for vy in range(obs.viewport_h):
                for vx in range(obs.viewport_w):
                    gx = obs.viewport_x + vx
                    gy = obs.viewport_y + vy
                    if 0 <= gx < state.width and 0 <= gy < state.height:
                        observed_terrain = int(obs.grid[vy, vx])
                        observed_class = TERRAIN_TO_CLASS.get(observed_terrain, 0)
                        prob = smoothed[gy, gx, observed_class]
                        total_nll -= np.log(max(prob, 1e-10))
                        total_cells += 1

    avg_nll = total_nll / max(total_cells, 1)
    return avg_nll
