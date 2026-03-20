"""Loss function for parameter inference — NLL of observed terrain under model."""

from __future__ import annotations

import numpy as np

from ..constants import TERRAIN_TO_CLASS, NUM_CLASSES
from ..prediction.monte_carlo import generate_prediction
from ..simulator.params import SimParams
from ..types import Observation, WorldState

# Vectorized terrain-to-class lookup table (max terrain code is 11)
_TERRAIN_CLASS_LUT = np.zeros(12, dtype=np.int32)
for _terrain, _cls in TERRAIN_TO_CLASS.items():
    _TERRAIN_CLASS_LUT[_terrain] = _cls


def compute_loss(
    observations: list[Observation],
    initial_states: list[WorldState],
    params: SimParams,
    num_runs: int = 20,
    base_seed: int = 42,
) -> float:
    """Compute negative log-likelihood of observations under the model.

    Uses generate_prediction (Rust-accelerated) to build empirical distributions,
    then scores observed terrain against those distributions via vectorized numpy.
    """
    total_nll = 0.0
    total_cells = 0

    # Group observations by seed to avoid redundant MC runs
    obs_by_seed: dict[int, list[Observation]] = {}
    for obs in observations:
        obs_by_seed.setdefault(obs.seed_index, []).append(obs)

    for seed_idx, seed_obs in obs_by_seed.items():
        state = initial_states[seed_idx]

        # Run MC once per seed (Rust-accelerated)
        pred = generate_prediction(
            state, params, num_runs=num_runs,
            base_seed=base_seed + seed_idx * 10000,
        )

        # Apply Laplace smoothing
        smoothed = (pred * num_runs + 0.5) / (num_runs + 0.5 * NUM_CLASSES)

        for obs in seed_obs:
            # Extract the viewport region from the full prediction
            gy_start = obs.viewport_y
            gx_start = obs.viewport_x
            gy_end = min(gy_start + obs.viewport_h, state.height)
            gx_end = min(gx_start + obs.viewport_w, state.width)
            vh = gy_end - gy_start
            vw = gx_end - gx_start

            # Get observed terrain classes (vectorized via LUT)
            obs_terrain = obs.grid[:vh, :vw].clip(0, 11)
            obs_classes = _TERRAIN_CLASS_LUT[obs_terrain]  # shape (vh, vw)

            # Get predicted probabilities for the observed classes
            pred_region = smoothed[gy_start:gy_end, gx_start:gx_end]  # (vh, vw, 6)
            # Advanced indexing: pick the probability of the observed class per cell
            y_idx, x_idx = np.mgrid[:vh, :vw]
            probs = pred_region[y_idx, x_idx, obs_classes]  # (vh, vw)

            # NLL
            probs = np.maximum(probs, 1e-10)
            total_nll -= np.sum(np.log(probs))
            total_cells += vh * vw

    avg_nll = total_nll / max(total_cells, 1)
    return avg_nll
