"""Monte Carlo prediction — run many sims and accumulate terrain distributions."""

from __future__ import annotations

import numpy as np

from ..constants import NUM_CLASSES, TERRAIN_TO_CLASS
from ..simulator.engine import simulate
from ..simulator.params import SimParams
from ..types import WorldState


def generate_prediction(
    initial_state: WorldState,
    params: SimParams,
    num_runs: int = 500,
    base_seed: int = 0,
) -> np.ndarray:
    """Run Monte Carlo simulations and produce a (H, W, 6) probability tensor.

    Args:
        initial_state: Starting world state for this seed.
        params: Inferred simulation parameters.
        num_runs: Number of simulations to run.
        base_seed: Base random seed (each run uses base_seed + run_idx).

    Returns:
        np.ndarray of shape (height, width, NUM_CLASSES) with probability distributions.
    """
    height = initial_state.height
    width = initial_state.width
    class_counts = np.zeros((height, width, NUM_CLASSES), dtype=np.float64)

    for run_idx in range(num_runs):
        sim_seed = base_seed + run_idx
        final_state = simulate(initial_state, params, sim_seed)

        for y in range(height):
            for x in range(width):
                terrain = int(final_state.grid[y, x])
                class_idx = TERRAIN_TO_CLASS.get(terrain, 0)
                class_counts[y, x, class_idx] += 1

    # Convert counts to probabilities
    prediction = class_counts / num_runs
    return prediction
