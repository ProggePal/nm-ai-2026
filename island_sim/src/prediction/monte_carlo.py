"""Monte Carlo prediction — uses Rust backend if available."""

from __future__ import annotations

import numpy as np

from ..constants import NUM_CLASSES, TERRAIN_TO_CLASS
from ..simulator.params import SimParams
from ..types import WorldState

try:
    import island_sim_core as _rust
    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False


def generate_prediction(
    initial_state: WorldState,
    params: SimParams,
    num_runs: int = 500,
    base_seed: int = 0,
) -> np.ndarray:
    """Run Monte Carlo simulations and produce a (H, W, 6) probability tensor."""
    if _HAS_RUST:
        return _generate_rust(initial_state, params, num_runs, base_seed)
    return _generate_python(initial_state, params, num_runs, base_seed)


def _generate_rust(
    initial_state: WorldState,
    params: SimParams,
    num_runs: int,
    base_seed: int,
) -> np.ndarray:
    """Use Rust backend for Monte Carlo."""
    params_array = params.to_array()
    return _rust.generate_prediction(
        initial_state.grid,
        initial_state.settlements,
        params_array,
        num_runs,
        base_seed,
    )


def _generate_python(
    initial_state: WorldState,
    params: SimParams,
    num_runs: int,
    base_seed: int,
) -> np.ndarray:
    """Pure Python fallback."""
    from ..simulator.engine import simulate

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

    return class_counts / num_runs
