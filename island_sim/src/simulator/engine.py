"""Main simulation engine — runs 50 years of the 5-phase lifecycle."""

from __future__ import annotations

import numpy as np

from ..types import WorldState
from .params import SimParams
from .phases.conflict import run_conflict
from .phases.environment import run_environment
from .phases.growth import run_growth
from .phases.trade import run_trade
from .phases.winter import run_winter

NUM_YEARS = 50
MAX_ALIVE_SETTLEMENTS = 200  # Safety cap to prevent runaway expansion


def simulate(
    initial_state: WorldState,
    params: SimParams,
    sim_seed: int,
    years: int = NUM_YEARS,
) -> WorldState:
    """Run the full simulation for `years` time steps.

    Args:
        initial_state: The starting world state (will be deep-copied).
        params: Simulation parameters.
        sim_seed: Random seed for reproducibility.
        years: Number of simulation years to run (default 50).

    Returns:
        The final world state after simulation.
    """
    state = initial_state.copy()
    rng = np.random.default_rng(sim_seed)

    for _ in range(years):
        run_growth(state, params, rng)
        run_conflict(state, params, rng)
        run_trade(state, params, rng)
        run_winter(state, params, rng)
        run_environment(state, params, rng)

    return state
