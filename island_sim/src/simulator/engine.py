"""Main simulation engine — uses Rust backend if available, falls back to Python."""

from __future__ import annotations

import numpy as np

from ..types import Settlement, WorldState
from .params import SimParams

NUM_YEARS = 50

try:
    import island_sim_core as _rust
    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False


def simulate(
    initial_state: WorldState,
    params: SimParams,
    sim_seed: int,
    years: int = NUM_YEARS,
) -> WorldState:
    """Run the full simulation for `years` time steps."""
    if _HAS_RUST and years == NUM_YEARS:
        return _simulate_rust(initial_state, params, sim_seed)
    return _simulate_python(initial_state, params, sim_seed, years)


def _simulate_rust(
    initial_state: WorldState,
    params: SimParams,
    sim_seed: int,
) -> WorldState:
    """Run simulation using the Rust backend."""
    params_array = params.to_array()
    result_grid, result_settlements = _rust.simulate(
        initial_state.grid, initial_state.settlements, params_array, sim_seed
    )

    # Convert settlement dicts back to Settlement objects
    settlements = []
    for sdata in result_settlements:
        settlements.append(Settlement(
            x=sdata["x"],
            y=sdata["y"],
            population=sdata["population"],
            food=sdata["food"],
            wealth=sdata["wealth"],
            defense=sdata["defense"],
            tech_level=sdata["tech_level"],
            has_port=sdata["has_port"],
            has_longship=sdata["has_longship"],
            owner_id=sdata["owner_id"],
            alive=sdata["alive"],
        ))

    return WorldState(
        grid=result_grid,
        settlements=settlements,
        width=initial_state.width,
        height=initial_state.height,
    )


def _simulate_python(
    initial_state: WorldState,
    params: SimParams,
    sim_seed: int,
    years: int = NUM_YEARS,
) -> WorldState:
    """Pure Python fallback."""
    from .phases.conflict import run_conflict
    from .phases.environment import run_environment
    from .phases.growth import run_growth
    from .phases.trade import run_trade
    from .phases.winter import run_winter

    state = initial_state.copy()
    rng = np.random.default_rng(sim_seed)

    for _ in range(years):
        run_growth(state, params, rng)
        run_conflict(state, params, rng)
        run_trade(state, params, rng)
        run_winter(state, params, rng)
        run_environment(state, params, rng)

    return state
