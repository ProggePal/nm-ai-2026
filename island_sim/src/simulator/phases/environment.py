"""Environment phase: forest reclaim ruins, settlement rebuild ruins."""

from __future__ import annotations

import numpy as np

from ...constants import FOREST, PLAINS, RUIN
from ...types import WorldState
from ..params import SimParams
from ..world import (
    find_settlements_in_range,
    get_adjacent_terrain_counts,
    is_coastal,
    place_settlement,
)


def run_environment(
    state: WorldState, params: SimParams, rng: np.random.Generator
) -> None:
    """Execute the environment phase — ruins reclaimed or rebuilt."""
    ruin_positions = []
    for y in range(state.height):
        for x in range(state.width):
            if state.grid[y, x] == RUIN:
                ruin_positions.append((x, y))

    # Shuffle to avoid order bias
    rng.shuffle(ruin_positions)

    for x, y in ruin_positions:
        if state.grid[y, x] != RUIN:
            continue

        # Check if a nearby thriving settlement can rebuild (probabilistic)
        if _try_rebuild(x, y, state, params, rng):
            continue

        # Otherwise, nature may reclaim (slowly)
        _try_nature_reclaim(x, y, state, params, rng)


def _try_rebuild(
    x: int,
    y: int,
    state: WorldState,
    params: SimParams,
    rng: np.random.Generator,
) -> bool:
    """Try to rebuild a ruin from a nearby thriving settlement."""
    nearby = find_settlements_in_range(
        state, x, y, params.ruin_rebuild_range, alive_only=True
    )

    thriving = [s for s in nearby if s.population >= params.ruin_rebuild_threshold]
    if not thriving:
        return False

    # Probabilistic: not every ruin gets rebuilt every turn
    # Probability scales with number of thriving neighbors
    rebuild_prob = min(0.3, 0.1 * len(thriving))
    if rng.random() > rebuild_prob:
        return False

    # Pick patron randomly from thriving (not always strongest — adds variance)
    patron = thriving[rng.integers(0, len(thriving))]

    transfer_pop = patron.population * params.ruin_rebuild_fraction
    patron.population -= transfer_pop

    has_port = is_coastal(state, x, y) and patron.has_port
    place_settlement(
        state,
        x,
        y,
        owner_id=patron.owner_id,
        population=transfer_pop,
        food=0.3,
        wealth=0.0,
        defense=0.2,
        tech_level=patron.tech_level * 0.3,
        has_port=has_port,
    )
    return True


def _try_nature_reclaim(
    x: int,
    y: int,
    state: WorldState,
    params: SimParams,
    rng: np.random.Generator,
) -> None:
    """Nature reclaims a ruin — either forest grows or reverts to plains."""
    adj = get_adjacent_terrain_counts(state, x, y)

    if adj.get(FOREST, 0) > 0:
        if rng.random() < params.forest_reclaim_probability:
            state.grid[y, x] = FOREST
            return

    if rng.random() < params.ruin_to_plains_probability:
        state.grid[y, x] = PLAINS
