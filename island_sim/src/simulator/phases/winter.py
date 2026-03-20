"""Winter phase: food loss, collapse, population dispersal."""

from __future__ import annotations

import numpy as np

from ...types import Settlement, WorldState
from ..params import SimParams
from ..world import distance, kill_settlement


def run_winter(
    state: WorldState, params: SimParams, rng: np.random.Generator
) -> None:
    """Execute the winter phase — food loss and potential collapses."""
    # Sample winter severity for this year
    severity = rng.normal(params.winter_severity_mean, params.winter_severity_variance)
    severity = max(severity, 0.0)

    alive = [s for s in state.settlements if s.alive]

    for settlement in alive:
        _apply_winter(settlement, severity, params)

    # Check for collapses after food loss
    for settlement in list(alive):
        if not settlement.alive:
            continue
        _check_collapse(settlement, state, params, rng)


def _apply_winter(settlement: Settlement, severity: float, params: SimParams) -> None:
    """Apply winter food loss proportional to population and severity."""
    # Larger settlements need more food to survive winter
    food_loss = severity * (1.0 + settlement.population * 0.2)
    settlement.food -= food_loss
    settlement.food = max(settlement.food, 0.0)

    # Population also suffers directly from harsh winters
    if severity > params.winter_severity_mean * 1.5:
        pop_loss = (severity - params.winter_severity_mean) * 0.1
        settlement.population -= pop_loss
        settlement.population = max(settlement.population, 0.1)


def _check_collapse(
    settlement: Settlement,
    state: WorldState,
    params: SimParams,
    rng: np.random.Generator,
) -> None:
    """Check if a settlement collapses from starvation."""
    if settlement.food >= params.collapse_food_threshold:
        return

    # Collapse probability increases the more food-deprived the settlement is
    # Base probability when food == 0, scales down as food approaches threshold
    food_ratio = settlement.food / max(params.collapse_food_threshold, 0.01)
    adjusted_prob = params.collapse_probability * (1.0 - food_ratio)

    # Small settlements are more vulnerable
    if settlement.population < 0.5:
        adjusted_prob *= 1.5

    adjusted_prob = min(adjusted_prob, 0.95)

    if rng.random() < adjusted_prob:
        _collapse(settlement, state, params, rng)


def _collapse(
    settlement: Settlement,
    state: WorldState,
    params: SimParams,
    rng: np.random.Generator,
) -> None:
    """Collapse a settlement into a ruin and disperse population."""
    # Find nearby friendly settlements to receive refugees
    friendly = [
        s
        for s in state.settlements
        if s.alive
        and s is not settlement
        and s.owner_id == settlement.owner_id
        and distance(settlement.x, settlement.y, s.x, s.y) <= params.dispersal_range
    ]

    if friendly:
        dispersed_pop = settlement.population * params.dispersal_fraction
        share = dispersed_pop / len(friendly)
        for target in friendly:
            target.population += share

    # Kill the settlement — turns to ruin
    kill_settlement(state, settlement)
