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
        _check_collapse(settlement, state, params, rng, severity)


def _apply_winter(settlement: Settlement, severity: float, params: SimParams) -> None:
    """Apply winter food loss proportional to population and severity."""
    # Larger settlements need more food to survive winter
    food_loss = severity * (1.0 + settlement.population * 0.2)
    settlement.food -= food_loss
    settlement.food = max(settlement.food, 0.0)

    # Population also suffers directly from harsh winters
    if severity > params.harsh_winter_collapse_factor * params.winter_severity_mean:
        pop_loss = (severity - params.winter_severity_mean) * 0.1
        settlement.population -= pop_loss
        settlement.population = max(settlement.population, 0.1)


def _check_collapse(
    settlement: Settlement,
    state: WorldState,
    params: SimParams,
    rng: np.random.Generator,
    severity: float = 0.0,
) -> None:
    """Check if a settlement collapses from starvation, harsh winter, or sustained raids."""
    collapse_prob = 0.0

    # Trigger 1: Starvation (original mechanic)
    if settlement.food < params.collapse_food_threshold:
        food_ratio = settlement.food / max(params.collapse_food_threshold, 0.01)
        collapse_prob += params.collapse_probability * (1.0 - food_ratio)

    # Trigger 2: Harsh winter — severity exceeds threshold
    harsh_threshold = params.harsh_winter_collapse_factor * max(params.winter_severity_mean, 0.01)
    if severity > harsh_threshold:
        excess = (severity - harsh_threshold) / max(params.winter_severity_mean, 0.01)
        collapse_prob += params.collapse_probability * min(excess, 1.0)

    # Trigger 3: Sustained raids — accumulated raid damage this turn
    if settlement.raid_damage_taken > params.raid_collapse_threshold:
        excess = (settlement.raid_damage_taken - params.raid_collapse_threshold) / max(params.raid_collapse_threshold, 0.01)
        collapse_prob += params.collapse_probability * min(excess, 1.0)

    if collapse_prob <= 0.0:
        return

    # Small settlements are more vulnerable
    if settlement.population < 0.5:
        collapse_prob *= 1.5

    # Large, established settlements resist collapse (survival bonus)
    if settlement.population > 1.0:
        survival_factor = 1.0 / (1.0 + (settlement.population - 1.0) * params.survival_bonus)
        collapse_prob *= survival_factor

    collapse_prob = min(collapse_prob, 0.95)

    if rng.random() < collapse_prob:
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
