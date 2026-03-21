"""Growth phase: food production, population growth, port development, expansion."""

from __future__ import annotations

import numpy as np

from ...constants import FOREST, PLAINS, PORT
from ...types import Settlement, WorldState
from ..params import SimParams
from ..world import (
    find_buildable_cells,
    get_adjacent_terrain_counts,
    is_coastal,
    place_settlement,
)

MAX_ALIVE_SETTLEMENTS = 150  # Safety cap


def run_growth(state: WorldState, params: SimParams, rng: np.random.Generator) -> None:
    """Execute the growth phase for all alive settlements."""
    # Snapshot the alive settlements — new settlements founded this turn don't act
    alive = [s for s in state.settlements if s.alive]

    for settlement in alive:
        _produce_and_consume_food(settlement, state, params)
        _produce_wealth(settlement, params)
        _grow_population(settlement, state, params)
        _develop_port(settlement, state, params)
        _build_longship(settlement, params)
        _try_expand(settlement, state, params, rng)


def _compute_food_production(
    settlement: Settlement, state: WorldState, params: SimParams
) -> float:
    """Calculate raw food production from adjacent terrain."""
    counts = get_adjacent_terrain_counts(state, settlement.x, settlement.y)
    production = params.base_food_production
    production += counts.get(FOREST, 0) * params.forest_food_bonus
    production += counts.get(PLAINS, 0) * params.plains_food_bonus
    return production


def _produce_and_consume_food(
    settlement: Settlement, state: WorldState, params: SimParams
) -> None:
    """Produce food from terrain, consume based on population."""
    production = _compute_food_production(settlement, state, params)
    consumption = settlement.population * params.food_consumption_rate
    settlement.food += production - consumption
    settlement.food = max(settlement.food, 0.0)


def _produce_wealth(settlement: Settlement, params: SimParams) -> None:
    """Prosperous settlements generate wealth from surplus food."""
    if settlement.food > params.pop_growth_food_threshold:
        settlement.wealth += params.wealth_production_rate * settlement.population


def _grow_population(
    settlement: Settlement, state: WorldState, params: SimParams
) -> None:
    """Grow population if food surplus, shrink if starving. Logistic growth with carrying capacity."""
    # Carrying capacity based on local food production potential
    food_production = _compute_food_production(settlement, state, params)
    carrying_cap = food_production * params.carrying_capacity_per_food
    carrying_cap = max(carrying_cap, 0.5)  # Even barren land supports a tiny population

    if settlement.food >= params.pop_growth_food_threshold:
        # Logistic growth: slows as pop approaches carrying capacity
        growth_factor = 1.0 - (settlement.population / carrying_cap)
        growth_factor = max(growth_factor, 0.0)
        settlement.population += params.pop_growth_rate * settlement.population * growth_factor
    elif settlement.food <= 0:
        # Starvation: population declines
        settlement.population *= 0.9

    settlement.population = max(settlement.population, 0.1)

    # Defense grows slowly with population
    settlement.defense += 0.02 * settlement.population
    settlement.defense = min(settlement.defense, settlement.population)


def _develop_port(
    settlement: Settlement, state: WorldState, params: SimParams
) -> None:
    """Develop port status if coastal and population threshold met."""
    if settlement.has_port:
        return
    if settlement.population >= params.port_development_threshold and is_coastal(
        state, settlement.x, settlement.y
    ):
        settlement.has_port = True
        state.grid[settlement.y, settlement.x] = PORT


def _build_longship(settlement: Settlement, params: SimParams) -> None:
    """Build a longship if port and wealth/tech threshold met."""
    if settlement.has_longship:
        return
    if settlement.has_port and settlement.wealth >= params.longship_build_threshold:
        settlement.has_longship = True
        settlement.wealth -= params.longship_build_threshold * 0.5


def _try_expand(
    settlement: Settlement,
    state: WorldState,
    params: SimParams,
    rng: np.random.Generator,
) -> None:
    """Found a new settlement if population exceeds expansion threshold."""
    if settlement.population < params.expansion_threshold:
        return

    # Safety cap
    alive_count = sum(1 for s in state.settlements if s.alive)
    if alive_count >= MAX_ALIVE_SETTLEMENTS:
        return

    radius = int(params.expansion_range)
    candidates = find_buildable_cells(state, settlement.x, settlement.y, radius)
    if not candidates:
        return

    # Pick a random buildable cell
    idx = rng.integers(0, len(candidates))
    nx, ny = candidates[idx]

    # Transfer population — parent loses more than child gets (cost of expansion)
    transfer_pop = settlement.population * params.expansion_pop_transfer
    settlement.population -= transfer_pop

    # New settlement starts small
    is_port = is_coastal(state, nx, ny) and settlement.has_port
    place_settlement(
        state,
        nx,
        ny,
        owner_id=settlement.owner_id,
        population=transfer_pop * 0.7,  # Some people lost in transit
        food=0.3,
        wealth=0.0,
        defense=0.2,
        tech_level=settlement.tech_level * 0.3,
        has_port=is_port,
    )
