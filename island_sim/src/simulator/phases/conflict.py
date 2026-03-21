"""Conflict phase: raiding, looting, conquest, destruction."""

from __future__ import annotations

import numpy as np

from ...types import Settlement, WorldState
from ..params import SimParams
from ..world import kill_settlement


def run_conflict(
    state: WorldState, params: SimParams, rng: np.random.Generator
) -> None:
    """Execute the conflict phase — raids between factions."""
    # Reset per-turn tracking
    state.war_pairs.clear()
    for settlement in state.settlements:
        if settlement.alive:
            settlement.raid_damage_taken = 0.0

    alive = [s for s in state.settlements if s.alive]

    # Build spatial lookup for O(1) position checks
    alive_by_cell: dict[tuple[int, int], Settlement] = {
        (s.x, s.y): s for s in alive
    }

    # Shuffle to avoid order bias
    order = list(range(len(alive)))
    rng.shuffle(order)

    for idx in order:
        attacker = alive[idx]
        if not attacker.alive:
            continue
        _try_raid(attacker, state, params, rng, alive_by_cell)


def _try_raid(
    attacker: Settlement,
    state: WorldState,
    params: SimParams,
    rng: np.random.Generator,
    alive_by_cell: dict[tuple[int, int], Settlement],
) -> None:
    """Attempt to raid nearby enemy settlements."""
    raid_range = params.raid_range_base
    if attacker.has_longship:
        raid_range = params.raid_range_longship
    raid_range_int = int(raid_range)

    # Find potential targets using spatial bounds
    targets = []
    for dy in range(-raid_range_int, raid_range_int + 1):
        for dx in range(-raid_range_int, raid_range_int + 1):
            if dx == 0 and dy == 0:
                continue
            nx, ny = attacker.x + dx, attacker.y + dy
            target = alive_by_cell.get((nx, ny))
            if target is None or not target.alive:
                continue
            if target.owner_id == attacker.owner_id:
                continue
            targets.append(target)

    if not targets:
        return

    # Desperate settlements always raid; others roll against raid_probability
    is_desperate = attacker.food < params.raid_desperation_threshold
    if not is_desperate and rng.random() >= params.raid_probability:
        return
    desperation_bonus = 1.5 if is_desperate else 1.0

    # Pick a target (prefer weakest)
    targets.sort(key=lambda s: s.defense)
    target = targets[0]

    # Record war between these factions
    pair = (min(attacker.owner_id, target.owner_id), max(attacker.owner_id, target.owner_id))
    state.war_pairs.add(pair)

    # Compute attack strength
    attack_strength = (
        attacker.population * params.raid_strength_factor * desperation_bonus
    )

    # Compare to defense
    if attack_strength <= target.defense * target.population:
        # Raid fails — attacker takes some damage
        attacker.population -= 0.1 * params.raid_damage_factor * attacker.population
        attacker.population = max(attacker.population, 0.1)
        return

    # Raid succeeds — loot resources
    loot_food = target.food * params.raid_loot_fraction
    loot_wealth = target.wealth * params.raid_loot_fraction
    target.food -= loot_food
    target.wealth -= loot_wealth
    attacker.food += loot_food
    attacker.wealth += loot_wealth

    # Damage the defender and track accumulated raid damage
    pop_damage = params.raid_damage_factor * target.population
    target.population -= pop_damage
    target.defense -= params.raid_damage_factor * target.defense
    target.population = max(target.population, 0.1)
    target.defense = max(target.defense, 0.0)
    target.raid_damage_taken += pop_damage

    # Check for conquest or destruction
    if target.population < params.raid_kill_threshold:
        # Settlement destroyed by raid — becomes ruin
        kill_settlement(state, target)
        alive_by_cell.pop((target.x, target.y), None)
    elif target.population < params.conquest_threshold:
        _conquer(attacker, target)


def _conquer(attacker: Settlement, target: Settlement) -> None:
    """Conquered settlement changes allegiance to attacker's faction."""
    target.owner_id = attacker.owner_id
    target.defense = 0.2
