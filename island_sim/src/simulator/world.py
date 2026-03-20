"""WorldState utility methods for querying and modifying the grid."""

from __future__ import annotations

import numpy as np

from ..constants import (
    BUILDABLE_TERRAINS,
    FOREST,
    MOUNTAIN,
    OCEAN,
    PLAINS,
    PORT,
    RUIN,
    SETTLEMENT,
)
from ..types import Settlement, WorldState


def get_neighbors(
    state: WorldState, x: int, y: int, radius: int = 1
) -> list[tuple[int, int]]:
    """Return all grid coordinates within Chebyshev distance `radius` of (x, y), excluding (x, y) itself."""
    result = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx == 0 and dy == 0:
                continue
            nx, ny = x + dx, y + dy
            if 0 <= nx < state.width and 0 <= ny < state.height:
                result.append((nx, ny))
    return result


def get_adjacent_terrain_counts(state: WorldState, x: int, y: int) -> dict[int, int]:
    """Count terrain types in the 8 cells adjacent to (x, y)."""
    counts: dict[int, int] = {}
    for nx, ny in get_neighbors(state, x, y, radius=1):
        terrain = int(state.grid[ny, nx])
        counts[terrain] = counts.get(terrain, 0) + 1
    return counts


def is_coastal(state: WorldState, x: int, y: int) -> bool:
    """Check if position (x, y) is adjacent to ocean."""
    for nx, ny in get_neighbors(state, x, y, radius=1):
        if state.grid[ny, nx] == OCEAN:
            return True
    return False


def distance(x1: int, y1: int, x2: int, y2: int) -> float:
    """Chebyshev distance between two points."""
    return max(abs(x1 - x2), abs(y1 - y2))


def find_settlements_in_range(
    state: WorldState, x: int, y: int, radius: float, alive_only: bool = True
) -> list[Settlement]:
    """Find all settlements within Chebyshev distance of (x, y)."""
    result = []
    for settlement in state.settlements:
        if alive_only and not settlement.alive:
            continue
        if distance(x, y, settlement.x, settlement.y) <= radius:
            result.append(settlement)
    return result


def get_settlement_at(state: WorldState, x: int, y: int) -> Settlement | None:
    """Get the alive settlement at (x, y), if any."""
    for settlement in state.settlements:
        if settlement.alive and settlement.x == x and settlement.y == y:
            return settlement
    return None


def find_buildable_cells(
    state: WorldState, x: int, y: int, radius: int
) -> list[tuple[int, int]]:
    """Find all cells within radius that can receive a new settlement."""
    result = []
    for nx, ny in get_neighbors(state, x, y, radius):
        terrain = int(state.grid[ny, nx])
        if terrain in BUILDABLE_TERRAINS and get_settlement_at(state, nx, ny) is None:
            result.append((nx, ny))
    return result


def place_settlement(
    state: WorldState,
    x: int,
    y: int,
    owner_id: int,
    population: float = 1.0,
    food: float = 0.5,
    wealth: float = 0.0,
    defense: float = 0.3,
    tech_level: float = 0.0,
    has_port: bool = False,
) -> Settlement:
    """Place a new settlement at (x, y) and update the grid."""
    settlement = Settlement(
        x=x,
        y=y,
        population=population,
        food=food,
        wealth=wealth,
        defense=defense,
        tech_level=tech_level,
        has_port=has_port,
        has_longship=False,
        owner_id=owner_id,
        alive=True,
    )
    state.settlements.append(settlement)
    state.grid[y, x] = PORT if has_port else SETTLEMENT
    return settlement


def kill_settlement(state: WorldState, settlement: Settlement) -> None:
    """Mark a settlement as dead and convert its grid cell to a ruin."""
    settlement.alive = False
    state.grid[settlement.y, settlement.x] = RUIN
