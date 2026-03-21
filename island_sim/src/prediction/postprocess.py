"""Post-processing: static cell overrides, smart probability floor, normalization."""

from __future__ import annotations

import numpy as np

from ..constants import (
    CLASS_EMPTY,
    CLASS_MOUNTAIN,
    CLASS_PORT,
    CLASS_RUIN,
    CLASS_SETTLEMENT,
    MOUNTAIN,
    NUM_CLASSES,
    OCEAN,
)
from ..types import WorldState

# Two-tier probability floor: near settlements vs remote
PROB_FLOOR = 0.005          # standard floor for plausible classes near settlements
PROB_FLOOR_REMOTE = 0.001   # minimal floor for unlikely classes far from settlements
SETTLEMENT_PROXIMITY = 6    # Chebyshev distance threshold for "near a settlement"


def postprocess(
    prediction: np.ndarray,
    initial_state: WorldState,
) -> np.ndarray:
    """Apply smart probability floor and static cell overrides.

    Key insight: only apply probability floor to classes that are actually
    possible for each cell. Mountains never appear on non-mountain cells,
    ports only appear on coastal cells. Cells far from settlements almost
    never become Settlement/Port/Ruin, so they get a smaller floor.
    """
    result = prediction.copy()
    grid = initial_state.grid

    # Precompute coastal mask (adjacent to ocean)
    coastal_mask = np.zeros((initial_state.height, initial_state.width), dtype=bool)
    for y in range(initial_state.height):
        for x in range(initial_state.width):
            if grid[y, x] in (OCEAN, MOUNTAIN):
                continue
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < initial_state.height and 0 <= nx < initial_state.width:
                        if grid[ny, nx] == OCEAN:
                            coastal_mask[y, x] = True

    # Precompute distance to nearest initial settlement (Chebyshev)
    near_settlement = np.zeros((initial_state.height, initial_state.width), dtype=bool)
    for settlement in initial_state.settlements:
        if not settlement.alive:
            continue
        sx, sy = settlement.x, settlement.y
        for dy in range(-SETTLEMENT_PROXIMITY, SETTLEMENT_PROXIMITY + 1):
            for dx in range(-SETTLEMENT_PROXIMITY, SETTLEMENT_PROXIMITY + 1):
                ny, nx = sy + dy, sx + dx
                if 0 <= ny < initial_state.height and 0 <= nx < initial_state.width:
                    near_settlement[ny, nx] = True

    for y in range(initial_state.height):
        for x in range(initial_state.width):
            terrain = int(grid[y, x])

            if terrain == OCEAN:
                result[y, x] = _make_exact_static(CLASS_EMPTY)
                continue
            if terrain == MOUNTAIN:
                result[y, x] = _make_exact_static(CLASS_MOUNTAIN)
                continue

            # Dynamic cell: apply floor to all classes.
            # Never assign 0.0 — the ground truth may have small nonzero
            # probabilities for any class, and a zero prediction creates
            # infinite KL divergence.
            is_near = near_settlement[y, x]

            for cls in range(NUM_CLASSES):
                # Mountain on non-mountain, port on non-coastal: very unlikely
                if cls == CLASS_MOUNTAIN:
                    floor = PROB_FLOOR_REMOTE
                elif cls == CLASS_PORT and not coastal_mask[y, x]:
                    floor = PROB_FLOOR_REMOTE
                elif cls in (CLASS_SETTLEMENT, CLASS_PORT, CLASS_RUIN) and not is_near:
                    floor = PROB_FLOOR_REMOTE
                else:
                    floor = PROB_FLOOR

                result[y, x, cls] = max(result[y, x, cls], floor)

            # Renormalize
            row_sum = result[y, x].sum()
            if row_sum > 1e-10:
                result[y, x] /= row_sum

    return result


def _make_exact_static(dominant_class: int) -> np.ndarray:
    """Create a probability vector heavily favoring the dominant class.

    Uses a tiny floor for all classes to avoid infinite KL divergence
    if the ground truth has any nonzero probability for other classes.
    """
    dist = np.full(NUM_CLASSES, PROB_FLOOR_REMOTE)
    dist[dominant_class] = 1.0
    dist /= dist.sum()
    return dist
