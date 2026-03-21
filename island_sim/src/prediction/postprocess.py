"""Post-processing: static cell overrides, smart probability floor, normalization."""

from __future__ import annotations

import numpy as np

from ..constants import (
    CLASS_EMPTY,
    CLASS_MOUNTAIN,
    CLASS_PORT,
    FOREST,
    MOUNTAIN,
    NUM_CLASSES,
    OCEAN,
)
from ..types import WorldState

# Probability floor to prevent catastrophic KL divergence
PROB_FLOOR = 0.01


def postprocess(
    prediction: np.ndarray,
    initial_state: WorldState,
) -> np.ndarray:
    """Apply smart probability floor and static cell overrides.

    Key insight: only apply probability floor to classes that are actually
    possible for each cell. Mountains never appear on non-mountain cells,
    ports only appear on coastal cells, etc.
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

    for y in range(initial_state.height):
        for x in range(initial_state.width):
            terrain = int(grid[y, x])

            if terrain == OCEAN:
                result[y, x] = _make_exact_static(CLASS_EMPTY)
                continue
            if terrain == MOUNTAIN:
                result[y, x] = _make_exact_static(CLASS_MOUNTAIN)
                continue

            # Dynamic cell: apply floor only to plausible classes
            # Class 5 (mountain) is NEVER possible on non-mountain cells
            result[y, x, CLASS_MOUNTAIN] = 0.0

            # Class 2 (port) is only possible on coastal cells
            if not coastal_mask[y, x]:
                result[y, x, CLASS_PORT] = 0.0

            # Apply floor to remaining classes
            for cls in range(NUM_CLASSES):
                if result[y, x, cls] > 0.0 or cls in (CLASS_EMPTY, 1, 3, 4):  # empty, settlement, ruin, forest always possible
                    if cls == CLASS_MOUNTAIN:
                        continue  # already zeroed
                    if cls == CLASS_PORT and not coastal_mask[y, x]:
                        continue  # already zeroed
                    result[y, x, cls] = max(result[y, x, cls], PROB_FLOOR)

            # Renormalize
            row_sum = result[y, x].sum()
            if row_sum > 1e-10:
                result[y, x] /= row_sum

    return result


def _make_exact_static(dominant_class: int) -> np.ndarray:
    """Create a probability vector with 1.0 for the dominant class, 0.0 elsewhere."""
    dist = np.zeros(NUM_CLASSES)
    dist[dominant_class] = 1.0
    return dist
