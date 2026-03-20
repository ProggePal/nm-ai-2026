"""Post-processing: static cell overrides, probability floor, normalization."""

from __future__ import annotations

import numpy as np

from ..constants import (
    CLASS_EMPTY,
    CLASS_FOREST,
    CLASS_MOUNTAIN,
    MOUNTAIN,
    NUM_CLASSES,
    OCEAN,
)
from ..types import WorldState

# Probability floor to prevent catastrophic KL divergence
PROB_FLOOR = 0.01

# High-confidence probability for static cells
STATIC_CONFIDENCE = 0.98


def postprocess(
    prediction: np.ndarray,
    initial_state: WorldState,
) -> np.ndarray:
    """Apply static cell overrides, probability floor, and normalization.

    Args:
        prediction: Raw (H, W, 6) probability tensor from Monte Carlo.
        initial_state: Initial world state to identify static cells.

    Returns:
        Processed (H, W, 6) probability tensor ready for submission.
    """
    result = prediction.copy()

    # Static cell overrides
    for y in range(initial_state.height):
        for x in range(initial_state.width):
            terrain = int(initial_state.grid[y, x])

            if terrain == OCEAN:
                # Ocean cells stay as empty class with high confidence
                result[y, x] = _make_static_distribution(CLASS_EMPTY)
            elif terrain == MOUNTAIN:
                # Mountains never change
                result[y, x] = _make_static_distribution(CLASS_MOUNTAIN)

    # Apply probability floor
    result = np.maximum(result, PROB_FLOOR)

    # Renormalize each cell to sum to 1.0
    row_sums = result.sum(axis=-1, keepdims=True)
    result = result / row_sums

    return result


def _make_static_distribution(dominant_class: int) -> np.ndarray:
    """Create a probability vector heavily favoring one class."""
    dist = np.full(NUM_CLASSES, (1.0 - STATIC_CONFIDENCE) / (NUM_CLASSES - 1))
    dist[dominant_class] = STATIC_CONFIDENCE
    return dist
