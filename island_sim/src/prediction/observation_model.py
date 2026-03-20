"""Aggregate viewport observations into per-cell empirical distributions."""

from __future__ import annotations

import numpy as np

from ..constants import NUM_CLASSES, TERRAIN_TO_CLASS
from ..types import Observation


def aggregate_observations(
    observations: list[Observation],
    seed_index: int,
    height: int,
    width: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate observations for one seed into terrain class counts.

    Each observation is from a different sim_seed (different stochastic outcome),
    so overlapping viewports give us independent samples of the true distribution.

    Args:
        observations: All observations (will be filtered to this seed).
        seed_index: Which seed to aggregate for.
        height: Full map height.
        width: Full map width.

    Returns:
        class_counts: (H, W, 6) — number of times each class was observed per cell.
        obs_counts: (H, W) — number of times each cell was observed.
    """
    class_counts = np.zeros((height, width, NUM_CLASSES), dtype=np.float64)
    obs_counts = np.zeros((height, width), dtype=np.int32)

    seed_obs = [obs for obs in observations if obs.seed_index == seed_index]

    for obs in seed_obs:
        for vy in range(obs.viewport_h):
            for vx in range(obs.viewport_w):
                gx = obs.viewport_x + vx
                gy = obs.viewport_y + vy
                if 0 <= gx < width and 0 <= gy < height:
                    terrain = int(obs.grid[vy, vx])
                    class_idx = TERRAIN_TO_CLASS.get(terrain, 0)
                    class_counts[gy, gx, class_idx] += 1
                    obs_counts[gy, gx] += 1

    return class_counts, obs_counts


def empirical_distribution(
    class_counts: np.ndarray,
    obs_counts: np.ndarray,
    smoothing: float = 0.5,
) -> np.ndarray:
    """Convert observation counts to smoothed probability distributions.

    Uses additive (Laplace) smoothing so we never assign zero probability.

    Args:
        class_counts: (H, W, 6) terrain class counts.
        obs_counts: (H, W) observation counts per cell.
        smoothing: Additive smoothing parameter (0.5 = Jeffreys prior).

    Returns:
        (H, W, 6) probability tensor for observed cells.
        Cells with 0 observations get uniform distribution.
    """
    # Add smoothing to counts
    smoothed = class_counts + smoothing
    # Denominator: total observations + smoothing * num_classes
    denom = obs_counts[..., np.newaxis] + smoothing * NUM_CLASSES
    # Where obs_counts == 0, we'll get uniform from smoothing alone
    probs = smoothed / denom
    return probs
