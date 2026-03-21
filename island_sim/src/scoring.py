"""Local implementation of the competition scoring metric."""

from __future__ import annotations

import numpy as np

from .constants import ENTROPY_THRESHOLD, NUM_CLASSES


def entropy(probs: np.ndarray) -> np.ndarray:
    """Compute per-cell entropy. Input shape: (H, W, C). Output shape: (H, W)."""
    safe = np.clip(probs, 1e-15, 1.0)
    return -np.sum(probs * np.log(safe), axis=-1)


def kl_divergence(ground_truth: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    """Compute per-cell KL(ground_truth || prediction). Shape: (H, W, C) → (H, W)."""
    gt = np.clip(ground_truth, 1e-15, 1.0)
    pred = np.clip(prediction, 1e-15, 1.0)
    return np.sum(gt * np.log(gt / pred), axis=-1)


def score_prediction(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
) -> dict:
    """Score a prediction against ground truth using entropy-weighted KL divergence.

    Args:
        ground_truth: (H, W, 6) ground truth probability tensor.
        prediction: (H, W, 6) predicted probability tensor.

    Returns:
        Dict with score, weighted_kl, num_dynamic_cells, and per-cell details.
    """
    assert ground_truth.shape == prediction.shape
    assert ground_truth.shape[-1] == NUM_CLASSES

    cell_entropy = entropy(ground_truth)
    cell_kl = kl_divergence(ground_truth, prediction)

    # Dynamic cells are those with non-trivial entropy
    dynamic_mask = cell_entropy > ENTROPY_THRESHOLD
    num_dynamic = int(np.sum(dynamic_mask))

    if num_dynamic == 0:
        return {
            "score": 100.0,
            "weighted_kl": 0.0,
            "num_dynamic_cells": 0,
            "total_cells": ground_truth.shape[0] * ground_truth.shape[1],
        }

    # Entropy-weighted KL
    weighted_kl_sum = np.sum(cell_entropy[dynamic_mask] * cell_kl[dynamic_mask])
    entropy_sum = np.sum(cell_entropy[dynamic_mask])
    weighted_kl = weighted_kl_sum / entropy_sum

    # Score formula from docs
    score = max(0.0, min(100.0, 100.0 * np.exp(-3.0 * weighted_kl)))

    return {
        "score": score,
        "weighted_kl": float(weighted_kl),
        "num_dynamic_cells": num_dynamic,
        "total_cells": ground_truth.shape[0] * ground_truth.shape[1],
        "mean_entropy": float(np.mean(cell_entropy[dynamic_mask])),
        "mean_kl": float(np.mean(cell_kl[dynamic_mask])),
    }
