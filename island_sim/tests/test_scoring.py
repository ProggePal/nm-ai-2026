"""Tests for scoring.py — entropy-weighted KL divergence."""

import numpy as np
import pytest

from src.scoring import score_prediction, entropy, kl_divergence
from src.constants import NUM_CLASSES


def _make_tensor(height, width, dominant_class=0, confidence=0.9):
    """Create a simple H×W×6 tensor with one dominant class."""
    other = (1.0 - confidence) / (NUM_CLASSES - 1)
    tensor = np.full((height, width, NUM_CLASSES), other)
    tensor[:, :, dominant_class] = confidence
    return tensor


class TestEntropy:
    def test_certain_distribution_has_zero_entropy(self):
        probs = np.array([[[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]])
        assert entropy(probs)[0, 0] == pytest.approx(0.0, abs=1e-10)

    def test_uniform_has_max_entropy(self):
        probs = np.array([[[1/6, 1/6, 1/6, 1/6, 1/6, 1/6]]])
        expected = -6 * (1/6) * np.log(1/6)
        assert entropy(probs)[0, 0] == pytest.approx(expected, rel=1e-6)

    def test_binary_entropy(self):
        probs = np.array([[[0.5, 0.5, 0.0, 0.0, 0.0, 0.0]]])
        expected = -2 * 0.5 * np.log(0.5)
        assert entropy(probs)[0, 0] == pytest.approx(expected, rel=1e-6)


class TestKLDivergence:
    def test_identical_distributions_have_zero_kl(self):
        gt = np.array([[[0.5, 0.3, 0.1, 0.05, 0.03, 0.02]]])
        assert kl_divergence(gt, gt)[0, 0] == pytest.approx(0.0, abs=1e-10)

    def test_kl_is_nonnegative(self):
        gt = np.array([[[0.5, 0.3, 0.1, 0.05, 0.03, 0.02]]])
        pred = np.array([[[0.2, 0.2, 0.2, 0.2, 0.1, 0.1]]])
        assert kl_divergence(gt, pred)[0, 0] >= 0.0


class TestScorePrediction:
    def test_perfect_prediction_scores_100(self):
        gt = _make_tensor(5, 5, dominant_class=1, confidence=0.8)
        result = score_prediction(gt, gt)
        assert result["score"] == pytest.approx(100.0, abs=0.1)

    def test_uniform_prediction_scores_low(self):
        gt = _make_tensor(5, 5, dominant_class=1, confidence=0.8)
        pred = np.full((5, 5, NUM_CLASSES), 1/6)
        result = score_prediction(gt, pred)
        assert result["score"] < 50.0

    def test_all_static_cells_score_100(self):
        # If all cells have zero entropy (fully deterministic), score is 100
        gt = np.zeros((5, 5, NUM_CLASSES))
        gt[:, :, 0] = 1.0  # All cells are 100% empty
        pred = np.full((5, 5, NUM_CLASSES), 1/6)
        result = score_prediction(gt, pred)
        assert result["score"] == pytest.approx(100.0, abs=0.1)
        assert result["num_dynamic_cells"] == 0

    def test_score_between_0_and_100(self):
        gt = _make_tensor(10, 10, dominant_class=1, confidence=0.6)
        pred = _make_tensor(10, 10, dominant_class=4, confidence=0.6)
        result = score_prediction(gt, pred)
        assert 0.0 <= result["score"] <= 100.0
