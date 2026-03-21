"""Tests for postprocess.py — static overrides, smart probability floor."""

import numpy as np
import pytest

from src.constants import MOUNTAIN, OCEAN, PLAINS, FOREST, NUM_CLASSES
from src.prediction.postprocess import postprocess
from src.types import Settlement, WorldState


def _make_state(grid):
    """Create a WorldState from a grid array."""
    h, w = grid.shape
    return WorldState(grid=grid, settlements=[], width=w, height=h)


class TestStaticCells:
    def test_ocean_cells_are_100_percent_empty(self):
        grid = np.full((5, 5), PLAINS, dtype=np.int32)
        grid[0, :] = OCEAN
        state = _make_state(grid)
        pred = np.full((5, 5, NUM_CLASSES), 1/6)

        result = postprocess(pred, state)

        for x in range(5):
            assert result[0, x, 0] == pytest.approx(1.0)  # CLASS_EMPTY
            for cls in range(1, NUM_CLASSES):
                assert result[0, x, cls] == pytest.approx(0.0)

    def test_mountain_cells_are_100_percent_mountain(self):
        grid = np.full((5, 5), PLAINS, dtype=np.int32)
        grid[2, 2] = MOUNTAIN
        state = _make_state(grid)
        pred = np.full((5, 5, NUM_CLASSES), 1/6)

        result = postprocess(pred, state)

        assert result[2, 2, 5] == pytest.approx(1.0)  # CLASS_MOUNTAIN
        for cls in range(5):
            assert result[2, 2, cls] == pytest.approx(0.0)


class TestSmartFloor:
    def test_mountain_prob_zero_on_non_mountain_cells(self):
        grid = np.full((5, 5), PLAINS, dtype=np.int32)
        grid[0, :] = OCEAN
        state = _make_state(grid)
        pred = np.full((5, 5, NUM_CLASSES), 1/6)

        result = postprocess(pred, state)

        # All non-mountain land cells should have 0 mountain probability
        for y in range(1, 5):
            for x in range(5):
                assert result[y, x, 5] == pytest.approx(0.0), f"Cell ({y},{x}) has mountain prob {result[y, x, 5]}"

    def test_port_prob_zero_on_non_coastal_cells(self):
        grid = np.full((5, 5), PLAINS, dtype=np.int32)
        grid[0, :] = OCEAN
        state = _make_state(grid)
        pred = np.full((5, 5, NUM_CLASSES), 1/6)

        result = postprocess(pred, state)

        # Row 1 is coastal (adjacent to ocean row 0)
        # Rows 2-4 are inland — should have 0 port probability
        for y in range(2, 5):
            for x in range(5):
                assert result[y, x, 2] == pytest.approx(0.0), f"Cell ({y},{x}) has port prob {result[y, x, 2]}"

    def test_coastal_cells_can_have_port_prob(self):
        grid = np.full((5, 5), PLAINS, dtype=np.int32)
        grid[0, :] = OCEAN
        state = _make_state(grid)
        pred = np.full((5, 5, NUM_CLASSES), 1/6)

        result = postprocess(pred, state)

        # Row 1 is coastal — port probability should be >= floor
        assert result[1, 2, 2] >= 0.01


class TestNormalization:
    def test_all_rows_sum_to_one(self):
        grid = np.full((10, 10), PLAINS, dtype=np.int32)
        grid[0, :] = OCEAN
        grid[5, 5] = MOUNTAIN
        grid[3, 3] = FOREST
        state = _make_state(grid)
        pred = np.random.dirichlet(np.ones(NUM_CLASSES), size=(10, 10))

        result = postprocess(pred, state)

        sums = result.sum(axis=-1)
        np.testing.assert_allclose(sums, 1.0, atol=1e-6)

    def test_no_negative_probabilities(self):
        grid = np.full((5, 5), PLAINS, dtype=np.int32)
        grid[0, :] = OCEAN
        state = _make_state(grid)
        pred = np.zeros((5, 5, NUM_CLASSES))

        result = postprocess(pred, state)

        assert (result >= 0.0).all()

    def test_probability_floor_applied(self):
        grid = np.full((5, 5), PLAINS, dtype=np.int32)
        grid[0, :] = OCEAN  # Need ocean so row 1 is coastal
        state = _make_state(grid)
        pred = np.zeros((5, 5, NUM_CLASSES))
        pred[:, :, 0] = 1.0  # All empty

        result = postprocess(pred, state)

        # Non-static, non-impossible classes should have at least floor
        # Row 2 is inland — classes 0, 1, 3, 4 should have >= floor
        for cls in [0, 1, 3, 4]:  # empty, settlement, ruin, forest
            assert result[2, 2, cls] >= 0.009  # slightly less than 0.01 due to normalization
