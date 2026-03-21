"""Shared test fixtures."""

import numpy as np
import pytest

from src.constants import EMPTY, FOREST, MOUNTAIN, OCEAN, PLAINS, SETTLEMENT, PORT
from src.simulator.params import SimParams
from src.types import Settlement, WorldState


@pytest.fixture
def small_state():
    """A small 10x10 world state for testing."""
    grid = np.full((10, 10), PLAINS, dtype=np.int32)
    grid[0, :] = OCEAN
    grid[-1, :] = OCEAN
    grid[:, 0] = OCEAN
    grid[:, -1] = OCEAN
    grid[3, 3:6] = FOREST
    grid[7, 7] = MOUNTAIN
    grid[4, 4] = SETTLEMENT

    settlements = [
        Settlement(x=4, y=4, population=2.0, food=1.0, wealth=0.5, defense=0.5, owner_id=0, alive=True),
    ]

    return WorldState(grid=grid, settlements=settlements, width=10, height=10)


@pytest.fixture
def default_params():
    return SimParams.default()
