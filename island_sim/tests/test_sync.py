"""Cross-validation tests: Python and Rust simulators must produce compatible results.

Note: Python uses np.random (PCG64), Rust uses ChaCha8Rng — different random sequences.
We can't compare exact grids, but MC prediction distributions should converge.
"""

import numpy as np
import pytest

from src.constants import NUM_CLASSES
from src.simulator.params import SimParams
from src.types import Settlement, WorldState

try:
    import island_sim_core
    HAS_RUST = True
except ImportError:
    HAS_RUST = False


pytestmark = pytest.mark.skipif(not HAS_RUST, reason="Rust backend not built")


@pytest.fixture
def round_001_state():
    """Load round 001 seed 0 if available, otherwise create a test state."""
    from pathlib import Path
    import json

    detail_path = Path("data/rounds/round_001/detail.json")
    if detail_path.exists():
        detail = json.loads(detail_path.read_text())
        return WorldState.from_api(detail["initial_states"][0], 40, 40)

    # Fallback: synthetic state
    grid = np.full((20, 20), 11, dtype=np.int32)
    grid[0, :] = 10
    grid[-1, :] = 10
    grid[:, 0] = 10
    grid[:, -1] = 10
    grid[5:8, 5:8] = 4
    grid[10, 10] = 5
    grid[3, 3] = 1

    settlements = [
        Settlement(x=3, y=3, population=2.0, food=1.0, wealth=0.5,
                  defense=0.5, tech_level=0.0, has_port=False,
                  has_longship=False, owner_id=0, alive=True),
    ]
    return WorldState(grid=grid, settlements=settlements, width=20, height=20)


class TestRustPythonSync:
    def test_param_count_matches(self):
        """Rust accepts the same number of params as Python."""
        params = SimParams.default()
        arr = params.to_array()
        grid = np.full((5, 5), 11, dtype=np.int32)
        grid[0, :] = 10

        # Should not raise
        island_sim_core.generate_prediction(grid, [], arr, 1, 0)

    def test_mc_distributions_converge(self, round_001_state):
        """With enough MC runs, Rust and Python predictions should be similar."""
        from src.prediction.monte_carlo import _generate_python
        params = SimParams.default()
        num_runs = 200

        # Python prediction
        py_pred = _generate_python(round_001_state, params, num_runs, base_seed=0)

        # Rust prediction
        rust_pred = np.array(island_sim_core.generate_prediction(
            round_001_state.grid, round_001_state.settlements,
            params.to_array(), num_runs, 0
        ))

        # Both should be valid probability tensors
        assert py_pred.shape == rust_pred.shape
        np.testing.assert_allclose(py_pred.sum(axis=-1), 1.0, atol=0.01)
        np.testing.assert_allclose(rust_pred.sum(axis=-1), 1.0, atol=0.01)

        # Distributions should be similar (not identical due to different RNG)
        # Use generous tolerance — we're testing behavioral equivalence, not exact match
        diff = np.abs(py_pred - rust_pred)
        assert diff.mean() < 0.1, f"Mean prediction difference too large: {diff.mean():.4f}"

    def test_static_cells_identical(self, round_001_state):
        """Ocean and mountain cells should produce identical results."""
        params = SimParams.default()

        rust_pred = np.array(island_sim_core.generate_prediction(
            round_001_state.grid, round_001_state.settlements,
            params.to_array(), 50, 0
        ))

        grid = round_001_state.grid
        ocean_mask = grid == 10
        mountain_mask = grid == 5

        # Ocean cells: 100% class 0
        if ocean_mask.any():
            np.testing.assert_allclose(rust_pred[ocean_mask, 0], 1.0, atol=0.01)

        # Mountain cells: 100% class 5
        if mountain_mask.any():
            np.testing.assert_allclose(rust_pred[mountain_mask, 5], 1.0, atol=0.01)

    def test_generate_prediction_shape(self, round_001_state):
        """Rust generate_prediction returns correct shape."""
        params = SimParams.default()
        pred = island_sim_core.generate_prediction(
            round_001_state.grid, round_001_state.settlements,
            params.to_array(), 10, 0
        )
        pred = np.array(pred)
        assert pred.shape == (round_001_state.height, round_001_state.width, NUM_CLASSES)

    def test_simulate_returns_valid_grid(self, round_001_state):
        """Rust simulate returns a grid with valid terrain codes."""
        params = SimParams.default()
        result_grid, result_settlements = island_sim_core.simulate(
            round_001_state.grid, round_001_state.settlements,
            params.to_array(), 42
        )

        assert result_grid.shape == (round_001_state.height, round_001_state.width)

        # All terrain codes should be valid
        valid_codes = {0, 1, 2, 3, 4, 5, 10, 11}
        unique = set(np.unique(result_grid))
        assert unique.issubset(valid_codes), f"Invalid terrain codes: {unique - valid_codes}"
