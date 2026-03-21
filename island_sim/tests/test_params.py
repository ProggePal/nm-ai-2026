"""Tests for SimParams — round-trip, bounds, naming."""

import numpy as np
import pytest

from src.simulator.params import SimParams


class TestParamRoundTrip:
    def test_array_round_trip(self):
        params = SimParams.default()
        arr = params.to_array()
        restored = SimParams.from_array(arr)
        np.testing.assert_allclose(restored.to_array(), arr)

    def test_dict_round_trip(self):
        params = SimParams.default()
        named = params.to_dict()
        restored = SimParams.from_dict(named)
        np.testing.assert_allclose(restored.to_array(), params.to_array())

    def test_random_params_round_trip(self):
        params = SimParams.random(np.random.default_rng(42))
        arr = params.to_array()
        restored = SimParams.from_array(arr)
        np.testing.assert_allclose(restored.to_array(), arr)

    def test_dict_handles_missing_params(self):
        """Missing params should use defaults."""
        partial = {"base_food_production": 1.0}
        params = SimParams.from_dict(partial)
        assert params.base_food_production == 1.0
        assert params.forest_food_bonus == SimParams.default().forest_food_bonus


class TestParamConsistency:
    def test_ndim_matches_param_names(self):
        assert SimParams.ndim() == len(SimParams.param_names())

    def test_all_params_have_bounds(self):
        for name in SimParams.param_names():
            assert name in SimParams.BOUNDS, f"Missing bounds for {name}"

    def test_bounds_count_matches_ndim(self):
        assert len(SimParams.BOUNDS) == SimParams.ndim()

    def test_defaults_within_bounds(self):
        params = SimParams.default()
        lower, upper = SimParams.bounds_arrays()
        arr = params.to_array()
        for idx, name in enumerate(SimParams.param_names()):
            assert arr[idx] >= lower[idx], f"{name}={arr[idx]} < lower={lower[idx]}"
            assert arr[idx] <= upper[idx], f"{name}={arr[idx]} > upper={upper[idx]}"

    def test_from_array_clamps_to_bounds(self):
        lower, upper = SimParams.bounds_arrays()
        # Way below bounds
        params = SimParams.from_array(np.full(SimParams.ndim(), -100.0))
        np.testing.assert_allclose(params.to_array(), lower)
        # Way above bounds
        params = SimParams.from_array(np.full(SimParams.ndim(), 1000.0))
        np.testing.assert_allclose(params.to_array(), upper)


class TestRustSync:
    def test_rust_param_count_matches(self):
        """Verify island_sim_core accepts the right number of params."""
        try:
            import island_sim_core
        except ImportError:
            pytest.skip("Rust backend not built")

        params = SimParams.default()
        grid = np.full((5, 5), 11, dtype=np.int32)
        grid[0, :] = 10

        # This will fail if Rust NDIM doesn't match Python
        from src.types import Settlement
        settlements = []
        island_sim_core.generate_prediction(grid, settlements, params.to_array(), 1, 0)
