"""CMA-ES optimizer for parameter inference."""

from __future__ import annotations

import numpy as np

from ..simulator.params import SimParams
from ..types import Observation, WorldState
from .loss import compute_loss

try:
    import island_sim_core
    HAS_RUST = True
except ImportError:
    HAS_RUST = False


def infer_params(
    observations: list[Observation],
    initial_states: list[WorldState],
    num_runs_per_eval: int = 20,
    max_evaluations: int = 300,
    population_size: int | None = None,
    sigma0: float = 0.3,
    warm_start: SimParams | None = None,
) -> SimParams:
    """Infer simulation parameters from observations using CMA-ES.

    Uses Rust CMA-ES backend if available (10-20x faster), falls back to Python.

    Args:
        observations: List of viewport observations from the API.
        initial_states: Initial world states for each seed.
        num_runs_per_eval: Sims per candidate evaluation (higher = more accurate but slower).
        max_evaluations: Maximum number of candidate evaluations.
        population_size: CMA-ES population size (None = auto).
        sigma0: Initial step size (in normalized [0,1] space).
        warm_start: Starting params (e.g. from grid search). Falls back to default if None.

    Returns:
        Best-fit SimParams.
    """
    if HAS_RUST:
        return _infer_rust(
            observations, initial_states, num_runs_per_eval,
            max_evaluations, population_size, sigma0, warm_start,
        )
    return _infer_python(
        observations, initial_states, num_runs_per_eval,
        max_evaluations, population_size, sigma0, warm_start,
    )


def _infer_rust(
    observations: list[Observation],
    initial_states: list[WorldState],
    num_runs_per_eval: int,
    max_evaluations: int,
    population_size: int | None,
    sigma0: float,
    warm_start: SimParams | None,
) -> SimParams:
    """Run CMA-ES inference via Rust backend."""
    # Marshal observations to list of dicts
    obs_dicts = []
    for obs in observations:
        obs_dicts.append({
            "seed_index": obs.seed_index,
            "viewport_x": obs.viewport_x,
            "viewport_y": obs.viewport_y,
            "viewport_w": obs.viewport_w,
            "viewport_h": obs.viewport_h,
            "grid": obs.grid.astype(np.int32),
        })

    # Marshal initial states to list of (grid, settlements) tuples
    state_tuples = []
    for state in initial_states:
        state_tuples.append((
            state.grid.astype(np.int32),
            state.settlements,
        ))

    # Warm start array (empty = use defaults)
    warm_arr = warm_start.to_array() if warm_start is not None else np.array([], dtype=np.float64)

    best_params_arr, best_loss = island_sim_core.run_cmaes_inference(
        obs_dicts,
        state_tuples,
        num_runs_per_eval,
        max_evaluations,
        population_size or 0,  # 0 = auto
        sigma0,
        warm_arr,
        42,  # CMA-ES seed
    )

    print(f"Rust CMA-ES finished: best loss = {best_loss:.4f}")
    return SimParams.from_array(np.array(best_params_arr))


def _infer_python(
    observations: list[Observation],
    initial_states: list[WorldState],
    num_runs_per_eval: int,
    max_evaluations: int,
    population_size: int | None,
    sigma0: float,
    warm_start: SimParams | None,
) -> SimParams:
    """Run CMA-ES inference via Python cma library (fallback)."""
    import cma

    lower, upper = SimParams.bounds_arrays()
    ndim = SimParams.ndim()

    # Normalize to [0, 1] space for CMA-ES — start from warm-start if available
    start_params = warm_start or SimParams.default()
    x0_raw = start_params.to_array()
    x0 = (x0_raw - lower) / (upper - lower)

    eval_count = 0
    best_loss = float("inf")
    best_params = start_params

    def objective(x_normalized: np.ndarray) -> float:
        nonlocal eval_count, best_loss, best_params

        # Map back to original space
        x_raw = lower + x_normalized * (upper - lower)
        params = SimParams.from_array(x_raw)

        loss = compute_loss(
            observations, initial_states, params, num_runs=num_runs_per_eval
        )
        eval_count += 1

        if loss < best_loss:
            best_loss = loss
            best_params = params

        return loss

    # CMA-ES options
    opts = {
        "maxfevals": max_evaluations,
        "bounds": [np.zeros(ndim).tolist(), np.ones(ndim).tolist()],
        "verbose": -1,  # Suppress output
        "seed": 42,
    }
    if population_size is not None:
        opts["popsize"] = population_size

    es = cma.CMAEvolutionStrategy(x0.tolist(), sigma0, opts)

    while not es.stop():
        solutions = es.ask()
        fitnesses = [objective(np.array(x)) for x in solutions]
        es.tell(solutions, fitnesses)

        if eval_count >= max_evaluations:
            break

    print(f"CMA-ES finished: {eval_count} evaluations, best loss = {best_loss:.4f}")
    return best_params
