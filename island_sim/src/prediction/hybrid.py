"""Hybrid predictor: blend observation-based and simulator-based predictions."""

from __future__ import annotations

import numpy as np

from ..constants import NUM_CLASSES
from ..simulator.params import SimParams
from ..types import Observation, WorldState
from .monte_carlo import generate_prediction
from .observation_model import aggregate_observations, empirical_distribution
from .postprocess import postprocess


def hybrid_prediction(
    initial_state: WorldState,
    seed_index: int,
    observations: list[Observation],
    params: SimParams,
    num_mc_runs: int = 500,
    mc_base_seed: int = 0,
    obs_trust_threshold: int = 3,
) -> np.ndarray:
    """Generate a hybrid prediction blending observations and simulator.

    Strategy:
    - Cells observed >= obs_trust_threshold times: primarily use empirical distribution
    - Cells observed 1 to threshold-1 times: blend empirical and simulator
    - Cells never observed: use simulator prediction
    - Static cells (ocean, mountain): override regardless

    Args:
        initial_state: Starting world state for this seed.
        seed_index: Which seed this is for.
        observations: All observations (filtered internally to this seed).
        params: Inferred simulation parameters for Monte Carlo fallback.
        num_mc_runs: Number of Monte Carlo simulations for unobserved cells.
        mc_base_seed: Base seed for Monte Carlo runs.
        obs_trust_threshold: Minimum observations to fully trust empirical data.

    Returns:
        (H, W, 6) postprocessed probability tensor ready for submission.
    """
    height = initial_state.height
    width = initial_state.width

    # Build empirical distribution from observations
    class_counts, obs_counts = aggregate_observations(
        observations, seed_index, height, width
    )
    obs_probs = empirical_distribution(class_counts, obs_counts)

    # Build simulator-based prediction
    sim_probs = generate_prediction(
        initial_state, params, num_runs=num_mc_runs, base_seed=mc_base_seed
    )

    # Blend based on observation density (vectorized)
    # Consistent formula: obs_weight = n_obs / (n_obs + obs_trust_threshold)
    # This naturally scales: 0 obs → 0 weight, 3 obs → 0.5, 10 obs → 0.77, etc.
    n_obs = obs_counts.astype(np.float64)
    obs_weight = n_obs / (n_obs + obs_trust_threshold)
    obs_weight = np.clip(obs_weight, 0.0, 0.95)[..., np.newaxis]  # cap at 95%, shape (H,W,1)

    prediction = obs_weight * obs_probs + (1 - obs_weight) * sim_probs

    # Postprocess: static overrides, probability floor, normalization
    prediction = postprocess(prediction, initial_state)
    return prediction
