"""Emergency submission — skip CMA-ES, use best available params directly.

Usage:
    python scripts/emergency_submit.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.api_client import get_active_round, get_round_detail, submit_prediction
from src.orchestrator import load_best_warm_start, load_observations
from src.prediction.hybrid import hybrid_prediction
from src.prediction.postprocess import postprocess
from src.prediction.monte_carlo import generate_prediction
from src.simulator.params import SimParams
from src.types import WorldState

NUM_SEEDS = 5
MC_RUNS = 2000


def main():
    print("=== Emergency Submit ===")

    active_round = get_active_round()
    if active_round is None:
        print("No active round.")
        return

    round_id = active_round["id"]
    width = active_round["map_width"]
    height = active_round["map_height"]
    print(f"Round {active_round['round_number']} ({width}x{height})")

    detail = get_round_detail(round_id)
    initial_states = [
        WorldState.from_api(state, width, height)
        for state in detail["initial_states"]
    ]

    # Best available params (no CMA-ES)
    params = load_best_warm_start()
    if params is None:
        params = SimParams.default()
        print("Using default params")
    else:
        print("Using best available warm-start params")

    # Load any saved observations (may be empty)
    observations = load_observations(round_id)
    print(f"Loaded {len(observations)} observations")

    for seed_idx in range(NUM_SEEDS):
        print(f"  Seed {seed_idx}: generating prediction ({MC_RUNS} MC runs)...")
        if observations:
            prediction = hybrid_prediction(
                initial_states[seed_idx],
                seed_idx,
                observations,
                params,
                num_mc_runs=MC_RUNS,
                mc_base_seed=seed_idx * 100000,
            )
        else:
            prediction = generate_prediction(initial_states[seed_idx], params, MC_RUNS)
            prediction = postprocess(prediction, initial_states[seed_idx])

        assert prediction.shape == (height, width, 6)
        assert np.allclose(prediction.sum(axis=-1), 1.0, atol=0.01)

        print(f"  Submitting seed {seed_idx}...")
        try:
            result = submit_prediction(round_id, seed_idx, prediction.tolist())
            print(f"  Seed {seed_idx}: {result.get('status', 'unknown')}")
        except Exception as exc:
            print(f"  Seed {seed_idx} FAILED: {exc}")

    print("\nDone!")


if __name__ == "__main__":
    main()
