"""Main orchestrator: discover → observe → infer → predict → submit.

Automatically fetches completed round data for warm-starting parameter inference.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from .api_client import (
    get_active_round,
    get_analysis,
    get_budget,
    get_my_rounds,
    get_round_detail,
    simulate_query,
    submit_prediction,
)
from .inference.observer import plan_adaptive_viewports, plan_viewports
from .inference.optimizer import infer_params
from .prediction.hybrid import hybrid_prediction
from .prediction.monte_carlo import generate_prediction
from .prediction.postprocess import postprocess
from .scoring import score_prediction
from .simulator.params import SimParams
from .types import Observation, WorldState

NUM_SEEDS = 5
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "rounds"
OBS_DIR = Path(__file__).resolve().parent.parent / "data" / "observations"


# ---------------------------------------------------------------------------
# History management — fetch and store completed rounds for warm-starting
# ---------------------------------------------------------------------------

def sync_history() -> None:
    """Fetch any new completed rounds and store locally."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Syncing round history...")
    try:
        rounds = get_my_rounds()
    except Exception as exc:
        print(f"  Could not fetch round history: {exc}")
        return

    completed = [r for r in rounds if r["status"] in ("completed", "scoring")]
    new_count = 0

    for rnd in completed:
        round_id = rnd["id"]
        round_num = rnd["round_number"]
        round_dir = DATA_DIR / f"round_{round_num:03d}"

        # Skip if already fetched
        if round_dir.exists() and (round_dir / "detail.json").exists():
            continue

        round_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Fetching round {round_num} (score={rnd.get('round_score')})...")

        try:
            (round_dir / "summary.json").write_text(json.dumps(rnd, indent=2))
            detail = get_round_detail(round_id)
            (round_dir / "detail.json").write_text(json.dumps(detail, indent=2))

            seeds_count = rnd.get("seeds_count", detail.get("seeds_count", NUM_SEEDS))
            for seed_idx in range(seeds_count):
                try:
                    analysis = get_analysis(round_id, seed_idx)
                    (round_dir / f"analysis_seed_{seed_idx}.json").write_text(
                        json.dumps(analysis, indent=2)
                    )
                except Exception:
                    pass
                time.sleep(0.3)

            new_count += 1
        except Exception as exc:
            print(f"  Failed: {exc}")

    if new_count:
        print(f"  Fetched {new_count} new round(s)")
    else:
        print("  History up to date")


def load_history_params() -> SimParams | None:
    """Try to load best-fit params from previous calibration runs."""
    params_path = DATA_DIR / "best_params.json"
    if not params_path.exists():
        return None

    try:
        data = json.loads(params_path.read_text())
        # Support both named dict format (new) and array format (legacy)
        if "named_params" in data:
            params = SimParams.from_dict(data["named_params"])
        elif "params" in data:
            arr = np.array(data["params"])
            params = SimParams.from_array(arr)
        else:
            return None
        print(f"  Loaded warm-start params (score={data.get('score', '?')})")
        return params
    except Exception:
        return None


def save_observations(round_id: str, observations: list[Observation]) -> None:
    """Save observations to disk for later analysis."""
    OBS_DIR.mkdir(parents=True, exist_ok=True)
    obs_file = OBS_DIR / f"{round_id}.json"

    # Convert to serializable format
    data = []
    for obs in observations:
        data.append({
            "seed_index": obs.seed_index,
            "viewport_x": obs.viewport_x,
            "viewport_y": obs.viewport_y,
            "viewport_w": obs.viewport_w,
            "viewport_h": obs.viewport_h,
            "grid": obs.grid.tolist(),
            "settlements": [
                {
                    "x": s.x, "y": s.y,
                    "population": s.population, "food": s.food,
                    "wealth": s.wealth, "defense": s.defense,
                    "has_port": s.has_port, "alive": s.alive,
                    "owner_id": s.owner_id,
                }
                for s in obs.settlements
            ],
        })

    # Append to existing observations if file exists
    existing = []
    if obs_file.exists():
        existing = json.loads(obs_file.read_text())
    existing.extend(data)
    obs_file.write_text(json.dumps(existing, indent=2))
    print(f"  Saved {len(data)} observations ({len(existing)} total for this round)")


def load_observations(round_id: str) -> list[Observation]:
    """Load previously saved observations for a round."""
    obs_file = OBS_DIR / f"{round_id}.json"
    if not obs_file.exists():
        return []

    data = json.loads(obs_file.read_text())
    observations = []
    for entry in data:
        observations.append(Observation.from_dict(entry))
    return observations


def save_params(params: SimParams, round_id: str, score: float | None = None) -> None:
    """Save CMA-ES inferred params per-round (does NOT overwrite grid search defaults)."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    data = {
        "params": params.to_array().tolist(),
        "named_params": params.to_dict(),
        "score": score,
        "param_names": SimParams.param_names(),
        "round_id": round_id,
        "source": "cma_es",
    }
    (DATA_DIR / f"params_round_{round_id[:8]}.json").write_text(json.dumps(data, indent=2))


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    """Full pipeline: sync history → discover → observe → infer → predict → submit."""
    # Step 0: Sync history for warm-starting
    sync_history()
    warm_params = load_history_params()

    # Step 1: Discover active round
    print("\nDiscovering active round...")
    active_round = get_active_round()
    if active_round is None:
        print("No active round found. Exiting.")
        sys.exit(0)

    round_id = active_round["id"]
    width = active_round["map_width"]
    height = active_round["map_height"]
    print(f"Active round: {active_round['round_number']} ({width}x{height})")

    # Step 2: Get initial states
    print("Fetching round details...")
    detail = get_round_detail(round_id)
    initial_states = [
        WorldState.from_api(state, width, height)
        for state in detail["initial_states"]
    ]
    print(f"Loaded {len(initial_states)} initial states")

    # Step 3: Check budget
    budget_info = get_budget()
    queries_remaining = budget_info["queries_max"] - budget_info["queries_used"]
    print(f"Budget: {budget_info['queries_used']}/{budget_info['queries_max']} used, {queries_remaining} remaining")

    # Load any previously saved observations for this round
    prior_observations = load_observations(round_id)
    if prior_observations:
        print(f"Loaded {len(prior_observations)} prior observations for this round")

    new_observations: list[Observation] = []

    if queries_remaining <= 0:
        print("No queries remaining! Skipping observation phase.")
        observations: list[Observation] = prior_observations
        params = warm_params or SimParams.default()
    else:
        # Step 4: Phase 1 queries — systematic viewport placement
        # With small budgets, skip the split and use everything systematically
        if queries_remaining <= 10:
            systematic_budget = queries_remaining
        else:
            systematic_budget = int(queries_remaining * 0.6)
        print(f"\nPhase 1: {systematic_budget} systematic queries...")
        systematic_queries = plan_viewports(initial_states, budget=systematic_budget)

        observations = []
        for idx, query in enumerate(systematic_queries):
            print(f"  Query {idx + 1}/{len(systematic_queries)}: seed={query['seed_index']} "
                  f"({query['viewport_x']},{query['viewport_y']})")
            try:
                obs = simulate_query(
                    round_id,
                    query["seed_index"],
                    query["viewport_x"],
                    query["viewport_y"],
                    query["viewport_w"],
                    query["viewport_h"],
                )
                observations.append(obs)
            except Exception as exc:
                print(f"  Query failed: {exc}")
                break

        # Track new observations from this run
        new_observations = list(observations)
        # Merge with prior observations
        observations = prior_observations + observations
        print(f"Total observations: {len(observations)}")

        # Step 5: Infer parameters (only if enough observations to be useful)
        MIN_OBS_FOR_INFERENCE = 10
        if len(observations) >= MIN_OBS_FOR_INFERENCE:
            print(f"\nInferring parameters from {len(observations)} observations...")
            params = infer_params(
                observations,
                initial_states,
                num_runs_per_eval=50,
                max_evaluations=200,
            )
        else:
            print(f"\nOnly {len(observations)} observations — skipping inference, using {'warm-start' if warm_params else 'default'} params")
            params = warm_params or SimParams.default()

        # Step 6: Phase 2 queries — adaptive viewport placement
        budget_info = get_budget()
        adaptive_remaining = budget_info["queries_max"] - budget_info["queries_used"]
        if adaptive_remaining > 0:
            print(f"\nPhase 2: {adaptive_remaining} adaptive queries...")
            adaptive_queries = plan_adaptive_viewports(
                initial_states, observations, adaptive_remaining
            )
            for idx, query in enumerate(adaptive_queries):
                print(f"  Query {idx + 1}/{len(adaptive_queries)}: seed={query['seed_index']} "
                      f"({query['viewport_x']},{query['viewport_y']})")
                try:
                    obs = simulate_query(
                        round_id,
                        query["seed_index"],
                        query["viewport_x"],
                        query["viewport_y"],
                        query["viewport_w"],
                        query["viewport_h"],
                    )
                    observations.append(obs)
                except Exception as exc:
                    print(f"  Query failed: {exc}")
                    break

            # Track phase 2 observations (added after prior + phase 1)
            new_observations = observations[len(prior_observations):]

            # Step 7: Re-infer with all observations (only if enough)
            if len(observations) >= MIN_OBS_FOR_INFERENCE:
                print(f"\nRe-inferring parameters with {len(observations)} total observations...")
                params = infer_params(
                    observations,
                    initial_states,
                    num_runs_per_eval=50,
                    max_evaluations=300,
                )

    # Save all new observations from this run
    if new_observations:
        save_observations(round_id, new_observations)

    # Step 8: Generate hybrid predictions for all 5 seeds
    print("\nGenerating hybrid predictions (observations + simulator)...")
    seed_obs_counts = {i: sum(1 for o in observations if o.seed_index == i) for i in range(NUM_SEEDS)}
    for seed_idx in range(NUM_SEEDS):
        n_obs = seed_obs_counts[seed_idx]
        print(f"  Seed {seed_idx}: {n_obs} observations + 2000 MC sims...")
        prediction = hybrid_prediction(
            initial_states[seed_idx],
            seed_idx,
            observations,
            params,
            num_mc_runs=2000,
            mc_base_seed=seed_idx * 100000,
        )

        # Validate
        assert prediction.shape == (height, width, 6), f"Bad shape: {prediction.shape}"
        sums = prediction.sum(axis=-1)
        assert np.allclose(sums, 1.0, atol=0.01), f"Row sums not 1.0: {sums.min()}-{sums.max()}"
        assert (prediction >= 0).all(), "Negative probabilities"

        # Step 9: Submit
        print(f"  Submitting seed {seed_idx}...")
        try:
            result = submit_prediction(round_id, seed_idx, prediction.tolist())
            print(f"  Seed {seed_idx}: {result.get('status', 'unknown')}")
        except Exception as exc:
            print(f"  Seed {seed_idx} submission failed: {exc}")

    # Step 11: Save CMA-ES params for this round (doesn't overwrite grid search defaults)
    save_params(params, round_id)
    print("\nRound params saved.")

    # Step 12: Post-round — wait for scoring and fetch results
    print("Run sync_history() after the round completes to fetch ground truth.")
    print("\nDone!")


if __name__ == "__main__":
    main()
