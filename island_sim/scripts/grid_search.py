"""Parallelized parameter grid search against ground truth.

Designed for GCP Compute Engine (high core count). Run with:
    python scripts/grid_search.py --workers 96 --candidates 2000

Also works locally with fewer workers:
    python scripts/grid_search.py --workers 8 --candidates 200
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.constants import NUM_CLASSES, TERRAIN_TO_CLASS
from src.prediction.postprocess import postprocess
from src.scoring import score_prediction
from src.simulator.engine import simulate
from src.simulator.params import SimParams
from src.types import WorldState

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "rounds"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "data" / "grid_search"

MC_RUNS_PER_SEED = 30  # Per candidate — balance speed vs accuracy


# ---------------------------------------------------------------------------
# Data loading (done once in main process)
# ---------------------------------------------------------------------------

def load_all_rounds() -> list[dict]:
    """Load all rounds that have ground truth."""
    rounds = []
    for round_dir in sorted(DATA_DIR.iterdir()):
        if not round_dir.is_dir():
            continue

        detail_path = round_dir / "detail.json"
        if not detail_path.exists():
            continue

        detail = json.loads(detail_path.read_text())
        width = detail["map_width"]
        height = detail["map_height"]

        initial_states = [
            WorldState.from_api(state, width, height)
            for state in detail["initial_states"]
        ]

        seeds = []
        for seed_idx in range(len(initial_states)):
            analysis_path = round_dir / f"analysis_seed_{seed_idx}.json"
            if not analysis_path.exists():
                continue
            analysis = json.loads(analysis_path.read_text())
            gt = analysis.get("ground_truth")
            if gt is None:
                continue
            seeds.append({
                "seed_index": seed_idx,
                "ground_truth": np.array(gt, dtype=np.float64),
            })

        if seeds:
            rounds.append({
                "name": round_dir.name,
                "width": width,
                "height": height,
                "initial_states": initial_states,
                "seeds": seeds,
            })

    return rounds


# ---------------------------------------------------------------------------
# Worker function — evaluates one candidate param set
# ---------------------------------------------------------------------------

def evaluate_candidate(args: tuple) -> dict:
    """Evaluate a single candidate parameter set against all ground truth.

    This runs in a worker process. Fully self-contained, no shared state.
    """
    candidate_idx, param_array, rounds_data = args

    params = SimParams.from_array(np.array(param_array))
    scores_by_round = {}
    all_scores = []

    for rnd in rounds_data:
        round_scores = []
        for seed_data in rnd["seeds"]:
            seed_idx = seed_data["seed_index"]
            ground_truth = seed_data["ground_truth"]
            initial_state = rnd["initial_states"][seed_idx]

            # Run Monte Carlo
            height, width = initial_state.height, initial_state.width
            class_counts = np.zeros((height, width, NUM_CLASSES), dtype=np.float64)

            for run_idx in range(MC_RUNS_PER_SEED):
                sim_seed = seed_idx * 100000 + run_idx
                final_state = simulate(initial_state, params, sim_seed)
                for y in range(height):
                    for x in range(width):
                        terrain = int(final_state.grid[y, x])
                        class_idx = TERRAIN_TO_CLASS.get(terrain, 0)
                        class_counts[y, x, class_idx] += 1

            pred = class_counts / MC_RUNS_PER_SEED
            pred = postprocess(pred, initial_state)
            metrics = score_prediction(ground_truth, pred)
            round_scores.append(metrics["score"])
            all_scores.append(metrics["score"])

        scores_by_round[rnd["name"]] = np.mean(round_scores)

    return {
        "candidate_idx": candidate_idx,
        "mean_score": float(np.mean(all_scores)),
        "median_score": float(np.median(all_scores)),
        "min_score": float(np.min(all_scores)),
        "per_round": {k: float(v) for k, v in scores_by_round.items()},
        "params": param_array,
    }


# ---------------------------------------------------------------------------
# Candidate generation strategies
# ---------------------------------------------------------------------------

def generate_candidates(num_candidates: int) -> list[np.ndarray]:
    """Generate candidate parameter sets to evaluate."""
    rng = np.random.default_rng(42)
    lower, upper = SimParams.bounds_arrays()
    candidates = []

    # Always include the current default
    candidates.append(SimParams.default().to_array())

    # Latin hypercube sampling for better coverage than pure random
    ndim = SimParams.ndim()
    remaining = num_candidates - 1

    # Stratified random: divide each dimension into `remaining` equal strata
    for dim in range(ndim):
        perm = rng.permutation(remaining)
        # This will be used to build LHS below

    # Simple LHS implementation
    lhs = np.zeros((remaining, ndim))
    for dim in range(ndim):
        perm = rng.permutation(remaining)
        for idx in range(remaining):
            lo = perm[idx] / remaining
            hi = (perm[idx] + 1) / remaining
            lhs[idx, dim] = rng.uniform(lo, hi)

    # Scale to parameter bounds
    for idx in range(remaining):
        param_array = lower + lhs[idx] * (upper - lower)
        candidates.append(param_array)

    # Also add perturbations of the default (local search)
    default_arr = SimParams.default().to_array()
    num_local = min(remaining // 4, 200)
    for _ in range(num_local):
        noise = rng.normal(0, 0.1, size=ndim)
        perturbed = default_arr + noise * (upper - lower)
        perturbed = np.clip(perturbed, lower, upper)
        candidates.append(perturbed)

    return candidates


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Parallel parameter grid search")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel workers")
    parser.add_argument("--candidates", type=int, default=200, help="Number of candidate param sets")
    parser.add_argument("--mc-runs", type=int, default=30, help="MC runs per seed per candidate")
    parser.add_argument("--top", type=int, default=20, help="Number of top results to show")
    args = parser.parse_args()

    global MC_RUNS_PER_SEED
    MC_RUNS_PER_SEED = args.mc_runs

    print(f"Loading ground truth data...")
    rounds_data = load_all_rounds()
    total_seeds = sum(len(r["seeds"]) for r in rounds_data)
    print(f"Loaded {len(rounds_data)} rounds, {total_seeds} seeds")

    print(f"Generating {args.candidates} candidates...")
    candidates = generate_candidates(args.candidates)
    print(f"Generated {len(candidates)} candidates (including local perturbations)")

    # Prepare serializable round data for workers
    # Workers need initial_states which contain numpy arrays — these serialize fine with fork
    print(f"\nStarting grid search with {args.workers} workers...")
    print(f"  {len(candidates)} candidates × {total_seeds} seeds × {args.mc_runs} MC runs")
    print(f"  = {len(candidates) * total_seeds * args.mc_runs} total simulations")
    print()

    work_items = [
        (idx, candidate.tolist(), rounds_data)
        for idx, candidate in enumerate(candidates)
    ]

    results = []
    t0 = time.time()
    completed = 0

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(evaluate_candidate, item): item[0]
            for item in work_items
        }

        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
                completed += 1

                if completed % 10 == 0 or completed == len(candidates):
                    elapsed = time.time() - t0
                    rate = completed / elapsed
                    eta = (len(candidates) - completed) / rate if rate > 0 else 0
                    best_so_far = max(results, key=lambda r: r["mean_score"])
                    print(
                        f"  [{completed}/{len(candidates)}] "
                        f"{elapsed:.0f}s elapsed, {eta:.0f}s ETA | "
                        f"best mean={best_so_far['mean_score']:.1f}"
                    )
            except Exception as exc:
                print(f"  Candidate failed: {exc}")

    elapsed = time.time() - t0
    print(f"\nCompleted in {elapsed:.0f}s ({elapsed/60:.1f} min)")

    # Sort by mean score
    results.sort(key=lambda r: r["mean_score"], reverse=True)

    # Show top results
    print(f"\n{'='*70}")
    print(f"TOP {args.top} RESULTS")
    print(f"{'='*70}")

    for rank, result in enumerate(results[:args.top], 1):
        print(f"\n#{rank}: mean={result['mean_score']:.1f}, median={result['median_score']:.1f}, min={result['min_score']:.1f}")
        for rnd_name, rnd_score in sorted(result["per_round"].items()):
            print(f"  {rnd_name}: {rnd_score:.1f}")

    # Save best params
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # Save full results
    results_file = RESULTS_DIR / f"search_{timestamp}.json"
    serializable_results = []
    for result in results[:100]:  # Top 100
        serializable_results.append({
            "mean_score": result["mean_score"],
            "median_score": result["median_score"],
            "min_score": result["min_score"],
            "per_round": result["per_round"],
            "params": result["params"],
        })
    results_file.write_text(json.dumps(serializable_results, indent=2))
    print(f"\nResults saved to {results_file}")

    # Save best params in the format the orchestrator can load
    best = results[0]
    best_params_file = DATA_DIR / "best_params.json"
    best_params_file.write_text(json.dumps({
        "params": best["params"],
        "score": best["mean_score"],
        "param_names": SimParams.param_names(),
        "source": "grid_search",
        "timestamp": timestamp,
    }, indent=2))
    print(f"Best params saved to {best_params_file}")
    print(f"Best mean score: {best['mean_score']:.1f}")

    # Print the best params for reference
    best_sim = SimParams.from_array(np.array(best["params"]))
    print(f"\nBest params:")
    for name in SimParams.param_names():
        print(f"  {name}: {getattr(best_sim, name):.4f}")


if __name__ == "__main__":
    main()
