"""CMA-ES powered parameter search — much smarter than random grid search.

Uses the Rust backend for fast evaluation, CMA-ES for intelligent exploration.
Finds good params in ~5,000 evaluations instead of 50,000.

Usage:
    python scripts/smart_search.py --mc-runs 50 --max-evals 5000 --popsize 100
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cma
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import island_sim_core
from src.simulator.params import SimParams
from src.types import WorldState

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "rounds"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "data" / "grid_search"


def load_all_rounds():
    """Load all rounds with ground truth."""
    rounds = []
    for round_dir in sorted(DATA_DIR.iterdir()):
        if not round_dir.is_dir() or not (round_dir / "detail.json").exists():
            continue

        detail = json.loads((round_dir / "detail.json").read_text())
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

            state = initial_states[seed_idx]
            gt_array = np.array(gt, dtype=np.float64)
            seeds.append((state.grid, state.settlements, gt_array))

        if seeds:
            rounds.append(seeds)

    return rounds


def main():
    parser = argparse.ArgumentParser(description="CMA-ES parameter search")
    parser.add_argument("--mc-runs", type=int, default=50)
    parser.add_argument("--max-evals", type=int, default=5000)
    parser.add_argument("--popsize", type=int, default=100)
    parser.add_argument("--sigma", type=float, default=0.3)
    parser.add_argument("--top", type=int, default=10)
    args = parser.parse_args()

    print("Loading ground truth data...")
    rounds_data = load_all_rounds()
    total_seeds = sum(len(r) for r in rounds_data)
    print(f"Loaded {len(rounds_data)} rounds, {total_seeds} seeds")
    print(f"CMA-ES: popsize={args.popsize}, max_evals={args.max_evals}, mc_runs={args.mc_runs}")
    print()

    lower, upper = SimParams.bounds_arrays()
    ndim = SimParams.ndim()

    # Start from current best params if available
    best_params_file = DATA_DIR / "best_params.json"
    if best_params_file.exists():
        data = json.loads(best_params_file.read_text())
        if len(data["params"]) == ndim:
            x0_raw = np.array(data["params"])
            print(f"Starting from saved params (score={data.get('score', '?')})")
        else:
            x0_raw = SimParams.default().to_array()
            print("Saved params have wrong dimension, using defaults")
    else:
        x0_raw = SimParams.default().to_array()
        print("No saved params, using defaults")

    # Normalize to [0, 1] for CMA-ES
    x0 = (x0_raw - lower) / (upper - lower)

    eval_count = 0
    best_score = -float("inf")
    best_params_arr = x0_raw.copy()
    all_results = []

    t0 = time.time()

    # Import once outside the loop
    from src.prediction.postprocess import postprocess
    from src.types import WorldState as WS
    from src.scoring import score_prediction

    def objective(x_normalized):
        nonlocal eval_count, best_score, best_params_arr

        x_raw = lower + np.clip(x_normalized, 0, 1) * (upper - lower)
        param_list = x_raw.tolist()

        scores = []
        for round_seeds in rounds_data:
            for grid, settlements, gt in round_seeds:
                pred = island_sim_core.generate_prediction(
                    grid, settlements, np.array(param_list), args.mc_runs, eval_count * 1000
                )
                pred = np.array(pred)

                state = WS(grid=grid, settlements=settlements,
                          width=grid.shape[1], height=grid.shape[0])
                pred_p = postprocess(pred, state)

                score = score_prediction(gt, pred_p)
                scores.append(score["score"])

        mean_score = np.mean(scores)
        eval_count += 1

        if mean_score > best_score:
            best_score = mean_score
            best_params_arr = x_raw.copy()
            all_results.append({
                "mean_score": float(mean_score),
                "params": x_raw.tolist(),
            })

        if eval_count % 10 == 0:
            elapsed = time.time() - t0
            print(f"  [{eval_count}/{args.max_evals}] {elapsed:.0f}s | best={best_score:.1f} | current={mean_score:.1f}")

        return -mean_score  # CMA-ES minimizes

    # Run CMA-ES
    opts = {
        "maxfevals": args.max_evals,
        "popsize": args.popsize,
        "bounds": [[0.0] * ndim, [1.0] * ndim],
        "verbose": -1,
        "seed": 42,
        "tolfun": 1e-4,
    }

    es = cma.CMAEvolutionStrategy(x0.tolist(), args.sigma, opts)

    while not es.stop():
        solutions = es.ask()
        fitnesses = [objective(np.array(x)) for x in solutions]
        es.tell(solutions, fitnesses)

        if eval_count >= args.max_evals:
            break

    elapsed = time.time() - t0
    print(f"\nCompleted in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"Total evaluations: {eval_count}")
    print(f"Best mean score: {best_score:.1f}")

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # Sort all results
    all_results.sort(key=lambda r: r["mean_score"], reverse=True)

    results_file = RESULTS_DIR / f"smart_search_{timestamp}.json"
    results_file.write_text(json.dumps(all_results[:args.top], indent=2))
    print(f"Results saved to {results_file}")

    # Save best params
    best_params_file.write_text(json.dumps({
        "params": best_params_arr.tolist(),
        "score": float(best_score),
        "param_names": SimParams.param_names(),
        "source": "smart_search_cmaes",
        "timestamp": timestamp,
    }, indent=2))
    print(f"Best params saved to {best_params_file}")

    best_sim = SimParams.from_array(best_params_arr)
    print(f"\nBest params:")
    for name in SimParams.param_names():
        print(f"  {name}: {getattr(best_sim, name):.4f}")


if __name__ == "__main__":
    main()
