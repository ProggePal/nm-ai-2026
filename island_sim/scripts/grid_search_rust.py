"""Grid search using fully Rust backend — ~10-50x faster than Python multiprocessing.

Usage:
    python scripts/grid_search_rust.py --candidates 50000 --mc-runs 50 --top 20
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import island_sim_core
from src.simulator.params import SimParams
from src.types import WorldState

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "rounds"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "data" / "grid_search"


def load_all_rounds():
    """Load all rounds with ground truth, returning data in the format Rust expects."""
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

            state = initial_states[seed_idx]
            gt_array = np.array(gt, dtype=np.float64)

            # Pack as (grid, settlements, ground_truth) tuple for Rust
            seeds.append((state.grid, state.settlements, gt_array))

        if seeds:
            rounds.append(seeds)

    return rounds


def main():
    parser = argparse.ArgumentParser(description="Rust-powered parallel grid search")
    parser.add_argument("--candidates", type=int, default=50000)
    parser.add_argument("--mc-runs", type=int, default=50)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for candidate generation")
    args = parser.parse_args()

    print("Loading ground truth data...")
    rounds = load_all_rounds()
    total_seeds = sum(len(r) for r in rounds)
    print(f"Loaded {len(rounds)} rounds, {total_seeds} seeds")
    print(f"Running {args.candidates} candidates × {total_seeds} seeds × {args.mc_runs} MC runs (seed={args.seed})")
    print(f"= {args.candidates * total_seeds * args.mc_runs:,} total simulations\n")

    t0 = time.time()
    results = island_sim_core.run_grid_search(rounds, args.candidates, args.mc_runs, args.top, args.seed)
    elapsed = time.time() - t0

    print(f"\nCompleted in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"\n{'='*70}")
    print(f"TOP {args.top} RESULTS")
    print(f"{'='*70}")

    round_names = [d.name for d in sorted(DATA_DIR.iterdir()) if d.is_dir() and (d / "detail.json").exists()]

    for rank, result in enumerate(results, 1):
        mean_score = result["mean_score"]
        per_round = result["per_round_scores"]
        print(f"\n#{rank}: mean={mean_score:.1f}")
        for idx, score in enumerate(per_round):
            name = round_names[idx] if idx < len(round_names) else f"round_{idx}"
            print(f"  {name}: {score:.1f}")

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    results_file = RESULTS_DIR / f"search_{timestamp}_seed{args.seed}.json"
    serializable = []
    for r in results:
        serializable.append({
            "mean_score": r["mean_score"],
            "per_round_scores": list(r["per_round_scores"]),
            "params": list(r["params"]),
        })
    results_file.write_text(json.dumps(serializable, indent=2))
    print(f"\nResults saved to {results_file}")

    # Save best params
    best = results[0]
    best_params_file = DATA_DIR / "best_params.json"
    best_sim = SimParams.from_array(np.array(best["params"]))
    best_params_file.write_text(json.dumps({
        "params": list(best["params"]),
        "named_params": best_sim.to_dict(),
        "score": best["mean_score"],
        "param_names": SimParams.param_names(),
        "source": "grid_search_rust",
        "timestamp": timestamp,
    }, indent=2))
    print(f"Best params saved to {best_params_file}")
    print(f"Best mean score: {best['mean_score']:.1f}")

    print(f"\nBest params:")
    for name in SimParams.param_names():
        print(f"  {name}: {getattr(best_sim, name):.4f}")


if __name__ == "__main__":
    main()
