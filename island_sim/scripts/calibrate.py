"""Calibrate our simulator against ground truth from completed rounds.

Runs Monte Carlo predictions with default params on all fetched rounds,
compares to ground truth, and reports where we're accurate vs biased.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.constants import NUM_CLASSES, TERRAIN_TO_CLASS, OCEAN, MOUNTAIN
from src.prediction.hybrid import hybrid_prediction
from src.prediction.monte_carlo import generate_prediction
from src.prediction.observation_model import aggregate_observations, empirical_distribution
from src.prediction.postprocess import postprocess
from src.scoring import score_prediction, entropy, kl_divergence
from src.simulator.params import SimParams
from src.types import Observation, WorldState

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "rounds"
NUM_MC_RUNS = 20  # Per seed — low for fast iteration, increase once sim is tuned


def load_round(round_dir: Path) -> dict | None:
    """Load a round's detail and analysis data."""
    detail_path = round_dir / "detail.json"
    if not detail_path.exists():
        return None

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
            "server_score": analysis.get("score"),
        })

    if not seeds:
        return None

    return {
        "round_dir": round_dir,
        "detail": detail,
        "width": width,
        "height": height,
        "initial_states": initial_states,
        "seeds": seeds,
    }


def calibrate_round(
    round_data: dict, params: SimParams, num_runs: int
) -> list[dict]:
    """Run our simulator on one round and compare to ground truth."""
    results = []

    for seed_data in round_data["seeds"]:
        seed_idx = seed_data["seed_index"]
        ground_truth = seed_data["ground_truth"]
        initial_state = round_data["initial_states"][seed_idx]

        t0 = time.time()
        raw_pred = generate_prediction(
            initial_state, params, num_runs=num_runs, base_seed=seed_idx * 100000
        )
        pred = postprocess(raw_pred, initial_state)
        elapsed = time.time() - t0

        # Score against ground truth
        metrics = score_prediction(ground_truth, pred)
        metrics["seed_index"] = seed_idx
        metrics["server_score"] = seed_data["server_score"]
        metrics["elapsed_sec"] = round(elapsed, 1)

        # Per-class analysis: where are we most wrong?
        class_kl = _per_class_kl(ground_truth, pred, initial_state)
        metrics["class_kl"] = class_kl

        results.append(metrics)

    return results


def _per_class_kl(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    initial_state: WorldState,
) -> dict[str, float]:
    """Compute average KL contribution per class on dynamic cells."""
    class_names = ["empty", "settlement", "port", "ruin", "forest", "mountain"]
    cell_entropy = entropy(ground_truth)
    dynamic_mask = cell_entropy > 0.01

    result = {}
    for cls_idx, cls_name in enumerate(class_names):
        gt_cls = np.clip(ground_truth[..., cls_idx], 1e-15, 1.0)
        pred_cls = np.clip(prediction[..., cls_idx], 1e-15, 1.0)
        # Per-cell KL contribution from this class
        kl_contrib = gt_cls * np.log(gt_cls / pred_cls)
        if np.sum(dynamic_mask) > 0:
            result[cls_name] = float(np.mean(kl_contrib[dynamic_mask]))
        else:
            result[cls_name] = 0.0

    return result


def _per_class_bias(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    initial_state: WorldState,
) -> dict[str, float]:
    """Mean (pred - gt) per class on dynamic cells. Positive = we over-predict."""
    class_names = ["empty", "settlement", "port", "ruin", "forest", "mountain"]
    cell_entropy = entropy(ground_truth)
    dynamic_mask = cell_entropy > 0.01

    result = {}
    for cls_idx, cls_name in enumerate(class_names):
        diff = prediction[..., cls_idx] - ground_truth[..., cls_idx]
        if np.sum(dynamic_mask) > 0:
            result[cls_name] = float(np.mean(diff[dynamic_mask]))
        else:
            result[cls_name] = 0.0
    return result


def main() -> None:
    if not DATA_DIR.exists():
        print("No data directory. Run scripts/fetch_history.py first.")
        sys.exit(1)

    params = SimParams.default()
    print(f"Calibrating with default params, {NUM_MC_RUNS} MC runs per seed\n")

    all_scores = []
    all_class_kl: dict[str, list[float]] = {}
    all_biases: dict[str, list[float]] = {}

    for round_dir in sorted(DATA_DIR.iterdir()):
        if not round_dir.is_dir():
            continue

        round_data = load_round(round_dir)
        if round_data is None:
            continue

        round_name = round_dir.name
        print(f"=== {round_name} ({round_data['width']}x{round_data['height']}) ===")

        results = calibrate_round(round_data, params, NUM_MC_RUNS)

        for res in results:
            seed = res["seed_index"]
            our_score = res["score"]
            server = res.get("server_score")
            dyn = res["num_dynamic_cells"]
            total = res["total_cells"]
            elapsed = res["elapsed_sec"]

            server_str = f"{server:.1f}" if server is not None else "n/a"
            print(
                f"  Seed {seed}: our_score={our_score:.1f}, "
                f"server_score={server_str}, "
                f"dynamic={dyn}/{total}, "
                f"weighted_kl={res['weighted_kl']:.4f}, "
                f"time={elapsed}s"
            )

            all_scores.append(our_score)

            for cls_name, kl_val in res["class_kl"].items():
                all_class_kl.setdefault(cls_name, []).append(kl_val)

        # Compute bias for this round too
        for seed_data in round_data["seeds"]:
            seed_idx = seed_data["seed_index"]
            gt = seed_data["ground_truth"]
            initial_state = round_data["initial_states"][seed_idx]
            raw_pred = generate_prediction(
                initial_state, params, num_runs=NUM_MC_RUNS, base_seed=seed_idx * 100000 + 999
            )
            pred = postprocess(raw_pred, initial_state)
            bias = _per_class_bias(gt, pred, initial_state)
            for cls_name, bias_val in bias.items():
                all_biases.setdefault(cls_name, []).append(bias_val)

        print()

    if not all_scores:
        print("No scores computed.")
        return

    print("=" * 60)
    print(f"OVERALL: mean score = {np.mean(all_scores):.1f}, "
          f"median = {np.median(all_scores):.1f}, "
          f"min = {np.min(all_scores):.1f}, max = {np.max(all_scores):.1f}")

    print("\nPer-class KL contribution (higher = we're worse at predicting this class):")
    for cls_name in ["empty", "settlement", "port", "ruin", "forest", "mountain"]:
        vals = all_class_kl.get(cls_name, [])
        if vals:
            print(f"  {cls_name:12s}: mean_kl={np.mean(vals):.4f}")

    print("\nPer-class bias (positive = we over-predict, negative = under-predict):")
    for cls_name in ["empty", "settlement", "port", "ruin", "forest", "mountain"]:
        vals = all_biases.get(cls_name, [])
        if vals:
            mean_bias = np.mean(vals)
            sign = "+" if mean_bias >= 0 else ""
            print(f"  {cls_name:12s}: {sign}{mean_bias:.4f}")


def _simulate_observations_from_gt(
    ground_truth: np.ndarray,
    seed_index: int,
    num_queries: int,
    height: int,
    width: int,
) -> list[Observation]:
    """Simulate viewport observations by sampling from the ground truth.

    This mimics what the API would return: for each query, we sample one
    stochastic outcome per cell from the GT distribution.
    """
    rng = np.random.default_rng(seed_index * 1000 + 7)
    observations = []
    vp_size = 15

    # Place viewports in a grid pattern
    positions = []
    for vy in range(0, height, vp_size - 2):
        for vx in range(0, width, vp_size - 2):
            positions.append((vx, vy))
    rng.shuffle(positions)

    for query_idx in range(min(num_queries, len(positions))):
        vx, vy = positions[query_idx]
        vw = min(vp_size, width - vx)
        vh = min(vp_size, height - vy)

        # Sample one outcome per cell from GT distribution
        grid = np.zeros((vh, vw), dtype=np.int32)
        # Reverse mapping: class index -> terrain code
        class_to_terrain = [0, 1, 2, 3, 4, 5]  # empty, settlement, port, ruin, forest, mountain
        for ly in range(vh):
            for lx in range(vw):
                probs = ground_truth[vy + ly, vx + lx]
                cls = rng.choice(NUM_CLASSES, p=probs)
                grid[ly, lx] = class_to_terrain[cls]

        observations.append(Observation(
            seed_index=seed_index,
            viewport_x=vx,
            viewport_y=vy,
            viewport_w=vw,
            viewport_h=vh,
            grid=grid,
            settlements=[],
        ))

    return observations


def main_hybrid() -> None:
    """Test the hybrid approach by simulating observations from ground truth."""
    if not DATA_DIR.exists():
        print("No data directory. Run scripts/fetch_history.py first.")
        sys.exit(1)

    params = SimParams.default()

    # Test with different observation counts to see diminishing returns
    obs_counts_to_test = [2, 5, 10]

    for n_obs in obs_counts_to_test:
        print(f"\n{'='*60}")
        print(f"HYBRID with {n_obs} observations per seed, {NUM_MC_RUNS} MC runs")
        print(f"{'='*60}\n")

        all_scores = []

        for round_dir in sorted(DATA_DIR.iterdir()):
            if not round_dir.is_dir():
                continue

            round_data = load_round(round_dir)
            if round_data is None:
                continue

            round_name = round_dir.name
            print(f"=== {round_name} ===")

            for seed_data in round_data["seeds"]:
                seed_idx = seed_data["seed_index"]
                gt = seed_data["ground_truth"]
                initial_state = round_data["initial_states"][seed_idx]

                # Simulate observations from GT
                fake_obs = _simulate_observations_from_gt(
                    gt, seed_idx, n_obs,
                    round_data["height"], round_data["width"],
                )

                t0 = time.time()
                pred = hybrid_prediction(
                    initial_state, seed_idx, fake_obs, params,
                    num_mc_runs=NUM_MC_RUNS,
                    mc_base_seed=seed_idx * 100000,
                )
                elapsed = time.time() - t0

                metrics = score_prediction(gt, pred)
                server = seed_data.get("server_score")
                server_str = f"{server:.1f}" if server is not None else "n/a"
                print(
                    f"  Seed {seed_idx}: hybrid={metrics['score']:.1f}, "
                    f"server={server_str}, time={elapsed:.1f}s"
                )
                all_scores.append(metrics["score"])

            print()

        if all_scores:
            print(f"OVERALL ({n_obs} obs): mean={np.mean(all_scores):.1f}, "
                  f"median={np.median(all_scores):.1f}, "
                  f"min={np.min(all_scores):.1f}, max={np.max(all_scores):.1f}")


if __name__ == "__main__":
    import sys as _sys
    if len(_sys.argv) > 1 and _sys.argv[1] == "--hybrid":
        main_hybrid()
    else:
        main()
