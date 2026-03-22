"""
Astar Island — Post-round analysis

After a round completes, compare your predictions against ground truth.
Shows where you were wrong and why.

Usage:
    python analyze.py <JWT_TOKEN>                  # analyze latest completed round
    python analyze.py <JWT_TOKEN> --round <id>     # specific round
"""

import sys
import argparse
import numpy as np
import requests
from astar import make_session, BASE, TERRAIN, N_CLASSES

TERRAIN_SHORT = ["Emp", "Set", "Por", "Rui", "For", "Mnt"]


def get_completed_rounds(session):
    rounds = session.get(f"{BASE}/astar-island/rounds").json()
    return [r for r in rounds if r["status"] == "completed"]


def get_analysis(session, round_id: str, seed_index: int) -> dict:
    resp = session.get(f"{BASE}/astar-island/analysis/{round_id}/{seed_index}")
    resp.raise_for_status()
    return resp.json()


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """KL(p || q) — ground truth vs prediction."""
    eps = 1e-10
    return float(np.sum(p * np.log((p + eps) / (q + eps))))


def entropy(p: np.ndarray) -> float:
    eps = 1e-10
    return float(-np.sum(p * np.log(p + eps)))


def analyze_seed(ground_truth: np.ndarray, prediction: np.ndarray,
                 seed_idx: int):
    """Print per-class error breakdown and worst cells."""
    height, width = ground_truth.shape[:2]

    cell_kl = np.zeros((height, width))
    cell_entropy = np.zeros((height, width))

    for y in range(height):
        for x in range(width):
            p = ground_truth[y, x]
            q = prediction[y, x]
            cell_kl[y, x] = kl_divergence(p, q)
            cell_entropy[y, x] = entropy(p)

    # Only care about dynamic cells (entropy > 0.05)
    dynamic = cell_entropy > 0.05
    n_dynamic = dynamic.sum()

    weighted_kl = (cell_entropy * cell_kl).sum() / cell_entropy.sum()
    score = max(0, min(100, 100 * np.exp(-3 * weighted_kl)))

    print(f"\n=== Seed {seed_idx} ===")
    print(f"  Dynamic cells: {n_dynamic}/{height*width}")
    print(f"  Weighted KL:   {weighted_kl:.4f}")
    print(f"  Estimated score: {score:.1f}/100")

    # Per-class accuracy: how often was the argmax right?
    gt_argmax = np.argmax(ground_truth, axis=2)
    pred_argmax = np.argmax(prediction, axis=2)
    correct = (gt_argmax == pred_argmax)
    print(f"  Argmax accuracy: {correct.mean()*100:.1f}%")

    # Worst 5 cells
    flat_kl = cell_kl.flatten()
    worst_idx = np.argsort(flat_kl)[-5:][::-1]
    print(f"\n  Top 5 worst cells (by KL):")
    print(f"  {'(y,x)':10} {'entropy':>8} {'KL':>8} {'GT argmax':>12} {'Pred argmax':>12}")
    for idx in worst_idx:
        y, x = divmod(int(idx), width)
        gt_cls = TERRAIN[int(gt_argmax[y, x])]
        pr_cls = TERRAIN[int(pred_argmax[y, x])]
        print(f"  ({y:2d},{x:2d})    "
              f"{cell_entropy[y,x]:8.3f} {cell_kl[y,x]:8.3f} "
              f"{gt_cls:>12} {pr_cls:>12}")

    # Per-class breakdown
    print(f"\n  Per-class breakdown (dynamic cells only):")
    print(f"  {'Class':12} {'GT cells':>8} {'KL mean':>8} {'Accuracy':>9}")
    for cls_i, cls_name in enumerate(TERRAIN):
        gt_this = gt_argmax == cls_i
        mask = dynamic & gt_this
        if mask.sum() == 0:
            continue
        kl_mean = cell_kl[mask].mean()
        acc = correct[mask].mean() * 100
        print(f"  {cls_name:12} {mask.sum():8d} {kl_mean:8.3f} {acc:8.1f}%")

    return score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("token")
    parser.add_argument("--round", default=None, help="Round ID (default: latest completed)")
    args = parser.parse_args()

    session = make_session(args.token)

    if args.round:
        round_id = args.round
        detail = session.get(f"{BASE}/astar-island/rounds/{round_id}").json()
    else:
        completed = get_completed_rounds(session)
        if not completed:
            print("No completed rounds found.")
            sys.exit(1)
        # Most recent
        detail = completed[-1]
        round_id = detail["id"]

    seeds = detail.get("seeds_count", 5)
    print(f"Analyzing round {detail.get('round_number')} — {seeds} seeds")

    seed_scores = []
    for seed_idx in range(seeds):
        try:
            data = get_analysis(session, round_id, seed_idx)
            gt = np.array(data["ground_truth"])
            pred = np.array(data["prediction"])
            score = analyze_seed(gt, pred, seed_idx)
            seed_scores.append(score)
        except Exception as e:
            print(f"  Seed {seed_idx}: error — {e}")

    if seed_scores:
        avg = np.mean(seed_scores)
        print(f"\nEstimated round score: {avg:.1f}/100")
        print("\nNext steps:")
        print("  - Run again next round to see if score improves")
        print("  - Worst cells above show where the model fails")
        print("  - High entropy + high KL = cells we're confident but wrong about")


if __name__ == "__main__":
    main()
