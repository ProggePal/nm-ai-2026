"""Visualize predictions vs ground truth for any round.

Usage:
    python scripts/visualize.py                    # Latest scored round
    python scripts/visualize.py --round 14         # Specific round
    python scripts/visualize.py --round 14 --seed 0  # Specific seed
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.constants import NUM_CLASSES

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "rounds"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "visualizations"

# Terrain class colors (matching the webapp)
CLASS_COLORS = {
    0: "#D4C89A",  # Empty/Plains/Ocean — sand
    1: "#D4802A",  # Settlement — orange
    2: "#6B7FB5",  # Port — blue-gray
    3: "#8B3A3A",  # Ruin — dark red
    4: "#2D7A2D",  # Forest — green
    5: "#6B6B6B",  # Mountain — gray
}

CLASS_NAMES = ["Empty", "Settlement", "Port", "Ruin", "Forest", "Mountain"]


def argmax_grid(tensor):
    """Convert (H, W, 6) probability tensor to argmax class grid."""
    return np.argmax(tensor, axis=-1)


def confidence_grid(tensor):
    """Convert (H, W, 6) to max probability per cell."""
    return np.max(tensor, axis=-1)


def render_class_grid(grid, ax, title, show_confidence=None):
    """Render a class-index grid as a colored image."""
    h, w = grid.shape
    rgb = np.zeros((h, w, 3))

    for cls_idx, hex_color in CLASS_COLORS.items():
        r, g, b = mcolors.hex2color(hex_color)
        mask = grid == cls_idx
        rgb[mask] = [r, g, b]

    if show_confidence is not None:
        # Dim uncertain cells
        alpha = 0.3 + 0.7 * show_confidence
        rgb = rgb * alpha[:, :, np.newaxis]

    ax.imshow(rgb, interpolation="nearest")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])


def render_heatmap(data, ax, title, cmap="RdBu_r", vmin=None, vmax=None):
    """Render a heatmap."""
    im = ax.imshow(data, cmap=cmap, interpolation="nearest", vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def render_class_comparison(gt, pred, ax, class_idx, class_name):
    """Show GT vs prediction for a single class as side-by-side heatmaps."""
    diff = pred[:, :, class_idx] - gt[:, :, class_idx]
    vmax = max(abs(diff.min()), abs(diff.max()), 0.1)
    im = ax.imshow(diff, cmap="RdBu_r", interpolation="nearest", vmin=-vmax, vmax=vmax)
    ax.set_title(f"{class_name} error", fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def visualize_seed(round_num, seed_idx, detail, analysis):
    """Generate full visualization for one seed."""
    gt = np.array(analysis["ground_truth"])
    init_grid = np.array(detail["initial_states"][seed_idx]["grid"])
    score = analysis.get("score", "?")

    has_pred = analysis.get("prediction") is not None
    if has_pred:
        pred = np.array(analysis["prediction"])
    else:
        pred = None

    # --- Figure 1: Overview ---
    ncols = 4 if has_pred else 2
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 5))
    fig.suptitle(f"Round {round_num}, Seed {seed_idx + 1} — Score: {score}",
                 fontsize=14, fontweight="bold", y=1.02)

    # Initial map
    from src.constants import TERRAIN_TO_CLASS
    init_classes = np.vectorize(lambda t: TERRAIN_TO_CLASS.get(t, 0))(init_grid)
    render_class_grid(init_classes, axes[0], "Initial Map")

    # GT argmax
    gt_argmax = argmax_grid(gt)
    gt_conf = confidence_grid(gt)
    render_class_grid(gt_argmax, axes[1], "Ground Truth", show_confidence=gt_conf)

    if has_pred:
        # Our prediction argmax
        pred_argmax = argmax_grid(pred)
        pred_conf = confidence_grid(pred)
        render_class_grid(pred_argmax, axes[2], "Our Prediction", show_confidence=pred_conf)

        # Error map: cells where argmax differs
        error = (pred_argmax != gt_argmax).astype(float)
        # Weight by GT entropy (only show errors on dynamic cells)
        from src.scoring import entropy
        cell_entropy = entropy(gt)
        error_weighted = error * (cell_entropy / max(cell_entropy.max(), 1e-10))
        render_heatmap(error_weighted, axes[3], "Weighted Errors", cmap="Reds", vmin=0)

    # Legend
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor=CLASS_COLORS[i], label=CLASS_NAMES[i])
        for i in range(NUM_CLASSES)
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=6,
              fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    return fig


def visualize_class_errors(round_num, seed_idx, detail, analysis):
    """Generate per-class error heatmaps."""
    gt = np.array(analysis["ground_truth"])
    pred = analysis.get("prediction")
    if pred is None:
        return None
    pred = np.array(pred)
    score = analysis.get("score", "?")

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f"Round {round_num}, Seed {seed_idx + 1} — Per-Class Errors (score: {score})",
                 fontsize=14, fontweight="bold")

    for cls_idx in range(NUM_CLASSES):
        ax = axes[cls_idx // 3, cls_idx % 3]
        render_class_comparison(gt, pred, ax, cls_idx, CLASS_NAMES[cls_idx])

    plt.tight_layout()
    return fig


def visualize_probability_bars(round_num, seed_idx, detail, analysis):
    """Bar chart comparing mean probabilities per class."""
    gt = np.array(analysis["ground_truth"])
    pred = analysis.get("prediction")
    if pred is None:
        return None
    pred = np.array(pred)
    score = analysis.get("score", "?")

    init_grid = np.array(detail["initial_states"][seed_idx]["grid"])
    land_mask = ~np.isin(init_grid, [10, 5])

    gt_means = gt[land_mask].mean(axis=0)
    pred_means = pred[land_mask].mean(axis=0)

    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(NUM_CLASSES)
    width = 0.35

    bars_gt = ax.bar(x - width/2, gt_means, width, label="Ground Truth",
                     color=[CLASS_COLORS[i] for i in range(NUM_CLASSES)], edgecolor="black")
    bars_pred = ax.bar(x + width/2, pred_means, width, label="Our Prediction",
                       color=[CLASS_COLORS[i] for i in range(NUM_CLASSES)], edgecolor="black",
                       alpha=0.6, hatch="//")

    ax.set_ylabel("Mean probability on land cells")
    ax.set_title(f"Round {round_num}, Seed {seed_idx + 1} — Class Probabilities (score: {score})")
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES)
    ax.legend()

    # Add value labels
    for bar, val in zip(bars_gt, gt_means):
        if val > 0.005:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                   f"{val:.1%}", ha="center", va="bottom", fontsize=8)
    for bar, val in zip(bars_pred, pred_means):
        if val > 0.005:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                   f"{val:.1%}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description="Visualize predictions vs ground truth")
    parser.add_argument("--round", type=int, default=None, help="Round number (default: latest)")
    parser.add_argument("--seed", type=int, default=None, help="Seed index 0-4 (default: all)")
    parser.add_argument("--no-show", action="store_true", help="Save only, don't display")
    args = parser.parse_args()

    # Find the round
    if args.round is not None:
        round_dir = DATA_DIR / f"round_{args.round:03d}"
    else:
        # Find latest round with scores
        round_dirs = sorted(DATA_DIR.iterdir())
        round_dir = None
        for d in reversed(round_dirs):
            if d.is_dir() and (d / "analysis_seed_0.json").exists():
                analysis = json.loads((d / "analysis_seed_0.json").read_text())
                if analysis.get("score") is not None:
                    round_dir = d
                    break
        if round_dir is None:
            print("No scored rounds found")
            sys.exit(1)

    round_num = int(round_dir.name.split("_")[1])
    print(f"Visualizing round {round_num}")

    detail = json.loads((round_dir / "detail.json").read_text())

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    seeds = [args.seed] if args.seed is not None else range(5)

    for seed_idx in seeds:
        analysis_path = round_dir / f"analysis_seed_{seed_idx}.json"
        if not analysis_path.exists():
            print(f"  Seed {seed_idx}: no analysis data")
            continue

        analysis = json.loads(analysis_path.read_text())
        score = analysis.get("score", "n/a")
        print(f"  Seed {seed_idx}: score={score}")

        # Overview
        fig1 = visualize_seed(round_num, seed_idx, detail, analysis)
        fig1.savefig(OUTPUT_DIR / f"round_{round_num:03d}_seed_{seed_idx}_overview.png",
                    dpi=150, bbox_inches="tight")

        # Per-class errors
        fig2 = visualize_class_errors(round_num, seed_idx, detail, analysis)
        if fig2:
            fig2.savefig(OUTPUT_DIR / f"round_{round_num:03d}_seed_{seed_idx}_class_errors.png",
                        dpi=150, bbox_inches="tight")

        # Probability bars
        fig3 = visualize_probability_bars(round_num, seed_idx, detail, analysis)
        if fig3:
            fig3.savefig(OUTPUT_DIR / f"round_{round_num:03d}_seed_{seed_idx}_probabilities.png",
                        dpi=150, bbox_inches="tight")

        if not args.no_show:
            plt.show()
        else:
            plt.close("all")

    print(f"\nSaved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
