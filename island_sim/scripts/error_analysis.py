"""
Systematic error analysis across all rounds and seeds.

Computes per-class KL divergence, miss rates, probability calibration,
and spatial error patterns.

Class indices: 0=Empty, 1=Settlement, 2=Port, 3=Ruin, 4=Forest, 5=Mountain
Terrain codes in initial_grid: 0=Empty, 1=Settlement, 2=Port, 3=Ruin,
                                4=Forest, 5=Mountain, 10=Ocean, 11=Plains
"""

import json
import os
import sys
from collections import defaultdict

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "rounds")
CLASS_NAMES = ["Empty", "Settlement", "Port", "Ruin", "Forest", "Mountain"]
NUM_CLASSES = 6
ENTROPY_THRESHOLD = 0.01  # min ground-truth entropy to be "dynamic"
EPS = 1e-12               # numerical stability for log

TERRAIN_NAMES = {
    0: "Empty",
    1: "Settlement",
    2: "Port",
    3: "Ruin",
    4: "Forest",
    5: "Mountain",
    10: "Ocean",
    11: "Plains",
}

# Map initial terrain code → "display group" for spatial analysis
COASTAL_CODES = {10}     # Ocean
OPEN_CODES = {0, 11}     # Empty / Plains
SETTLEMENT_CODES = {1, 2}  # Settlement / Port (initially)
FOREST_CODES = {4}
MOUNTAIN_CODES = {5}
RUIN_CODES = {3}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def entropy(probs: np.ndarray) -> np.ndarray:
    """Shannon entropy along last axis, shape (...,)."""
    p = np.clip(probs, EPS, 1.0)
    return -np.sum(p * np.log(p), axis=-1)


def kl_div_per_cell(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """
    KL(gt || pred) per cell, summed over classes.
    gt, pred: (..., C).  Returns (...,).
    """
    p = np.clip(gt, EPS, 1.0)
    q = np.clip(pred, EPS, 1.0)
    return np.sum(p * (np.log(p) - np.log(q)), axis=-1)


def kl_div_per_class(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """
    Contribution of each class to KL(gt || pred) per cell.
    Returns (..., C).
    """
    p = np.clip(gt, EPS, 1.0)
    q = np.clip(pred, EPS, 1.0)
    return p * (np.log(p) - np.log(q))


def load_analysis(filepath: str):
    with open(filepath) as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Per-file analysis
# ---------------------------------------------------------------------------

def analyse_file(filepath: str, round_num: str, seed: int) -> dict | None:
    data = load_analysis(filepath)
    gt_raw = np.array(data["ground_truth"], dtype=np.float64)   # H×W×6
    pred_raw = np.array(data["prediction"], dtype=np.float64)   # H×W×6
    initial_grid = np.array(data["initial_grid"], dtype=np.int32)  # H×W
    score = data.get("score")

    # Some rounds have no prediction submitted (prediction is null/scalar)
    if pred_raw.ndim != 3 or gt_raw.ndim != 3:
        return None

    H, W, C = gt_raw.shape
    assert C == NUM_CLASSES, f"Unexpected class count {C} in {filepath}"

    # Normalise rows that don't sum to 1 (defensive)
    gt_sum = gt_raw.sum(axis=-1, keepdims=True)
    pred_sum = pred_raw.sum(axis=-1, keepdims=True)
    gt = gt_raw / np.where(gt_sum > 0, gt_sum, 1.0)
    pred = pred_raw / np.where(pred_sum > 0, pred_sum, 1.0)

    # Dynamic cells: gt entropy > threshold
    gt_entropy = entropy(gt)                        # H×W
    dynamic_mask = gt_entropy > ENTROPY_THRESHOLD   # H×W, bool

    n_dynamic = dynamic_mask.sum()
    if n_dynamic == 0:
        return None  # nothing to learn from this file

    # ---- KL divergence ----
    kl_per_cell = kl_div_per_cell(gt, pred)          # H×W
    kl_class_contrib = kl_div_per_class(gt, pred)    # H×W×C

    dyn_kl_total = kl_per_cell[dynamic_mask]         # (n_dynamic,)
    dyn_kl_class = kl_class_contrib[dynamic_mask]    # (n_dynamic, C)

    mean_kl_total = dyn_kl_total.mean()
    mean_kl_per_class = dyn_kl_class.mean(axis=0)    # (C,)

    # ---- Miss rate ----
    gt_argmax = np.argmax(gt, axis=-1)     # H×W
    pred_argmax = np.argmax(pred, axis=-1) # H×W
    mismatch = (gt_argmax != pred_argmax)  # H×W
    miss_rate_overall = mismatch[dynamic_mask].mean()

    # Per-class miss rate: among dynamic cells where GT argmax == class c
    per_class_miss = {}
    for c in range(NUM_CLASSES):
        mask_c = dynamic_mask & (gt_argmax == c)
        n_c = mask_c.sum()
        if n_c > 0:
            per_class_miss[c] = mismatch[mask_c].mean()
        else:
            per_class_miss[c] = np.nan

    # ---- Probability calibration (dynamic cells only) ----
    mean_gt_prob = gt[dynamic_mask].mean(axis=0)    # (C,)
    mean_pred_prob = pred[dynamic_mask].mean(axis=0) # (C,)

    # ---- Frequency counts (dynamic cells) ----
    gt_freq = np.bincount(gt_argmax[dynamic_mask], minlength=NUM_CLASSES) / n_dynamic
    pred_freq = np.bincount(pred_argmax[dynamic_mask], minlength=NUM_CLASSES) / n_dynamic

    # ---- Initial terrain → outcome analysis ----
    # For each initial terrain code, record GT distribution over 6 classes
    terrain_to_gt_dist = defaultdict(list)    # code → list of (C,) arrays
    terrain_to_kl = defaultdict(list)
    terrain_to_miss = defaultdict(list)

    flat_gt = gt.reshape(-1, C)
    flat_pred = pred.reshape(-1, C)
    flat_ig = initial_grid.reshape(-1)
    flat_dyn = dynamic_mask.reshape(-1)
    flat_kl = kl_per_cell.reshape(-1)
    flat_gt_argmax = gt_argmax.reshape(-1)
    flat_pred_argmax = pred_argmax.reshape(-1)
    flat_mismatch = mismatch.reshape(-1)

    for idx in range(H * W):
        code = flat_ig[idx]
        terrain_to_gt_dist[code].append(flat_gt[idx])
        if flat_dyn[idx]:
            terrain_to_kl[code].append(flat_kl[idx])
            terrain_to_miss[code].append(flat_mismatch[idx])

    # Summarise terrain outcomes
    terrain_summary = {}
    for code, dist_list in terrain_to_gt_dist.items():
        arr = np.stack(dist_list)  # (N, C)
        dyn_kls = terrain_to_kl[code]
        dyn_miss = terrain_to_miss[code]
        terrain_summary[code] = {
            "n_cells": len(dist_list),
            "n_dynamic": len(dyn_kls),
            "mean_gt_dist": arr.mean(axis=0),
            "mean_kl": np.mean(dyn_kls) if dyn_kls else np.nan,
            "miss_rate": np.mean(dyn_miss) if dyn_miss else np.nan,
        }

    # ---- Spatial: distance to nearest settlement ----
    settle_mask = np.isin(initial_grid, list(SETTLEMENT_CODES))
    settle_ys, settle_xs = np.where(settle_mask)

    # Compute min-distance to settlement for each cell
    gy, gx = np.mgrid[0:H, 0:W]
    if len(settle_ys) > 0:
        # Vectorised: (H*W) vs (n_settlements)
        gy_flat = gy.reshape(-1)
        gx_flat = gx.reshape(-1)
        dy = gy_flat[:, None] - settle_ys[None, :]
        dx = gx_flat[:, None] - settle_xs[None, :]
        dist_to_settle = np.sqrt(dy**2 + dx**2).min(axis=1).reshape(H, W)
    else:
        dist_to_settle = np.full((H, W), np.inf)

    # Bin cells by distance band (dynamic only)
    dist_bands = [(0, 2), (2, 5), (5, 10), (10, 20), (20, 999)]
    spatial_kl = {}
    spatial_miss = {}
    for lo, hi in dist_bands:
        band_mask = dynamic_mask & (dist_to_settle >= lo) & (dist_to_settle < hi)
        n_band = band_mask.sum()
        label = f"dist_{lo}_{hi}"
        if n_band > 0:
            spatial_kl[label] = kl_per_cell[band_mask].mean()
            spatial_miss[label] = mismatch[band_mask].mean()
        else:
            spatial_kl[label] = np.nan
            spatial_miss[label] = np.nan

    # ---- Coastal proximity: is the cell adjacent to ocean? ----
    ocean_mask = (initial_grid == 10).astype(np.float32)
    from scipy.ndimage import uniform_filter
    ocean_neighbor = uniform_filter(ocean_mask, size=3) > 0.05  # has ocean neighbour
    coastal_mask = ocean_neighbor & ~settle_mask

    spatial_kl["coastal"] = kl_per_cell[dynamic_mask & coastal_mask].mean() if (dynamic_mask & coastal_mask).any() else np.nan
    spatial_miss["coastal"] = mismatch[dynamic_mask & coastal_mask].mean() if (dynamic_mask & coastal_mask).any() else np.nan
    spatial_kl["inland"] = kl_per_cell[dynamic_mask & ~coastal_mask].mean() if (dynamic_mask & ~coastal_mask).any() else np.nan
    spatial_miss["inland"] = mismatch[dynamic_mask & ~coastal_mask].mean() if (dynamic_mask & ~coastal_mask).any() else np.nan

    return {
        "round": round_num,
        "seed": seed,
        "score": score,
        "n_dynamic": int(n_dynamic),
        "n_total": H * W,
        "mean_kl_total": float(mean_kl_total),
        "mean_kl_per_class": mean_kl_per_class.tolist(),
        "miss_rate_overall": float(miss_rate_overall),
        "per_class_miss": {c: float(v) for c, v in per_class_miss.items()},
        "mean_gt_prob": mean_gt_prob.tolist(),
        "mean_pred_prob": mean_pred_prob.tolist(),
        "gt_freq": gt_freq.tolist(),
        "pred_freq": pred_freq.tolist(),
        "terrain_summary": {
            int(k): {
                "n_cells": v["n_cells"],
                "n_dynamic": v["n_dynamic"],
                "mean_gt_dist": v["mean_gt_dist"].tolist(),
                "mean_kl": float(v["mean_kl"]) if not np.isnan(v["mean_kl"]) else None,
                "miss_rate": float(v["miss_rate"]) if not np.isnan(v["miss_rate"]) else None,
            }
            for k, v in terrain_summary.items()
        },
        "spatial_kl": {k: (float(v) if not np.isnan(v) else None) for k, v in spatial_kl.items()},
        "spatial_miss": {k: (float(v) if not np.isnan(v) else None) for k, v in spatial_miss.items()},
    }


# ---------------------------------------------------------------------------
# Aggregate across all files
# ---------------------------------------------------------------------------

def aggregate(results: list[dict]) -> dict:
    """Aggregate a list of per-file result dicts into global summaries."""

    def nanmean(vals):
        arr = [v for v in vals if v is not None and not np.isnan(v)]
        return np.mean(arr) if arr else np.nan

    all_kl_total = [r["mean_kl_total"] for r in results]
    all_miss = [r["miss_rate_overall"] for r in results]
    all_scores = [r["score"] for r in results if r["score"] is not None]

    kl_per_class = np.array([r["mean_kl_per_class"] for r in results])   # (N, C)
    gt_prob = np.array([r["mean_gt_prob"] for r in results])              # (N, C)
    pred_prob = np.array([r["mean_pred_prob"] for r in results])          # (N, C)
    gt_freq = np.array([r["gt_freq"] for r in results])                   # (N, C)
    pred_freq = np.array([r["pred_freq"] for r in results])               # (N, C)

    per_class_miss_agg = {}
    for c in range(NUM_CLASSES):
        vals = [r["per_class_miss"].get(c) for r in results]
        per_class_miss_agg[c] = nanmean(vals)

    # Terrain summaries
    terrain_kl = defaultdict(list)
    terrain_miss = defaultdict(list)
    terrain_gt_dist = defaultdict(list)
    terrain_n_cells = defaultdict(int)
    terrain_n_dynamic = defaultdict(int)

    for r in results:
        for code_str, ts in r["terrain_summary"].items():
            code = int(code_str)
            terrain_n_cells[code] += ts["n_cells"]
            terrain_n_dynamic[code] += ts["n_dynamic"]
            terrain_gt_dist[code].append(np.array(ts["mean_gt_dist"]))
            if ts["mean_kl"] is not None:
                terrain_kl[code].append(ts["mean_kl"])
            if ts["miss_rate"] is not None:
                terrain_miss[code].append(ts["miss_rate"])

    terrain_agg = {}
    for code in sorted(terrain_n_cells.keys()):
        dist_arr = np.stack(terrain_gt_dist[code])
        terrain_agg[code] = {
            "n_cells_total": terrain_n_cells[code],
            "n_dynamic_total": terrain_n_dynamic[code],
            "mean_gt_dist": dist_arr.mean(axis=0),
            "mean_kl": nanmean(terrain_kl[code]),
            "miss_rate": nanmean(terrain_miss[code]),
        }

    # Spatial KL / miss agg
    spatial_keys = set()
    for r in results:
        spatial_keys.update(r["spatial_kl"].keys())

    spatial_kl_agg = {}
    spatial_miss_agg = {}
    for key in sorted(spatial_keys):
        kl_vals = [r["spatial_kl"].get(key) for r in results]
        miss_vals = [r["spatial_miss"].get(key) for r in results]
        spatial_kl_agg[key] = nanmean(kl_vals)
        spatial_miss_agg[key] = nanmean(miss_vals)

    return {
        "n_files": len(results),
        "mean_score": np.mean(all_scores) if all_scores else np.nan,
        "std_score": np.std(all_scores) if all_scores else np.nan,
        "min_score": np.min(all_scores) if all_scores else np.nan,
        "max_score": np.max(all_scores) if all_scores else np.nan,
        "mean_kl_total": nanmean(all_kl_total),
        "mean_miss_overall": nanmean(all_miss),
        "mean_kl_per_class": kl_per_class.mean(axis=0),
        "mean_gt_prob": gt_prob.mean(axis=0),
        "mean_pred_prob": pred_prob.mean(axis=0),
        "mean_gt_freq": gt_freq.mean(axis=0),
        "mean_pred_freq": pred_freq.mean(axis=0),
        "per_class_miss": per_class_miss_agg,
        "terrain_agg": terrain_agg,
        "spatial_kl": spatial_kl_agg,
        "spatial_miss": spatial_miss_agg,
    }


# ---------------------------------------------------------------------------
# Pretty printer
# ---------------------------------------------------------------------------

def print_separator(char="=", width=72):
    print(char * width)


def print_section(title: str):
    print()
    print_separator()
    print(f"  {title}")
    print_separator()


def fmt(val, decimals=4):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "  N/A  "
    return f"{val:.{decimals}f}"


def print_report(agg: dict, per_file: list[dict]):
    print_section("OVERVIEW")
    print(f"  Files analysed    : {agg['n_files']}")
    print(f"  Score  mean±std   : {fmt(agg['mean_score'], 2)} ± {fmt(agg['std_score'], 2)}")
    print(f"  Score  range      : [{fmt(agg['min_score'], 2)}, {fmt(agg['max_score'], 2)}]")
    print(f"  Mean KL (dynamic) : {fmt(agg['mean_kl_total'])}")
    print(f"  Mean miss rate    : {fmt(agg['mean_miss_overall'])}")

    # ----------------------------------------------------------------
    print_section("PER-CLASS KL DIVERGENCE (mean over dynamic cells)")
    kl = agg["mean_kl_per_class"]
    total_kl = kl.sum()
    header = f"  {'Class':<12} {'Mean KL':>10} {'% of total':>12}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for c, name in enumerate(CLASS_NAMES):
        pct = (kl[c] / total_kl * 100) if total_kl > 0 else 0
        print(f"  {name:<12} {kl[c]:>10.4f} {pct:>11.1f}%")
    print(f"  {'TOTAL':<12} {total_kl:>10.4f} {'100.0':>11}%")

    # ----------------------------------------------------------------
    print_section("PER-CLASS MISS RATE (dynamic cells, GT argmax == class)")
    header = f"  {'Class':<12} {'Miss Rate':>10}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for c, name in enumerate(CLASS_NAMES):
        val = agg["per_class_miss"].get(c, np.nan)
        print(f"  {name:<12} {fmt(val):>10}")

    # ----------------------------------------------------------------
    print_section("PROBABILITY CALIBRATION (mean over dynamic cells)")
    print(f"  {'Class':<12} {'GT prob':>10} {'Pred prob':>10} {'Bias':>10}")
    print("  " + "-" * 46)
    gtp = agg["mean_gt_prob"]
    pp = agg["mean_pred_prob"]
    for c, name in enumerate(CLASS_NAMES):
        bias = pp[c] - gtp[c]
        direction = "OVER" if bias > 0.005 else ("UNDER" if bias < -0.005 else "ok")
        print(f"  {name:<12} {gtp[c]:>10.4f} {pp[c]:>10.4f} {bias:>+10.4f}  {direction}")

    # ----------------------------------------------------------------
    print_section("CLASS FREQUENCY: GT argmax vs PRED argmax (dynamic cells)")
    print(f"  {'Class':<12} {'GT freq':>10} {'Pred freq':>10} {'Diff':>10}")
    print("  " + "-" * 46)
    gf = agg["mean_gt_freq"]
    pf = agg["mean_pred_freq"]
    for c, name in enumerate(CLASS_NAMES):
        diff = pf[c] - gf[c]
        print(f"  {name:<12} {gf[c]:>10.4f} {pf[c]:>10.4f} {diff:>+10.4f}")

    # ----------------------------------------------------------------
    print_section("INITIAL TERRAIN → OUTCOME ANALYSIS")
    print("  (What GT distribution does each starting terrain lead to?)")
    print()
    for code in sorted(agg["terrain_agg"].keys()):
        tname = TERRAIN_NAMES.get(code, f"code_{code}")
        ts = agg["terrain_agg"][code]
        n_dyn = ts["n_dynamic_total"]
        n_tot = ts["n_cells_total"]
        dyn_pct = 100 * n_dyn / n_tot if n_tot > 0 else 0
        mean_kl = ts["mean_kl"]
        miss_rate = ts["miss_rate"]
        dist = ts["mean_gt_dist"]
        dominant = CLASS_NAMES[int(np.argmax(dist))]
        print(f"  [{tname:<10}]  cells={n_tot:5d}  dynamic={n_dyn:5d} ({dyn_pct:4.1f}%)  "
              f"mean_KL={fmt(mean_kl, 4)}  miss={fmt(miss_rate, 3)}  "
              f"dominant_outcome={dominant}")
        # Show full distribution if dynamic cells exist
        if n_dyn > 0:
            dist_str = "  ".join(
                f"{CLASS_NAMES[c]}={dist[c]:.3f}" for c in range(NUM_CLASSES)
            )
            print(f"    GT dist: {dist_str}")
    print()

    # ---- Special focus: settlements ----
    print_section("SETTLEMENT FATE (cells initially coded as Settlement=1 or Port=2)")
    for code in [1, 2]:
        tname = TERRAIN_NAMES[code]
        if code not in agg["terrain_agg"]:
            print(f"  No {tname} cells found.")
            continue
        ts = agg["terrain_agg"][code]
        dist = ts["mean_gt_dist"]
        print(f"\n  Initial {tname} cells: {ts['n_cells_total']} total, "
              f"{ts['n_dynamic_total']} dynamic  "
              f"(miss={fmt(ts['miss_rate'], 3)})")
        print("  GT outcome distribution:")
        for c in range(NUM_CLASSES):
            bar = "#" * int(dist[c] * 40)
            print(f"    {CLASS_NAMES[c]:<12} {dist[c]:.3f}  {bar}")

    # ---- Special focus: forest ----
    print_section("FOREST FATE (cells initially coded as Forest=4)")
    if 4 in agg["terrain_agg"]:
        ts = agg["terrain_agg"][4]
        dist = ts["mean_gt_dist"]
        print(f"  Forest cells: {ts['n_cells_total']} total, "
              f"{ts['n_dynamic_total']} dynamic  "
              f"(miss={fmt(ts['miss_rate'], 3)})")
        print("  GT outcome distribution:")
        for c in range(NUM_CLASSES):
            bar = "#" * int(dist[c] * 40)
            print(f"    {CLASS_NAMES[c]:<12} {dist[c]:.3f}  {bar}")
    else:
        print("  No Forest cells found.")

    # ----------------------------------------------------------------
    print_section("SPATIAL ERROR PATTERNS")
    print("  (Mean KL divergence and miss rate by zone, dynamic cells only)")
    print()
    ordered_keys = [
        "dist_0_2", "dist_2_5", "dist_5_10", "dist_10_20", "dist_20_999",
        "coastal", "inland",
    ]
    labels = {
        "dist_0_2": "0–2 cells from settlement",
        "dist_2_5": "2–5 cells from settlement",
        "dist_5_10": "5–10 cells from settlement",
        "dist_10_20": "10–20 cells from settlement",
        "dist_20_999": ">20 cells from settlement",
        "coastal": "Coastal (ocean-adjacent)",
        "inland": "Inland (non-coastal)",
    }
    print(f"  {'Zone':<30} {'Mean KL':>10} {'Miss rate':>10}")
    print("  " + "-" * 52)
    for key in ordered_keys:
        label = labels.get(key, key)
        kl_val = agg["spatial_kl"].get(key, np.nan)
        miss_val = agg["spatial_miss"].get(key, np.nan)
        print(f"  {label:<30} {fmt(kl_val, 4):>10} {fmt(miss_val, 3):>10}")

    # ----------------------------------------------------------------
    print_section("PER-ROUND SUMMARY")
    # Group by round
    by_round = defaultdict(list)
    for r in per_file:
        by_round[r["round"]].append(r)

    print(f"  {'Round':<10} {'Seeds':>6} {'Score':>8} {'Mean KL':>10} {'Miss':>8}")
    print("  " + "-" * 44)
    for round_id in sorted(by_round.keys()):
        files = by_round[round_id]
        scores = [f["score"] for f in files if f["score"] is not None]
        kls = [f["mean_kl_total"] for f in files]
        misses = [f["miss_rate_overall"] for f in files]
        score_str = f"{np.mean(scores):.2f}" if scores else "N/A"
        print(f"  {round_id:<10} {len(files):>6} {score_str:>8} "
              f"{np.mean(kls):>10.4f} {np.mean(misses):>8.4f}")

    # ----------------------------------------------------------------
    print_section("TOP-5 WORST FILES (by mean KL)")
    worst = sorted(per_file, key=lambda r: r["mean_kl_total"], reverse=True)[:5]
    for r in worst:
        print(f"  round={r['round']} seed={r['seed']:2d}  "
              f"KL={r['mean_kl_total']:.4f}  miss={r['miss_rate_overall']:.4f}  "
              f"score={r['score']}")

    print_section("TOP-5 BEST FILES (by mean KL)")
    best = sorted(per_file, key=lambda r: r["mean_kl_total"])[:5]
    for r in best:
        print(f"  round={r['round']} seed={r['seed']:2d}  "
              f"KL={r['mean_kl_total']:.4f}  miss={r['miss_rate_overall']:.4f}  "
              f"score={r['score']}")

    # ----------------------------------------------------------------
    print_section("KEY FINDINGS SUMMARY")
    # Find the worst class by KL
    kl = agg["mean_kl_per_class"]
    worst_class_kl = CLASS_NAMES[int(np.argmax(kl))]
    # Find the worst class by miss rate
    miss_vals = {c: v for c, v in agg["per_class_miss"].items()
                 if v is not None and not np.isnan(v)}
    worst_class_miss = CLASS_NAMES[max(miss_vals, key=lambda c: miss_vals[c])] if miss_vals else "N/A"
    # Bias direction
    bias = np.array(agg["mean_pred_prob"]) - np.array(agg["mean_gt_prob"])
    most_over = CLASS_NAMES[int(np.argmax(bias))]
    most_under = CLASS_NAMES[int(np.argmin(bias))]
    # Spatial worst zone
    spatial_kl_clean = {k: v for k, v in agg["spatial_kl"].items()
                        if v is not None and not np.isnan(v)}
    worst_zone = max(spatial_kl_clean, key=lambda k: spatial_kl_clean[k]) if spatial_kl_clean else "N/A"

    print(f"  1. Worst class by KL contribution  : {worst_class_kl}")
    print(f"  2. Worst class by miss rate        : {worst_class_miss}")
    print(f"  3. Most over-predicted class       : {most_over}  (bias={bias[CLASS_NAMES.index(most_over)]:+.4f})")
    print(f"  4. Most under-predicted class      : {most_under}  (bias={bias[CLASS_NAMES.index(most_under)]:+.4f})")
    print(f"  5. Worst spatial zone (KL)         : {worst_zone}")
    print()
    print_separator()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    results = []
    errors = []

    round_dirs = sorted(
        d for d in os.listdir(DATA_DIR)
        if d.startswith("round_") and os.path.isdir(os.path.join(DATA_DIR, d))
    )

    for round_dir in round_dirs:
        round_path = os.path.join(DATA_DIR, round_dir)
        seed = 0
        while True:
            fname = f"analysis_seed_{seed}.json"
            fpath = os.path.join(round_path, fname)
            if not os.path.exists(fpath):
                break
            try:
                res = analyse_file(fpath, round_dir, seed)
                if res is not None:
                    results.append(res)
                else:
                    print(f"  [skip] {round_dir}/seed_{seed}: no dynamic cells", file=sys.stderr)
            except Exception as exc:
                errors.append((fpath, str(exc)))
                print(f"  [ERROR] {fpath}: {exc}", file=sys.stderr)
            seed += 1

    if errors:
        print(f"\nEncountered {len(errors)} errors:", file=sys.stderr)
        for path, msg in errors:
            print(f"  {path}: {msg}", file=sys.stderr)

    if not results:
        print("No valid analysis files found. Exiting.")
        sys.exit(1)

    print(f"Loaded {len(results)} analysis files from {len(round_dirs)} rounds.")
    agg = aggregate(results)
    print_report(agg, results)


if __name__ == "__main__":
    main()
