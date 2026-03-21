"""
Analyze why certain rounds score high vs low.
No API calls — only reads local files.

Terrain codes in initial grid:
  0=Empty, 1=Settlement, 2=Port, 3=Ruin, 4=Forest, 5=Mountain, 10=Ocean, 11=Plains

Class indices in prediction/ground_truth tensors:
  0=Empty, 1=Settlement, 2=Port, 3=Ruin, 4=Forest, 5=Mountain
"""

import json
import math
import os
from collections import Counter

# ── config ────────────────────────────────────────────────────────────────────

DATA_ROOT = "/Users/erikankerkilbergskallevold/Develop/git/nm-ai-2026/island_sim/data/rounds"

HIGH_ROUNDS = [1, 5, 8, 9, 13]
LOW_ROUNDS  = [6, 7, 10, 14]

KNOWN_SCORES = {
    1: 72.2, 5: 72.7, 8: 71.3, 9: 82.7, 13: 85.8,
    6: 53.1, 7: 43.0, 10: 63.7, 14: 46.7,
}

TERRAIN_NAMES = {
    0: "Empty", 1: "Settlement", 2: "Port", 3: "Ruin",
    4: "Forest", 5: "Mountain", 10: "Ocean", 11: "Plains",
}
CLASS_NAMES = ["Empty", "Settlement", "Port", "Ruin", "Forest", "Mountain"]
NUM_CLASSES = 6

EPS = 1e-12

# ── helpers ───────────────────────────────────────────────────────────────────

def round_dir(n):
    return os.path.join(DATA_ROOT, f"round_{n:03d}")


def load_detail(n):
    with open(os.path.join(round_dir(n), "detail.json")) as f:
        return json.load(f)


def load_analysis(n, seed):
    path = os.path.join(round_dir(n), f"analysis_seed_{seed}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def entropy(probs):
    """Shannon entropy of a probability vector (nats)."""
    return -sum(p * math.log(p + EPS) for p in probs)


def kl_div(p, q):
    """KL(p||q) — p is ground truth, q is prediction."""
    return sum(pk * math.log((pk + EPS) / (qk + EPS)) for pk, qk in zip(p, q))


def euclidean(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def settlement_clustering(settlements, width, height):
    """
    Returns a dict with:
      n_settlements, mean_nn_dist (mean nearest-neighbour distance),
      spread_frac (bbox area / map area)
    """
    n = len(settlements)
    if n == 0:
        return {"n": 0, "mean_nn_dist": 0.0, "spread_frac": 0.0}

    coords = [(s["x"], s["y"]) for s in settlements]

    # nearest-neighbour distances
    nn_dists = []
    for i, (x1, y1) in enumerate(coords):
        best = float("inf")
        for j, (x2, y2) in enumerate(coords):
            if i != j:
                d = euclidean(x1, y1, x2, y2)
                if d < best:
                    best = d
        nn_dists.append(best)

    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    bbox_area = max(1, (max(xs) - min(xs)) * (max(ys) - min(ys)))
    map_area  = width * height

    return {
        "n": n,
        "mean_nn_dist": sum(nn_dists) / len(nn_dists),
        "spread_frac": bbox_area / map_area,
    }


def terrain_stats(grid, width, height):
    """Count each terrain type and return fractions."""
    flat = [grid[row][col] for row in range(height) for col in range(width)]
    total = len(flat)
    counts = Counter(flat)
    return {
        TERRAIN_NAMES.get(k, f"code_{k}"): v / total
        for k, v in sorted(counts.items())
    }


def is_coastal(grid, x, y, height, width):
    """True if cell (x,y) is adjacent to Ocean (code 10)."""
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            ny, nx = y + dy, x + dx
            if 0 <= ny < height and 0 <= nx < width:
                if grid[ny][nx] == 10:
                    return True
    return False


def coastal_settlement_fraction(settlements, grid, width, height):
    if not settlements:
        return 0.0
    coastal = sum(
        1 for s in settlements
        if is_coastal(grid, s["x"], s["y"], height, width)
    )
    return coastal / len(settlements)


# ── per-round analysis ────────────────────────────────────────────────────────

def analyze_round(n):
    detail = load_detail(n)
    width  = detail["map_width"]
    height = detail["map_height"]
    states = detail["initial_states"]

    # ── map characteristics (averaged over seeds) ──────────────────────────
    all_n_settlements   = []
    all_mean_nn_dist    = []
    all_spread_frac     = []
    all_coastal_frac    = []
    all_port_frac       = []       # fraction of settlements that are ports
    all_terrain_fracs   = []

    for state in states:
        grid        = state["grid"]
        settlements = state["settlements"]

        clust = settlement_clustering(settlements, width, height)
        all_n_settlements.append(clust["n"])
        all_mean_nn_dist.append(clust["mean_nn_dist"])
        all_spread_frac.append(clust["spread_frac"])
        all_coastal_frac.append(
            coastal_settlement_fraction(settlements, grid, width, height)
        )
        port_count = sum(1 for s in settlements if s.get("has_port"))
        port_frac  = port_count / len(settlements) if settlements else 0.0
        all_port_frac.append(port_frac)
        all_terrain_fracs.append(terrain_stats(grid, width, height))

    def avg(lst):
        return sum(lst) / len(lst) if lst else 0.0

    # average terrain fracs across seeds
    terrain_keys = set()
    for tf in all_terrain_fracs:
        terrain_keys.update(tf.keys())
    avg_terrain = {
        k: avg([tf.get(k, 0.0) for tf in all_terrain_fracs])
        for k in sorted(terrain_keys)
    }

    map_info = {
        "n_settlements":  avg(all_n_settlements),
        "mean_nn_dist":   avg(all_mean_nn_dist),
        "spread_frac":    avg(all_spread_frac),
        "coastal_frac":   avg(all_coastal_frac),
        "port_frac":      avg(all_port_frac),
        "terrain":        avg_terrain,
    }

    # ── ground truth dynamics (from analysis files) ────────────────────────
    all_change_frac    = []
    all_entropy        = []      # entropy of GT distribution per cell, flattened
    all_class_probs    = [[] for _ in range(NUM_CLASSES)]   # on dynamic cells
    all_gt_class_probs = [[] for _ in range(NUM_CLASSES)]   # all cells
    all_kl_per_class   = [[] for _ in range(NUM_CLASSES)]
    all_seed_scores    = []

    for seed in range(5):
        ana = load_analysis(n, seed)
        if ana is None:
            continue

        gt   = ana["ground_truth"]    # H×W×6
        pred = ana["prediction"]      # H×W×6
        ig   = ana["initial_grid"]    # H×W  (int terrain code)
        seed_score = ana.get("score")
        if seed_score is not None:
            all_seed_scores.append(seed_score)

        changed_cells = 0
        total_cells   = height * width

        for row in range(height):
            for col in range(width):
                gt_vec   = gt[row][col]    # 6 floats
                pred_vec = pred[row][col]  # 6 floats
                init_code = ig[row][col]

                # dominant GT class
                gt_class = gt_vec.index(max(gt_vec))

                # initial class (map terrain code to class index; ocean/plains→empty)
                init_class_map = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5,
                                  10: 0, 11: 0}
                init_class = init_class_map.get(init_code, 0)

                changed = (gt_class != init_class)
                if changed:
                    changed_cells += 1

                # entropy of GT distribution
                ent = entropy(gt_vec)
                all_entropy.append(ent)

                # per-class GT distribution across all cells
                for cls in range(NUM_CLASSES):
                    all_gt_class_probs[cls].append(gt_vec[cls])

                # on dynamic cells: per-class GT distribution
                if changed:
                    for cls in range(NUM_CLASSES):
                        all_class_probs[cls].append(gt_vec[cls])

                # KL per class (pointwise contribution)
                for cls in range(NUM_CLASSES):
                    pk = gt_vec[cls]
                    qk = pred_vec[cls]
                    kl_contrib = pk * math.log((pk + EPS) / (qk + EPS))
                    all_kl_per_class[cls].append(kl_contrib)

        all_change_frac.append(changed_cells / total_cells)

    gt_dynamics = {
        "change_frac":       avg(all_change_frac),
        "mean_entropy":      avg(all_entropy),
        "seed_scores":       all_seed_scores,
        "mean_seed_score":   avg(all_seed_scores) if all_seed_scores else None,
    }

    # Per-class stats on dynamic cells
    dynamic_class_means = {}
    for cls in range(NUM_CLASSES):
        vals = all_class_probs[cls]
        dynamic_class_means[CLASS_NAMES[cls]] = avg(vals) if vals else 0.0

    # Per-class stats across ALL cells
    all_cell_class_means = {}
    for cls in range(NUM_CLASSES):
        vals = all_gt_class_probs[cls]
        all_cell_class_means[CLASS_NAMES[cls]] = avg(vals) if vals else 0.0

    # KL per class (mean contribution per cell)
    kl_per_class = {}
    for cls in range(NUM_CLASSES):
        vals = all_kl_per_class[cls]
        kl_per_class[CLASS_NAMES[cls]] = avg(vals) if vals else 0.0
    total_kl = sum(kl_per_class.values())
    kl_frac  = {
        cls: (v / total_kl if total_kl > 0 else 0.0)
        for cls, v in kl_per_class.items()
    }

    return {
        "round":            n,
        "score":            KNOWN_SCORES.get(n),
        "map":              map_info,
        "gt_dynamics":      gt_dynamics,
        "dynamic_class_means":  dynamic_class_means,
        "all_cell_class_means": all_cell_class_means,
        "kl_per_class":     kl_per_class,
        "kl_frac":          kl_frac,
        "total_kl":         total_kl,
    }


# ── printing helpers ──────────────────────────────────────────────────────────

def fmt(v, digits=3):
    if v is None:
        return "N/A"
    return f"{v:.{digits}f}"


def print_section(title):
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_round_summary(r):
    n  = r["round"]
    sc = r["score"]
    m  = r["map"]
    gd = r["gt_dynamics"]
    print(f"\nRound {n:02d}  score={fmt(sc,1)}  (mean_seed_score={fmt(gd['mean_seed_score'],1)})")
    print(f"  Map: {m['n_settlements']:.1f} settlements | "
          f"nn_dist={fmt(m['mean_nn_dist'],1)} | "
          f"spread={fmt(m['spread_frac'],2)} | "
          f"coastal={fmt(m['coastal_frac'],2)} | "
          f"port_frac={fmt(m['port_frac'],2)}")
    t = m["terrain"]
    print(f"  Terrain: Ocean={fmt(t.get('Ocean',0),2)} "
          f"Forest={fmt(t.get('Forest',0),2)} "
          f"Mountain={fmt(t.get('Mountain',0),2)} "
          f"Plains={fmt(t.get('Plains',0),2)} "
          f"Empty={fmt(t.get('Empty',0),2)} "
          f"Settlement={fmt(t.get('Settlement',0),2)}")
    print(f"  GT dynamics: change_frac={fmt(gd['change_frac'],3)} | "
          f"mean_entropy={fmt(gd['mean_entropy'],4)}")
    print(f"  Dynamic cells class means: "
          + "  ".join(f"{cls}={fmt(v,3)}"
                      for cls, v in r["dynamic_class_means"].items()))
    print(f"  All cells class means:     "
          + "  ".join(f"{cls}={fmt(v,3)}"
                      for cls, v in r["all_cell_class_means"].items()))
    print(f"  KL total={fmt(r['total_kl'],4)}  "
          f"per-class fracs: "
          + "  ".join(f"{cls}={fmt(v,2)}"
                      for cls, v in r["kl_frac"].items()))


def group_avg(results, key_path):
    """Average a nested key across a list of result dicts."""
    vals = []
    for r in results:
        obj = r
        for k in key_path:
            obj = obj[k]
        if obj is not None:
            vals.append(obj)
    return sum(vals) / len(vals) if vals else None


def compare_groups(high_results, low_results):
    print_section("GROUP COMPARISON: HIGH vs LOW SCORING ROUNDS")

    metrics = [
        ("Map: n_settlements",   ["map", "n_settlements"]),
        ("Map: mean_nn_dist",    ["map", "mean_nn_dist"]),
        ("Map: spread_frac",     ["map", "spread_frac"]),
        ("Map: coastal_frac",    ["map", "coastal_frac"]),
        ("Map: port_frac",       ["map", "port_frac"]),
        ("Terrain: Ocean",       ["map", "terrain", "Ocean"]),
        ("Terrain: Forest",      ["map", "terrain", "Forest"]),
        ("Terrain: Mountain",    ["map", "terrain", "Mountain"]),
        ("Terrain: Plains",      ["map", "terrain", "Plains"]),
        ("GT: change_frac",      ["gt_dynamics", "change_frac"]),
        ("GT: mean_entropy",     ["gt_dynamics", "mean_entropy"]),
        ("Total KL",             ["total_kl"]),
    ]

    print(f"\n{'Metric':<30}  {'HIGH avg':>10}  {'LOW avg':>10}  {'diff (H-L)':>12}")
    print("-" * 68)
    for label, path in metrics:
        try:
            h = group_avg(high_results, path)
            l = group_avg(low_results,  path)
            diff = (h - l) if (h is not None and l is not None) else None
            sign = "+" if (diff is not None and diff > 0) else ""
            print(f"  {label:<28}  {fmt(h,4):>10}  {fmt(l,4):>10}  {sign+fmt(diff,4):>12}")
        except (KeyError, TypeError):
            print(f"  {label:<28}  {'N/A':>10}  {'N/A':>10}  {'N/A':>12}")

    # Per-class KL fracs
    print(f"\n{'Per-class KL fraction':<30}  {'HIGH avg':>10}  {'LOW avg':>10}  {'diff (H-L)':>12}")
    print("-" * 68)
    for cls in CLASS_NAMES:
        h = group_avg(high_results, ["kl_frac", cls])
        l = group_avg(low_results,  ["kl_frac", cls])
        diff = (h - l) if (h is not None and l is not None) else None
        sign = "+" if (diff is not None and diff > 0) else ""
        print(f"  KL frac [{cls}]{'':>14}  {fmt(h,4):>10}  {fmt(l,4):>10}  {sign+fmt(diff,4):>12}")

    # Dynamic cell class distribution
    print(f"\n{'Dynamic cells class mean':<30}  {'HIGH avg':>10}  {'LOW avg':>10}  {'diff (H-L)':>12}")
    print("-" * 68)
    for cls in CLASS_NAMES:
        h = group_avg(high_results, ["dynamic_class_means", cls])
        l = group_avg(low_results,  ["dynamic_class_means", cls])
        diff = (h - l) if (h is not None and l is not None) else None
        sign = "+" if (diff is not None and diff > 0) else ""
        print(f"  dyn_mean [{cls}]{'':>15}  {fmt(h,4):>10}  {fmt(l,4):>10}  {sign+fmt(diff,4):>12}")

    # All-cell class distribution
    print(f"\n{'All-cell GT class mean':<30}  {'HIGH avg':>10}  {'LOW avg':>10}  {'diff (H-L)':>12}")
    print("-" * 68)
    for cls in CLASS_NAMES:
        h = group_avg(high_results, ["all_cell_class_means", cls])
        l = group_avg(low_results,  ["all_cell_class_means", cls])
        diff = (h - l) if (h is not None and l is not None) else None
        sign = "+" if (diff is not None and diff > 0) else ""
        print(f"  all_mean [{cls}]{'':>15}  {fmt(h,4):>10}  {fmt(l,4):>10}  {sign+fmt(diff,4):>12}")


def print_kl_breakdown_table(all_results):
    print_section("KL CONTRIBUTION BREAKDOWN (absolute, per cell avg)")
    print(f"\n  {'Rnd':>4}  {'Score':>6}  {'KL_tot':>7}  "
          + "  ".join(f"{cls[:5]:>7}" for cls in CLASS_NAMES))
    print("  " + "-" * 70)
    for r in sorted(all_results, key=lambda x: x["score"] or 0, reverse=True):
        tag = "HIGH" if r["round"] in HIGH_ROUNDS else " LOW"
        row = (f"  {r['round']:>4}  {fmt(r['score'],1):>6}  "
               f"{fmt(r['total_kl'],4):>7}  "
               + "  ".join(f"{fmt(r['kl_per_class'].get(cls,0),4):>7}"
                            for cls in CLASS_NAMES))
        print(f"[{tag}]{row}")


def print_diagnoses(high_results, low_results):
    print_section("DIAGNOSIS SUMMARY")

    def avg_score(results):
        scores = [r["score"] for r in results if r["score"]]
        return sum(scores) / len(scores)

    h_score = avg_score(high_results)
    l_score = avg_score(low_results)
    print(f"\n  Average score — HIGH: {h_score:.1f}   LOW: {l_score:.1f}"
          f"   gap: {h_score - l_score:.1f}")

    # Find largest absolute differences in any KL class
    kl_diffs = {}
    for cls in CLASS_NAMES:
        h = group_avg(high_results, ["kl_per_class", cls])
        l = group_avg(low_results,  ["kl_per_class", cls])
        if h is not None and l is not None:
            kl_diffs[cls] = l - h   # positive = low rounds have more KL in this class

    print("\n  Classes with higher KL in LOW rounds (sorted by extra KL per cell):")
    for cls, diff in sorted(kl_diffs.items(), key=lambda x: -x[1]):
        direction = "MORE error in low rounds" if diff > 0 else "more error in high rounds"
        print(f"    {cls:<12}  extra KL={diff:+.5f}  ({direction})")

    # Change fraction
    h_cf = group_avg(high_results, ["gt_dynamics", "change_frac"])
    l_cf = group_avg(low_results,  ["gt_dynamics", "change_frac"])
    print(f"\n  Change fraction — HIGH: {h_cf:.4f}   LOW: {l_cf:.4f}")

    # Entropy
    h_ent = group_avg(high_results, ["gt_dynamics", "mean_entropy"])
    l_ent = group_avg(low_results,  ["gt_dynamics", "mean_entropy"])
    print(f"  Mean GT entropy  — HIGH: {h_ent:.4f}   LOW: {l_ent:.4f}")

    # Settlement count
    h_ns = group_avg(high_results, ["map", "n_settlements"])
    l_ns = group_avg(low_results,  ["map", "n_settlements"])
    print(f"  Mean # settlements — HIGH: {h_ns:.1f}   LOW: {l_ns:.1f}")

    # Dynamic Settlement mean
    h_ds = group_avg(high_results, ["dynamic_class_means", "Settlement"])
    l_ds = group_avg(low_results,  ["dynamic_class_means", "Settlement"])
    print(f"  Dyn-cell Settlement mean — HIGH: {h_ds:.4f}   LOW: {l_ds:.4f}")

    h_dr = group_avg(high_results, ["dynamic_class_means", "Ruin"])
    l_dr = group_avg(low_results,  ["dynamic_class_means", "Ruin"])
    print(f"  Dyn-cell Ruin mean       — HIGH: {h_dr:.4f}   LOW: {l_dr:.4f}")

    print("\n  INTERPRETATION:")
    findings = []

    if l_cf > h_cf * 1.05:
        findings.append(
            f"- Low rounds have higher cell-change fraction ({l_cf:.3f} vs {h_cf:.3f}), "
            "meaning more cells flip — the model's static prior hurts more."
        )
    elif h_cf > l_cf * 1.05:
        findings.append(
            f"- High rounds have higher cell-change fraction ({h_cf:.3f} vs {l_cf:.3f}). "
            "More change doesn't explain the score gap here."
        )
    else:
        findings.append(
            f"- Change fraction is similar (HIGH={h_cf:.3f}, LOW={l_cf:.3f}). "
            "Absolute dynamics volume is not the differentiator."
        )

    if l_ent > h_ent * 1.01:
        findings.append(
            f"- Low rounds have higher GT entropy ({l_ent:.4f} vs {h_ent:.4f}): "
            "outcomes are more uncertain/spread, which is inherently harder to predict."
        )
    elif h_ent > l_ent * 1.01:
        findings.append(
            f"- High rounds have higher GT entropy ({h_ent:.4f} vs {l_ent:.4f}): "
            "entropy alone doesn't explain performance gaps."
        )

    worst_cls = max(kl_diffs, key=kl_diffs.get)
    if kl_diffs[worst_cls] > 0:
        findings.append(
            f"- The largest extra KL error in low rounds comes from the '{worst_cls}' class "
            f"(+{kl_diffs[worst_cls]:.5f} per cell). The model mis-predicts this class most."
        )

    if l_ds > h_ds * 1.05:
        findings.append(
            f"- On dynamic cells, low rounds show higher Settlement probability in ground truth "
            f"({l_ds:.3f} vs {h_ds:.3f}). This suggests settlement growth/persistence is harder."
        )
    elif h_ds > l_ds * 1.05:
        findings.append(
            f"- On dynamic cells, high rounds have more Settlement in GT ({h_ds:.3f} vs {l_ds:.3f}): "
            "settlement dynamics are not the culprit in low rounds."
        )

    if l_dr > h_dr * 1.05:
        findings.append(
            f"- Ruin probability is higher on dynamic cells in low rounds ({l_dr:.3f} vs {h_dr:.3f}): "
            "settlement-to-ruin transitions may be over/under-predicted."
        )

    # Ocean coverage
    h_oc = group_avg(high_results, ["map", "terrain", "Ocean"])
    l_oc = group_avg(low_results,  ["map", "terrain", "Ocean"])
    if abs(h_oc - l_oc) > 0.03:
        more = "Low" if l_oc > h_oc else "High"
        findings.append(
            f"- {more} rounds have noticeably different ocean coverage "
            f"(HIGH={h_oc:.2f}, LOW={l_oc:.2f}). "
            "Ocean cells are deterministic (don't change), so more ocean = easier round."
        )

    for f in findings:
        print(f"    {f}")

    print()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("Analyzing round difficulty...")
    all_round_nums = sorted(set(HIGH_ROUNDS + LOW_ROUNDS))

    results = {}
    for n in all_round_nums:
        print(f"  Processing round {n:02d}...", end="", flush=True)
        results[n] = analyze_round(n)
        print(" done")

    high_results = [results[n] for n in HIGH_ROUNDS]
    low_results  = [results[n] for n in LOW_ROUNDS]

    print_section("PER-ROUND DETAILS — HIGH SCORING ROUNDS")
    for n in HIGH_ROUNDS:
        print_round_summary(results[n])

    print_section("PER-ROUND DETAILS — LOW SCORING ROUNDS")
    for n in LOW_ROUNDS:
        print_round_summary(results[n])

    print_kl_breakdown_table(list(results.values()))
    compare_groups(high_results, low_results)
    print_diagnoses(high_results, low_results)


if __name__ == "__main__":
    main()
