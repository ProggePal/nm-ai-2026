"""
Astar Island — Viking Civilisation Prediction

Ground truth = Monte Carlo distribution over hundreds of sim runs.
We estimate it via stochastic observations + Laplace smoothing.

Query strategy — two-phase breadth-first:
  Phase 1 (10 queries): 2 per seed → submit ALL seeds immediately
  Phase 2 (40 queries): 8 per seed → resubmit with better data

This ensures all seeds are always on the board. If the round closes
early or the script crashes, you've already scored something on every seed.

Terrain codes (internal) vs prediction class indices:
  Ocean(10), Plains(11), Empty(0) → class 0
  Settlement(1) → 1, Port(2) → 2, Ruin(3) → 3
  Forest(4) → 4, Mountain(5) → 5
"""

import time
import numpy as np
import requests
from store import save_round_detail, save_query, save_prediction, \
    rebuild_counts_from_queries, get_settlement_signals

BASE = "https://api.ainm.no"

TERRAIN = ["Empty", "Settlement", "Port", "Ruin", "Forest", "Mountain"]
N_CLASSES = 6
MIN_PROB = 0.01  # floor prevents KL divergence blowing up

TERRAIN_TO_CLASS = {
    0: 0, 10: 0, 11: 0,  # Empty / Ocean / Plains → 0
    1: 1, 2: 2, 3: 3, 4: 4, 5: 5,
}

STATIC_CLASSES = frozenset({0, 5})  # Ocean/Plains/Empty and Mountain — never change


# ── Auth ──────────────────────────────────────────────────────────────────────

def make_session(token: str) -> requests.Session:
    session = requests.Session()
    session.headers["Authorization"] = f"Bearer {token}"
    return session


# ── API ───────────────────────────────────────────────────────────────────────

def get_active_round(session) -> str:
    rounds = session.get(f"{BASE}/astar-island/rounds").json()
    active = next((r for r in rounds if r["status"] == "active"), None)
    if not active:
        raise RuntimeError("No active round found")
    return active["id"]


def get_round_detail(session, round_id: str) -> dict:
    return session.get(f"{BASE}/astar-island/rounds/{round_id}").json()


def simulate(session, round_id: str, seed_index: int,
             vx: int, vy: int, vw: int = 15, vh: int = 15) -> dict:
    payload = {
        "round_id": round_id,
        "seed_index": seed_index,
        "viewport_x": vx,
        "viewport_y": vy,
        "viewport_w": max(5, min(15, vw)),
        "viewport_h": max(5, min(15, vh)),
    }
    resp = session.post(f"{BASE}/astar-island/simulate", json=payload)
    if resp.status_code == 429:
        print("  Rate limited — waiting 1s…")
        time.sleep(1.0)
        resp = session.post(f"{BASE}/astar-island/simulate", json=payload)
    resp.raise_for_status()
    return resp.json()


def submit_prediction(session, round_id: str, seed_index: int,
                      prediction: np.ndarray) -> dict:
    resp = session.post(f"{BASE}/astar-island/submit", json={
        "round_id": round_id,
        "seed_index": seed_index,
        "prediction": prediction.tolist(),
    })
    resp.raise_for_status()
    return resp.json()


# ── Map utilities ─────────────────────────────────────────────────────────────

def tile_positions(map_size: int, vp_size: int = 15) -> list[int]:
    """Non-overlapping positions that tile [0, map_size) with vp_size windows."""
    positions = list(range(0, map_size, vp_size))
    if positions[-1] + vp_size > map_size:
        positions[-1] = map_size - vp_size
    return positions


def all_tile_viewports(width: int, height: int, vp_size: int = 15) -> list[tuple]:
    """All 9 viewport positions for full 40×40 coverage."""
    return [(vx, vy)
            for vy in tile_positions(height, vp_size)
            for vx in tile_positions(width, vp_size)]


def parse_initial_classes(state: dict) -> np.ndarray:
    """Convert raw initial_states grid to H×W class index array (0–5)."""
    raw = np.array(state["grid"], dtype=int)
    return np.vectorize(TERRAIN_TO_CLASS.__getitem__)(raw)


def rank_viewports_by_volatility(initial_classes: np.ndarray,
                                 width: int, height: int,
                                 vp_size: int = 15) -> list[tuple]:
    """
    Sort all tile viewports by number of volatile (non-static) cells.
    Best 2 viewports first → used for the quick Phase 1 baseline.
    """
    volatile = ~np.isin(initial_classes, list(STATIC_CLASSES))
    viewports = all_tile_viewports(width, height, vp_size)

    def score(pos):
        vx, vy = pos
        return volatile[vy:vy+vp_size, vx:vx+vp_size].sum()

    return sorted(viewports, key=score, reverse=True)


# ── Observation accumulator ───────────────────────────────────────────────────

def accumulate(counts: np.ndarray, result: dict, vx: int, vy: int):
    """Add one viewport observation into H×W×6 counts array."""
    for row_i, row in enumerate(result["grid"]):
        for col_i, raw_code in enumerate(row):
            cls = TERRAIN_TO_CLASS.get(raw_code, 0)
            counts[vy + row_i][vx + col_i][cls] += 1


# ── Prediction builder ────────────────────────────────────────────────────────

def build_prediction(counts: np.ndarray,
                     initial_classes: np.ndarray,
                     height: int, width: int) -> np.ndarray:
    """
    Build H×W×6 probability tensor.

      Static cells    → near-certain on initial class
      Observed cells  → Laplace-smoothed empirical distribution
      Unobserved      → soft prior on initial class (uncertain)

    Laplace smoothing (+1 per class) ensures no class gets probability 0,
    which would cause KL divergence = infinity and destroy the cell score.
    """
    total_obs = counts.sum(axis=2)
    is_static = np.isin(initial_classes, list(STATIC_CLASSES))
    is_observed = total_obs > 0

    # Static prior: near-certain on initial class
    static_p = np.full((height, width, N_CLASSES), MIN_PROB)
    for cls in range(N_CLASSES):
        static_p[initial_classes == cls, cls] = 1.0 - (N_CLASSES - 1) * MIN_PROB

    # Empirical: Laplace-smoothed counts
    n = total_obs[:, :, np.newaxis].clip(1)
    empirical_p = (counts + 1.0) / (n + N_CLASSES)

    # Unobserved volatile: soft prior on initial class
    unobs_p = np.full((height, width, N_CLASSES), 0.05)
    for cls in range(N_CLASSES):
        unobs_p[initial_classes == cls, cls] = 0.75

    prediction = np.where(
        is_static[:, :, np.newaxis], static_p,
        np.where(is_observed[:, :, np.newaxis], empirical_p, unobs_p)
    )

    # Floor and renormalize
    prediction = np.maximum(prediction, MIN_PROB)
    prediction /= prediction.sum(axis=2, keepdims=True)
    return prediction


def validate(prediction: np.ndarray, height: int, width: int):
    assert prediction.shape == (height, width, N_CLASSES), \
        f"Wrong shape: {prediction.shape}"
    assert np.allclose(prediction.sum(axis=2), 1.0, atol=1e-5), \
        f"Rows don't sum to 1"
    assert prediction.min() >= MIN_PROB - 1e-9, \
        f"Floor violation: {prediction.min():.6f}"


# ── Core run logic ────────────────────────────────────────────────────────────

def run(token: str, total_budget: int = 50):
    """
    Two-phase breadth-first submission.

    Phase 1 — get all seeds on the board immediately (2 queries each):
      Uses the 2 most volatile viewports per seed so the quick submission
      is as informative as possible.

    Phase 2 — improve all seeds with remaining budget (8 queries each):
      Completes full map tiling + 1 targeted volatile resample per seed.
    """
    session = make_session(token)

    round_id = get_active_round(session)
    detail = get_round_detail(session, round_id)

    width   = detail["map_width"]   # 40
    height  = detail["map_height"]  # 40
    seeds   = detail["seeds_count"] # 5

    print(f"Round {detail['round_number']}: {width}×{height}, {seeds} seeds")
    print(f"Total budget: {total_budget} queries → {total_budget//seeds}/seed\n")

    # Parse initial states (free — no queries used)
    initial_classes_per_seed = []
    ranked_viewports_per_seed = []
    for i, state in enumerate(detail["initial_states"]):
        cls_map = parse_initial_classes(state)
        initial_classes_per_seed.append(cls_map)
        ranked = rank_viewports_by_volatility(cls_map, width, height)
        ranked_viewports_per_seed.append(ranked)
        n_volatile = (~np.isin(cls_map, list(STATIC_CLASSES))).sum()
        print(f"  Seed {i}: {n_volatile}/{width*height} volatile cells")

    # Per-seed observation counts — persist across phases
    counts_per_seed = [
        np.zeros((height, width, N_CLASSES), dtype=int)
        for _ in range(seeds)
    ]

    PHASE1_QUERIES = 2  # per seed — get on the board fast
    budget_per_seed = total_budget // seeds  # 10
    PHASE2_QUERIES = budget_per_seed - PHASE1_QUERIES  # 8

    # ── Phase 1: quick baseline, submit all seeds ─────────────────────────
    print(f"\n── Phase 1: {PHASE1_QUERIES} queries/seed → submit all ──")
    for seed_idx in range(seeds):
        ranked = ranked_viewports_per_seed[seed_idx]
        counts = counts_per_seed[seed_idx]

        for vx, vy in ranked[:PHASE1_QUERIES]:
            vw = min(15, width - vx)
            vh = min(15, height - vy)
            result = simulate(session, round_id, seed_idx, vx, vy, vw, vh)
            accumulate(counts, result, vx, vy)
            save_query(round_id, seed_idx, vx, vy, result)  # persist
            time.sleep(0.22)

        prediction = build_prediction(counts, initial_classes_per_seed[seed_idx],
                                      height, width)
        validate(prediction, height, width)
        resp = submit_prediction(session, round_id, seed_idx, prediction)
        save_prediction(round_id, seed_idx, prediction)      # persist
        obs = (counts.sum(axis=2) > 0).sum()
        print(f"  Seed {seed_idx}: submitted ({obs}/{width*height} cells observed)")
        time.sleep(0.6)

    print("  ✓ All seeds on the board\n")

    # ── Phase 2: improve all seeds, resubmit each ────────────────────────
    print(f"── Phase 2: {PHASE2_QUERIES} more queries/seed → resubmit all ──")
    for seed_idx in range(seeds):
        ranked = ranked_viewports_per_seed[seed_idx]
        counts = counts_per_seed[seed_idx]
        remaining_viewports = ranked[PHASE1_QUERIES:]

        for vx, vy in remaining_viewports[:PHASE2_QUERIES]:
            vw = min(15, width - vx)
            vh = min(15, height - vy)
            result = simulate(session, round_id, seed_idx, vx, vy, vw, vh)
            accumulate(counts, result, vx, vy)
            save_query(round_id, seed_idx, vx, vy, result)  # persist
            time.sleep(0.22)

        prediction = build_prediction(counts, initial_classes_per_seed[seed_idx],
                                      height, width)
        validate(prediction, height, width)
        resp = submit_prediction(session, round_id, seed_idx, prediction)
        save_prediction(round_id, seed_idx, prediction)      # persist
        obs = (counts.sum(axis=2) > 0).sum()
        avg = counts.sum(axis=2).mean()
        print(f"  Seed {seed_idx}: resubmitted "
              f"({obs}/{width*height} cells, {avg:.1f} avg obs/cell)")
        time.sleep(0.6)

    print("\n  ✓ All seeds improved and resubmitted")
    print(f"  Queries used: {total_budget} / {total_budget}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python astar.py <JWT_TOKEN>")
        sys.exit(1)
    run(sys.argv[1])
