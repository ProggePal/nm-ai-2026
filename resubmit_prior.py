"""
Resubmit with a better spatial prior model — no queries needed.

Current submission used Laplace-smoothed empirical from ~1 observation per cell.
Problem: with only 1 obs, P(Forest) ≈ 0.29 when it should be ~0.85.

This script builds predictions using:
  - Terrain-specific priors from simulation mechanics docs
  - Spatial features: distance to nearest settlement, coastal adjacency
  - Ocean/Mountain treated as truly static (huge win on those cells)
  - Plains near settlements get elevated Settlement/Ruin probability

No queries consumed. Pure resubmit.
"""

import time
import numpy as np
import requests

BASE = "https://api.ainm.no"
MIN_PROB = 0.01
N_CLASSES = 6
TERRAIN = ["Empty", "Settlement", "Port", "Ruin", "Forest", "Mountain"]


def make_session(token):
    s = requests.Session()
    s.headers["Authorization"] = f"Bearer {token}"
    return s


def dist_to_nearest(targets: list[tuple], height: int, width: int) -> np.ndarray:
    """
    Compute minimum Manhattan distance from every cell to the nearest target.
    targets = list of (x, y) positions.
    Returns H×W array of distances (inf if no targets).
    """
    dist = np.full((height, width), np.inf)
    if not targets:
        return dist
    for tx, ty in targets:
        for dy in range(-height, height):
            for dx in range(-width, width):
                ny, nx = ty + dy, tx + dx
                if 0 <= ny < height and 0 <= nx < width:
                    d = abs(dy) + abs(dx)
                    if d < dist[ny, nx]:
                        dist[ny, nx] = d
    return dist


def dist_to_nearest_fast(targets: list[tuple], height: int, width: int) -> np.ndarray:
    """Fast BFS-based distance computation."""
    from collections import deque
    dist = np.full((height, width), 999, dtype=int)
    q = deque()
    for tx, ty in targets:
        if 0 <= ty < height and 0 <= tx < width:
            dist[ty, tx] = 0
            q.append((tx, ty))
    while q:
        cx, cy = q.popleft()
        for nx, ny in [(cx+1,cy),(cx-1,cy),(cx,cy+1),(cx,cy-1)]:
            if 0 <= ny < height and 0 <= nx < width and dist[ny, nx] == 999:
                dist[ny, nx] = dist[cy, cx] + 1
                q.append((nx, ny))
    return dist


def is_coastal_mask(raw_grid: np.ndarray, height: int, width: int) -> np.ndarray:
    """True for cells adjacent (4-way) to Ocean (code 10)."""
    ocean = (raw_grid == 10)
    coastal = np.zeros((height, width), dtype=bool)
    coastal[1:] |= ocean[:-1]
    coastal[:-1] |= ocean[1:]
    coastal[:, 1:] |= ocean[:, :-1]
    coastal[:, :-1] |= ocean[:, 1:]
    return coastal & ~ocean  # coastal but not ocean itself


def terrain_prior(raw_code: int, dist_settle: int, coastal: bool) -> np.ndarray:
    """
    Prior P = [Empty, Settlement, Port, Ruin, Forest, Mountain]
    Based on simulation mechanics:
      - Ocean, Mountain: truly static
      - Forest: mostly stays forest
      - Settlement/Port: volatile (collapse, expand, raid)
      - Ruin: volatile (reclaimed or overgrown)
      - Plains/Empty: mostly static, more volatile near settlements
    """
    # Ocean — truly static, just borders map
    if raw_code == 10:
        p = [0.990, 0.002, 0.002, 0.002, 0.002, 0.002]

    # Mountain — truly static
    elif raw_code == 5:
        p = [0.002, 0.002, 0.002, 0.002, 0.002, 0.990]

    # Forest — mostly stays, can expand into ruins / be cleared for settlement
    elif raw_code == 4:
        p = [0.05, 0.03, 0.01, 0.04, 0.85, 0.02]

    # Settlement — very volatile: thrives, raids, starves, collapses
    elif raw_code == 1:
        if coastal:
            # coastal settlements often develop ports
            p = [0.07, 0.35, 0.22, 0.27, 0.07, 0.02]
        else:
            p = [0.08, 0.42, 0.08, 0.30, 0.10, 0.02]

    # Port — volatile: depends on parent settlement surviving
    elif raw_code == 2:
        p = [0.07, 0.18, 0.45, 0.24, 0.04, 0.02]

    # Ruin — volatile: reclaimed by settlement, overgrown by forest, or stays
    elif raw_code == 3:
        p = [0.18, 0.15, 0.04, 0.38, 0.23, 0.02]

    # Plains / Empty (code 11 or 0) — mostly static but proximity matters
    else:
        if dist_settle <= 2:
            if coastal:
                p = [0.45, 0.20, 0.10, 0.12, 0.08, 0.05]
            else:
                p = [0.50, 0.30, 0.02, 0.10, 0.06, 0.02]
        elif dist_settle <= 4:
            p = [0.65, 0.20, 0.02, 0.07, 0.05, 0.01]
        elif dist_settle <= 8:
            p = [0.80, 0.09, 0.01, 0.04, 0.05, 0.01]
        else:
            p = [0.88, 0.04, 0.01, 0.02, 0.04, 0.01]

    arr = np.array(p, dtype=float)
    # Iterative floor + renormalize until stable
    for _ in range(5):
        arr = np.maximum(arr, MIN_PROB)
        arr = arr / arr.sum()
    return arr


def build_prior_prediction(state: dict, width: int, height: int) -> np.ndarray:
    raw_grid = np.array(state["grid"], dtype=int)
    settlement_positions = [(s["x"], s["y"]) for s in state["settlements"]]

    dist = dist_to_nearest_fast(settlement_positions, height, width)
    coastal = is_coastal_mask(raw_grid, height, width)

    prediction = np.zeros((height, width, N_CLASSES))
    for y in range(height):
        for x in range(width):
            code = raw_grid[y, x]
            d = int(dist[y, x])
            c = bool(coastal[y, x])
            prediction[y, x] = terrain_prior(code, d, c)

    return prediction


def submit_prediction(session, round_id, seed_index, prediction):
    resp = session.post(f"{BASE}/astar-island/submit", json={
        "round_id": round_id,
        "seed_index": seed_index,
        "prediction": prediction.tolist(),
    })
    resp.raise_for_status()
    return resp.json()


def run(token: str):
    session = make_session(token)

    # Get active round
    rounds = session.get(f"{BASE}/astar-island/rounds").json()
    active = next((r for r in rounds if r["status"] == "active"), None)
    if not active:
        print("No active round — nothing to resubmit")
        return

    round_id = active["id"]
    detail = session.get(f"{BASE}/astar-island/rounds/{round_id}").json()

    width = detail["map_width"]
    height = detail["map_height"]
    seeds = detail["seeds_count"]
    closes_at = detail.get("closes_at", "unknown")

    print(f"Round {active['round_number']}: {width}×{height}, closes {closes_at}")
    print(f"Building prior-based predictions for {seeds} seeds...\n")

    for seed_idx in range(seeds):
        state = detail["initial_states"][seed_idx]
        n_settlements = len(state["settlements"])

        prediction = build_prior_prediction(state, width, height)

        # Enforce floor and renormalize across whole tensor
        prediction = np.maximum(prediction, MIN_PROB)
        prediction /= prediction.sum(axis=2, keepdims=True)

        # Validate
        assert prediction.shape == (height, width, N_CLASSES)
        assert np.allclose(prediction.sum(axis=2), 1.0, atol=1e-4)

        # Show what we're predicting
        argmax = np.argmax(prediction, axis=2)
        counts = {TERRAIN[c]: int((argmax == c).sum()) for c in range(N_CLASSES)}
        avg_conf = float(prediction.max(axis=2).mean())

        resp = submit_prediction(session, round_id, seed_idx, prediction)
        print(f"Seed {seed_idx} ({n_settlements} initial settlements):")
        print(f"  Predicted: {counts}")
        print(f"  Avg confidence: {avg_conf:.3f}")
        print(f"  Submitted: {resp}")
        time.sleep(0.6)

    print("\nDone — all seeds resubmitted with spatial prior model")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python resubmit_prior.py <JWT_TOKEN>")
        sys.exit(1)
    run(sys.argv[1])
