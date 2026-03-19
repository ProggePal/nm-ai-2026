"""
Persistent storage for Astar Island observations and ground truth.

Saves everything needed to:
  1. Resubmit better predictions within the same round (uses saved observations)
  2. Learn across rounds by accumulating (initial_state → ground_truth) pairs

Storage layout:
  data/
    {round_id}/
      round_detail.json          # map, seeds, initial states
      seed_{i}/
        queries.jsonl            # every simulate response (grid + settlements)
        ground_truth.npy         # H×W×6 ground truth (after round completes)
        initial_classes.npy      # H×W initial class indices
      predictions/
        seed_{i}_v{n}.npy        # every prediction tensor we submitted
"""

import json
import time
import numpy as np
from pathlib import Path
from datetime import datetime

DATA_DIR = Path("data")


# ── Save / load round detail ───────────────────────────────────────────────

def save_round_detail(round_id: str, detail: dict):
    path = DATA_DIR / round_id / "round_detail.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(detail, indent=2))


def load_round_detail(round_id: str) -> dict:
    return json.loads((DATA_DIR / round_id / "round_detail.json").read_text())


# ── Save / load queries ────────────────────────────────────────────────────

def save_query(round_id: str, seed_idx: int, vx: int, vy: int, result: dict):
    """Append one simulate response to the query log."""
    path = DATA_DIR / round_id / f"seed_{seed_idx}" / "queries.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "ts": datetime.utcnow().isoformat(),
        "vx": vx, "vy": vy,
        "result": result,
    }
    with path.open("a") as f:
        f.write(json.dumps(entry) + "\n")


def load_queries(round_id: str, seed_idx: int) -> list[dict]:
    """Load all saved query results for a seed."""
    path = DATA_DIR / round_id / f"seed_{seed_idx}" / "queries.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().strip().split("\n") if line]


def rebuild_counts_from_queries(round_id: str, seed_idx: int,
                                height: int, width: int,
                                terrain_to_class: dict) -> np.ndarray:
    """
    Reconstruct H×W×6 observation counts from saved queries.
    This is what lets us resubmit with empirical distributions within a round.
    """
    counts = np.zeros((height, width, 6), dtype=int)
    for entry in load_queries(round_id, seed_idx):
        vx, vy = entry["vx"], entry["vy"]
        for row_i, row in enumerate(entry["result"]["grid"]):
            for col_i, raw_code in enumerate(row):
                cls = terrain_to_class.get(raw_code, 0)
                counts[vy + row_i][vx + col_i][cls] += 1
    return counts


def get_settlement_signals(round_id: str, seed_idx: int) -> dict:
    """
    Extract settlement health signals from all saved queries.
    Returns {(x, y): {"alive": bool, "food": float, "population": float, ...}}
    for every settlement ever seen in any query for this seed.
    Uses the LAST observation of each settlement (most recent sim run).
    """
    signals = {}
    for entry in load_queries(round_id, seed_idx):
        for s in entry["result"].get("settlements", []):
            signals[(s["x"], s["y"])] = {
                "alive": s.get("alive", True),
                "food": s.get("food", 1.0),
                "population": s.get("population", 1.0),
                "wealth": s.get("wealth", 0.5),
                "defense": s.get("defense", 0.5),
                "has_port": s.get("has_port", False),
            }
    return signals


# ── Save / load predictions ────────────────────────────────────────────────

def save_prediction(round_id: str, seed_idx: int, prediction: np.ndarray):
    """Save a prediction tensor. Keeps all versions (v0, v1, v2...)."""
    base = DATA_DIR / round_id / "predictions"
    base.mkdir(parents=True, exist_ok=True)
    existing = sorted(base.glob(f"seed_{seed_idx}_v*.npy"))
    version = len(existing)
    path = base / f"seed_{seed_idx}_v{version}.npy"
    np.save(str(path), prediction)
    return path


def load_latest_prediction(round_id: str, seed_idx: int) -> np.ndarray | None:
    base = DATA_DIR / round_id / "predictions"
    existing = sorted(base.glob(f"seed_{seed_idx}_v*.npy"))
    if not existing:
        return None
    return np.load(str(existing[-1]))


# ── Ground truth (post-round) ──────────────────────────────────────────────

def save_ground_truth(round_id: str, seed_idx: int, ground_truth: np.ndarray):
    path = DATA_DIR / round_id / f"seed_{seed_idx}" / "ground_truth.npy"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(path), ground_truth)
    print(f"  Saved ground truth: {path}")


def load_ground_truth(round_id: str, seed_idx: int) -> np.ndarray | None:
    path = DATA_DIR / round_id / f"seed_{seed_idx}" / "ground_truth.npy"
    return np.load(str(path)) if path.exists() else None


def fetch_and_save_all_ground_truth(session, round_id: str, seeds: int, base_url: str):
    """
    Call /analysis for each seed and save ground truth tensors.
    Only works after the round is completed/scoring.
    """
    import requests
    saved = 0
    for seed_idx in range(seeds):
        existing = load_ground_truth(round_id, seed_idx)
        if existing is not None:
            print(f"  Seed {seed_idx}: ground truth already saved")
            continue
        try:
            resp = session.get(f"{base_url}/astar-island/analysis/{round_id}/{seed_idx}")
            resp.raise_for_status()
            data = resp.json()
            gt = np.array(data["ground_truth"])
            save_ground_truth(round_id, seed_idx, gt)
            saved += 1
            time.sleep(0.3)
        except Exception as e:
            print(f"  Seed {seed_idx}: {e}")
    return saved


# ── Training data export ───────────────────────────────────────────────────

def list_training_examples() -> list[dict]:
    """
    List all (initial_state, ground_truth) pairs across all rounds.
    This is the dataset for training a proper ML model.
    """
    examples = []
    for round_dir in sorted(DATA_DIR.iterdir()):
        if not round_dir.is_dir():
            continue
        round_id = round_dir.name
        detail_path = round_dir / "round_detail.json"
        if not detail_path.exists():
            continue
        detail = json.loads(detail_path.read_text())
        seeds = detail.get("seeds_count", 5)
        for seed_idx in range(seeds):
            gt = load_ground_truth(round_id, seed_idx)
            if gt is None:
                continue
            initial_states = detail.get("initial_states", [])
            if seed_idx >= len(initial_states):
                continue
            examples.append({
                "round_id": round_id,
                "seed_idx": seed_idx,
                "initial_grid": initial_states[seed_idx]["grid"],
                "settlements": initial_states[seed_idx]["settlements"],
                "ground_truth": gt,
            })
    return examples


def training_data_summary():
    examples = list_training_examples()
    print(f"Training examples: {len(examples)}")
    print(f"  = {len(examples) // 5} completed rounds × 5 seeds")
    return examples
