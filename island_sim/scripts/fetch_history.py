"""Fetch all completed round data (ground truths, initial states, scores) for offline calibration."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.api_client import get_my_rounds, get_round_detail, get_analysis

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "rounds"
NUM_SEEDS = 5


def fetch_all() -> None:
    print("Fetching round history...")
    rounds = get_my_rounds()
    print(f"Found {len(rounds)} rounds")

    completed = [r for r in rounds if r["status"] in ("completed", "scoring")]
    print(f"{len(completed)} completed/scoring rounds")

    if not completed:
        print("No completed rounds to fetch.")
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for rnd in completed:
        round_id = rnd["id"]
        round_num = rnd["round_number"]
        round_dir = DATA_DIR / f"round_{round_num:03d}"

        if round_dir.exists() and (round_dir / "detail.json").exists():
            print(f"Round {round_num}: already fetched, skipping")
            continue

        round_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nRound {round_num} (score: {rnd.get('round_score')}, rank: {rnd.get('rank')})")

        # Save round summary
        (round_dir / "summary.json").write_text(json.dumps(rnd, indent=2))

        # Fetch round detail (initial states)
        print(f"  Fetching round detail...")
        try:
            detail = get_round_detail(round_id)
            (round_dir / "detail.json").write_text(json.dumps(detail, indent=2))
            print(f"  Saved detail with {len(detail.get('initial_states', []))} initial states")
        except Exception as exc:
            print(f"  Failed to fetch detail: {exc}")
            continue

        # Fetch analysis (ground truth) for each seed
        seeds_count = rnd.get("seeds_count", detail.get("seeds_count", NUM_SEEDS))
        for seed_idx in range(seeds_count):
            print(f"  Fetching analysis for seed {seed_idx}...")
            try:
                analysis = get_analysis(round_id, seed_idx)
                (round_dir / f"analysis_seed_{seed_idx}.json").write_text(
                    json.dumps(analysis, indent=2)
                )
                score = analysis.get("score")
                print(f"    Seed {seed_idx}: score={score}")
            except Exception as exc:
                print(f"    Seed {seed_idx} failed: {exc}")

            # Be nice to the API
            time.sleep(0.3)

    print("\nDone! Data saved to", DATA_DIR)
    _print_summary()


def _print_summary() -> None:
    """Print a summary of all fetched data."""
    if not DATA_DIR.exists():
        return

    print("\n--- Fetched Data Summary ---")
    for round_dir in sorted(DATA_DIR.iterdir()):
        if not round_dir.is_dir():
            continue

        summary_path = round_dir / "summary.json"
        if not summary_path.exists():
            continue

        summary = json.loads(summary_path.read_text())
        score = summary.get("round_score", "n/a")
        rank = summary.get("rank", "n/a")
        seeds_submitted = summary.get("seeds_submitted", 0)
        queries_used = summary.get("queries_used", 0)

        analysis_files = list(round_dir.glob("analysis_seed_*.json"))
        seed_scores = []
        for af in sorted(analysis_files):
            analysis = json.loads(af.read_text())
            seed_scores.append(analysis.get("score"))

        print(
            f"  {round_dir.name}: score={score}, rank={rank}, "
            f"seeds={seeds_submitted}, queries={queries_used}, "
            f"ground_truths={len(analysis_files)}, seed_scores={seed_scores}"
        )


if __name__ == "__main__":
    fetch_all()
