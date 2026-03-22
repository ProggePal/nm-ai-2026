"""
Astar Island — Auto-submit daemon

Polls for active rounds and automatically runs the full pipeline.
Leave this running overnight.

Usage:
    python daemon.py <JWT_TOKEN>
    python daemon.py <JWT_TOKEN> --poll 60       # check every 60 seconds
    python daemon.py <JWT_TOKEN> --interactive   # pause for human approval
"""

import sys
import time
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from astar import (
    make_session, get_round_detail,
    build_prediction, submit_prediction, parse_initial_classes,
    simulate, accumulate, rank_viewports_by_volatility,
    BASE, N_CLASSES, STATIC_CLASSES,
)
from store import (
    save_round_detail, save_prediction, save_query,
    fetch_and_save_all_ground_truth, training_data_summary,
)

LOG_FILE = Path("daemon_log.jsonl")

TOTAL_BUDGET = 50
PHASE1_QUERIES = 2   # per seed — get all seeds on the board fast


# ── Logging ───────────────────────────────────────────────────────────────────

def log(msg: str, data: dict = None):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")
    if data:
        entry = {"ts": datetime.utcnow().isoformat(), "msg": msg, **data}
        with LOG_FILE.open("a") as f:
            f.write(json.dumps(entry) + "\n")


def get_round_closes_at(detail: dict):
    closes_at_str = detail.get("closes_at")
    if not closes_at_str:
        return None
    return datetime.fromisoformat(closes_at_str.replace("Z", "+00:00"))


# ── Interactive plan display ───────────────────────────────────────────────────

def print_query_plan(round_num, seeds, initial_classes_per_seed, ranked_per_seed,
                     budget_per_seed):
    border = "═" * 42
    print(f"\n{border}")
    print(f"  ROUND {round_num} — QUERY PLAN (approve?)")
    print(f"{border}")

    for seed_idx in range(seeds):
        cls_map = initial_classes_per_seed[seed_idx]
        n_volatile = int((~np.isin(cls_map, list(STATIC_CLASSES))).sum())
        n_settlements = int((cls_map == 1).sum())
        ranked = ranked_per_seed[seed_idx]

        phase1 = ranked[:PHASE1_QUERIES]
        phase2 = ranked[PHASE1_QUERIES:budget_per_seed]

        p1_str = ", ".join(f"({vx},{vy})" for vx, vy in phase1)
        p2_str = ", ".join(f"({vx},{vy})" for vx, vy in phase2[:4])
        if len(phase2) > 4:
            p2_str += "…"

        print(f"\n  Seed {seed_idx}: {n_settlements} settlements, {n_volatile} volatile cells")
        print(f"    Phase 1 viewports: {p1_str}  [{PHASE1_QUERIES} queries]")
        print(f"    Phase 2 viewports: {p2_str}  [{budget_per_seed - PHASE1_QUERIES} queries]")

    print(f"\n  Total: {TOTAL_BUDGET} queries across {seeds} seeds")
    print(f"\n  Approve? [Y/n]: ", end="", flush=True)


# ── Round runner ──────────────────────────────────────────────────────────────

def run_round(session, round_id: str, interactive: bool = False):
    """Run full observe + submit pipeline for one active round."""
    detail = get_round_detail(session, round_id)
    save_round_detail(round_id, detail)

    width = detail["map_width"]
    height = detail["map_height"]
    seeds = detail["seeds_count"]
    round_num = detail["round_number"]
    closes_at = get_round_closes_at(detail)

    log(f"Round {round_num} active — {width}×{height}, {seeds} seeds",
        {"round_id": round_id, "closes_at": str(closes_at)})

    if closes_at:
        now = datetime.now(timezone.utc)
        minutes_left = (closes_at - now).total_seconds() / 60
        log(f"Time remaining: {minutes_left:.0f} minutes")
        if minutes_left < 5:
            log("WARNING: Less than 5 minutes left — skipping round")
            return

    budget_per_seed = TOTAL_BUDGET // seeds

    # Parse initial states + rank viewports (free — no queries used)
    initial_classes_per_seed = []
    ranked_per_seed = []
    for i, state in enumerate(detail["initial_states"]):
        cls_map = parse_initial_classes(state)
        initial_classes_per_seed.append(cls_map)
        ranked = rank_viewports_by_volatility(cls_map, width, height)
        ranked_per_seed.append(ranked)
        n_volatile = int((~np.isin(cls_map, list(STATIC_CLASSES))).sum())
        n_settlements = int((cls_map == 1).sum())
        log(f"  Seed {i}: {n_settlements} settlements, {n_volatile} volatile cells")

    # ── Interactive checkpoint ─────────────────────────────────────────────────
    if interactive:
        print_query_plan(
            round_num, seeds,
            initial_classes_per_seed, ranked_per_seed, budget_per_seed,
        )
        answer = sys.stdin.readline().strip().lower()
        if answer in ("n", "no"):
            log("Round skipped by user (interactive mode)")
            return
        log("Query plan approved — proceeding")

    counts_per_seed = [
        np.zeros((height, width, N_CLASSES), dtype=int)
        for _ in range(seeds)
    ]

    # ── Phase 1: quick baseline, submit all seeds ──────────────────────────────
    log(f"Phase 1: {PHASE1_QUERIES} queries/seed → submit all seeds")
    for seed_idx in range(seeds):
        ranked = ranked_per_seed[seed_idx]
        counts = counts_per_seed[seed_idx]

        for vx, vy in ranked[:PHASE1_QUERIES]:
            vw = min(15, width - vx)
            vh = min(15, height - vy)
            result = simulate(session, round_id, seed_idx, vx, vy, vw, vh)
            accumulate(counts, result, vx, vy)
            save_query(round_id, seed_idx, vx, vy, result)
            time.sleep(0.22)

        prediction = build_prediction(
            counts, initial_classes_per_seed[seed_idx], height, width
        )
        submit_prediction(session, round_id, seed_idx, prediction)
        save_prediction(round_id, seed_idx, prediction)
        obs = int((counts.sum(axis=2) > 0).sum())
        log(f"  Seed {seed_idx}: submitted ({obs}/{width*height} cells observed)")
        time.sleep(0.6)

    log("✓ All seeds on the board")

    # ── Phase 2: improve all seeds, resubmit each ─────────────────────────────
    phase2_per_seed = budget_per_seed - PHASE1_QUERIES
    log(f"Phase 2: {phase2_per_seed} more queries/seed → resubmit all")
    for seed_idx in range(seeds):
        ranked = ranked_per_seed[seed_idx]
        counts = counts_per_seed[seed_idx]

        for vx, vy in ranked[PHASE1_QUERIES:budget_per_seed]:
            vw = min(15, width - vx)
            vh = min(15, height - vy)
            result = simulate(session, round_id, seed_idx, vx, vy, vw, vh)
            accumulate(counts, result, vx, vy)
            save_query(round_id, seed_idx, vx, vy, result)
            time.sleep(0.22)

        prediction = build_prediction(
            counts, initial_classes_per_seed[seed_idx], height, width
        )
        submit_prediction(session, round_id, seed_idx, prediction)
        save_prediction(round_id, seed_idx, prediction)
        obs = int((counts.sum(axis=2) > 0).sum())
        avg = counts.sum(axis=2).mean()
        log(f"  Seed {seed_idx}: resubmitted ({obs}/{width*height} cells, {avg:.1f} avg obs/cell)")
        time.sleep(0.6)

    log(f"Round {round_num} done — all {seeds} seeds submitted",
        {"round_id": round_id, "seeds_submitted": seeds})


# ── Already-ran guard ─────────────────────────────────────────────────────────

def already_ran(round_id: str) -> bool:
    """Check log to avoid re-running a round we already submitted."""
    if not LOG_FILE.exists():
        return False
    with LOG_FILE.open() as f:
        for line in f:
            try:
                entry = json.loads(line)
                if entry.get("round_id") == round_id and "seeds_submitted" in entry:
                    return True
            except json.JSONDecodeError:
                pass
    return False


# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("token", help="JWT token from app.ainm.no")
    parser.add_argument("--poll", type=int, default=120,
                        help="Seconds between round checks (default: 120)")
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="Pause before spending queries for human approval")
    args = parser.parse_args()

    session = make_session(args.token)
    mode = "interactive" if args.interactive else "autonomous"
    log(f"Daemon started — polling for active rounds [{mode} mode]")

    while True:
        try:
            rounds = session.get(f"{BASE}/astar-island/rounds").json()
            active = next((r for r in rounds if r["status"] == "active"), None)

            if active:
                round_id = active["id"]
                if already_ran(round_id):
                    log(f"Round {active['round_number']} already submitted — waiting")
                else:
                    run_round(session, round_id, interactive=args.interactive)
            else:
                # Collect ground truth from completed rounds
                completed = [r for r in rounds if r["status"] == "completed"]
                for r in completed:
                    n = fetch_and_save_all_ground_truth(
                        session, r["id"], r["seeds_count"], BASE
                    )
                    if n:
                        log(f"Saved ground truth for round {r['round_number']} ({n} seeds)")
                        training_data_summary()

                pending = [r for r in rounds if r["status"] == "pending"]
                if pending:
                    log(f"No active round. {len(pending)} pending. Waiting…")
                else:
                    log("No active or pending rounds. Waiting…")

        except KeyboardInterrupt:
            log("Daemon stopped by user")
            sys.exit(0)
        except Exception as e:
            log(f"ERROR: {e} — will retry in {args.poll}s")

        time.sleep(args.poll)


if __name__ == "__main__":
    main()
