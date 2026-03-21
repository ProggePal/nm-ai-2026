"""Astar Island competition dashboard — lightweight Flask app for Cloud Run."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import requests
from flask import Flask, jsonify, render_template

app = Flask(__name__)

BASE_URL = "https://api.ainm.no"
JWT_TOKEN = os.environ.get("JWT_TOKEN", "")
DATA_DIR = Path(os.environ.get("DATA_DIR", "../island_sim/data"))


def _session() -> requests.Session:
    s = requests.Session()
    s.headers["Authorization"] = f"Bearer {JWT_TOKEN}"
    return s


def _api_get(path: str) -> dict | list:
    resp = _session().get(f"{BASE_URL}{path}", timeout=10)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# API routes — called by the frontend JS
# ---------------------------------------------------------------------------


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/state")
def api_state():
    """Single endpoint that returns everything the dashboard needs."""
    try:
        my_rounds = _api_get("/astar-island/my-rounds")
    except Exception as e:
        my_rounds = []
        print(f"API error (my-rounds): {e}")

    try:
        all_rounds = _api_get("/astar-island/rounds")
    except Exception as e:
        all_rounds = []
        print(f"API error (rounds): {e}")

    # Find active round
    active = next((r for r in all_rounds if r.get("status") == "active"), None)

    # Budget info
    budget = None
    if active:
        try:
            budget = _api_get("/astar-island/budget")
        except Exception:
            pass

    # Best params from disk (if available)
    best_params_file = DATA_DIR / "rounds" / "best_params.json"
    best_params = None
    if best_params_file.exists():
        try:
            best_params = json.loads(best_params_file.read_text())
        except Exception:
            pass

    # Enrich rounds with score data from my-rounds
    my_rounds_map = {r.get("round_number", r.get("id")): r for r in my_rounds}

    # Build round history
    rounds_data = []
    for r in sorted(my_rounds, key=lambda x: x.get("round_number", 0)):
        rounds_data.append({
            "round_number": r.get("round_number"),
            "status": r.get("status"),
            "event_date": r.get("event_date"),
            "round_score": r.get("round_score"),
            "seed_scores": r.get("seed_scores", []),
            "rank": r.get("rank"),
            "total_teams": r.get("total_teams"),
            "round_weight": r.get("round_weight"),
            "queries_used": r.get("queries_used"),
            "queries_max": r.get("queries_max"),
            "seeds_submitted": r.get("seeds_submitted"),
            "started_at": r.get("started_at"),
            "closes_at": r.get("closes_at"),
        })

    # Active round details
    active_info = None
    if active:
        closes_at = active.get("closes_at")
        remaining_sec = None
        if closes_at:
            try:
                deadline = datetime.fromisoformat(closes_at)
                remaining_sec = max(0, (deadline - datetime.now(timezone.utc)).total_seconds())
            except Exception:
                pass

        # Find matching my-round for the active round
        active_my = my_rounds_map.get(active.get("round_number"))
        active_info = {
            "round_number": active.get("round_number"),
            "status": active.get("status"),
            "started_at": active.get("started_at"),
            "closes_at": closes_at,
            "remaining_seconds": remaining_sec,
            "round_weight": active.get("round_weight"),
            "seeds_count": active.get("seeds_count", 5),
            "prediction_window_minutes": active.get("prediction_window_minutes", 165),
            "budget": budget,
            "round_score": active_my.get("round_score") if active_my else None,
            "seed_scores": active_my.get("seed_scores") if active_my else None,
            "seeds_submitted": (active_my.get("seeds_submitted") if active_my else None) or 0,
            "rank": active_my.get("rank") if active_my else None,
            "total_teams": active_my.get("total_teams") if active_my else None,
        }

    # Compute weighted best score
    completed = [r for r in rounds_data if r["status"] == "completed" and r["round_score"]]
    weighted_score = None
    if completed:
        total_weight = sum(r["round_weight"] for r in completed if r["round_weight"])
        if total_weight > 0:
            weighted_score = sum(
                r["round_score"] * r["round_weight"]
                for r in completed
                if r["round_score"] and r["round_weight"]
            ) / total_weight

    return jsonify({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "active_round": active_info,
        "rounds": rounds_data,
        "weighted_score": weighted_score,
        "best_params": {
            "score": best_params.get("score") if best_params else None,
            "source": best_params.get("source") if best_params else None,
            "timestamp": best_params.get("timestamp") if best_params else None,
        } if best_params else None,
        "total_rounds_completed": len(completed),
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=os.environ.get("DEBUG") == "1")
