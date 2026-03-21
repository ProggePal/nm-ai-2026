"""Thin API client for Astar Island endpoints."""

from __future__ import annotations

import os
import time

import requests
from dotenv import load_dotenv

from .types import Observation, WorldState

load_dotenv()

BASE_URL = "https://api.ainm.no"

# Rate limit tracking
_last_simulate_time: float = 0.0
_last_submit_time: float = 0.0
SIMULATE_INTERVAL = 0.22  # ~4.5 req/sec, safely under 5/sec limit
SUBMIT_INTERVAL = 0.55    # ~1.8 req/sec, safely under 2/sec limit


def _get_session() -> requests.Session:
    """Create an authenticated session."""
    token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJkNmFmNDU4NC0xYmNmLTQxZGMtYWU3Yy04NzEyNmVmMmVmMTIiLCJlbWFpbCI6ImVyaWtAYXBwZmFybS5pbyIsImlzX2FkbWluIjpmYWxzZSwiZXhwIjoxNzc0NjI2NjAwfQ.bsUzF162S2DQUbWg_3W8C1ohAj9kI8j1qxrt_ko95cY"
    if not token:
        raise RuntimeError(
            "JWT_TOKEN not set. Set it in .env or as an environment variable."
        )
    session = requests.Session()
    session.headers["Authorization"] = f"Bearer {token}"
    return session


_session: requests.Session | None = None


def get_session() -> requests.Session:
    """Get or create the global authenticated session."""
    global _session
    if _session is None:
        _session = _get_session()
    return _session


def _rate_limit_simulate() -> None:
    """Enforce rate limit for simulate endpoint."""
    global _last_simulate_time
    elapsed = time.time() - _last_simulate_time
    if elapsed < SIMULATE_INTERVAL:
        time.sleep(SIMULATE_INTERVAL - elapsed)
    _last_simulate_time = time.time()


def _rate_limit_submit() -> None:
    """Enforce rate limit for submit endpoint."""
    global _last_submit_time
    elapsed = time.time() - _last_submit_time
    if elapsed < SUBMIT_INTERVAL:
        time.sleep(SUBMIT_INTERVAL - elapsed)
    _last_submit_time = time.time()


def get_rounds() -> list[dict]:
    """List all rounds."""
    resp = get_session().get(f"{BASE_URL}/astar-island/rounds")
    resp.raise_for_status()
    return resp.json()


def get_active_round() -> dict | None:
    """Get the active round, or None if no round is active."""
    rounds = get_rounds()
    return next((r for r in rounds if r["status"] == "active"), None)


def get_round_detail(round_id: str) -> dict:
    """Get round details including initial states."""
    resp = get_session().get(f"{BASE_URL}/astar-island/rounds/{round_id}")
    resp.raise_for_status()
    return resp.json()


def get_budget() -> dict:
    """Get remaining query budget for the active round."""
    resp = get_session().get(f"{BASE_URL}/astar-island/budget")
    resp.raise_for_status()
    return resp.json()


def simulate_query(
    round_id: str,
    seed_index: int,
    viewport_x: int,
    viewport_y: int,
    viewport_w: int = 15,
    viewport_h: int = 15,
) -> Observation:
    """Run a simulation query and return an Observation."""
    _rate_limit_simulate()
    resp = get_session().post(
        f"{BASE_URL}/astar-island/simulate",
        json={
            "round_id": round_id,
            "seed_index": seed_index,
            "viewport_x": viewport_x,
            "viewport_y": viewport_y,
            "viewport_w": viewport_w,
            "viewport_h": viewport_h,
        },
    )
    resp.raise_for_status()
    return Observation.from_api(seed_index, resp.json())


def submit_prediction(
    round_id: str, seed_index: int, prediction: list[list[list[float]]]
) -> dict:
    """Submit a prediction tensor for one seed."""
    _rate_limit_submit()
    resp = get_session().post(
        f"{BASE_URL}/astar-island/submit",
        json={
            "round_id": round_id,
            "seed_index": seed_index,
            "prediction": prediction,
        },
    )
    resp.raise_for_status()
    return resp.json()


def get_my_rounds() -> list[dict]:
    """Get rounds with team scores and budget info."""
    resp = get_session().get(f"{BASE_URL}/astar-island/my-rounds")
    resp.raise_for_status()
    return resp.json()


def get_analysis(round_id: str, seed_index: int) -> dict:
    """Get post-round analysis for a specific seed."""
    resp = get_session().get(
        f"{BASE_URL}/astar-island/analysis/{round_id}/{seed_index}"
    )
    resp.raise_for_status()
    return resp.json()
