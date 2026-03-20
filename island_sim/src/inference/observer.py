"""Viewport placement strategy for maximizing observation value."""

from __future__ import annotations

import numpy as np

from ..constants import MOUNTAIN, OCEAN, SETTLEMENT, PORT
from ..types import WorldState

MAX_VIEWPORT = 15
TOTAL_BUDGET = 50
NUM_SEEDS = 5


def plan_viewports(
    initial_states: list[WorldState],
    budget: int = TOTAL_BUDGET,
    systematic_fraction: float = 0.6,
) -> list[dict]:
    """Plan viewport queries to maximize coverage of dynamic cells.

    Returns a list of query dicts: {seed_index, viewport_x, viewport_y, viewport_w, viewport_h}.
    Split into systematic (cover settlement clusters) and adaptive (reserved for later).
    """
    systematic_budget = int(budget * systematic_fraction)
    queries = []

    # Distribute queries across seeds round-robin (handles budget < NUM_SEEDS)
    per_seed_base = systematic_budget // NUM_SEEDS
    extra = systematic_budget % NUM_SEEDS

    for seed_idx, state in enumerate(initial_states):
        n = per_seed_base + (1 if seed_idx < extra else 0)
        if n <= 0:
            continue
        seed_queries = _plan_seed_viewports(state, seed_idx, n)
        queries.extend(seed_queries)

    return queries


def _plan_seed_viewports(
    state: WorldState, seed_index: int, num_queries: int
) -> list[dict]:
    """Plan viewports for a single seed, focusing on settlement clusters."""
    # Build a "dynamic cell" heatmap — cells likely to change
    dynamic_map = _build_dynamic_heatmap(state)

    # Greedily place viewports to cover the most dynamic cells
    covered = np.zeros_like(dynamic_map, dtype=bool)
    queries = []

    for _ in range(num_queries):
        best_score = -1
        best_x, best_y = 0, 0

        # Search over possible viewport positions
        for vy in range(0, state.height - MAX_VIEWPORT + 1, 3):
            for vx in range(0, state.width - MAX_VIEWPORT + 1, 3):
                # Score = sum of uncovered dynamic cell values in this viewport
                region = dynamic_map[vy : vy + MAX_VIEWPORT, vx : vx + MAX_VIEWPORT]
                mask = ~covered[vy : vy + MAX_VIEWPORT, vx : vx + MAX_VIEWPORT]
                score = float(np.sum(region * mask))
                if score > best_score:
                    best_score = score
                    best_x, best_y = vx, vy

        # Mark covered
        covered[best_y : best_y + MAX_VIEWPORT, best_x : best_x + MAX_VIEWPORT] = True

        queries.append({
            "seed_index": seed_index,
            "viewport_x": best_x,
            "viewport_y": best_y,
            "viewport_w": MAX_VIEWPORT,
            "viewport_h": MAX_VIEWPORT,
        })

    return queries


def _build_dynamic_heatmap(state: WorldState) -> np.ndarray:
    """Build a heatmap where higher values = more likely to be dynamic."""
    heatmap = np.zeros((state.height, state.width), dtype=np.float32)

    # Settlements and their neighborhoods are the most dynamic
    for settlement in state.settlements:
        if not settlement.alive:
            continue
        cx, cy = settlement.x, settlement.y

        # High value at settlement, decreasing with distance
        for dy in range(-8, 9):
            for dx in range(-8, 9):
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < state.width and 0 <= ny < state.height:
                    dist = max(abs(dx), abs(dy))
                    terrain = int(state.grid[ny, nx])
                    # Skip static terrain
                    if terrain in (OCEAN, MOUNTAIN):
                        continue
                    # Settlement cells get highest weight
                    if dist == 0:
                        heatmap[ny, nx] += 10.0
                    else:
                        # Decay with distance, closer cells more likely to change
                        heatmap[ny, nx] += max(0, 5.0 - dist * 0.5)

    return heatmap


def plan_adaptive_viewports(
    initial_states: list[WorldState],
    observations: list,
    remaining_budget: int,
) -> list[dict]:
    """Plan adaptive viewports based on initial observations.

    Focus on areas where the model is most uncertain or where
    observations revealed unexpected results.
    """
    queries = []
    per_seed = remaining_budget // NUM_SEEDS
    extra = remaining_budget % NUM_SEEDS

    for seed_idx, state in enumerate(initial_states):
        n = per_seed + (1 if seed_idx < extra else 0)
        if n <= 0:
            continue

        # For adaptive, focus on areas with settlements we haven't observed well
        seed_obs = [obs for obs in observations if obs.seed_index == seed_idx]
        covered = np.zeros((state.height, state.width), dtype=bool)
        for obs in seed_obs:
            covered[
                obs.viewport_y : obs.viewport_y + obs.viewport_h,
                obs.viewport_x : obs.viewport_x + obs.viewport_w,
            ] = True

        # Place viewports on uncovered dynamic areas
        dynamic_map = _build_dynamic_heatmap(state)
        dynamic_map[covered] = 0  # Zero out already-covered cells

        for _ in range(n):
            best_score = -1
            best_x, best_y = 0, 0

            for vy in range(0, state.height - MAX_VIEWPORT + 1, 2):
                for vx in range(0, state.width - MAX_VIEWPORT + 1, 2):
                    region = dynamic_map[
                        vy : vy + MAX_VIEWPORT, vx : vx + MAX_VIEWPORT
                    ]
                    score = float(np.sum(region))
                    if score > best_score:
                        best_score = score
                        best_x, best_y = vx, vy

            # If nothing left to observe, use a random position
            if best_score <= 0:
                best_x = int(np.random.randint(0, max(1, state.width - MAX_VIEWPORT)))
                best_y = int(np.random.randint(0, max(1, state.height - MAX_VIEWPORT)))

            dynamic_map[
                best_y : best_y + MAX_VIEWPORT, best_x : best_x + MAX_VIEWPORT
            ] = 0

            queries.append({
                "seed_index": seed_idx,
                "viewport_x": best_x,
                "viewport_y": best_y,
                "viewport_w": MAX_VIEWPORT,
                "viewport_h": MAX_VIEWPORT,
            })

    return queries
