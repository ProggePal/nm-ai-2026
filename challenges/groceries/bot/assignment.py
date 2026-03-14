"""
Task assignment: maps bots to items using the Hungarian algorithm.

Cost of assigning bot B to item I:
  dist(B.position → I.position) + dist_to_nearest_dropoff(I.position)

This minimises total travel cost across all bots simultaneously,
preventing the "all bots race the same item" failure of greedy approaches.
"""
from __future__ import annotations
from typing import Optional
import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

from state import Bot, Item, GameState
from pathfinding import precompute_drop_distances, bfs_distance


def assign_tasks(
    bots: list[Bot],
    items: list[Item],
    state: GameState,
    drop_dist: dict[tuple[int, int], int],
    walls: frozenset[tuple[int, int]],
) -> dict[int, Optional[Item]]:
    """
    Returns a mapping of bot_id → Item (or None if no item assigned).

    Only bots with inventory space (<3 items) are considered as collectors.
    Items already claimed (in any bot's inventory) are excluded.
    """
    # Filter to bots that can still pick up items
    collecting_bots = [b for b in bots if len(b.inventory) < 3]

    if not collecting_bots or not items:
        return {b.id: None for b in bots}

    n_bots = len(collecting_bots)
    n_items = len(items)

    # Build cost matrix  [bots × items]
    INF = 9999
    cost = np.full((n_bots, n_items), INF, dtype=float)

    w, h = state.grid.width, state.grid.height

    for bi, bot in enumerate(collecting_bots):
        for ii, item in enumerate(items):
            dist_to_item = bfs_distance(bot.position, item.position, walls, w, h)
            dist_to_drop = drop_dist.get(item.position, INF)
            cost[bi, ii] = dist_to_item + dist_to_drop

    # Pad to square if needed (Hungarian requires square or handled by scipy)
    assignment: dict[int, Optional[Item]] = {b.id: None for b in bots}

    if _HAS_SCIPY:
        row_ind, col_ind = linear_sum_assignment(cost)
        for ri, ci in zip(row_ind, col_ind):
            if cost[ri, ci] < INF:
                assignment[collecting_bots[ri].id] = items[ci]
    else:
        # Fallback: greedy nearest (if scipy not installed)
        claimed: set[int] = set()
        for bi, bot in enumerate(collecting_bots):
            best_cost = INF
            best_item_idx = None
            for ii in range(n_items):
                if ii not in claimed and cost[bi, ii] < best_cost:
                    best_cost = cost[bi, ii]
                    best_item_idx = ii
            if best_item_idx is not None:
                assignment[bot.id] = items[best_item_idx]
                claimed.add(best_item_idx)

    return assignment
