"""
Per-bot state machine.

States:
  COLLECTING   — navigate to assigned item and pick it up
  DELIVERING   — navigate to drop-off zone and deliver
  PRE_FETCHING — pick up preview-order items when we have spare slots

Inventory fill rule:
  Don't deliver after 1 item — fill to min(3, remaining_needed) first,
  unless end-game or already at drop-off.
"""
from __future__ import annotations
from typing import Optional
from enum import Enum, auto

from state import Bot, Item, GameState, Order
from pathfinding import next_step_toward, precompute_drop_distances


class BotState(Enum):
    COLLECTING = auto()
    DELIVERING = auto()
    PRE_FETCHING = auto()
    IDLE = auto()


def nearest_dropoff(
    pos: tuple[int, int],
    drop_off_zones: list[tuple[int, int]],
    bot_counts: dict[tuple[int, int], int],
) -> tuple[int, int]:
    """
    Returns the least-congested nearest drop-off zone.
    Congestion penalty: 2 per bot within radius 3 of the zone.
    """
    def cost(zone):
        from pathfinding import bfs_distance
        # We only have position here, use Manhattan for tie-breaking
        manhattan = abs(pos[0] - zone[0]) + abs(pos[1] - zone[1])
        congestion = bot_counts.get(zone, 0) * 2
        return manhattan + congestion

    return min(drop_off_zones, key=cost)


def plan_bot_action(
    bot: Bot,
    state: GameState,
    assigned_item: Optional[Item],
    drop_dist: dict[tuple[int, int], int],
    walls: frozenset[tuple[int, int]],
    blocked: frozenset[tuple[int, int]],
    dropoff_congestion: dict[tuple[int, int], int],
) -> dict:
    """
    Returns an action dict for this bot:
      {"action": "move_up"} or {"action": "pick_up", "item_id": "item_0"} etc.
    """
    pos = bot.position
    inventory = bot.inventory
    active = state.active_order
    preview = state.preview_order
    drop_zones = state.drop_off_zones
    w, h = state.grid.width, state.grid.height

    # ── Determine what's useful to deliver ──────────────────────────────────
    active_needed = active.still_needed if active else []
    useful_in_inv = [i for i in inventory if i in active_needed]

    # ── Check if we're standing on a drop-off zone ──────────────────────────
    on_dropoff = pos in drop_zones

    # ── END-GAME: deliver immediately if holding anything useful ────────────
    if state.end_game and useful_in_inv:
        if on_dropoff:
            return {"action": "drop_off"}
        target = nearest_dropoff(pos, drop_zones, dropoff_congestion)
        step = next_step_toward(pos, target, walls, w, h, blocked)
        return {"action": step or "wait"}

    # ── DELIVERING: if holding useful items and should go deliver ────────────
    if useful_in_inv:
        # Inventory fill check: should we pick up more first?
        truly_needed = state.truly_needed(active)
        should_fill_more = (
            len(inventory) < 3
            and len(truly_needed) > 0
            and assigned_item is not None
            and not state.end_game
        )

        if not should_fill_more:
            # Go deliver
            if on_dropoff:
                return {"action": "drop_off"}
            target = nearest_dropoff(pos, drop_zones, dropoff_congestion)
            step = next_step_toward(pos, target, walls, w, h, blocked)
            return {"action": step or "wait"}

    # ── COLLECTING: go pick up assigned item ────────────────────────────────
    if assigned_item is not None and len(inventory) < 3:
        item_pos = assigned_item.position
        dist = abs(pos[0] - item_pos[0]) + abs(pos[1] - item_pos[1])

        if dist == 1:
            # Adjacent — pick up
            return {"action": "pick_up", "item_id": assigned_item.id}
        else:
            step = next_step_toward(pos, item_pos, walls, w, h, blocked)
            return {"action": step or "wait"}

    # ── PRE_FETCHING: pick up preview items if we have slots ────────────────
    if preview and len(inventory) < 3:
        preview_needed_types = preview.still_needed
        preview_items = state.available_items_for(preview_needed_types)

        # Only pre-fetch if the active order is nearly done
        truly_needed = state.truly_needed(active)
        if len(truly_needed) <= 1 and preview_items:
            # Find nearest preview item
            best = min(
                preview_items,
                key=lambda i: abs(pos[0] - i.position[0]) + abs(pos[1] - i.position[1]),
            )
            dist = abs(pos[0] - best.position[0]) + abs(pos[1] - best.position[1])
            if dist == 1:
                return {"action": "pick_up", "item_id": best.id}
            step = next_step_toward(pos, best.position, walls, w, h, blocked)
            if step:
                return {"action": step}

    # ── IDLE: move toward drop-off if holding anything, else wait ────────────
    if inventory:
        if on_dropoff:
            return {"action": "drop_off"}
        target = nearest_dropoff(pos, drop_zones, dropoff_congestion)
        step = next_step_toward(pos, target, walls, w, h, blocked)
        return {"action": step or "wait"}

    return {"action": "wait"}
