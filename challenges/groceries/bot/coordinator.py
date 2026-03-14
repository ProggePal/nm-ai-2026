"""
Multi-bot coordinator.

Processes bots in priority order and resolves movement conflicts via a
reservation table. Bots holding deliverable items move first.

Priority score (higher = moves first):
  holding_useful * 1000 + inventory_count * 100 - dist_to_nearest_dropoff
"""
from __future__ import annotations
from typing import Optional

from state import Bot, Item, GameState
from pathfinding import precompute_drop_distances
from assignment import assign_tasks
from planner import plan_bot_action


DIRECTION_DELTAS = {
    "move_up":    (0, -1),
    "move_down":  (0,  1),
    "move_left":  (-1, 0),
    "move_right": (1,  0),
}


def apply_move(pos: tuple[int, int], action: str) -> tuple[int, int]:
    if action in DIRECTION_DELTAS:
        dx, dy = DIRECTION_DELTAS[action]
        return (pos[0] + dx, pos[1] + dy)
    return pos  # pick_up / drop_off / wait — bot stays in place


def priority_score(
    bot: Bot,
    state: GameState,
    drop_dist: dict[tuple[int, int], int],
) -> int:
    active = state.active_order
    active_needed = active.still_needed if active else []
    holding_useful = any(i in active_needed for i in bot.inventory)
    inv_count = len(bot.inventory)
    dist = drop_dist.get(bot.position, 9999)
    return int(holding_useful) * 1000 + inv_count * 100 - dist


def plan_round(
    state: GameState,
    drop_dist: dict[tuple[int, int], int],
) -> list[dict]:
    """
    Plans actions for all bots for this round.
    Returns a list of action dicts suitable for the WebSocket response.
    """
    walls = state.grid.walls
    w, h = state.grid.width, state.grid.height
    drop_zones = state.drop_off_zones

    # Needed items for active order (excluding in-transit)
    active = state.active_order
    truly_needed_types = state.truly_needed(active) if active else []
    available_items = state.available_items_for(truly_needed_types)

    # Task assignment: bot_id → Item | None
    assignment = assign_tasks(state.bots, available_items, state, drop_dist, walls)

    # Sort bots by priority (highest first)
    bots_ordered = sorted(
        state.bots,
        key=lambda b: -priority_score(b, state, drop_dist),
    )

    # Reservation table: next_cell → bot_id
    reserved: dict[tuple[int, int], int] = {}

    # Current positions (to detect swap deadlocks)
    current_positions = {b.id: b.position for b in state.bots}

    # Congestion counter for drop-off zones
    dropoff_congestion: dict[tuple[int, int], int] = {}
    for bot in state.bots:
        if bot.position in drop_zones:
            dropoff_congestion[bot.position] = dropoff_congestion.get(bot.position, 0) + 1

    results: dict[int, dict] = {}

    for bot in bots_ordered:
        pos = bot.position

        # Cells already reserved by higher-priority bots
        blocked = frozenset(reserved.keys()) - {pos}

        action_dict = plan_bot_action(
            bot=bot,
            state=state,
            assigned_item=assignment.get(bot.id),
            drop_dist=drop_dist,
            walls=walls,
            blocked=blocked,
            dropoff_congestion=dropoff_congestion,
        )

        action = action_dict.get("action", "wait")
        desired_cell = apply_move(pos, action)

        # Check for swap deadlock: if desired_cell is occupied by a bot that
        # wants to move to our current cell
        swap_deadlock = False
        for other_id, other_result in results.items():
            other_action = other_result.get("action", "wait")
            other_pos = current_positions[other_id]
            other_desired = apply_move(other_pos, other_action)
            if desired_cell == other_pos and other_desired == pos:
                swap_deadlock = True
                break

        # Resolve conflicts
        if desired_cell in reserved or swap_deadlock:
            # Try perpendicular sidestep
            sidestep = _find_sidestep(pos, action, walls, reserved, w, h)
            if sidestep:
                action = sidestep
                desired_cell = apply_move(pos, sidestep)
                action_dict = {"action": sidestep}
            else:
                # Give up — wait in place
                action = "wait"
                desired_cell = pos
                action_dict = {"action": "wait"}

        reserved[desired_cell] = bot.id
        results[bot.id] = action_dict

    # Build final output list
    output = []
    for bot in state.bots:
        action_dict = results[bot.id]
        entry = {"bot": bot.id}
        entry.update(action_dict)
        output.append(entry)

    return output


def _find_sidestep(
    pos: tuple[int, int],
    intended_action: str,
    walls: frozenset[tuple[int, int]],
    reserved: dict[tuple[int, int], int],
    width: int,
    height: int,
) -> Optional[str]:
    """Try perpendicular moves to the intended direction."""
    perp = {
        "move_up":    ["move_left", "move_right"],
        "move_down":  ["move_left", "move_right"],
        "move_left":  ["move_up", "move_down"],
        "move_right": ["move_up", "move_down"],
    }
    candidates = perp.get(intended_action, [])
    for action in candidates:
        dx, dy = DIRECTION_DELTAS[action]
        nx, ny = pos[0] + dx, pos[1] + dy
        npos = (nx, ny)
        if (
            0 <= nx < width
            and 0 <= ny < height
            and npos not in walls
            and npos not in reserved
        ):
            return action
    return None
