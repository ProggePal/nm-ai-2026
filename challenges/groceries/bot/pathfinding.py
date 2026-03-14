"""
Pathfinding utilities.

- bfs(): wall-aware BFS returning full action list
- reverse_bfs(): flood-fill from a set of goal cells — precompute distance matrix
- precompute_drop_distances(): O(1) dist-to-dropoff lookup after setup
"""
from __future__ import annotations
from collections import deque
from typing import Optional


MOVES = [
    (0, -1, "move_up"),
    (0,  1, "move_down"),
    (-1, 0, "move_left"),
    (1,  0, "move_right"),
]


def bfs(
    start: tuple[int, int],
    goal: tuple[int, int],
    walls: frozenset[tuple[int, int]],
    width: int,
    height: int,
    blocked: frozenset[tuple[int, int]] = frozenset(),
) -> Optional[list[str]]:
    """
    Returns the shortest list of action strings from start to goal,
    or None if no path exists.

    `blocked` = cells currently reserved by other bots (treated as walls for
    this call). Do NOT include `start` itself in blocked.
    """
    if start == goal:
        return []

    visited: set[tuple[int, int]] = {start}
    # Queue: (current_pos, path_so_far)
    queue: deque[tuple[tuple[int, int], list[str]]] = deque([(start, [])])

    impassable = walls | blocked

    while queue:
        pos, path = queue.popleft()
        for dx, dy, action in MOVES:
            nx, ny = pos[0] + dx, pos[1] + dy
            npos = (nx, ny)
            if (
                0 <= nx < width
                and 0 <= ny < height
                and npos not in impassable
                and npos not in visited
            ):
                new_path = path + [action]
                if npos == goal:
                    return new_path
                visited.add(npos)
                queue.append((npos, new_path))

    return None  # no path


def bfs_distance(
    start: tuple[int, int],
    goal: tuple[int, int],
    walls: frozenset[tuple[int, int]],
    width: int,
    height: int,
) -> int:
    """Returns BFS distance (ignoring other bots), or 9999 if unreachable."""
    path = bfs(start, goal, walls, width, height)
    return len(path) if path is not None else 9999


def reverse_bfs(
    goal: tuple[int, int],
    walls: frozenset[tuple[int, int]],
    width: int,
    height: int,
) -> dict[tuple[int, int], int]:
    """
    Flood-fill BFS starting from `goal`, returning a dict mapping every
    reachable cell to its distance from `goal`.

    Use this to precompute drop-off distances once per game.
    """
    dist: dict[tuple[int, int], int] = {goal: 0}
    queue: deque[tuple[tuple[int, int], int]] = deque([(goal, 0)])

    while queue:
        pos, d = queue.popleft()
        for dx, dy, _ in MOVES:
            nx, ny = pos[0] + dx, pos[1] + dy
            npos = (nx, ny)
            if (
                0 <= nx < width
                and 0 <= ny < height
                and npos not in walls
                and npos not in dist
            ):
                dist[npos] = d + 1
                queue.append((npos, d + 1))

    return dist


def precompute_drop_distances(
    drop_off_zones: list[tuple[int, int]],
    walls: frozenset[tuple[int, int]],
    width: int,
    height: int,
) -> dict[tuple[int, int], int]:
    """
    For each navigable cell, returns the distance to the NEAREST drop-off zone.
    Computed once at round 0; used as O(1) lookup every subsequent round.
    """
    # Start BFS from all drop zones simultaneously (multi-source BFS)
    dist: dict[tuple[int, int], int] = {}
    queue: deque[tuple[tuple[int, int], int]] = deque()

    for zone in drop_off_zones:
        dist[zone] = 0
        queue.append((zone, 0))

    while queue:
        pos, d = queue.popleft()
        for dx, dy, _ in MOVES:
            nx, ny = pos[0] + dx, pos[1] + dy
            npos = (nx, ny)
            if (
                0 <= nx < width
                and 0 <= ny < height
                and npos not in walls
                and npos not in dist
            ):
                dist[npos] = d + 1
                queue.append((npos, d + 1))

    return dist


def next_step_toward(
    start: tuple[int, int],
    goal: tuple[int, int],
    walls: frozenset[tuple[int, int]],
    width: int,
    height: int,
    blocked: frozenset[tuple[int, int]] = frozenset(),
) -> Optional[str]:
    """Convenience: returns just the first action of the BFS path, or None."""
    path = bfs(start, goal, walls, width, height, blocked)
    if not path:
        return None
    return path[0]
