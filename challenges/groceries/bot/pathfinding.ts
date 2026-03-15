import type { Position, DistanceMap } from "./types.ts";
import { posKey, parsePos } from "./utils.ts";

export const MOVES: [number, number, string][] = [
  [0, -1, "move_up"],
  [0, 1, "move_down"],
  [-1, 0, "move_left"],
  [1, 0, "move_right"],
];

export function bfs(
  start: Position,
  goal: Position,
  walls: Set<string>,
  width: number,
  height: number,
  blocked: Set<string> = new Set(),
): string[] | null {
  const startKey = posKey(start);
  const goalKey = posKey(goal);

  if (startKey === goalKey) return [];

  // Parent map for O(n) path reconstruction instead of O(n²) path copying
  const parent = new Map<string, [string, string]>();
  const visited = new Set<string>([startKey]);
  const queue: string[] = [startKey];
  let head = 0;

  while (head < queue.length) {
    const currentKey = queue[head++];
    const current = parsePos(currentKey);

    for (const [dx, dy, action] of MOVES) {
      const nx = current[0] + dx;
      const ny = current[1] + dy;
      if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;

      const nKey = posKey([nx, ny]);

      // Check goal BEFORE wall/visited — allows reaching items on wall cells (shelves)
      if (nKey === goalKey) {
        const path: string[] = [action];
        let key = currentKey;
        while (parent.has(key)) {
          const [pKey, pAction] = parent.get(key)!;
          path.push(pAction);
          key = pKey;
        }
        path.reverse();
        return path;
      }

      if (visited.has(nKey) || walls.has(nKey) || blocked.has(nKey)) continue;

      visited.add(nKey);
      parent.set(nKey, [currentKey, action]);
      queue.push(nKey);
    }
  }

  return null;
}

export function bfsDistance(
  start: Position,
  goal: Position,
  walls: Set<string>,
  width: number,
  height: number,
): number {
  const startKey = posKey(start);
  const goalKey = posKey(goal);

  if (startKey === goalKey) return 0;

  // Distance-only BFS — no path tracking, no allocations per node
  const visited = new Set<string>([startKey]);
  const queue: [string, number][] = [[startKey, 0]];
  let head = 0;

  while (head < queue.length) {
    const [currentKey, dist] = queue[head++];
    const current = parsePos(currentKey);

    for (const [dx, dy] of MOVES) {
      const nx = current[0] + dx;
      const ny = current[1] + dy;
      if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;

      const nKey = posKey([nx, ny]);

      // Goal check before wall check (matches bfs behavior for wall-cell items)
      if (nKey === goalKey) return dist + 1;
      if (visited.has(nKey) || walls.has(nKey)) continue;

      visited.add(nKey);
      queue.push([nKey, dist + 1]);
    }
  }

  return 9999;
}

export function reverseBfs(
  goal: Position,
  walls: Set<string>,
  width: number,
  height: number,
): DistanceMap {
  const dist: DistanceMap = new Map();
  const goalKey = posKey(goal);
  dist.set(goalKey, 0);

  const queue: [string, number][] = [[goalKey, 0]];
  let head = 0;

  while (head < queue.length) {
    const [currentKey, d] = queue[head++];
    const current = parsePos(currentKey);

    for (const [dx, dy] of MOVES) {
      const nx = current[0] + dx;
      const ny = current[1] + dy;
      const nKey = posKey([nx, ny]);

      if (
        nx >= 0 &&
        nx < width &&
        ny >= 0 &&
        ny < height &&
        !walls.has(nKey) &&
        !dist.has(nKey)
      ) {
        dist.set(nKey, d + 1);
        queue.push([nKey, d + 1]);
      }
    }
  }

  return dist;
}

export function precomputeDropDistances(
  dropOffZones: Position[],
  walls: Set<string>,
  width: number,
  height: number,
): DistanceMap {
  const dist: DistanceMap = new Map();
  const queue: [string, number][] = [];

  for (const zone of dropOffZones) {
    const key = posKey(zone);
    dist.set(key, 0);
    queue.push([key, 0]);
  }

  let head = 0;
  while (head < queue.length) {
    const [currentKey, d] = queue[head++];
    const current = parsePos(currentKey);

    for (const [dx, dy] of MOVES) {
      const nx = current[0] + dx;
      const ny = current[1] + dy;
      const nKey = posKey([nx, ny]);

      if (
        nx >= 0 &&
        nx < width &&
        ny >= 0 &&
        ny < height &&
        !walls.has(nKey) &&
        !dist.has(nKey)
      ) {
        dist.set(nKey, d + 1);
        queue.push([nKey, d + 1]);
      }
    }
  }

  return dist;
}

export function nextStepToward(
  start: Position,
  goal: Position,
  walls: Set<string>,
  width: number,
  height: number,
  blocked: Set<string> = new Set(),
): string | null {
  const path = bfs(start, goal, walls, width, height, blocked);
  if (!path || path.length === 0) return null;
  return path[0];
}

/**
 * Follow a precomputed distance gradient — O(1) per call.
 * Picks the unblocked neighbor with the lowest distance value.
 * Used for drop-off navigation to avoid expensive per-round BFS.
 */
export function nextStepViaGradient(
  pos: Position,
  distMap: DistanceMap,
  walls: Set<string>,
  width: number,
  height: number,
  blocked: Set<string> = new Set(),
): string | null {
  const currentDist = distMap.get(posKey(pos)) ?? Infinity;
  let bestAction: string | null = null;
  let bestDist = currentDist;

  for (const [dx, dy, action] of MOVES) {
    const nx = pos[0] + dx;
    const ny = pos[1] + dy;
    if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;

    const nKey = posKey([nx, ny]);
    if (walls.has(nKey) || blocked.has(nKey)) continue;

    const d = distMap.get(nKey);
    if (d !== undefined && d < bestDist) {
      bestDist = d;
      bestAction = action;
    }
  }

  return bestAction;
}
