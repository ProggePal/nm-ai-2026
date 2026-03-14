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

  const visited = new Set<string>([startKey]);
  // Queue entries: [posKey, path]
  const queue: [string, string[]][] = [[startKey, []]];
  let head = 0;

  while (head < queue.length) {
    const [currentKey, path] = queue[head++];
    const current = parsePos(currentKey);

    for (const [dx, dy, action] of MOVES) {
      const nx = current[0] + dx;
      const ny = current[1] + dy;
      const nKey = posKey([nx, ny]);

      if (
        nx >= 0 &&
        nx < width &&
        ny >= 0 &&
        ny < height &&
        !walls.has(nKey) &&
        !blocked.has(nKey) &&
        !visited.has(nKey)
      ) {
        const newPath = [...path, action];
        if (nKey === goalKey) return newPath;
        visited.add(nKey);
        queue.push([nKey, newPath]);
      }
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
  const path = bfs(start, goal, walls, width, height);
  return path !== null ? path.length : 9999;
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
