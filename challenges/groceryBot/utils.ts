import type { Position } from "./types.ts";

export function posKey(pos: Position): string {
  return `${pos[0]},${pos[1]}`;
}

export function parsePos(key: string): Position {
  const parts = key.split(",");
  return [Number(parts[0]), Number(parts[1])];
}

export function manhattanDistance(a: Position, b: Position): number {
  return Math.abs(a[0] - b[0]) + Math.abs(a[1] - b[1]);
}

export function posEquals(a: Position, b: Position): boolean {
  return a[0] === b[0] && a[1] === b[1];
}

export function positionsInclude(positions: Position[], target: Position): boolean {
  return positions.some((pos) => posEquals(pos, target));
}
