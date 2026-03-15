import { describe, it, expect } from "vitest";
import { bfs, bfsDistance, reverseBfs, precomputeDropDistances, nextStepToward, nextStepViaGradient } from "../pathfinding.ts";
import type { Position } from "../types.ts";
import { posKey } from "../utils.ts";

// 5x5 grid with a wall in the middle
// . . . . .
// . . . . .
// . . W . .
// . . . . .
// . . . . .
const walls = new Set([posKey([2, 2])]);
const width = 5;
const height = 5;

describe("bfs", () => {
  it("returns empty path when start equals goal", () => {
    expect(bfs([1, 1], [1, 1], walls, width, height)).toEqual([]);
  });

  it("finds shortest path on open grid", () => {
    const path = bfs([0, 0], [4, 0], new Set(), width, height);
    expect(path).not.toBeNull();
    expect(path!.length).toBe(4);
    // Should be all move_right
    expect(path!.every((a) => a === "move_right")).toBe(true);
  });

  it("navigates around walls", () => {
    const path = bfs([1, 2], [3, 2], walls, width, height);
    expect(path).not.toBeNull();
    // Must go around the wall at (2,2) — length > 2
    expect(path!.length).toBeGreaterThan(2);
  });

  it("reaches wall-cell goals via adjacent cells", () => {
    // Wall at (2,2) — BFS can now reach it from adjacent walkable cells
    const path = bfs([0, 0], [2, 2], walls, width, height);
    expect(path).not.toBeNull();
    expect(path!.length).toBeGreaterThan(0);
  });

  it("returns null when goal is surrounded by walls", () => {
    // Goal (2,2) with all adjacent cells also walled off
    const boxedWalls = new Set([
      posKey([2, 2]),
      posKey([1, 2]),
      posKey([3, 2]),
      posKey([2, 1]),
      posKey([2, 3]),
    ]);
    const path = bfs([0, 0], [2, 2], boxedWalls, width, height);
    expect(path).toBeNull();
  });

  it("respects blocked cells", () => {
    const blocked = new Set([posKey([1, 0])]);
    const path = bfs([0, 0], [2, 0], walls, width, height, blocked);
    expect(path).not.toBeNull();
    // Can't go through (1,0), must detour
    expect(path!.length).toBeGreaterThan(2);
  });

  it("returns null when fully enclosed", () => {
    // Block all neighbors of (0,0)
    const allBlocked = new Set([posKey([1, 0]), posKey([0, 1])]);
    const path = bfs([0, 0], [4, 4], walls, width, height, allBlocked);
    expect(path).toBeNull();
  });
});

describe("bfsDistance", () => {
  it("returns 0 for same position", () => {
    expect(bfsDistance([1, 1], [1, 1], walls, width, height)).toBe(0);
  });

  it("returns correct distance", () => {
    expect(bfsDistance([0, 0], [4, 0], new Set(), width, height)).toBe(4);
  });

  it("reaches wall-cell goals", () => {
    // Wall at (2,2) — distance-only BFS can now reach it via adjacent cell
    const dist = bfsDistance([0, 0], [2, 2], walls, width, height);
    expect(dist).toBeLessThan(9999);
    expect(dist).toBeGreaterThan(0);
  });

  it("returns 9999 for unreachable goal", () => {
    const boxedWalls = new Set([
      posKey([2, 2]),
      posKey([1, 2]),
      posKey([3, 2]),
      posKey([2, 1]),
      posKey([2, 3]),
    ]);
    expect(bfsDistance([0, 0], [2, 2], boxedWalls, width, height)).toBe(9999);
  });
});

describe("reverseBfs", () => {
  it("computes distances from goal to all reachable cells", () => {
    const dist = reverseBfs([0, 0], new Set(), width, height);
    expect(dist.get(posKey([0, 0]))).toBe(0);
    expect(dist.get(posKey([1, 0]))).toBe(1);
    expect(dist.get(posKey([4, 4]))).toBe(8);
  });

  it("does not include walls in distances", () => {
    const dist = reverseBfs([0, 0], walls, width, height);
    expect(dist.has(posKey([2, 2]))).toBe(false);
  });
});

describe("precomputeDropDistances", () => {
  it("computes distances from multiple drop zones", () => {
    const dropZones: Position[] = [[0, 0], [4, 4]];
    const dist = precomputeDropDistances(dropZones, new Set(), width, height);

    // Both sources should have distance 0
    expect(dist.get(posKey([0, 0]))).toBe(0);
    expect(dist.get(posKey([4, 4]))).toBe(0);

    // Middle should be 4 from either (manhattan = 4)
    expect(dist.get(posKey([2, 2]))).toBe(4);

    // Corner cells should use nearest source
    expect(dist.get(posKey([1, 0]))).toBe(1);
    expect(dist.get(posKey([3, 4]))).toBe(1);
  });

  it("handles single drop zone", () => {
    const dist = precomputeDropDistances([[0, 0]], new Set(), width, height);
    expect(dist.get(posKey([0, 0]))).toBe(0);
    expect(dist.get(posKey([4, 4]))).toBe(8);
  });
});

describe("nextStepToward", () => {
  it("returns first action of path", () => {
    const step = nextStepToward([0, 0], [0, 2], new Set(), width, height);
    expect(step).toBe("move_down");
  });

  it("returns null when start equals goal", () => {
    expect(nextStepToward([1, 1], [1, 1], new Set(), width, height)).toBeNull();
  });

  it("navigates toward wall-cell goals", () => {
    const step = nextStepToward([0, 0], [2, 2], walls, width, height);
    expect(step).not.toBeNull();
  });

  it("returns null when goal is surrounded by walls", () => {
    const boxedWalls = new Set([
      posKey([2, 2]),
      posKey([1, 2]),
      posKey([3, 2]),
      posKey([2, 1]),
      posKey([2, 3]),
    ]);
    expect(nextStepToward([0, 0], [2, 2], boxedWalls, width, height)).toBeNull();
  });

  it("respects blocked cells", () => {
    const blocked = new Set([posKey([1, 0])]);
    const step = nextStepToward([0, 0], [2, 0], new Set(), width, height, blocked);
    // Can't go right through (1,0), must go down first
    expect(step).toBe("move_down");
  });
});

describe("nextStepViaGradient", () => {
  it("follows distance gradient toward goal", () => {
    const distMap = precomputeDropDistances([[0, 0]], new Set(), width, height);
    const step = nextStepViaGradient([2, 2], distMap, new Set(), width, height);
    expect(step).not.toBeNull();
    expect(["move_up", "move_left"]).toContain(step);
  });

  it("avoids blocked cells", () => {
    const distMap = precomputeDropDistances([[0, 0]], new Set(), width, height);
    // Block the two improving neighbors (left and up)
    const blocked = new Set([posKey([1, 2]), posKey([2, 1])]);
    const step = nextStepViaGradient([2, 2], distMap, new Set(), width, height, blocked);
    // Remaining neighbors [3,2] and [2,3] have higher distance — no improvement
    expect(step).toBeNull();
  });

  it("returns null when already at goal", () => {
    const distMap = precomputeDropDistances([[0, 0]], new Set(), width, height);
    // At the goal, all neighbors have higher distance
    const step = nextStepViaGradient([0, 0], distMap, new Set(), width, height);
    expect(step).toBeNull();
  });

  it("avoids walls", () => {
    const distMap = precomputeDropDistances([[0, 0]], walls, width, height);
    // At (3,2), move_left would go to wall at (2,2) — should pick move_up instead
    const step = nextStepViaGradient([3, 2], distMap, walls, width, height);
    expect(step).not.toBeNull();
    expect(step).not.toBe("move_left"); // wall at (2,2)
  });
});
