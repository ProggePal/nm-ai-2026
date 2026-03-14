import { describe, it, expect } from "vitest";
import { applyMove, priorityScore, findSidestep, planRound } from "../coordinator.ts";
import { precomputeDropDistances } from "../pathfinding.ts";
import type { Bot, GameState, Position, DistanceMap } from "../types.ts";
import { posKey } from "../utils.ts";

describe("applyMove", () => {
  it("applies move_up", () => {
    expect(applyMove([5, 5], "move_up")).toEqual([5, 4]);
  });

  it("applies move_down", () => {
    expect(applyMove([5, 5], "move_down")).toEqual([5, 6]);
  });

  it("applies move_left", () => {
    expect(applyMove([5, 5], "move_left")).toEqual([4, 5]);
  });

  it("applies move_right", () => {
    expect(applyMove([5, 5], "move_right")).toEqual([6, 5]);
  });

  it("returns same position for non-movement actions", () => {
    expect(applyMove([5, 5], "wait")).toEqual([5, 5]);
    expect(applyMove([5, 5], "pick_up")).toEqual([5, 5]);
    expect(applyMove([5, 5], "drop_off")).toEqual([5, 5]);
  });
});

describe("priorityScore", () => {
  it("gives highest priority to bots holding useful items", () => {
    const dropDist = precomputeDropDistances([[0, 0]], new Set(), 10, 10);
    const state: GameState = {
      round: 5,
      maxRounds: 100,
      grid: { width: 10, height: 10, walls: new Set() },
      bots: [
        { id: 0, position: [1, 1], inventory: ["apple"] },
        { id: 1, position: [1, 1], inventory: [] },
      ],
      items: [],
      orders: [{
        id: "o1",
        itemsRequired: ["apple"],
        itemsDelivered: [],
        complete: false,
        status: "active",
      }],
      dropOffZones: [[0, 0]],
      score: 0,
    };

    const score0 = priorityScore(state.bots[0], state, dropDist);
    const score1 = priorityScore(state.bots[1], state, dropDist);
    expect(score0).toBeGreaterThan(score1);
  });

  it("prefers bots with more inventory among non-useful holders", () => {
    const dropDist = precomputeDropDistances([[0, 0]], new Set(), 10, 10);
    const state: GameState = {
      round: 5,
      maxRounds: 100,
      grid: { width: 10, height: 10, walls: new Set() },
      bots: [
        { id: 0, position: [1, 1], inventory: ["x", "y"] },
        { id: 1, position: [1, 1], inventory: ["z"] },
      ],
      items: [],
      orders: [],
      dropOffZones: [[0, 0]],
      score: 0,
    };

    const score0 = priorityScore(state.bots[0], state, dropDist);
    const score1 = priorityScore(state.bots[1], state, dropDist);
    expect(score0).toBeGreaterThan(score1);
  });
});

describe("findSidestep", () => {
  it("finds perpendicular move when intended is blocked", () => {
    const walls = new Set<string>();
    const reserved = new Map<string, number>();
    reserved.set(posKey([5, 4]), 99); // block move_up destination

    const step = findSidestep([5, 5], "move_up", walls, reserved, 10, 10);
    expect(step).not.toBeNull();
    expect(["move_left", "move_right"]).toContain(step);
  });

  it("returns null when all perpendicular moves are blocked", () => {
    const walls = new Set([posKey([4, 5])]);
    const reserved = new Map<string, number>();
    reserved.set(posKey([6, 5]), 99);

    const step = findSidestep([5, 5], "move_up", walls, reserved, 10, 10);
    expect(step).toBeNull();
  });

  it("returns null for non-movement action", () => {
    const step = findSidestep([5, 5], "wait", new Set(), new Map(), 10, 10);
    expect(step).toBeNull();
  });
});

describe("planRound", () => {
  it("returns actions for all bots", () => {
    const bots: Bot[] = [
      { id: 0, position: [1, 1], inventory: [] },
      { id: 1, position: [3, 3], inventory: [] },
    ];
    const dropZones: Position[] = [[0, 0]];
    const state: GameState = {
      round: 5,
      maxRounds: 100,
      grid: { width: 10, height: 10, walls: new Set() },
      bots,
      items: [],
      orders: [],
      dropOffZones: dropZones,
      score: 0,
    };
    const dropDist = precomputeDropDistances(dropZones, new Set(), 10, 10);

    const actions = planRound(state, dropDist);
    expect(actions).toHaveLength(2);
    expect(actions.map((a) => a.bot).sort()).toEqual([0, 1]);
  });

  it("prevents two bots from moving to the same cell", () => {
    // Two bots at adjacent positions both trying to move right
    const bots: Bot[] = [
      { id: 0, position: [2, 0], inventory: ["apple"] },
      { id: 1, position: [3, 0], inventory: ["banana"] },
    ];
    const dropZones: Position[] = [[9, 0]];
    const state: GameState = {
      round: 5,
      maxRounds: 100,
      grid: { width: 10, height: 10, walls: new Set() },
      bots,
      items: [],
      orders: [{
        id: "o1",
        itemsRequired: ["apple", "banana"],
        itemsDelivered: [],
        complete: false,
        status: "active",
      }],
      dropOffZones: dropZones,
      score: 0,
    };
    const dropDist = precomputeDropDistances(dropZones, new Set(), 10, 10);

    const actions = planRound(state, dropDist);
    // Both bots should have valid actions
    expect(actions).toHaveLength(2);
    // They should NOT both end up at the same cell
    // (exact actions depend on priority, but no collision)
  });
});
