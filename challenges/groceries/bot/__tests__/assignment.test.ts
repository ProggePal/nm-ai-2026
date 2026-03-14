import { describe, it, expect } from "vitest";
import { assignTasks } from "../assignment.ts";
import { precomputeDropDistances } from "../pathfinding.ts";
import type { Bot, Item, GameState, Position, DistanceMap } from "../types.ts";

function makeState(
  bots: Bot[],
  items: Item[],
  dropZones: Position[] = [[0, 0]],
): { state: GameState; dropDist: DistanceMap; walls: Set<string> } {
  const walls = new Set<string>();
  const state: GameState = {
    round: 5,
    maxRounds: 100,
    grid: { width: 10, height: 10, walls },
    bots,
    items,
    orders: [],
    dropOffZones: dropZones,
    score: 0,
  };
  const dropDist = precomputeDropDistances(dropZones, walls, 10, 10);
  return { state, dropDist, walls };
}

describe("assignTasks", () => {
  it("assigns one bot to one item", () => {
    const bots: Bot[] = [{ id: 0, position: [1, 1], inventory: [] }];
    const items: Item[] = [{ id: "item_0", type: "apple", position: [3, 3] }];
    const { state, dropDist, walls } = makeState(bots, items);

    const result = assignTasks(bots, items, state, dropDist, walls);
    expect(result.get(0)).toEqual(items[0]);
  });

  it("assigns multiple bots to different items", () => {
    const bots: Bot[] = [
      { id: 0, position: [1, 1], inventory: [] },
      { id: 1, position: [5, 5], inventory: [] },
    ];
    const items: Item[] = [
      { id: "item_0", type: "apple", position: [2, 1] },
      { id: "item_1", type: "banana", position: [6, 5] },
    ];
    const { state, dropDist, walls } = makeState(bots, items);

    const result = assignTasks(bots, items, state, dropDist, walls);
    // No duplicate assignments
    const assignedItems = [...result.values()].filter((v) => v !== null);
    const assignedIds = assignedItems.map((i) => i!.id);
    expect(new Set(assignedIds).size).toBe(assignedIds.length);
  });

  it("handles more bots than items", () => {
    const bots: Bot[] = [
      { id: 0, position: [1, 1], inventory: [] },
      { id: 1, position: [2, 2], inventory: [] },
      { id: 2, position: [3, 3], inventory: [] },
    ];
    const items: Item[] = [{ id: "item_0", type: "apple", position: [1, 2] }];
    const { state, dropDist, walls } = makeState(bots, items);

    const result = assignTasks(bots, items, state, dropDist, walls);
    const assigned = [...result.values()].filter((v) => v !== null);
    expect(assigned).toHaveLength(1);
  });

  it("handles more items than bots", () => {
    const bots: Bot[] = [{ id: 0, position: [1, 1], inventory: [] }];
    const items: Item[] = [
      { id: "item_0", type: "apple", position: [2, 2] },
      { id: "item_1", type: "banana", position: [3, 3] },
      { id: "item_2", type: "milk", position: [4, 4] },
    ];
    const { state, dropDist, walls } = makeState(bots, items);

    const result = assignTasks(bots, items, state, dropDist, walls);
    const assigned = [...result.values()].filter((v) => v !== null);
    expect(assigned).toHaveLength(1);
  });

  it("skips bots with full inventory", () => {
    const bots: Bot[] = [
      { id: 0, position: [1, 1], inventory: ["a", "b", "c"] },
      { id: 1, position: [2, 2], inventory: [] },
    ];
    const items: Item[] = [{ id: "item_0", type: "apple", position: [3, 3] }];
    const { state, dropDist, walls } = makeState(bots, items);

    const result = assignTasks(bots, items, state, dropDist, walls);
    expect(result.get(0)).toBeNull();
    expect(result.get(1)).toEqual(items[0]);
  });

  it("returns all nulls when no items", () => {
    const bots: Bot[] = [{ id: 0, position: [1, 1], inventory: [] }];
    const { state, dropDist, walls } = makeState(bots, []);

    const result = assignTasks(bots, [], state, dropDist, walls);
    expect(result.get(0)).toBeNull();
  });

  it("returns all nulls when no bots have space", () => {
    const bots: Bot[] = [
      { id: 0, position: [1, 1], inventory: ["a", "b", "c"] },
    ];
    const items: Item[] = [{ id: "item_0", type: "apple", position: [2, 2] }];
    const { state, dropDist, walls } = makeState(bots, items);

    const result = assignTasks(bots, items, state, dropDist, walls);
    expect(result.get(0)).toBeNull();
  });
});
