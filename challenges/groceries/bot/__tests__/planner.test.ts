import { describe, it, expect } from "vitest";
import { planBotAction, nearestDropoff } from "../planner.ts";
import { precomputeDropDistances } from "../pathfinding.ts";
import type { Bot, Item, GameState, Position, DistanceMap } from "../types.ts";
import { posKey } from "../utils.ts";

function makeState(overrides: Partial<{
  round: number;
  maxRounds: number;
  bots: Bot[];
  items: Item[];
  orders: GameState["orders"];
  dropOffZones: Position[];
  walls: Set<string>;
}>): { state: GameState; dropDist: DistanceMap } {
  const walls = overrides.walls ?? new Set<string>();
  const dropOffZones: Position[] = overrides.dropOffZones ?? [[0, 0]];
  const state: GameState = {
    round: overrides.round ?? 5,
    maxRounds: overrides.maxRounds ?? 100,
    grid: { width: 10, height: 10, walls },
    bots: overrides.bots ?? [{ id: 0, position: [5, 5], inventory: [] }],
    items: overrides.items ?? [],
    orders: overrides.orders ?? [],
    dropOffZones,
    score: 0,
  };
  const dropDist = precomputeDropDistances(dropOffZones, walls, 10, 10);
  return { state, dropDist };
}

describe("nearestDropoff", () => {
  it("returns the closest drop-off zone", () => {
    const zones: Position[] = [[0, 0], [9, 9]];
    const result = nearestDropoff([1, 1], zones, new Map());
    expect(result).toEqual([0, 0]);
  });

  it("accounts for congestion", () => {
    const zones: Position[] = [[0, 0], [2, 0]];
    const congestion = new Map([[posKey([0, 0]), 5]]);
    // (1,0) is distance 1 from both, but (0,0) has congestion penalty
    const result = nearestDropoff([1, 0], zones, congestion);
    expect(result).toEqual([2, 0]);
  });
});

describe("planBotAction", () => {
  it("drops off when on drop zone with useful inventory", () => {
    const bot: Bot = { id: 0, position: [0, 0], inventory: ["apple"] };
    const { state, dropDist } = makeState({
      bots: [bot],
      orders: [{
        id: "o1",
        itemsRequired: ["apple"],
        itemsDelivered: [],
        complete: false,
        status: "active",
      }],
    });

    const action = planBotAction(
      bot, state, null, dropDist,
      state.grid.walls, new Set(),
    );
    expect(action.action).toBe("drop_off");
  });

  it("picks up adjacent item", () => {
    const bot: Bot = { id: 0, position: [3, 3], inventory: [] };
    const item: Item = { id: "item_0", type: "apple", position: [3, 4] };
    const { state, dropDist } = makeState({
      bots: [bot],
      items: [item],
      orders: [{
        id: "o1",
        itemsRequired: ["apple"],
        itemsDelivered: [],
        complete: false,
        status: "active",
      }],
    });

    const action = planBotAction(
      bot, state, item, dropDist,
      state.grid.walls, new Set(),
    );
    expect(action.action).toBe("pick_up");
    expect(action.item_id).toBe("item_0");
  });

  it("picks up item when standing on it", () => {
    const bot: Bot = { id: 0, position: [3, 3], inventory: [] };
    const item: Item = { id: "item_0", type: "apple", position: [3, 3] };
    const { state, dropDist } = makeState({
      bots: [bot],
      items: [item],
      orders: [{
        id: "o1",
        itemsRequired: ["apple"],
        itemsDelivered: [],
        complete: false,
        status: "active",
      }],
    });

    const action = planBotAction(
      bot, state, item, dropDist,
      state.grid.walls, new Set(),
    );
    expect(action.action).toBe("pick_up");
    expect(action.item_id).toBe("item_0");
  });

  it("moves toward drop-off when delivering", () => {
    const bot: Bot = { id: 0, position: [5, 5], inventory: ["apple"] };
    const { state, dropDist } = makeState({
      bots: [bot],
      orders: [{
        id: "o1",
        itemsRequired: ["apple"],
        itemsDelivered: [],
        complete: false,
        status: "active",
      }],
    });

    const action = planBotAction(
      bot, state, null, dropDist,
      state.grid.walls, new Set(),
    );
    // Should move toward (0,0) drop zone
    expect(["move_up", "move_left"]).toContain(action.action);
  });

  it("delivers immediately in end game", () => {
    const bot: Bot = { id: 0, position: [5, 5], inventory: ["apple"] };
    const { state, dropDist } = makeState({
      round: 95,
      maxRounds: 100,
      bots: [bot],
      orders: [{
        id: "o1",
        itemsRequired: ["apple"],
        itemsDelivered: [],
        complete: false,
        status: "active",
      }],
    });

    const action = planBotAction(
      bot, state, null, dropDist,
      state.grid.walls, new Set(),
    );
    expect(["move_up", "move_left"]).toContain(action.action);
  });

  it("waits when idle with empty inventory", () => {
    const bot: Bot = { id: 0, position: [5, 5], inventory: [] };
    const { state, dropDist } = makeState({ bots: [bot] });

    const action = planBotAction(
      bot, state, null, dropDist,
      state.grid.walls, new Set(),
    );
    expect(action.action).toBe("wait");
  });

  it("pre-fetches preview items when active order nearly done", () => {
    const bot: Bot = { id: 0, position: [3, 3], inventory: [] };
    const previewItem: Item = { id: "item_p", type: "milk", position: [3, 4] };
    const { state, dropDist } = makeState({
      bots: [bot],
      items: [previewItem],
      orders: [
        {
          id: "o1",
          itemsRequired: ["apple"],
          itemsDelivered: ["apple"],
          complete: true,
          status: "active",
        },
        {
          id: "o2",
          itemsRequired: ["milk"],
          itemsDelivered: [],
          complete: false,
          status: "preview",
        },
      ],
    });

    const action = planBotAction(
      bot, state, null, dropDist,
      state.grid.walls, new Set(),
    );
    expect(action.action).toBe("pick_up");
    expect(action.item_id).toBe("item_p");
  });
});
