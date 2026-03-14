import { describe, it, expect } from "vitest";
import {
  stillNeeded,
  parseGameState,
  getActiveOrder,
  getPreviewOrder,
  getAllBotInventory,
  trulyNeeded,
  availableItemsFor,
  getBotById,
  isEndGame,
} from "../state.ts";
import type { Order, GameState } from "../types.ts";

function makeGameStateData(): Record<string, any> {
  return {
    round: 10,
    max_rounds: 100,
    grid: {
      width: 10,
      height: 10,
      walls: [[5, 5], [5, 6]],
    },
    bots: [
      { id: 0, position: [1, 1], inventory: ["apple"] },
      { id: 1, position: [2, 2], inventory: ["banana", "milk"] },
    ],
    items: [
      { id: "item_0", type: "apple", position: [3, 3] },
      { id: "item_1", type: "bread", position: [4, 4] },
      { id: "item_2", type: "milk", position: [6, 6] },
    ],
    orders: [
      {
        id: "order_0",
        items_required: ["apple", "banana", "bread"],
        items_delivered: ["banana"],
        complete: false,
        status: "active",
      },
      {
        id: "order_1",
        items_required: ["milk", "cheese"],
        items_delivered: [],
        complete: false,
        status: "preview",
      },
    ],
    drop_off_zones: [[0, 0], [9, 9]],
    score: 42,
  };
}

describe("stillNeeded", () => {
  it("returns items required but not yet delivered", () => {
    const order: Order = {
      id: "o1",
      itemsRequired: ["apple", "banana", "bread"],
      itemsDelivered: ["banana"],
      complete: false,
      status: "active",
    };
    expect(stillNeeded(order)).toEqual(["apple", "bread"]);
  });

  it("handles duplicates correctly", () => {
    const order: Order = {
      id: "o2",
      itemsRequired: ["apple", "apple", "banana"],
      itemsDelivered: ["apple"],
      complete: false,
      status: "active",
    };
    expect(stillNeeded(order)).toEqual(["apple", "banana"]);
  });

  it("returns empty array when all delivered", () => {
    const order: Order = {
      id: "o3",
      itemsRequired: ["apple"],
      itemsDelivered: ["apple"],
      complete: true,
      status: "active",
    };
    expect(stillNeeded(order)).toEqual([]);
  });
});

describe("parseGameState", () => {
  it("parses a full game state from JSON", () => {
    const state = parseGameState(makeGameStateData());

    expect(state.round).toBe(10);
    expect(state.maxRounds).toBe(100);
    expect(state.grid.width).toBe(10);
    expect(state.grid.height).toBe(10);
    expect(state.grid.walls.has("5,5")).toBe(true);
    expect(state.grid.walls.has("5,6")).toBe(true);
    expect(state.grid.walls.size).toBe(2);
    expect(state.bots).toHaveLength(2);
    expect(state.bots[0].position).toEqual([1, 1]);
    expect(state.items).toHaveLength(3);
    expect(state.orders).toHaveLength(2);
    expect(state.orders[0].itemsRequired).toEqual(["apple", "banana", "bread"]);
    expect(state.dropOffZones).toEqual([[0, 0], [9, 9]]);
    expect(state.score).toBe(42);
  });

  it("handles legacy drop_off format", () => {
    const data = makeGameStateData();
    delete data.drop_off_zones;
    data.drop_off = [5, 0];

    const state = parseGameState(data);
    expect(state.dropOffZones).toEqual([[5, 0]]);
  });

  it("handles missing drop_off", () => {
    const data = makeGameStateData();
    delete data.drop_off_zones;

    const state = parseGameState(data);
    expect(state.dropOffZones).toEqual([]);
  });

  it("defaults score to 0", () => {
    const data = makeGameStateData();
    delete data.score;

    const state = parseGameState(data);
    expect(state.score).toBe(0);
  });
});

describe("getActiveOrder", () => {
  it("returns the active order", () => {
    const state = parseGameState(makeGameStateData());
    const active = getActiveOrder(state);
    expect(active).not.toBeNull();
    expect(active!.id).toBe("order_0");
  });

  it("returns null if no active order", () => {
    const data = makeGameStateData();
    data.orders = [data.orders[1]]; // only preview
    const state = parseGameState(data);
    expect(getActiveOrder(state)).toBeNull();
  });
});

describe("getPreviewOrder", () => {
  it("returns the preview order", () => {
    const state = parseGameState(makeGameStateData());
    const preview = getPreviewOrder(state);
    expect(preview).not.toBeNull();
    expect(preview!.id).toBe("order_1");
  });
});

describe("getAllBotInventory", () => {
  it("returns flat list of all bot inventories", () => {
    const state = parseGameState(makeGameStateData());
    expect(getAllBotInventory(state)).toEqual(["apple", "banana", "milk"]);
  });
});

describe("trulyNeeded", () => {
  it("excludes items already in bot inventories", () => {
    const state = parseGameState(makeGameStateData());
    // active order needs: apple, bread (banana already delivered)
    // bots hold: apple, banana, milk → apple is in transit
    const needed = trulyNeeded(state);
    expect(needed).toEqual(["bread"]);
  });

  it("returns empty if no active order", () => {
    const data = makeGameStateData();
    data.orders = [];
    const state = parseGameState(data);
    expect(trulyNeeded(state)).toEqual([]);
  });
});

describe("availableItemsFor", () => {
  it("returns items matching given types", () => {
    const state = parseGameState(makeGameStateData());
    const items = availableItemsFor(state, ["apple", "milk"]);
    expect(items).toHaveLength(2);
    expect(items.map((i) => i.type)).toEqual(["apple", "milk"]);
  });

  it("returns empty for no matching types", () => {
    const state = parseGameState(makeGameStateData());
    expect(availableItemsFor(state, ["cheese"])).toEqual([]);
  });
});

describe("getBotById", () => {
  it("returns the correct bot", () => {
    const state = parseGameState(makeGameStateData());
    const bot = getBotById(state, 1);
    expect(bot).toBeDefined();
    expect(bot!.position).toEqual([2, 2]);
  });

  it("returns undefined for unknown id", () => {
    const state = parseGameState(makeGameStateData());
    expect(getBotById(state, 99)).toBeUndefined();
  });
});

describe("isEndGame", () => {
  it("returns false when plenty of rounds remain", () => {
    const state = parseGameState(makeGameStateData());
    expect(isEndGame(state)).toBe(false);
  });

  it("returns true when > 90% of rounds elapsed", () => {
    const data = makeGameStateData();
    data.round = 91;
    data.max_rounds = 100;
    const state = parseGameState(data);
    expect(isEndGame(state)).toBe(true);
  });

  it("returns false at exactly 90%", () => {
    const data = makeGameStateData();
    data.round = 90;
    data.max_rounds = 100;
    const state = parseGameState(data);
    expect(isEndGame(state)).toBe(false);
  });
});
