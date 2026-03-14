import type { Bot, Item, Order, Grid, GameState, Position } from "./types.ts";
import { posKey } from "./utils.ts";

export function stillNeeded(order: Order): string[] {
  const delivered = [...order.itemsDelivered];
  const needed: string[] = [];
  for (const item of order.itemsRequired) {
    const idx = delivered.indexOf(item);
    if (idx !== -1) {
      delivered.splice(idx, 1);
    } else {
      needed.push(item);
    }
  }
  return needed;
}

export function parseGameState(data: Record<string, any>): GameState {
  const grid: Grid = {
    width: data.grid.width,
    height: data.grid.height,
    walls: new Set(
      (data.grid.walls as number[][]).map((w) => posKey(w as Position)),
    ),
  };

  const bots: Bot[] = (data.bots as any[]).map((b) => ({
    id: b.id,
    position: [b.position[0], b.position[1]] as Position,
    inventory: [...b.inventory],
  }));

  const items: Item[] = (data.items as any[]).map((i) => ({
    id: i.id,
    type: i.type,
    position: [i.position[0], i.position[1]] as Position,
  }));

  const orders: Order[] = (data.orders as any[]).map((o) => ({
    id: o.id,
    itemsRequired: [...o.items_required],
    itemsDelivered: [...o.items_delivered],
    complete: o.complete,
    status: o.status,
  }));

  let dropOffZones: Position[];
  if (data.drop_off_zones) {
    dropOffZones = (data.drop_off_zones as number[][]).map(
      (z) => [z[0], z[1]] as Position,
    );
  } else if (data.drop_off) {
    dropOffZones = [[data.drop_off[0], data.drop_off[1]] as Position];
  } else {
    dropOffZones = [];
  }

  return {
    round: data.round,
    maxRounds: data.max_rounds,
    grid,
    bots,
    items,
    orders,
    dropOffZones,
    score: data.score ?? 0,
  };
}

export function getActiveOrder(state: GameState): Order | null {
  return state.orders.find((o) => o.status === "active") ?? null;
}

export function getPreviewOrder(state: GameState): Order | null {
  return state.orders.find((o) => o.status === "preview") ?? null;
}

export function getAllBotInventory(state: GameState): string[] {
  return state.bots.flatMap((bot) => bot.inventory);
}

export function trulyNeeded(state: GameState, order?: Order | null): string[] {
  const target = order ?? getActiveOrder(state);
  if (!target) return [];

  const needed = stillNeeded(target);
  const inTransit = getAllBotInventory(state);

  const result: string[] = [];
  for (const itemType of needed) {
    const idx = inTransit.indexOf(itemType);
    if (idx !== -1) {
      inTransit.splice(idx, 1);
    } else {
      result.push(itemType);
    }
  }
  return result;
}

export function availableItemsFor(state: GameState, itemTypes: string[]): Item[] {
  return state.items.filter((item) => itemTypes.includes(item.type));
}

export function getBotById(state: GameState, botId: number): Bot | undefined {
  return state.bots.find((b) => b.id === botId);
}

export function isEndGame(state: GameState): boolean {
  return state.round > state.maxRounds * 0.9;
}
