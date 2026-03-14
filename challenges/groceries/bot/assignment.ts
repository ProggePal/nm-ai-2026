import type { Bot, Item, GameState, DistanceMap } from "./types.ts";
import { posKey } from "./utils.ts";
import { bfsDistance } from "./pathfinding.ts";

const INF = 9999;

export function assignTasks(
  bots: Bot[],
  items: Item[],
  state: GameState,
  dropDist: DistanceMap,
  walls: Set<string>,
): Map<number, Item | null> {
  const assignment = new Map<number, Item | null>();
  for (const bot of bots) {
    assignment.set(bot.id, null);
  }

  const collectingBots = bots.filter((b) => b.inventory.length < 3);
  if (collectingBots.length === 0 || items.length === 0) {
    return assignment;
  }

  const { width, height } = state.grid;

  // Build cost matrix [bots × items]
  const cost: number[][] = [];
  for (const bot of collectingBots) {
    const row: number[] = [];
    for (const item of items) {
      const distToItem = bfsDistance(bot.position, item.position, walls, width, height);
      const distToDrop = dropDist.get(posKey(item.position)) ?? INF;
      row.push(distToItem + distToDrop);
    }
    cost.push(row);
  }

  // Greedy assignment (cost matrix is tiny: ~4 bots x ~20 items)
  const claimed = new Set<number>();
  for (let bi = 0; bi < collectingBots.length; bi++) {
    let bestCost = INF;
    let bestItemIdx: number | null = null;
    for (let ii = 0; ii < items.length; ii++) {
      if (!claimed.has(ii) && cost[bi][ii] < bestCost) {
        bestCost = cost[bi][ii];
        bestItemIdx = ii;
      }
    }
    if (bestItemIdx !== null) {
      assignment.set(collectingBots[bi].id, items[bestItemIdx]);
      claimed.add(bestItemIdx);
    }
  }

  return assignment;
}
