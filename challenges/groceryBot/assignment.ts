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

  // Build all (bot, item, cost) pairs for globally-optimal greedy matching
  const pairs: { botIdx: number; itemIdx: number; cost: number }[] = [];
  for (let bi = 0; bi < collectingBots.length; bi++) {
    for (let ii = 0; ii < items.length; ii++) {
      const distToItem = bfsDistance(collectingBots[bi].position, items[ii].position, walls, width, height);
      const distToDrop = dropDist.get(posKey(items[ii].position)) ?? INF;
      pairs.push({ botIdx: bi, itemIdx: ii, cost: distToItem + distToDrop });
    }
  }

  // Sort by cost (cheapest first) — assigns globally cheapest pairs first
  pairs.sort((a, b) => a.cost - b.cost);

  const claimedBots = new Set<number>();
  const claimedItems = new Set<number>();

  for (const { botIdx, itemIdx, cost } of pairs) {
    if (cost >= INF) continue;
    if (claimedBots.has(botIdx) || claimedItems.has(itemIdx)) continue;
    assignment.set(collectingBots[botIdx].id, items[itemIdx]);
    claimedBots.add(botIdx);
    claimedItems.add(itemIdx);
    if (claimedBots.size === collectingBots.length) break;
  }

  return assignment;
}
