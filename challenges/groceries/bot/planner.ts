import type { Bot, Item, GameState, Position, BotAction, DistanceMap } from "./types.ts";
import { posKey, positionsInclude, manhattanDistance } from "./utils.ts";
import { nextStepToward } from "./pathfinding.ts";
import { stillNeeded, getActiveOrder, getPreviewOrder, trulyNeeded, availableItemsFor, isEndGame } from "./state.ts";

export type BotStateName = "collecting" | "delivering" | "pre_fetching" | "idle";

export function nearestDropoff(
  pos: Position,
  dropOffZones: Position[],
  botCounts: Map<string, number>,
): Position {
  let bestZone = dropOffZones[0];
  let bestCost = Infinity;

  for (const zone of dropOffZones) {
    const manhattan = manhattanDistance(pos, zone);
    const congestion = (botCounts.get(posKey(zone)) ?? 0) * 2;
    const cost = manhattan + congestion;
    if (cost < bestCost) {
      bestCost = cost;
      bestZone = zone;
    }
  }

  return bestZone;
}

export function planBotAction(
  bot: Bot,
  state: GameState,
  assignedItem: Item | null,
  dropDist: DistanceMap,
  walls: Set<string>,
  blocked: Set<string>,
  dropoffCongestion: Map<string, number>,
): BotAction {
  const pos = bot.position;
  const inventory = bot.inventory;
  const active = getActiveOrder(state);
  const preview = getPreviewOrder(state);
  const dropZones = state.dropOffZones;
  const { width, height } = state.grid;

  // Determine what's useful to deliver
  const activeNeeded = active ? stillNeeded(active) : [];
  const usefulInInv = inventory.filter((item) => activeNeeded.includes(item));

  // Check if we're standing on a drop-off zone
  const onDropoff = positionsInclude(dropZones, pos);

  // END-GAME: deliver immediately if holding anything useful
  if (isEndGame(state) && usefulInInv.length > 0) {
    if (onDropoff) return { action: "drop_off" };
    const target = nearestDropoff(pos, dropZones, dropoffCongestion);
    const step = nextStepToward(pos, target, walls, width, height, blocked);
    return { action: step ?? "wait" };
  }

  // DELIVERING: if holding useful items and should go deliver
  if (usefulInInv.length > 0) {
    const trulyNeededItems = trulyNeeded(state, active);
    const shouldFillMore =
      inventory.length < 3 &&
      trulyNeededItems.length > 0 &&
      assignedItem !== null &&
      !isEndGame(state);

    if (!shouldFillMore) {
      if (onDropoff) return { action: "drop_off" };
      const target = nearestDropoff(pos, dropZones, dropoffCongestion);
      const step = nextStepToward(pos, target, walls, width, height, blocked);
      return { action: step ?? "wait" };
    }
  }

  // COLLECTING: go pick up assigned item
  if (assignedItem !== null && inventory.length < 3) {
    const itemPos = assignedItem.position;
    const dist = manhattanDistance(pos, itemPos);

    if (dist === 1) {
      return { action: "pick_up", item_id: assignedItem.id };
    }
    const step = nextStepToward(pos, itemPos, walls, width, height, blocked);
    return { action: step ?? "wait" };
  }

  // PRE_FETCHING: pick up preview items if we have slots
  if (preview && inventory.length < 3) {
    const previewNeededTypes = stillNeeded(preview);
    const previewItems = availableItemsFor(state, previewNeededTypes);

    const trulyNeededItems = trulyNeeded(state, active);
    if (trulyNeededItems.length <= 1 && previewItems.length > 0) {
      // Find nearest preview item
      const best = previewItems.reduce((closest, item) =>
        manhattanDistance(pos, item.position) < manhattanDistance(pos, closest.position)
          ? item
          : closest,
      );
      const dist = manhattanDistance(pos, best.position);
      if (dist === 1) {
        return { action: "pick_up", item_id: best.id };
      }
      const step = nextStepToward(pos, best.position, walls, width, height, blocked);
      if (step) return { action: step };
    }
  }

  // IDLE: move toward drop-off if holding anything, else wait
  if (inventory.length > 0) {
    if (onDropoff) return { action: "drop_off" };
    const target = nearestDropoff(pos, dropZones, dropoffCongestion);
    const step = nextStepToward(pos, target, walls, width, height, blocked);
    return { action: step ?? "wait" };
  }

  return { action: "wait" };
}
