import type { Bot, GameState, Position, BotAction, RoundAction, DistanceMap } from "./types.ts";
import { posKey } from "./utils.ts";
import { assignTasks } from "./assignment.ts";
import { planBotAction } from "./planner.ts";
import { stillNeeded, getActiveOrder, trulyNeeded, availableItemsFor } from "./state.ts";

export const DIRECTION_DELTAS: Record<string, [number, number]> = {
  move_up: [0, -1],
  move_down: [0, 1],
  move_left: [-1, 0],
  move_right: [1, 0],
};

export function applyMove(pos: Position, action: string): Position {
  const delta = DIRECTION_DELTAS[action];
  if (delta) {
    return [pos[0] + delta[0], pos[1] + delta[1]];
  }
  return pos; // pick_up / drop_off / wait — bot stays in place
}

export function priorityScore(
  bot: Bot,
  state: GameState,
  dropDist: DistanceMap,
): number {
  const active = getActiveOrder(state);
  const activeNeeded = active ? stillNeeded(active) : [];
  const holdingUseful = bot.inventory.some((item) => activeNeeded.includes(item));
  const invCount = bot.inventory.length;
  const dist = dropDist.get(posKey(bot.position)) ?? 9999;
  return (holdingUseful ? 1000 : 0) + invCount * 100 - dist;
}

export function planRound(
  state: GameState,
  dropDist: DistanceMap,
): RoundAction[] {
  const walls = state.grid.walls;
  const { width, height } = state.grid;

  // Needed items for active order (excluding in-transit)
  const active = getActiveOrder(state);
  const trulyNeededTypes = active ? trulyNeeded(state, active) : [];
  const availableItems = availableItemsFor(state, trulyNeededTypes);

  // Task assignment: bot_id → Item | null
  const assignment = assignTasks(state.bots, availableItems, state, dropDist, walls);

  // Sort bots by priority (highest first)
  const botsOrdered = [...state.bots].sort(
    (a, b) => priorityScore(b, state, dropDist) - priorityScore(a, state, dropDist),
  );

  // Reservation table: posKey → bot_id
  const reserved = new Map<string, number>();

  // Current positions (to detect swap deadlocks)
  const currentPositions = new Map<number, Position>();
  for (const bot of state.bots) {
    currentPositions.set(bot.id, bot.position);
  }

  const results = new Map<number, BotAction>();

  for (const bot of botsOrdered) {
    const pos = bot.position;

    // Cells already reserved by higher-priority bots (exclude own position)
    const blocked = new Set<string>();
    for (const [key] of reserved) {
      if (key !== posKey(pos)) {
        blocked.add(key);
      }
    }

    const actionDict = planBotAction(
      bot,
      state,
      assignment.get(bot.id) ?? null,
      dropDist,
      walls,
      blocked,
    );

    let action = actionDict.action;
    let desiredCell = applyMove(pos, action);

    // Check for swap deadlock
    let swapDeadlock = false;
    for (const [otherId, otherAction] of results) {
      const otherPos = currentPositions.get(otherId)!;
      const otherDesired = applyMove(otherPos, otherAction.action);
      const desiredKey = posKey(desiredCell);
      const otherPosKey = posKey(otherPos);
      const posKeyStr = posKey(pos);
      const otherDesiredKey = posKey(otherDesired);
      if (desiredKey === otherPosKey && otherDesiredKey === posKeyStr) {
        swapDeadlock = true;
        break;
      }
    }

    // Resolve conflicts
    if (reserved.has(posKey(desiredCell)) || swapDeadlock) {
      const sidestep = findSidestep(pos, action, walls, reserved, width, height);
      if (sidestep) {
        action = sidestep;
        desiredCell = applyMove(pos, sidestep);
        results.set(bot.id, { action: sidestep });
      } else {
        action = "wait";
        desiredCell = pos;
        results.set(bot.id, { action: "wait" });
      }
    } else {
      results.set(bot.id, actionDict);
    }

    reserved.set(posKey(desiredCell), bot.id);
  }

  // Build final output list (preserve original bot order)
  const output: RoundAction[] = [];
  for (const bot of state.bots) {
    const actionDict = results.get(bot.id)!;
    const entry: RoundAction = { bot: bot.id, action: actionDict.action };
    if (actionDict.item_id) {
      entry.item_id = actionDict.item_id;
    }
    output.push(entry);
  }

  return output;
}

export function findSidestep(
  pos: Position,
  intendedAction: string,
  walls: Set<string>,
  reserved: Map<string, number>,
  width: number,
  height: number,
): string | null {
  const perp: Record<string, string[]> = {
    move_up: ["move_left", "move_right", "move_down"],
    move_down: ["move_left", "move_right", "move_up"],
    move_left: ["move_up", "move_down", "move_right"],
    move_right: ["move_up", "move_down", "move_left"],
  };

  const candidates = perp[intendedAction] ?? [];
  for (const action of candidates) {
    const [dx, dy] = DIRECTION_DELTAS[action];
    const nx = pos[0] + dx;
    const ny = pos[1] + dy;
    const nKey = posKey([nx, ny]);
    if (
      nx >= 0 &&
      nx < width &&
      ny >= 0 &&
      ny < height &&
      !walls.has(nKey) &&
      !reserved.has(nKey)
    ) {
      return action;
    }
  }
  return null;
}
