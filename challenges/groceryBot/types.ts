export type Position = [number, number];

export interface Bot {
  id: number;
  position: Position;
  inventory: string[];
}

export interface Item {
  id: string;
  type: string;
  position: Position;
}

export interface Order {
  id: string;
  itemsRequired: string[];
  itemsDelivered: string[];
  complete: boolean;
  status: string; // "active" | "preview"
}

export interface Grid {
  width: number;
  height: number;
  walls: Set<string>;
}

export interface GameState {
  round: number;
  maxRounds: number;
  grid: Grid;
  bots: Bot[];
  items: Item[];
  orders: Order[];
  dropOffZones: Position[];
  score: number;
}

export interface BotAction {
  action: string;
  item_id?: string;
}

export interface RoundAction {
  bot: number;
  action: string;
  item_id?: string;
}

export type MoveDirection = "move_up" | "move_down" | "move_left" | "move_right";

export type DistanceMap = Map<string, number>;
