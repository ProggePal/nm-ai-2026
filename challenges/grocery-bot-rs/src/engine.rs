use std::collections::HashSet;

use crate::types::*;

const MAX_INVENTORY: usize = 3;

pub struct Engine {
    pub grid: Grid,
    pub bots: Vec<Bot>,
    pub items: Vec<Item>,
    pub drop_off_zones: Vec<Position>,
    pub orders: Vec<ScenarioOrder>,
    pub active_order_index: usize,
    /// Items delivered to the currently active order so far.
    pub active_order_delivered: Vec<String>,
    pub score: i64,
    pub items_delivered: i64,
    pub orders_completed: i64,
    pub round: u32,
    pub max_rounds: u32,
    wall_set: HashSet<(i32, i32)>,
}

impl Engine {
    pub fn new(scenario: &Scenario) -> Self {
        let wall_set: HashSet<(i32, i32)> = scenario
            .grid
            .walls
            .iter()
            .map(|w| (w[0], w[1]))
            .collect();

        Engine {
            grid: scenario.grid.clone(),
            bots: scenario.bots.clone(),
            items: scenario.items.clone(),
            drop_off_zones: scenario.drop_off_zones.clone(),
            orders: scenario.orders.clone(),
            active_order_index: 0,
            active_order_delivered: Vec::new(),
            score: 0,
            items_delivered: 0,
            orders_completed: 0,
            round: 0,
            max_rounds: scenario.max_rounds,
            wall_set,
        }
    }

    /// Build a GameState snapshot for the planner.
    pub fn game_state(&self) -> GameState {
        let mut orders = Vec::new();

        // Active order
        if let Some(active) = self.orders.get(self.active_order_index) {
            orders.push(Order {
                id: active.id.clone(),
                items_required: active.items_required.clone(),
                items_delivered: self.active_order_delivered.clone(),
                complete: false,
                status: "active".to_string(),
            });
        }

        // Preview order
        if let Some(preview) = self.orders.get(self.active_order_index + 1) {
            orders.push(Order {
                id: preview.id.clone(),
                items_required: preview.items_required.clone(),
                items_delivered: vec![],
                complete: false,
                status: "preview".to_string(),
            });
        }

        GameState {
            round: self.round,
            max_rounds: self.max_rounds,
            grid: self.grid.clone(),
            bots: self.bots.clone(),
            items: self.items.clone(),
            orders,
            drop_off_zones: self.drop_off_zones.clone(),
            score: self.score,
        }
    }

    /// Run the full game, calling the planner each round.
    pub fn run<F>(&mut self, mut planner: F) -> SimResult
    where
        F: FnMut(&GameState) -> Vec<RoundAction>,
    {
        while self.round < self.max_rounds {
            if self.active_order_index >= self.orders.len() {
                break; // All orders done
            }

            let state = self.game_state();
            let actions = planner(&state);
            self.apply_actions(&actions);
            self.round += 1;
        }

        SimResult {
            score: self.score,
            items_delivered: self.items_delivered,
            orders_completed: self.orders_completed,
            rounds_used: self.round,
        }
    }

    fn apply_actions(&mut self, actions: &[RoundAction]) {
        // Sort by bot ID to resolve in order (bot 0 first)
        let mut sorted: Vec<&RoundAction> = actions.iter().collect();
        sorted.sort_by_key(|a| a.bot);

        // Track occupied positions for collision detection
        let mut occupied: HashSet<(i32, i32)> = self
            .bots
            .iter()
            .map(|b| (b.position[0], b.position[1]))
            .collect();

        for action in &sorted {
            let bot_idx = match self.bots.iter().position(|b| b.id == action.bot) {
                Some(idx) => idx,
                None => continue,
            };

            match action.action.as_str() {
                "move_up" => self.apply_move(bot_idx, 0, -1, &mut occupied),
                "move_down" => self.apply_move(bot_idx, 0, 1, &mut occupied),
                "move_left" => self.apply_move(bot_idx, -1, 0, &mut occupied),
                "move_right" => self.apply_move(bot_idx, 1, 0, &mut occupied),
                "pick_up" => self.apply_pickup(bot_idx, action.item_id.as_deref()),
                "drop_off" => self.apply_dropoff(bot_idx),
                "wait" | _ => {}
            }
        }
    }

    fn apply_move(
        &mut self,
        bot_idx: usize,
        dx: i32,
        dy: i32,
        occupied: &mut HashSet<(i32, i32)>,
    ) {
        let bot = &self.bots[bot_idx];
        let new_x = bot.position[0] + dx;
        let new_y = bot.position[1] + dy;

        // Out of bounds
        if new_x < 0
            || new_y < 0
            || new_x >= self.grid.width as i32
            || new_y >= self.grid.height as i32
        {
            return;
        }

        // Wall/shelf collision
        if self.wall_set.contains(&(new_x, new_y)) {
            return;
        }

        // Bot collision
        if occupied.contains(&(new_x, new_y)) {
            return;
        }

        let old_pos = (bot.position[0], bot.position[1]);
        occupied.remove(&old_pos);
        occupied.insert((new_x, new_y));
        self.bots[bot_idx].position = [new_x, new_y];
    }

    fn apply_pickup(&mut self, bot_idx: usize, item_id: Option<&str>) {
        let item_id = match item_id {
            Some(id) => id,
            None => return,
        };

        let bot = &self.bots[bot_idx];

        if bot.inventory.len() >= MAX_INVENTORY {
            return;
        }

        let item_idx = match self.items.iter().position(|i| i.id == item_id) {
            Some(idx) => idx,
            None => return,
        };

        // Adjacent = Manhattan distance 1
        let item = &self.items[item_idx];
        let dist = (bot.position[0] - item.position[0]).abs()
            + (bot.position[1] - item.position[1]).abs();
        if dist != 1 {
            return;
        }

        let item_type = self.items[item_idx].item_type.clone();
        self.bots[bot_idx].inventory.push(item_type);
        self.items.remove(item_idx);
    }

    fn apply_dropoff(&mut self, bot_idx: usize) {
        let bot = &self.bots[bot_idx];

        if !self.drop_off_zones.contains(&bot.position) {
            return;
        }

        if bot.inventory.is_empty() {
            return;
        }

        self.deliver_matching_items(bot_idx);
    }

    fn deliver_matching_items(&mut self, bot_idx: usize) {
        loop {
            if self.active_order_index >= self.orders.len() {
                break;
            }

            let order = &self.orders[self.active_order_index];

            // Compute what's still needed: required minus delivered
            let mut still_needed = order.items_required.clone();
            for item in &self.active_order_delivered {
                if let Some(pos) = still_needed.iter().position(|n| n == item) {
                    still_needed.remove(pos);
                }
            }

            // Try to deliver matching items from bot inventory
            let mut delivered_this_pass = 0;
            let bot = &mut self.bots[bot_idx];
            let mut remaining_inventory = Vec::new();

            for item_type in bot.inventory.drain(..) {
                if let Some(pos) = still_needed.iter().position(|n| *n == item_type) {
                    still_needed.remove(pos);
                    self.score += 1;
                    self.items_delivered += 1;
                    self.active_order_delivered.push(item_type);
                    delivered_this_pass += 1;
                } else {
                    remaining_inventory.push(item_type);
                }
            }

            self.bots[bot_idx].inventory = remaining_inventory;

            if delivered_this_pass == 0 {
                break;
            }

            // Order complete?
            if still_needed.is_empty() {
                self.score += 5;
                self.orders_completed += 1;
                self.active_order_index += 1;
                self.active_order_delivered.clear();
                // Continue loop — re-check remaining inventory against next order
            } else {
                break;
            }
        }
    }
}

pub struct SimResult {
    pub score: i64,
    pub items_delivered: i64,
    pub orders_completed: i64,
    pub rounds_used: u32,
}
