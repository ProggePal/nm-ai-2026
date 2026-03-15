#![allow(dead_code)]

use serde::{Deserialize, Serialize};

pub type Position = [i32; 2];

#[derive(Debug, Deserialize)]
pub struct RawGameState {
    pub r#type: String,
    pub round: u32,
    pub max_rounds: u32,
    pub grid: RawGrid,
    pub bots: Vec<RawBot>,
    pub items: Vec<RawItem>,
    pub orders: Vec<RawOrder>,
    pub drop_off_zones: Option<Vec<Position>>,
    pub drop_off: Option<Position>,
    pub score: Option<i64>,
}

#[derive(Debug, Deserialize)]
pub struct RawGrid {
    pub width: u32,
    pub height: u32,
    pub walls: Vec<Position>,
}

#[derive(Debug, Deserialize)]
pub struct RawBot {
    pub id: u32,
    pub position: Position,
    pub inventory: Vec<String>,
}

#[derive(Debug, Deserialize)]
pub struct RawItem {
    pub id: String,
    pub r#type: String,
    pub position: Position,
}

#[derive(Debug, Deserialize)]
pub struct RawOrder {
    pub id: String,
    pub items_required: Vec<String>,
    pub items_delivered: Vec<String>,
    pub complete: bool,
    pub status: String,
}

// Parsed types

#[derive(Debug)]
pub struct GameState {
    pub round: u32,
    pub max_rounds: u32,
    pub grid: Grid,
    pub bots: Vec<Bot>,
    pub items: Vec<Item>,
    pub orders: Vec<Order>,
    pub drop_off_zones: Vec<Position>,
    pub score: i64,
}

#[derive(Debug)]
pub struct Grid {
    pub width: u32,
    pub height: u32,
    pub walls: Vec<Position>,
}

#[derive(Debug)]
pub struct Bot {
    pub id: u32,
    pub position: Position,
    pub inventory: Vec<String>,
}

#[derive(Debug)]
pub struct Item {
    pub id: String,
    pub item_type: String,
    pub position: Position,
}

#[derive(Debug)]
pub struct Order {
    pub id: String,
    pub items_required: Vec<String>,
    pub items_delivered: Vec<String>,
    pub complete: bool,
    pub status: String,
}

#[derive(Debug, Serialize)]
pub struct RoundAction {
    pub bot: u32,
    pub action: String,
}

#[derive(Debug, Serialize)]
pub struct ActionsResponse {
    pub actions: Vec<RoundAction>,
}

// Replay/map storage types

#[derive(Debug, Serialize)]
pub struct ReplayFrame {
    pub round: u32,
    pub score: i64,
    pub actions: Vec<RoundAction>,
    pub state_summary: StateSummary,
    pub planning_ms: u64,
    pub timestamp: String,
}

#[derive(Debug, Serialize)]
pub struct StateSummary {
    pub bots: Vec<BotSummary>,
    pub active_order: Option<String>,
    pub items_on_map: usize,
}

#[derive(Debug, Serialize)]
pub struct BotSummary {
    pub id: u32,
    pub position: Position,
    pub inventory: Vec<String>,
}

#[derive(Debug, Serialize)]
pub struct MapData {
    pub difficulty: String,
    pub grid: MapGrid,
    pub drop_off_zones: Vec<Position>,
    pub captured_at: String,
}

#[derive(Debug, Serialize)]
pub struct MapGrid {
    pub width: u32,
    pub height: u32,
    pub walls: Vec<Position>,
}
