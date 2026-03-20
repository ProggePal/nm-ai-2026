/// Core data types for the simulation.

#[derive(Clone, Debug)]
pub struct Settlement {
    pub x: i32,
    pub y: i32,
    pub population: f64,
    pub food: f64,
    pub wealth: f64,
    pub defense: f64,
    pub tech_level: f64,
    pub has_port: bool,
    pub has_longship: bool,
    pub owner_id: i32,
    pub alive: bool,
}

#[derive(Clone)]
pub struct WorldState {
    pub grid: Vec<Vec<i32>>, // grid[y][x]
    pub settlements: Vec<Settlement>,
    pub width: i32,
    pub height: i32,
}

// Terrain codes
pub const OCEAN: i32 = 10;
pub const PLAINS: i32 = 11;
pub const EMPTY: i32 = 0;
pub const SETTLEMENT: i32 = 1;
pub const PORT: i32 = 2;
pub const RUIN: i32 = 3;
pub const FOREST: i32 = 4;
pub const MOUNTAIN: i32 = 5;

// Prediction class indices
pub const NUM_CLASSES: usize = 6;

/// Map terrain code to prediction class index.
pub fn terrain_to_class(terrain: i32) -> usize {
    match terrain {
        OCEAN | PLAINS | EMPTY => 0,
        SETTLEMENT => 1,
        PORT => 2,
        RUIN => 3,
        FOREST => 4,
        MOUNTAIN => 5,
        _ => 0,
    }
}

/// Check if terrain is buildable (new settlement can be founded here).
pub fn is_buildable(terrain: i32) -> bool {
    matches!(terrain, PLAINS | EMPTY | FOREST | RUIN)
}
