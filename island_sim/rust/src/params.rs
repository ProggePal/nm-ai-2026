/// Simulation parameters — mirrors Python SimParams exactly.

#[derive(Clone, Debug)]
pub struct SimParams {
    // Growth
    pub base_food_production: f64,
    pub forest_food_bonus: f64,
    pub plains_food_bonus: f64,
    pub food_consumption_rate: f64,
    pub pop_growth_rate: f64,
    pub pop_growth_food_threshold: f64,
    pub carrying_capacity_per_food: f64,
    pub port_development_threshold: f64,
    pub longship_build_threshold: f64,
    pub expansion_threshold: f64,
    pub expansion_range: f64,
    pub expansion_pop_transfer: f64,
    // Conflict
    pub raid_range_base: f64,
    pub raid_range_longship: f64,
    pub raid_desperation_threshold: f64,
    pub raid_strength_factor: f64,
    pub raid_loot_fraction: f64,
    pub raid_damage_factor: f64,
    pub conquest_threshold: f64,
    pub raid_kill_threshold: f64,
    // Trade
    pub trade_range: f64,
    pub trade_food_gain: f64,
    pub trade_wealth_gain: f64,
    pub tech_diffusion_rate: f64,
    // Winter
    pub winter_severity_mean: f64,
    pub winter_severity_variance: f64,
    pub collapse_food_threshold: f64,
    pub collapse_probability: f64,
    pub survival_bonus: f64,
    pub dispersal_range: f64,
    pub dispersal_fraction: f64,
    // Environment
    pub forest_reclaim_probability: f64,
    pub ruin_rebuild_range: f64,
    pub ruin_rebuild_threshold: f64,
    pub ruin_rebuild_fraction: f64,
    pub ruin_to_plains_probability: f64,
    // Additional collapse triggers (appended to preserve indices)
    pub harsh_winter_collapse_factor: f64,
    pub raid_collapse_threshold: f64,
    // Raid probability gate
    pub raid_probability: f64,
}

impl SimParams {
    /// Number of parameters (must match Python SimParams.ndim()).
    pub const NDIM: usize = 39;

    /// Create from a flat array of f64 values. Order must match Python param_names().
    pub fn from_array(arr: &[f64]) -> Self {
        assert!(arr.len() >= Self::NDIM, "Expected {} params, got {}", Self::NDIM, arr.len());
        SimParams {
            base_food_production: arr[0],
            forest_food_bonus: arr[1],
            plains_food_bonus: arr[2],
            food_consumption_rate: arr[3],
            pop_growth_rate: arr[4],
            pop_growth_food_threshold: arr[5],
            carrying_capacity_per_food: arr[6],
            port_development_threshold: arr[7],
            longship_build_threshold: arr[8],
            expansion_threshold: arr[9],
            expansion_range: arr[10],
            expansion_pop_transfer: arr[11],
            raid_range_base: arr[12],
            raid_range_longship: arr[13],
            raid_desperation_threshold: arr[14],
            raid_strength_factor: arr[15],
            raid_loot_fraction: arr[16],
            raid_damage_factor: arr[17],
            conquest_threshold: arr[18],
            raid_kill_threshold: arr[19],
            trade_range: arr[20],
            trade_food_gain: arr[21],
            trade_wealth_gain: arr[22],
            tech_diffusion_rate: arr[23],
            winter_severity_mean: arr[24],
            winter_severity_variance: arr[25],
            collapse_food_threshold: arr[26],
            collapse_probability: arr[27],
            survival_bonus: arr[28],
            dispersal_range: arr[29],
            dispersal_fraction: arr[30],
            forest_reclaim_probability: arr[31],
            ruin_rebuild_range: arr[32],
            ruin_rebuild_threshold: arr[33],
            ruin_rebuild_fraction: arr[34],
            ruin_to_plains_probability: arr[35],
            harsh_winter_collapse_factor: arr[36],
            raid_collapse_threshold: arr[37],
            raid_probability: arr[38],
        }
    }
}
