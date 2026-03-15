use std::collections::HashSet;
use std::fs;
use std::path::Path;

use crate::types::*;

/// Extract a replayable scenario from a recording file.
///
/// Pulls the grid, items, bots, and drop-off zones from round 0,
/// and collects the full order sequence across all rounds.
pub fn load_scenario(path: &Path) -> Result<Scenario, String> {
    let content = fs::read_to_string(path)
        .map_err(|err| format!("Failed to read {}: {err}", path.display()))?;
    let recording: Recording = serde_json::from_str(&content)
        .map_err(|err| format!("Failed to parse {}: {err}", path.display()))?;

    extract_scenario(&recording)
}

fn extract_scenario(recording: &Recording) -> Result<Scenario, String> {
    // Find the first game_state message (round 0) for initial state
    let round0 = recording
        .messages
        .iter()
        .find(|msg| msg.message["type"].as_str() == Some("game_state"))
        .ok_or("No game_state message found in recording")?;

    let raw: RawGameState = serde_json::from_value(round0.message.clone())
        .map_err(|err| format!("Failed to parse round 0 game_state: {err}"))?;

    let grid = Grid {
        width: raw.grid.width,
        height: raw.grid.height,
        walls: raw.grid.walls,
    };

    let items: Vec<Item> = raw
        .items
        .iter()
        .map(|i| Item {
            id: i.id.clone(),
            item_type: i.r#type.clone(),
            position: i.position,
        })
        .collect();

    let bots: Vec<Bot> = raw
        .bots
        .iter()
        .map(|b| Bot {
            id: b.id,
            position: b.position,
            inventory: vec![],
        })
        .collect();

    let drop_off_zones = if let Some(zones) = &raw.drop_off_zones {
        zones.clone()
    } else if let Some(zone) = &raw.drop_off {
        vec![*zone]
    } else {
        vec![]
    };

    // Collect the full order sequence across all rounds (preserving first-appearance order)
    let mut seen_order_ids = HashSet::new();
    let mut orders = Vec::new();

    for msg in &recording.messages {
        if msg.message["type"].as_str() != Some("game_state") {
            continue;
        }

        if let Some(raw_orders) = msg.message["orders"].as_array() {
            for raw_order in raw_orders {
                let id = raw_order["id"].as_str().unwrap_or("").to_string();
                if id.is_empty() || seen_order_ids.contains(&id) {
                    continue;
                }
                seen_order_ids.insert(id.clone());

                let items_required: Vec<String> = raw_order["items_required"]
                    .as_array()
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| v.as_str().map(|s| s.to_string()))
                            .collect()
                    })
                    .unwrap_or_default();

                orders.push(ScenarioOrder {
                    id,
                    items_required,
                });
            }
        }
    }

    Ok(Scenario {
        difficulty: recording.difficulty.clone(),
        max_rounds: raw.max_rounds,
        grid,
        items,
        bots,
        drop_off_zones,
        orders,
    })
}
