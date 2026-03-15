use crate::types::*;

pub fn parse_game_state(raw: &RawGameState) -> GameState {
    let grid = Grid {
        width: raw.grid.width,
        height: raw.grid.height,
        walls: raw.grid.walls.clone(),
    };

    let bots = raw
        .bots
        .iter()
        .map(|b| Bot {
            id: b.id,
            position: b.position,
            inventory: b.inventory.clone(),
        })
        .collect();

    let items = raw
        .items
        .iter()
        .map(|i| Item {
            id: i.id.clone(),
            item_type: i.r#type.clone(),
            position: i.position,
        })
        .collect();

    let orders = raw
        .orders
        .iter()
        .map(|o| Order {
            id: o.id.clone(),
            items_required: o.items_required.clone(),
            items_delivered: o.items_delivered.clone(),
            complete: o.complete,
            status: o.status.clone(),
        })
        .collect();

    let drop_off_zones = if let Some(zones) = &raw.drop_off_zones {
        zones.clone()
    } else if let Some(zone) = &raw.drop_off {
        vec![*zone]
    } else {
        vec![]
    };

    GameState {
        round: raw.round,
        max_rounds: raw.max_rounds,
        grid,
        bots,
        items,
        orders,
        drop_off_zones,
        score: raw.score.unwrap_or(0),
    }
}
