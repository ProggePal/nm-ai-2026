use crate::types::*;

/// Plan actions for all bots given the current game state.
///
/// This is the function an AI agent should modify to improve the bot's score.
pub fn plan_round(state: &GameState) -> Vec<RoundAction> {
    state
        .bots
        .iter()
        .map(|bot| RoundAction {
            bot: bot.id,
            action: "wait".to_string(),
            item_id: None,
        })
        .collect()
}
