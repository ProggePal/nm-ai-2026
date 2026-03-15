# Agent Instructions

You are improving a grocery bot for the NM AI 2026 challenge. Your goal is to maximize the score by iterating on the planning logic.

## Setup

1. Read `README.md` for full context on the project, game rules, and data types.
2. Read `src/planner.rs` — this is the **only file you should modify**.
3. Read `src/types.rs` to understand `GameState`, `RoundAction`, and related types.

## Your loop

Repeat these steps until you are satisfied with the score or run out of ideas:

1. **Get a baseline score.** Run the simulator against all recordings:
   ```sh
   for f in recordings/*.json; do echo "$(cargo run --release --bin simulate -- --recording "$f" 2>/dev/null) $f"; done
   ```
   Note the scores.

2. **Analyze the game state.** If you need to understand what data a scenario contains (grid size, number of bots, items, orders), run with verbose:
   ```sh
   cargo run --release --bin simulate -- --recording recordings/<file>.json --verbose
   ```

3. **Plan your improvement.** Think about what the planner is doing wrong or could do better. Consider:
   - Are bots pathfinding around walls or getting stuck?
   - Are bots picking up items the active order actually needs?
   - Are bots coordinating (not duplicating targets)?
   - Are bots delivering efficiently (full loads, short paths)?
   - Is the bot handling order transitions (preview -> active)?

4. **Edit `src/planner.rs`.** Make one focused improvement at a time. The function signature is:
   ```rust
   pub fn plan_round(state: &GameState) -> Vec<RoundAction>
   ```
   You may add helper functions, structs, and modules as needed, but `plan_round` is the entry point called each round.

5. **Test your change.** Run against all recordings again:
   ```sh
   for f in recordings/*.json; do echo "$(cargo run --release --bin simulate -- --recording "$f" 2>/dev/null) $f"; done
   ```
   If compilation fails, fix the error and re-run.

6. **Evaluate.** Compare scores to the baseline:
   - Score improved across recordings? Keep the change.
   - Score regressed or only improved on one recording? Reconsider — you may be overfitting.
   - Score unchanged? The change may not be reaching the code path you think. Add `--verbose` and inspect.

7. **Go to step 3.**

## Rules

- Only modify `src/planner.rs` (and any new modules it imports).
- Do not modify `engine.rs`, `simulate.rs`, `types.rs`, `scenario.rs`, `state.rs`, or `main.rs`.
- Do not modify `Cargo.toml` unless you need a new dependency (ask first).
- Always test against **all** recordings, not just one.
- Always use `--release` for simulation runs.
- The working directory is `challenges/grocery-bot-rs`.

## Key game rules (quick reference)

- Bots move on a grid with walls. Moves into walls/bots/out-of-bounds fail silently (treated as wait).
- Actions resolve in bot ID order (bot 0 first).
- `pick_up` requires Manhattan distance 1 to the item's shelf cell + `item_id` field.
- `drop_off` requires standing on a drop-off zone. Only items matching the active order are delivered.
- +1 point per delivered item, +5 bonus per completed order.
- Max inventory: 3 items per bot.
- When an order completes, the next activates immediately and remaining inventory is re-checked.
- Coordinate system: (0,0) top-left, X right, Y down.
