# grocery-bot-rs

Rust implementation of the grocery bot for the NM AI 2026 challenge. Connects to a WebSocket game server, receives game state each round, sends back bot actions, and persists maps/replays/recordings to disk.

## Binaries

### `grocery-bot` (src/main.rs)

The bot client. Connects to the game server, plays the game, and saves output files.

```sh
cargo run --bin grocery-bot -- --token <JWT> --difficulty medium
```

**CLI flags:**

| Flag             | Default                      | Description                                      |
| ---------------- | ---------------------------- | ------------------------------------------------ |
| `--url`          | `wss://game.ainm.no/ws`     | WebSocket server URL                             |
| `--token`        | *(required)*                 | JWT authentication token                         |
| `--difficulty`   | `unknown`                    | Difficulty label (used in output filenames)       |
| `--no-save-map`  | `false`                      | Skip saving the map file                         |
| `--no-record`    | `false`                      | Skip saving raw server messages (used by simulate) |

### `simulate` (src/simulate.rs)

Local game engine that runs the bot against a recorded scenario. No network needed — uses the map, items, and order sequence extracted from a recording file.

```sh
cargo run --release --bin simulate -- --recording recordings/medium_20260315_170000.json
```

**CLI flags:**

| Flag            | Default  | Description                              |
| --------------- | -------- | ---------------------------------------- |
| `--recording`   | *(required)* | Path to a recording file             |
| `--verbose`     | `false`  | Print state every 50 rounds              |

Outputs the score to stdout (for easy parsing by scripts/agents) and detailed stats to stderr.

## AI agent iteration loop

This project is designed for an AI agent to autonomously improve the bot's score. Follow this loop:

### Prerequisites

One or more recording files must exist in `recordings/`. A human runs real games to capture them (requires a JWT from dev.ainm.no). Each recording captures a different game with different maps, items, and order sequences. After recording, no network access is needed.

### The loop

1. **Edit `src/planner.rs`** — this is the only file you need to change. It contains the `plan_round` function:
   ```rust
   pub fn plan_round(state: &GameState) -> Vec<RoundAction>
   ```
   It receives the full game state and must return one action per bot.

2. **Build and run the simulator** against a random recording:
   ```sh
   cargo run --release --bin simulate -- --recording "$(ls recordings/*.json | shuf -n1)" 2>/dev/null
   ```
   This prints **only the score** (an integer) to stdout. Stderr has detailed stats — redirect it to `/dev/null` for clean parsing, or keep it for debugging.

   To test against a specific recording:
   ```sh
   cargo run --release --bin simulate -- --recording recordings/medium_20260315_170000.json 2>/dev/null
   ```

   To test against **all** recordings and see aggregate results:
   ```sh
   for f in recordings/*.json; do echo "$(cargo run --release --bin simulate -- --recording "$f" 2>/dev/null) $f"; done
   ```

3. **Read the score** — higher is better. Evaluate whether your change improved or regressed. Test against multiple recordings to avoid overfitting to one scenario.

4. **Repeat** — go back to step 1.

### What `plan_round` receives

The `GameState` struct (defined in `src/types.rs`) contains:

| Field            | Type              | Description                                        |
| ---------------- | ----------------- | -------------------------------------------------- |
| `round`          | `u32`             | Current round (0-indexed)                          |
| `max_rounds`     | `u32`             | Total rounds in the game                           |
| `grid.width`     | `u32`             | Grid width                                         |
| `grid.height`    | `u32`             | Grid height                                        |
| `grid.walls`     | `Vec<[i32; 2]>`   | Wall/shelf positions (impassable)                  |
| `bots`           | `Vec<Bot>`        | Each bot has `id`, `position: [i32; 2]`, `inventory: Vec<String>` |
| `items`          | `Vec<Item>`       | Items on map: `id`, `item_type`, `position`        |
| `orders`         | `Vec<Order>`      | Active + preview orders with `items_required`, `items_delivered`, `status` |
| `drop_off_zones` | `Vec<[i32; 2]>`   | Where to deliver items                             |
| `score`          | `i64`             | Current score                                      |

### What `plan_round` returns

A `Vec<RoundAction>` — one per bot:

```rust
RoundAction {
    bot: u32,           // Bot ID
    action: String,     // "move_up", "move_down", "move_left", "move_right", "pick_up", "drop_off", "wait"
    item_id: Option<String>, // Required for "pick_up", None otherwise
}
```

### Strategy tips

- Each delivered item = +1 point. Each completed order = +5 bonus.
- Items are on wall/shelf cells. Bots pick up from an adjacent walkable cell (Manhattan distance 1).
- Bot inventory cap is 3 items.
- Only items matching the **active** order get delivered on drop-off. Non-matching items stay in inventory.
- When an order completes, the next one activates immediately — leftover inventory is re-checked.
- Actions resolve in bot ID order (bot 0 moves first). Moves into walls/bots fail silently.
- Use BFS/A* for pathfinding around walls. Coordinate bots to avoid collisions and duplicate work.

## Game rules (implemented in engine.rs)

- **Movement**: `move_up` (y-1), `move_down` (y+1), `move_left` (x-1), `move_right` (x+1). Moves into walls, out-of-bounds, or occupied cells fail silently.
- **Action resolution**: bot ID order (bot 0 first, then bot 1, etc.)
- **Pickup**: `pick_up` with `item_id`. Bot must be adjacent (Manhattan distance 1) to the item's shelf. Max inventory: 3 items.
- **Drop-off**: `drop_off` while standing on a drop-off zone. Only items matching the active order are delivered. Non-matching items stay in inventory.
- **Scoring**: +1 per item delivered, +5 bonus per completed order.
- **Order progression**: when the active order completes, the next order activates immediately and remaining inventory is re-checked against it.
- **Coordinate system**: origin (0,0) is top-left. X increases right, Y increases down.

## Output directories

All paths are relative to the working directory (run from this project root).

| Directory      | Contents                                                                 |
| -------------- | ------------------------------------------------------------------------ |
| `maps/`        | Grid layout snapshots (walls, dimensions, drop-off zones)                |
| `replays/`     | Per-round bot decisions and state summaries                              |
| `recordings/`  | Full raw server messages with timing — used by `simulate`                    |

### Session linking

All output files from a single game session share the same `YYYYMMDD_HHMMSS` timestamp:

```
maps/medium_20260315_170000.json
replays/medium_20260315_170000_score42.json
recordings/medium_20260315_170000.json
```

## Source structure

```
src/
  main.rs            # Bot client — WebSocket connection, game loop, file saving
  planner.rs         # Planning logic — THIS IS WHAT TO MODIFY to improve the bot
  engine.rs          # Local game engine — implements all game rules
  scenario.rs        # Extracts game scenario (map, items, orders) from a recording
  simulate.rs        # Simulation binary — runs planner against engine, outputs score
  types.rs           # All data types (server protocol, parsed state, storage formats)
  state.rs           # Parses raw server JSON into typed GameState
```

## WebSocket protocol

The game server sends JSON messages over WebSocket. Three message types:

- **`game_state`** — sent each round. Contains grid, bots, items, orders, drop-off zones, score, round number.
- **`game_over`** — sent when the game ends. Contains final score.
- **`error`** — server-side error.

The client responds to each `game_state` with:

```json
{
  "actions": [
    { "bot": 0, "action": "move_up" },
    { "bot": 1, "action": "pick_up", "item_id": "item_3" },
    { "bot": 2, "action": "wait" }
  ]
}
```

Valid actions: `move_up`, `move_down`, `move_left`, `move_right`, `pick_up`, `drop_off`, `wait`.

## Dependencies

- `tokio` + `tokio-tungstenite` — async WebSocket client
- `serde` + `serde_json` — JSON serialization
- `clap` — CLI argument parsing
- `chrono` — timestamps
- `futures-util` — stream/sink utilities for WebSocket
