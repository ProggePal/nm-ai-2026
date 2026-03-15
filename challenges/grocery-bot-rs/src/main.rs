mod planner;
mod state;
mod types;

use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use chrono::Utc;
use clap::Parser;
use futures_util::{SinkExt, StreamExt};
use tokio_tungstenite::{connect_async, tungstenite::Message};

use state::parse_game_state;
use types::*;

const DEFAULT_WS_URL: &str = "wss://game.ainm.no/ws";

fn log(level: &str, message: &str) {
    let timestamp = Utc::now().to_rfc3339();
    eprintln!("{timestamp} [{level}] {message}");
}

fn project_dir() -> PathBuf {
    std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
}

fn maps_dir() -> PathBuf {
    project_dir().join("maps")
}

fn replays_dir() -> PathBuf {
    project_dir().join("replays")
}

fn recordings_dir() -> PathBuf {
    project_dir().join("recordings")
}

#[derive(Parser, Debug)]
#[command(name = "grocery-bot-rs")]
struct Args {
    /// WebSocket server URL
    #[arg(long, default_value = DEFAULT_WS_URL)]
    url: String,

    /// JWT authentication token
    #[arg(long)]
    token: String,

    /// Game difficulty level
    #[arg(long, default_value = "unknown")]
    difficulty: String,

    /// Disable map saving
    #[arg(long)]
    no_save_map: bool,

    /// Disable recording raw server messages (used by test server)
    #[arg(long)]
    no_record: bool,
}

fn save_map(difficulty: &str, session_ts: &str, raw: &serde_json::Value) {
    let dir = maps_dir();
    fs::create_dir_all(&dir).ok();

    let map_data = MapData {
        difficulty: difficulty.to_string(),
        grid: MapGrid {
            width: raw["grid"]["width"].as_u64().unwrap_or(0) as u32,
            height: raw["grid"]["height"].as_u64().unwrap_or(0) as u32,
            walls: serde_json::from_value(raw["grid"]["walls"].clone()).unwrap_or_default(),
        },
        drop_off_zones: raw
            .get("drop_off_zones")
            .and_then(|v| serde_json::from_value(v.clone()).ok())
            .or_else(|| {
                raw.get("drop_off")
                    .and_then(|v| serde_json::from_value::<Position>(v.clone()).ok())
                    .map(|pos| vec![pos])
            })
            .unwrap_or_default(),
        captured_at: Utc::now().to_rfc3339(),
    };

    let path = dir.join(format!("{difficulty}_{session_ts}.json"));
    match serde_json::to_string_pretty(&map_data) {
        Ok(json) => {
            if let Err(err) = fs::write(&path, json) {
                log("ERROR", &format!("Failed to write map: {err}"));
            } else {
                log("INFO", &format!("Map saved to {}", path.display()));
            }
        }
        Err(err) => log("ERROR", &format!("Failed to serialize map: {err}")),
    }
}

fn save_recording(difficulty: &str, session_ts: &str, recording: &[RecordedMessage]) {
    let dir = recordings_dir();
    fs::create_dir_all(&dir).ok();

    let filename = format!("{difficulty}_{session_ts}.json");
    let path = dir.join(&filename);

    let data = Recording {
        difficulty: difficulty.to_string(),
        recorded_at: Utc::now().to_rfc3339(),
        messages: recording.to_vec(),
    };

    match serde_json::to_string_pretty(&data) {
        Ok(json) => {
            if let Err(err) = fs::write(&path, json) {
                log("ERROR", &format!("Failed to write recording: {err}"));
            } else {
                log("INFO", &format!("Recording saved to {}", path.display()));
            }
        }
        Err(err) => log("ERROR", &format!("Failed to serialize recording: {err}")),
    }
}

fn save_replay(difficulty: &str, session_ts: &str, final_score: i64, replay: &[ReplayFrame]) {
    let dir = replays_dir();
    fs::create_dir_all(&dir).ok();

    let filename = format!("{difficulty}_{session_ts}_score{final_score}.json");
    let path = dir.join(&filename);

    match serde_json::to_string_pretty(replay) {
        Ok(json) => {
            if let Err(err) = fs::write(&path, json) {
                log("ERROR", &format!("Failed to write replay: {err}"));
            } else {
                log("INFO", &format!("Replay saved to {}", path.display()));
            }
        }
        Err(err) => log("ERROR", &format!("Failed to serialize replay: {err}")),
    }
}

#[tokio::main]
async fn main() {
    let args = Args::parse();

    let url = format!("{}?token={}", args.url, args.token);
    log("INFO", &format!("Connecting to {url}"));

    let (ws_stream, _response) = connect_async(&url)
        .await
        .expect("Failed to connect to WebSocket");

    log("INFO", "Connected");

    let (mut write, mut read) = ws_stream.split();

    let session_ts = Utc::now().format("%Y%m%d_%H%M%S").to_string();
    let mut replay: Vec<ReplayFrame> = Vec::new();
    let mut recording: Vec<RecordedMessage> = Vec::new();
    let mut final_score: i64 = 0;
    let mut map_saved = false;
    let connection_start = Instant::now();

    while let Some(msg) = read.next().await {
        let msg = match msg {
            Ok(m) => m,
            Err(err) => {
                log("ERROR", &format!("WebSocket error: {err}"));
                break;
            }
        };

        let text = match msg {
            Message::Text(t) => t,
            Message::Close(_) => {
                log("INFO", "Server closed connection");
                break;
            }
            _ => continue,
        };

        let raw: serde_json::Value = match serde_json::from_str(&text) {
            Ok(v) => v,
            Err(err) => {
                log("ERROR", &format!("Failed to parse message: {err}"));
                continue;
            }
        };

        let msg_type = raw["type"].as_str().unwrap_or("");

        // Record raw message for test server replay
        if !args.no_record {
            recording.push(RecordedMessage {
                elapsed_ms: connection_start.elapsed().as_millis() as u64,
                message: raw.clone(),
            });
        }

        match msg_type {
            "game_state" => {
                let raw_state: RawGameState = match serde_json::from_value(raw.clone()) {
                    Ok(s) => s,
                    Err(err) => {
                        log("ERROR", &format!("Failed to parse game_state: {err}"));
                        continue;
                    }
                };

                let state = parse_game_state(&raw_state);
                final_score = state.score;

                if state.round == 0 {
                    log(
                        "INFO",
                        &format!(
                            "Round 0: grid={}x{}, bots={}, drop_zones={:?}",
                            state.grid.width,
                            state.grid.height,
                            state.bots.len(),
                            state.drop_off_zones,
                        ),
                    );

                    if !args.no_save_map && !map_saved {
                        save_map(&args.difficulty, &session_ts, &raw);
                        map_saved = true;
                    }
                }

                let planning_start = Instant::now();
                let actions = planner::plan_round(&state);
                let planning_ms = planning_start.elapsed().as_millis() as u64;

                // Send response (serialize before moving actions into replay)
                let response_json =
                    serde_json::to_string(&ActionsResponse { actions: &actions }).unwrap();
                if let Err(err) = write.send(Message::Text(response_json)).await {
                    log("ERROR", &format!("Failed to send actions: {err}"));
                    break;
                }

                // Record replay frame
                replay.push(ReplayFrame {
                    round: state.round,
                    score: state.score,
                    actions,
                    state_summary: StateSummary {
                        bots: state
                            .bots
                            .iter()
                            .map(|b| BotSummary {
                                id: b.id,
                                position: b.position,
                                inventory: b.inventory.clone(),
                            })
                            .collect(),
                        active_order: state
                            .orders
                            .iter()
                            .find(|o| o.status == "active")
                            .map(|o| o.id.clone()),
                        items_on_map: state.items.len(),
                    },
                    planning_ms,
                    timestamp: Utc::now().to_rfc3339(),
                });

                if state.round % 50 == 0 {
                    log(
                        "INFO",
                        &format!(
                            "Round {}/{} | Score: {} | Planning: {}ms",
                            state.round, state.max_rounds, state.score, planning_ms,
                        ),
                    );
                }
            }
            "game_over" => {
                if let Some(score) = raw["score"].as_i64() {
                    final_score = score;
                }
                log("INFO", &format!("Game over! Final score: {final_score}"));
                save_replay(&args.difficulty, &session_ts, final_score, &replay);
                if !args.no_record {
                    save_recording(&args.difficulty, &session_ts, &recording);
                }
                break;
            }
            "error" => {
                log("ERROR", &format!("Server error: {raw}"));
            }
            other => {
                log("WARN", &format!("Unknown message type: {other}"));
            }
        }
    }

    log("INFO", "Disconnected");
    println!("\nFinal score: {final_score}");
}
