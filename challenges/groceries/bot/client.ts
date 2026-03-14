import WebSocket from "ws";
import minimist from "minimist";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import { parseGameState } from "./state.ts";
import { precomputeDropDistances } from "./pathfinding.ts";
import { planRound } from "./coordinator.ts";
import type { DistanceMap, GameState } from "./types.ts";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const WS_URL = "wss://game.ainm.no/ws";
const REPLAYS_DIR = path.resolve(__dirname, "..", "replays");
const MAPS_DIR = path.resolve(__dirname, "..", "maps");
const PLANNING_TIMEOUT = 1.5; // seconds

function log(level: string, message: string): void {
  const timestamp = new Date().toISOString();
  console.error(`${timestamp} [${level}] ${message}`);
}

class GroceryBotClient {
  private token: string;
  private difficulty: string;
  private saveMap: boolean;
  private dropDist: DistanceMap = new Map();
  private replay: Record<string, any>[] = [];
  private finalScore = 0;

  constructor(token: string, difficulty = "unknown", saveMap = true) {
    this.token = token;
    this.difficulty = difficulty;
    this.saveMap = saveMap;
  }

  async run(): Promise<void> {
    const url = `${WS_URL}?token=${this.token}`;
    log("INFO", `Connecting to ${url}`);

    fs.mkdirSync(REPLAYS_DIR, { recursive: true });
    fs.mkdirSync(MAPS_DIR, { recursive: true });

    return new Promise<void>((resolve, reject) => {
      const ws = new WebSocket(url);

      ws.on("open", () => {
        log("INFO", "Connected");
      });

      ws.on("message", (rawMessage: Buffer) => {
        const data = JSON.parse(rawMessage.toString());
        const msgType = data.type;

        if (msgType === "game_state") {
          const state = parseGameState(data);
          this.finalScore = state.score;

          // Precompute drop distances on round 0
          if (state.round === 0 || this.dropDist.size === 0) {
            this.dropDist = precomputeDropDistances(
              state.dropOffZones,
              state.grid.walls,
              state.grid.width,
              state.grid.height,
            );
            log(
              "INFO",
              `Round 0: grid=${state.grid.width}x${state.grid.height}, ` +
                `bots=${state.bots.length}, ` +
                `drop_zones=${JSON.stringify(state.dropOffZones)}`,
            );
            if (this.saveMap) {
              this.saveMapData(data);
            }
          }

          // Plan with timeout guard
          const t0 = performance.now();
          let actions = planRound(state, this.dropDist);
          const elapsed = (performance.now() - t0) / 1000;

          if (elapsed > PLANNING_TIMEOUT) {
            log(
              "WARN",
              `Round ${state.round}: planning took ${elapsed.toFixed(2)}s — ` +
                "sending wait-all as safety measure",
            );
            actions = state.bots.map((b) => ({ bot: b.id, action: "wait" }));
          }

          const response = JSON.stringify({ actions });
          ws.send(response);

          // Save replay frame
          this.replay.push({
            round: state.round,
            score: state.score,
            actions,
            state_summary: {
              bots: state.bots.map((b) => [b.id, b.position, b.inventory]),
              active_order: state.orders.find((o) => o.status === "active")?.id ?? null,
              items_on_map: state.items.length,
            },
            planning_ms: Math.round(elapsed * 1000),
          });

          if (state.round % 50 === 0) {
            log(
              "INFO",
              `Round ${state.round}/${state.maxRounds} | ` +
                `Score: ${state.score} | ` +
                `Planning: ${(elapsed * 1000).toFixed(1)}ms`,
            );
          }
        } else if (msgType === "game_over") {
          this.finalScore = data.score ?? this.finalScore;
          log("INFO", `Game over! Final score: ${this.finalScore}`);
          this.saveReplay();
          ws.close();
          resolve();
        } else if (msgType === "error") {
          log("ERROR", `Server error: ${JSON.stringify(data)}`);
        }
      });

      ws.on("error", (err) => {
        log("ERROR", `WebSocket error: ${err.message}`);
        reject(err);
      });

      ws.on("close", () => {
        log("INFO", "Disconnected");
        resolve();
      });
    });
  }

  private saveMapData(gameStateData: Record<string, any>): void {
    const mapData = {
      difficulty: this.difficulty,
      grid: gameStateData.grid,
      drop_off_zones:
        gameStateData.drop_off_zones ?? [gameStateData.drop_off],
      captured_at: new Date().toISOString(),
    };
    const filePath = path.join(MAPS_DIR, `${this.difficulty}.json`);
    fs.writeFileSync(filePath, JSON.stringify(mapData, null, 2));
    log("INFO", `Map saved to ${filePath}`);
  }

  private saveReplay(): void {
    const timestamp = new Date()
      .toISOString()
      .replace(/[-:T]/g, "")
      .slice(0, 15)
      .replace(/(\d{8})(\d{6})/, "$1_$2");
    const filePath = path.join(
      REPLAYS_DIR,
      `${this.difficulty}_${timestamp}_score${this.finalScore}.json`,
    );
    fs.writeFileSync(filePath, JSON.stringify(this.replay, null, 2));
    log("INFO", `Replay saved to ${filePath}`);
  }
}

async function main(): Promise<void> {
  const args = minimist(process.argv.slice(2), {
    string: ["token", "difficulty"],
    boolean: ["no-save-map"],
    default: { difficulty: "unknown", "no-save-map": false },
  });

  if (!args.token) {
    console.error("Usage: node --experimental-transform-types client.ts --token <JWT> [--difficulty easy]");
    process.exit(1);
  }

  const client = new GroceryBotClient(
    args.token,
    args.difficulty,
    !args["no-save-map"],
  );
  await client.run();
  console.log(`\nFinal score: ${client["finalScore"]}`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
