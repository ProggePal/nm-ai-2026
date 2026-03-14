import { execFileSync } from "node:child_process";
import minimist from "minimist";

const COOLDOWN_SECONDS = 61;
const MAX_GAMES_PER_HOUR = 38;
const MAX_GAMES_PER_DAY = 280;

function log(level: string, message: string): void {
  const timestamp = new Date().toISOString();
  console.error(`${timestamp} [${level}] ${message}`);
}

function sleep(ms: number): void {
  Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, ms);
}

class RateLimitedRunner {
  private botScript: string;
  private gamesPerDay: number;
  private gameTimes: number[] = [];

  constructor(botScript: string, gamesPerDay = MAX_GAMES_PER_DAY) {
    this.botScript = botScript;
    this.gamesPerDay = Math.min(gamesPerDay, MAX_GAMES_PER_DAY);
  }

  private waitIfNeeded(): void {
    let now = Date.now() / 1000;

    // Enforce cooldown since last game
    if (this.gameTimes.length > 0) {
      const elapsed = now - this.gameTimes[this.gameTimes.length - 1];
      if (elapsed < COOLDOWN_SECONDS) {
        const wait = COOLDOWN_SECONDS - elapsed;
        log("INFO", `Cooldown: waiting ${wait.toFixed(1)}s...`);
        sleep(Math.ceil(wait * 1000));
        now = Date.now() / 1000;
      }
    }

    // Enforce hourly limit
    const oneHourAgo = now - 3600;
    const recentHour = this.gameTimes.filter((t) => t > oneHourAgo).length;
    if (recentHour >= MAX_GAMES_PER_HOUR) {
      const oldestInWindow = Math.min(
        ...this.gameTimes.filter((t) => t > oneHourAgo),
      );
      const wait = oldestInWindow + 3600 - now + 5;
      log("INFO", `Hourly limit reached. Waiting ${wait.toFixed(0)}s (${(wait / 60).toFixed(1)}min)...`);
      sleep(Math.max(Math.ceil(wait * 1000), 0));
      now = Date.now() / 1000;
    }

    // Enforce daily limit
    const oneDayAgo = now - 86400;
    const recentDay = this.gameTimes.filter((t) => t > oneDayAgo).length;
    if (recentDay >= this.gamesPerDay) {
      log("WARN", `Daily limit reached (${recentDay}/${this.gamesPerDay}). Stopping.`);
      process.exit(0);
    }
  }

  runGame(token: string, difficulty: string): number {
    this.waitIfNeeded();

    log("INFO", `Starting game: ${difficulty}`);
    const t0 = Date.now();

    try {
      const result = execFileSync(
        "node",
        ["--experimental-transform-types", this.botScript, "--token", token, "--difficulty", difficulty],
        { timeout: 400_000, encoding: "utf-8" },
      );

      const elapsed = (Date.now() - t0) / 1000;
      this.gameTimes.push(Date.now() / 1000);

      // Extract score from last line of output
      let score = 0;
      const lines = result.trim().split("\n");
      for (let idx = lines.length - 1; idx >= 0; idx--) {
        if (lines[idx].includes("Final score:")) {
          const match = lines[idx].split("Final score:")[1]?.trim();
          if (match) score = parseInt(match, 10) || 0;
          break;
        }
      }

      log("INFO", `Game complete: ${difficulty} | Score: ${score} | Time: ${elapsed.toFixed(0)}s`);
      return score;
    } catch (err: any) {
      const elapsed = (Date.now() - t0) / 1000;
      this.gameTimes.push(Date.now() / 1000);
      log("ERROR", `Game failed: ${err.stderr?.slice(-500) ?? err.message}`);
      return 0;
    }
  }

  runSchedule(schedule: [string, string][]): Record<string, any>[] {
    const results: Record<string, any>[] = [];
    for (let idx = 0; idx < schedule.length; idx++) {
      const [token, difficulty] = schedule[idx];
      log("INFO", `Game ${idx + 1}/${schedule.length}: ${difficulty}`);
      const score = this.runGame(token, difficulty);
      results.push({
        difficulty,
        score,
        timestamp: new Date().toISOString(),
      });
    }

    log("INFO", "\n=== Run Summary ===");
    for (const result of results) {
      log("INFO", `  ${result.difficulty.padEnd(12)}: ${result.score}`);
    }
    return results;
  }
}

function main(): void {
  const args = minimist(process.argv.slice(2), {
    string: ["bot-dir", "games"],
    default: { "games-per-day": MAX_GAMES_PER_DAY },
  });

  if (!args["bot-dir"]) {
    console.error("Usage: node --experimental-transform-types runner.ts --bot-dir DIR --games difficulty:TOKEN ...");
    process.exit(1);
  }

  const botScript = `${args["bot-dir"]}/client.ts`;
  const runner = new RateLimitedRunner(botScript, args["games-per-day"]);

  const gameSpecs: string[] = Array.isArray(args.games) ? args.games : args.games ? [args.games] : [];
  // Also grab any remaining positional args after --games
  const allSpecs = [...gameSpecs, ...(args._ as string[])];

  const schedule: [string, string][] = [];
  for (const spec of allSpecs) {
    const parts = spec.split(":", 2);
    if (parts.length === 2) {
      schedule.push([parts[1], parts[0]]); // [token, difficulty]
    } else {
      log("WARN", `Invalid game spec (expected difficulty:token): ${spec}`);
    }
  }

  if (schedule.length === 0) {
    log("ERROR", "No games specified. Use --games easy:TOKEN medium:TOKEN ...");
    process.exit(1);
  }

  runner.runSchedule(schedule);
}

main();
