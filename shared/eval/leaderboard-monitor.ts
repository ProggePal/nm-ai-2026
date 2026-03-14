import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import minimist from "minimist";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const LEADERBOARD_URL = "https://app.ainm.no/api/leaderboard";
const HISTORY_FILE = path.join(__dirname, "leaderboard_history.jsonl");

function log(level: string, message: string): void {
  const timestamp = new Date().toISOString();
  console.error(`${timestamp} [${level}] ${message}`);
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

class LeaderboardMonitor {
  private ourTeam: string;
  private interval: number;
  private lastScores = new Map<string, number>();
  private ourBest = 0;

  constructor(ourTeam: string, intervalSeconds = 120) {
    this.ourTeam = ourTeam;
    this.interval = intervalSeconds;
  }

  async run(): Promise<void> {
    log("INFO", `Monitoring leaderboard for team: ${this.ourTeam}`);
    log("INFO", `Poll interval: ${this.interval}s`);

    while (true) {
      try {
        await this.poll();
      } catch (err: any) {
        log("ERROR", `Poll failed: ${err.message}`);
      }
      await sleep(this.interval * 1000);
    }
  }

  private async poll(): Promise<void> {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 10_000);

    try {
      const resp = await fetch(LEADERBOARD_URL, { signal: controller.signal });
      if (!resp.ok) {
        log("WARN", `Leaderboard returned HTTP ${resp.status}`);
        return;
      }
      const data = await resp.json();
      const teams = this.parseLeaderboard(data);
      this.processUpdate(teams);
    } finally {
      clearTimeout(timeout);
    }
  }

  private parseLeaderboard(data: any): Record<string, any>[] {
    if (Array.isArray(data)) return data;
    if (data?.teams) return data.teams;
    if (data?.leaderboard) return data.leaderboard;
    log("WARN", `Unknown leaderboard format: ${typeof data}`);
    return [];
  }

  private processUpdate(teams: Record<string, any>[]): void {
    const timestamp = new Date().toISOString();
    const newScores = new Map<string, number>();

    for (const team of teams) {
      const name = team.team ?? team.name ?? team.team_name ?? "unknown";
      const score = team.score ?? team.total_score ?? 0;
      newScores.set(name, score);
    }

    const ourScore = newScores.get(this.ourTeam) ?? 0;
    if (ourScore > this.ourBest) {
      this.ourBest = ourScore;
    }

    // Find our rank
    const sortedTeams = [...newScores.entries()].sort((a, b) => b[1] - a[1]);
    const ourRank = sortedTeams.findIndex(([name]) => name === this.ourTeam) + 1 || null;

    // Detect score changes
    const changed: [string, number, number, number][] = [];
    for (const [teamName, score] of newScores) {
      const oldScore = this.lastScores.get(teamName) ?? 0;
      if (score !== oldScore) {
        changed.push([teamName, oldScore, score, score - oldScore]);
      }
    }

    if (changed.length > 0) {
      log("INFO", `=== Leaderboard update at ${timestamp} ===`);
      changed.sort((a, b) => b[3] - a[3]);
      for (const [teamName, old, current, delta] of changed) {
        const marker = teamName === this.ourTeam ? " <- US" : "";
        log("INFO", `  ${teamName}: ${old} -> ${current} (+${delta})${marker}`);
      }
    }

    // Alert if overtaken
    if (ourRank && ourRank > 1) {
      const [leaderName, leaderScore] = sortedTeams[0];
      const gap = leaderScore - ourScore;
      log("WARN", `We are rank #${ourRank} | Leader: ${leaderName} (${leaderScore}) | Gap: ${gap} points`);
    } else if (ourRank === 1) {
      log("INFO", `We are #1 with ${ourScore} points!`);
    }

    if (changed.length > 0) {
      log("INFO", `Top 5: ${JSON.stringify(sortedTeams.slice(0, 5))}`);
    }

    // Append to history file
    fs.appendFileSync(
      HISTORY_FILE,
      JSON.stringify({
        timestamp,
        our_rank: ourRank,
        our_score: ourScore,
        top_teams: sortedTeams.slice(0, 10),
        changes: changed,
      }) + "\n",
    );

    this.lastScores = newScores;
  }
}

async function main(): Promise<void> {
  const args = minimist(process.argv.slice(2), {
    string: ["team"],
    default: { interval: 120 },
  });

  if (!args.team) {
    console.error("Usage: node --experimental-transform-types leaderboard-monitor.ts --team TeamName [--interval 120]");
    process.exit(1);
  }

  const monitor = new LeaderboardMonitor(args.team, args.interval);
  await monitor.run();
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
