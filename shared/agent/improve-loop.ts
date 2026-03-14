import { execFileSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import Anthropic from "@anthropic-ai/sdk";
import minimist from "minimist";

function log(level: string, message: string): void {
  const timestamp = new Date().toISOString();
  console.error(`${timestamp} [${level}] ${message}`);
}

const MODEL = "claude-opus-4-6";
const client = new Anthropic();

function runSimGames(
  simScript: string,
  botDir: string,
  nGames: number,
  difficulty = "medium",
): number[] {
  try {
    const result = execFileSync(
      "node",
      ["--experimental-transform-types", simScript, "--bot-dir", botDir, "--games", String(nGames), "--difficulty", difficulty],
      { timeout: 300_000, encoding: "utf-8" },
    );
    const scores: number[] = [];
    for (const line of result.trim().split("\n")) {
      const num = parseInt(line.trim(), 10);
      if (!isNaN(num)) scores.push(num);
    }
    return scores;
  } catch (err: any) {
    log("ERROR", `Sim failed: ${err.stderr ?? err.message}`);
    return [];
  }
}

function medianScore(scores: number[]): number {
  if (scores.length === 0) return 0;
  const sorted = [...scores].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  if (sorted.length % 2 === 0) {
    return (sorted[mid - 1] + sorted[mid]) / 2;
  }
  return sorted[mid];
}

function readRecentReplays(replayDir: string, count = 5): any[] {
  const dir = replayDir;
  if (!fs.existsSync(dir)) return [];

  const files = fs
    .readdirSync(dir)
    .filter((f) => f.endsWith(".json"))
    .map((f) => ({ name: f, mtime: fs.statSync(path.join(dir, f)).mtimeMs }))
    .sort((a, b) => b.mtime - a.mtime)
    .slice(0, count);

  const replays: any[] = [];
  for (const file of files) {
    try {
      replays.push(JSON.parse(fs.readFileSync(path.join(dir, file.name), "utf-8")));
    } catch {
      // skip corrupt files
    }
  }
  return replays;
}

function readBotFiles(botDir: string): Record<string, string> {
  const files: Record<string, string> = {};
  for (const entry of fs.readdirSync(botDir)) {
    if (entry.endsWith(".ts") && !entry.endsWith(".test.ts")) {
      files[entry] = fs.readFileSync(path.join(botDir, entry), "utf-8");
    }
  }
  return files;
}

async function analyzeReplaysWithClaude(
  replays: any[],
  botFiles: Record<string, string>,
): Promise<Record<string, any>> {
  const replaySummaries: any[] = [];
  for (const replay of replays) {
    if (!replay || !Array.isArray(replay)) continue;
    const rounds = replay;
    const totalRounds = rounds.length;
    const finalScore = rounds[totalRounds - 1]?.score ?? 0;

    const waitRounds = rounds.filter((r: any) =>
      r.actions?.some((a: any) => a.action === "wait"),
    ).length;

    const maxPlanningMs = Math.max(
      ...rounds.map((r: any) => r.planning_ms ?? 0),
      0,
    );

    const step = Math.max(1, Math.floor(totalRounds / 10));
    const sampleRounds = rounds.filter((_: any, idx: number) => idx % step === 0).slice(0, 10);

    replaySummaries.push({
      total_rounds: totalRounds,
      final_score: finalScore,
      wait_rounds: waitRounds,
      wait_pct: Math.round((waitRounds / Math.max(totalRounds, 1)) * 1000) / 10,
      max_planning_ms: maxPlanningMs,
      sample_rounds: sampleRounds,
    });
  }

  const prompt = `You are analyzing an AI bot for a grocery delivery game competition.

The bot controls multiple robots that pick up items from shelves and deliver them to drop-off zones.
Scoring: +1 per item delivered, +5 per complete order. More completed orders = better score.

## Bot source code

${JSON.stringify(botFiles, null, 2)}

## Recent game replay summaries (last ${replaySummaries.length} games)

${JSON.stringify(replaySummaries, null, 2)}

## Your task

1. Identify the SINGLE most impactful failure pattern (the one that costs the most points)
2. Propose a SPECIFIC, MINIMAL code fix (edit ONE function if possible)
3. Explain exactly what to change and why it will help

Focus on real problems visible in the data:
- Bots waiting when they could be collecting or delivering
- Bots delivering with only 1 item when more are available
- Deadlocks in narrow aisles
- Poor task assignment (multiple bots chasing same item)
- Not using preview order to pre-fetch

Respond with a JSON object:
{
  "failure_pattern": "one-sentence description of the main problem",
  "impact": "estimated points lost per game",
  "file": "which .ts file to edit",
  "function": "which function to modify",
  "explanation": "why this fix helps",
  "new_code": "the complete replacement function (TypeScript code as string)"
}`;

  const response = await client.messages.create({
    model: MODEL,
    max_tokens: 4096,
    thinking: { type: "enabled", budget_tokens: 2048 },
    messages: [{ role: "user", content: prompt }],
  });

  const text = response.content.find((b) => b.type === "text")?.text ?? "";

  const jsonMatch = text.match(/\{[\s\S]*\}/);
  if (jsonMatch) {
    try {
      return JSON.parse(jsonMatch[0]);
    } catch {
      // fall through
    }
  }

  log("WARN", "Claude did not return valid JSON");
  return {};
}

function replaceFunction(
  lines: string[],
  funcName: string,
  newCode: string,
): string[] | null {
  let startIdx: number | null = null;
  for (let idx = 0; idx < lines.length; idx++) {
    const stripped = lines[idx].trimStart();
    if (
      stripped.startsWith(`function ${funcName}(`) ||
      stripped.startsWith(`export function ${funcName}(`) ||
      stripped.startsWith(`async function ${funcName}(`) ||
      stripped.startsWith(`export async function ${funcName}(`)
    ) {
      startIdx = idx;
      break;
    }
  }

  if (startIdx === null) return null;

  const baseIndent = lines[startIdx].length - lines[startIdx].trimStart().length;
  let endIdx = lines.length;
  for (let idx = startIdx + 1; idx < lines.length; idx++) {
    const line = lines[idx];
    if (!line.trim()) continue;
    const indent = line.length - line.trimStart().length;
    if (
      indent <= baseIndent &&
      (line.trimStart().startsWith("function ") ||
        line.trimStart().startsWith("export function ") ||
        line.trimStart().startsWith("async function ") ||
        line.trimStart().startsWith("export async function ") ||
        line.trimStart().startsWith("export class ") ||
        line.trimStart().startsWith("class "))
    ) {
      endIdx = idx;
      break;
    }
  }

  return [...lines.slice(0, startIdx), ...newCode.split("\n"), "", ...lines.slice(endIdx)];
}

function applyFix(botDir: string, fix: Record<string, any>): boolean {
  const fileName = fix.file;
  const newCode = fix.new_code;
  const functionName = fix.function;

  if (!fileName || !newCode || !functionName) {
    log("WARN", "Incomplete fix spec");
    return false;
  }

  const target = path.join(botDir, fileName);
  if (!fs.existsSync(target)) {
    log("WARN", `Target file not found: ${target}`);
    return false;
  }

  const original = fs.readFileSync(target, "utf-8");
  const backup = target.replace(/\.ts$/, ".ts.bak");
  fs.writeFileSync(backup, original);

  try {
    const lines = original.split("\n");
    const resultLines = replaceFunction(lines, functionName, newCode);
    if (resultLines) {
      fs.writeFileSync(target, resultLines.join("\n"));
      log("INFO", `Applied fix to ${fileName}:${functionName}`);
      return true;
    }
    log("WARN", `Could not locate function ${functionName} in ${fileName}`);
    return false;
  } catch (err: any) {
    log("ERROR", `Failed to apply fix: ${err.message}`);
    // Restore backup
    fs.copyFileSync(backup, target);
    return false;
  }
}

function revertFix(botDir: string, fileName: string): void {
  const target = path.join(botDir, fileName);
  const backup = target.replace(/\.ts$/, ".ts.bak");
  if (fs.existsSync(backup)) {
    fs.copyFileSync(backup, target);
    fs.unlinkSync(backup);
    log("INFO", `Reverted ${fileName}`);
  }
}

async function runImproveLoop(opts: {
  botDir: string;
  simScript: string;
  replayDir: string;
  gamesPerEval: number;
  maxIterations: number;
  minImprovement: number;
  difficulty: string;
}): Promise<void> {
  const history: any[] = [];

  log("INFO", `Starting improve loop: ${opts.maxIterations} iterations, ${opts.gamesPerEval} games/eval`);

  // Baseline
  log("INFO", "Running baseline evaluation...");
  const baselineScores = runSimGames(opts.simScript, opts.botDir, opts.gamesPerEval, opts.difficulty);
  const baseline = medianScore(baselineScores);
  log("INFO", `Baseline median score: ${baseline.toFixed(1)} (n=${baselineScores.length})`);

  let currentBest = baseline;

  for (let iteration = 0; iteration < opts.maxIterations; iteration++) {
    log("INFO", `\n${"=".repeat(60)}`);
    log("INFO", `Iteration ${iteration + 1}/${opts.maxIterations} | Best: ${currentBest.toFixed(1)}`);

    const replays = readRecentReplays(opts.replayDir, 5);
    const botFiles = readBotFiles(opts.botDir);

    log("INFO", "Asking Claude for improvement...");
    const fix = await analyzeReplaysWithClaude(replays, botFiles);

    if (!fix || Object.keys(fix).length === 0) {
      log("WARN", "No fix proposed — skipping iteration");
      continue;
    }

    log("INFO", `Failure pattern: ${fix.failure_pattern ?? "unknown"}`);
    log("INFO", `Proposed fix: ${fix.file}:${fix.function}`);
    log("INFO", `Explanation: ${fix.explanation ?? ""}`);

    const applied = applyFix(opts.botDir, fix);
    if (!applied) continue;

    log("INFO", `Evaluating fix (${opts.gamesPerEval} games)...`);
    const newScores = runSimGames(opts.simScript, opts.botDir, opts.gamesPerEval, opts.difficulty);
    const newMedian = medianScore(newScores);
    const improvement = newMedian - currentBest;

    log("INFO", `New median: ${newMedian.toFixed(1)} | Improvement: ${improvement >= 0 ? "+" : ""}${improvement.toFixed(1)}`);

    if (improvement >= opts.minImprovement) {
      currentBest = newMedian;
      log("INFO", `Fix accepted! New best: ${currentBest.toFixed(1)}`);
      const backup = path.join(opts.botDir, (fix.file ?? "").replace(/\.ts$/, ".ts.bak"));
      if (fs.existsSync(backup)) fs.unlinkSync(backup);
    } else {
      revertFix(opts.botDir, fix.file ?? "");
      log("INFO", `Fix rejected (improvement ${improvement >= 0 ? "+" : ""}${improvement.toFixed(1)} < threshold ${opts.minImprovement})`);
    }

    history.push({
      iteration: iteration + 1,
      timestamp: new Date().toISOString(),
      pattern: fix.failure_pattern,
      fix: `${fix.file}:${fix.function}`,
      before: improvement >= opts.minImprovement ? currentBest - improvement : currentBest,
      after: newMedian,
      accepted: improvement >= opts.minImprovement,
    });

    const historyFile = path.join(path.dirname(opts.botDir), "improve_history.json");
    fs.writeFileSync(historyFile, JSON.stringify(history, null, 2));
  }

  log("INFO", `\n${"=".repeat(60)}`);
  log("INFO", `Improve loop complete. Final best: ${currentBest.toFixed(1)} (started: ${baseline.toFixed(1)})`);
  log("INFO", `Total improvement: ${currentBest - baseline >= 0 ? "+" : ""}${(currentBest - baseline).toFixed(1)} points`);
}

function main(): void {
  const args = minimist(process.argv.slice(2), {
    string: ["bot-dir", "sim-script", "replay-dir", "difficulty"],
    default: {
      "replay-dir": "../../challenges/groceries/replays",
      "games-per-eval": 20,
      "max-iterations": 50,
      "min-improvement": 2,
      difficulty: "medium",
    },
  });

  if (!args["bot-dir"] || !args["sim-script"]) {
    console.error(
      "Usage: node --experimental-transform-types improve-loop.ts " +
        "--bot-dir DIR --sim-script SCRIPT [--games-per-eval 20] " +
        "[--max-iterations 50] [--min-improvement 2] [--difficulty medium]",
    );
    process.exit(1);
  }

  runImproveLoop({
    botDir: args["bot-dir"],
    simScript: args["sim-script"],
    replayDir: args["replay-dir"],
    gamesPerEval: args["games-per-eval"],
    maxIterations: args["max-iterations"],
    minImprovement: args["min-improvement"],
    difficulty: args.difficulty,
  }).catch((err) => {
    console.error(err);
    process.exit(1);
  });
}

main();
