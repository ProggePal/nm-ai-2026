"""
Autonomous Claude improve loop.

Runs overnight:
  1. Plays N local sim games → collects score distribution + replays
  2. Reads replays → asks Claude to identify failure patterns
  3. Claude proposes a targeted code fix
  4. Evaluates fix (N more games) → commits if score improved, reverts if not
  5. Repeats

Usage:
  python improve_loop.py \
    --bot-dir ../../challenges/groceries/bot \
    --sim-script ../../challenges/groceries/sim/run_sim.py \
    --games-per-eval 20 \
    --max-iterations 50 \
    --min-improvement 2
"""
from __future__ import annotations
import asyncio
import json
import subprocess
import shutil
import logging
import argparse
import time
from pathlib import Path
from datetime import datetime

import anthropic

logger = logging.getLogger("improve_loop")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# Model for code analysis and improvement
MODEL = "claude-opus-4-6"

client = anthropic.Anthropic()


def run_sim_games(sim_script: str, bot_dir: str, n_games: int, difficulty: str = "medium") -> list[int]:
    """
    Runs n_games in the local simulator and returns a list of scores.
    The simulator script should accept --bot-dir, --games, --difficulty flags
    and print one score per line to stdout.
    """
    result = subprocess.run(
        ["python", sim_script, "--bot-dir", bot_dir, "--games", str(n_games), "--difficulty", difficulty],
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode != 0:
        logger.error(f"Sim failed: {result.stderr}")
        return []
    scores = []
    for line in result.stdout.strip().split("\n"):
        try:
            scores.append(int(line.strip()))
        except ValueError:
            pass
    return scores


def median_score(scores: list[int]) -> float:
    if not scores:
        return 0.0
    s = sorted(scores)
    n = len(s)
    if n % 2 == 0:
        return (s[n // 2 - 1] + s[n // 2]) / 2
    return float(s[n // 2])


def read_recent_replays(replay_dir: Path, n: int = 5) -> list[dict]:
    """Reads the N most recent replay files for analysis."""
    replays = sorted(replay_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    result = []
    for path in replays[:n]:
        try:
            result.append(json.loads(path.read_text()))
        except Exception:
            pass
    return result


def read_bot_files(bot_dir: Path) -> dict[str, str]:
    """Reads all Python files in the bot directory."""
    files = {}
    for path in bot_dir.glob("*.py"):
        files[path.name] = path.read_text()
    return files


def analyze_replays_with_claude(replays: list[dict], bot_files: dict[str, str]) -> dict:
    """
    Asks Claude to analyze replays and identify the most impactful failure pattern.
    Returns {"pattern": str, "file": str, "fix": str, "code": str}
    """
    # Summarize replays (don't send entire replay — too large)
    replay_summaries = []
    for replay in replays:
        if not replay:
            continue
        rounds = replay
        total_rounds = len(rounds)
        final_score = rounds[-1].get("score", 0) if rounds else 0

        # Find rounds where bots waited when they shouldn't have
        wait_rounds = sum(
            1 for r in rounds
            if any(a.get("action") == "wait" for a in r.get("actions", []))
        )
        # Find max planning time
        max_planning_ms = max((r.get("planning_ms", 0) for r in rounds), default=0)

        replay_summaries.append({
            "total_rounds": total_rounds,
            "final_score": final_score,
            "wait_rounds": wait_rounds,
            "wait_pct": round(wait_rounds / max(total_rounds, 1) * 100, 1),
            "max_planning_ms": max_planning_ms,
            "sample_rounds": rounds[::max(1, total_rounds // 10)][:10],  # every 10th round
        })

    prompt = f"""You are analyzing an AI bot for a grocery delivery game competition.

The bot controls multiple robots that pick up items from shelves and deliver them to drop-off zones.
Scoring: +1 per item delivered, +5 per complete order. More completed orders = better score.

## Bot source code

{json.dumps({k: v for k, v in bot_files.items()}, indent=2)}

## Recent game replay summaries (last {len(replay_summaries)} games)

{json.dumps(replay_summaries, indent=2)}

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
{{
  "failure_pattern": "one-sentence description of the main problem",
  "impact": "estimated points lost per game",
  "file": "which .py file to edit",
  "function": "which function to modify",
  "explanation": "why this fix helps",
  "new_code": "the complete replacement function (Python code as string)"
}}"""

    response = client.messages.create(
        model=MODEL,
        max_tokens=4096,
        thinking={"type": "adaptive"},
        messages=[{"role": "user", "content": prompt}],
    )

    text = next((b.text for b in response.content if b.type == "text"), "")

    # Extract JSON from response
    import re
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    logger.warning("Claude did not return valid JSON")
    return {}


def apply_fix(bot_dir: Path, fix: dict) -> bool:
    """
    Applies the proposed fix to the bot code.
    Returns True if applied successfully.
    """
    file_name = fix.get("file")
    new_code = fix.get("new_code")
    function_name = fix.get("function")

    if not file_name or not new_code or not function_name:
        logger.warning("Incomplete fix spec")
        return False

    target = bot_dir / file_name
    if not target.exists():
        logger.warning(f"Target file not found: {target}")
        return False

    original = target.read_text()

    # Create backup
    backup = target.with_suffix(".py.bak")
    backup.write_text(original)

    # Replace the function
    import ast
    try:
        # Try to find and replace the function in the source
        lines = original.split("\n")
        result_lines = _replace_function(lines, function_name, new_code)
        if result_lines:
            target.write_text("\n".join(result_lines))
            logger.info(f"Applied fix to {file_name}:{function_name}")
            return True
        else:
            logger.warning(f"Could not locate function {function_name} in {file_name}")
            return False
    except Exception as e:
        logger.error(f"Failed to apply fix: {e}")
        backup.replace(target)  # restore
        return False


def _replace_function(lines: list[str], func_name: str, new_code: str) -> list[str] | None:
    """
    Finds `def func_name` in lines and replaces the entire function with new_code.
    Returns modified lines or None if function not found.
    """
    # Find the start of the function
    start_idx = None
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith(f"def {func_name}(") or stripped.startswith(f"async def {func_name}("):
            start_idx = i
            break

    if start_idx is None:
        return None

    # Find the end of the function (next def at same or lower indent level)
    base_indent = len(lines[start_idx]) - len(lines[start_idx].lstrip())
    end_idx = len(lines)
    for i in range(start_idx + 1, len(lines)):
        line = lines[i]
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        if indent <= base_indent and (line.lstrip().startswith("def ") or
                                       line.lstrip().startswith("async def ") or
                                       line.lstrip().startswith("class ")):
            end_idx = i
            break

    new_lines = lines[:start_idx] + new_code.split("\n") + [""] + lines[end_idx:]
    return new_lines


def revert_fix(bot_dir: Path, file_name: str):
    """Restores the backup file."""
    target = bot_dir / file_name
    backup = target.with_suffix(".py.bak")
    if backup.exists():
        backup.replace(target)
        logger.info(f"Reverted {file_name}")


async def run_improve_loop(
    bot_dir: str,
    sim_script: str,
    replay_dir: str,
    games_per_eval: int = 20,
    max_iterations: int = 50,
    min_improvement: float = 2.0,
    difficulty: str = "medium",
):
    bot_path = Path(bot_dir)
    replay_path = Path(replay_dir)
    history = []

    logger.info(f"Starting improve loop: {max_iterations} iterations, {games_per_eval} games/eval")

    # Baseline
    logger.info("Running baseline evaluation...")
    baseline_scores = run_sim_games(sim_script, bot_dir, games_per_eval, difficulty)
    baseline = median_score(baseline_scores)
    logger.info(f"Baseline median score: {baseline:.1f} (n={len(baseline_scores)})")

    current_best = baseline

    for iteration in range(max_iterations):
        logger.info(f"\n{'='*60}")
        logger.info(f"Iteration {iteration + 1}/{max_iterations} | Best: {current_best:.1f}")

        # Read recent replays and bot code
        replays = read_recent_replays(replay_path, n=5)
        bot_files = read_bot_files(bot_path)

        # Ask Claude for a fix
        logger.info("Asking Claude for improvement...")
        fix = analyze_replays_with_claude(replays, bot_files)

        if not fix:
            logger.warning("No fix proposed — skipping iteration")
            continue

        logger.info(f"Failure pattern: {fix.get('failure_pattern', 'unknown')}")
        logger.info(f"Proposed fix: {fix.get('file')}:{fix.get('function')}")
        logger.info(f"Explanation: {fix.get('explanation', '')}")

        # Apply the fix
        applied = apply_fix(bot_path, fix)
        if not applied:
            continue

        # Evaluate the fix
        logger.info(f"Evaluating fix ({games_per_eval} games)...")
        new_scores = run_sim_games(sim_script, bot_dir, games_per_eval, difficulty)
        new_median = median_score(new_scores)
        improvement = new_median - current_best

        logger.info(f"New median: {new_median:.1f} | Improvement: {improvement:+.1f}")

        if improvement >= min_improvement:
            # Keep the fix
            current_best = new_median
            logger.info(f"✅ Fix accepted! New best: {current_best:.1f}")
            # Clean up backup
            backup = bot_path / (fix.get("file", "").replace(".py", ".py.bak"))
            if backup.exists():
                backup.unlink()
        else:
            # Revert
            revert_fix(bot_path, fix.get("file", ""))
            logger.info(f"❌ Fix rejected (improvement {improvement:+.1f} < threshold {min_improvement})")

        history.append({
            "iteration": iteration + 1,
            "timestamp": datetime.now().isoformat(),
            "pattern": fix.get("failure_pattern"),
            "fix": f"{fix.get('file')}:{fix.get('function')}",
            "before": current_best if improvement < min_improvement else current_best - improvement,
            "after": new_median,
            "accepted": improvement >= min_improvement,
        })

        # Save history
        history_file = bot_path.parent / "improve_history.json"
        history_file.write_text(json.dumps(history, indent=2))

    logger.info(f"\n{'='*60}")
    logger.info(f"Improve loop complete. Final best: {current_best:.1f} (started: {baseline:.1f})")
    logger.info(f"Total improvement: {current_best - baseline:+.1f} points")


def main():
    parser = argparse.ArgumentParser(description="NM i AI 2026 — Autonomous Improve Loop")
    parser.add_argument("--bot-dir", required=True)
    parser.add_argument("--sim-script", required=True)
    parser.add_argument("--replay-dir", default="../../challenges/groceries/replays")
    parser.add_argument("--games-per-eval", type=int, default=20)
    parser.add_argument("--max-iterations", type=int, default=50)
    parser.add_argument("--min-improvement", type=float, default=2.0)
    parser.add_argument("--difficulty", default="medium")
    args = parser.parse_args()

    asyncio.run(run_improve_loop(
        bot_dir=args.bot_dir,
        sim_script=args.sim_script,
        replay_dir=args.replay_dir,
        games_per_eval=args.games_per_eval,
        max_iterations=args.max_iterations,
        min_improvement=args.min_improvement,
        difficulty=args.difficulty,
    ))


if __name__ == "__main__":
    main()
