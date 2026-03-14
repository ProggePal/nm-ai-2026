"""
Rate-limit-aware real server runner.

Respects:
  - 60s cooldown between games
  - 40 games/hour max
  - 300 games/day max

Usage:
  python runner.py \
    --tokens easy:TOKEN1 medium:TOKEN2 hard:TOKEN3 \
    --difficulties easy medium hard \
    --games-per-day 200 \
    --bot-dir ../../challenges/groceries/bot

Tokens are single-use — you need a fresh token from app.ainm.no/challenge for each game.
This script just enforces rate limits and calls client.py for each game.
"""
from __future__ import annotations
import subprocess
import time
import logging
import argparse
from datetime import datetime, timedelta
from collections import deque
from pathlib import Path

logger = logging.getLogger("runner")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

COOLDOWN_SECONDS = 61       # 60s minimum + 1s buffer
MAX_GAMES_PER_HOUR = 38     # 40 - 2 buffer
MAX_GAMES_PER_DAY = 280     # 300 - 20 buffer for manual games


class RateLimitedRunner:
    def __init__(
        self,
        bot_script: str,
        games_per_day: int = MAX_GAMES_PER_DAY,
    ):
        self.bot_script = bot_script
        self.games_per_day = min(games_per_day, MAX_GAMES_PER_DAY)
        self.game_times: deque[float] = deque()  # timestamps of recent games

    def _wait_if_needed(self):
        now = time.time()

        # Enforce cooldown since last game
        if self.game_times:
            elapsed = now - self.game_times[-1]
            if elapsed < COOLDOWN_SECONDS:
                wait = COOLDOWN_SECONDS - elapsed
                logger.info(f"Cooldown: waiting {wait:.1f}s...")
                time.sleep(wait)
                now = time.time()

        # Enforce hourly limit
        one_hour_ago = now - 3600
        recent_hour = sum(1 for t in self.game_times if t > one_hour_ago)
        if recent_hour >= MAX_GAMES_PER_HOUR:
            oldest_in_window = min(t for t in self.game_times if t > one_hour_ago)
            wait = (oldest_in_window + 3600) - now + 5
            logger.info(f"Hourly limit reached. Waiting {wait:.0f}s ({wait/60:.1f}min)...")
            time.sleep(max(wait, 0))
            now = time.time()

        # Enforce daily limit
        one_day_ago = now - 86400
        recent_day = sum(1 for t in self.game_times if t > one_day_ago)
        if recent_day >= self.games_per_day:
            logger.warning(f"Daily limit reached ({recent_day}/{self.games_per_day}). Stopping.")
            raise SystemExit(0)

    def run_game(self, token: str, difficulty: str) -> int:
        """Runs a single game and returns the final score."""
        self._wait_if_needed()

        logger.info(f"Starting game: {difficulty}")
        t0 = time.time()

        result = subprocess.run(
            ["python", self.bot_script, "--token", token, "--difficulty", difficulty],
            capture_output=True,
            text=True,
            timeout=400,  # max game time with buffer
        )

        elapsed = time.time() - t0
        self.game_times.append(time.time())

        if result.returncode != 0:
            logger.error(f"Game failed: {result.stderr[-500:]}")
            return 0

        # Extract score from last line of output
        score = 0
        for line in reversed(result.stdout.strip().split("\n")):
            if "Final score:" in line:
                try:
                    score = int(line.split("Final score:")[-1].strip())
                except ValueError:
                    pass
                break

        logger.info(f"Game complete: {difficulty} | Score: {score} | Time: {elapsed:.0f}s")
        return score

    def run_schedule(self, schedule: list[tuple[str, str]]):
        """
        Runs a list of (token, difficulty) tuples with rate limiting.
        schedule = [("TOKEN_A", "easy"), ("TOKEN_B", "medium"), ...]
        """
        results = []
        for i, (token, difficulty) in enumerate(schedule):
            logger.info(f"Game {i+1}/{len(schedule)}: {difficulty}")
            score = self.run_game(token, difficulty)
            results.append({"difficulty": difficulty, "score": score, "timestamp": datetime.now().isoformat()})

        # Summary
        logger.info("\n=== Run Summary ===")
        for r in results:
            logger.info(f"  {r['difficulty']:12s}: {r['score']}")
        return results


def main():
    parser = argparse.ArgumentParser(description="NM i AI 2026 — Rate-Limited Runner")
    parser.add_argument("--bot-dir", required=True, help="Path to bot directory")
    parser.add_argument(
        "--games",
        nargs="+",
        metavar="DIFFICULTY:TOKEN",
        help="Games to run as difficulty:token pairs, e.g. easy:JWT_TOKEN_HERE",
    )
    parser.add_argument("--games-per-day", type=int, default=MAX_GAMES_PER_DAY)
    args = parser.parse_args()

    bot_script = str(Path(args.bot_dir) / "client.py")
    runner = RateLimitedRunner(bot_script=bot_script, games_per_day=args.games_per_day)

    schedule = []
    for game_spec in (args.games or []):
        parts = game_spec.split(":", 1)
        if len(parts) == 2:
            schedule.append((parts[1], parts[0]))  # (token, difficulty)
        else:
            logger.warning(f"Invalid game spec (expected difficulty:token): {game_spec}")

    if not schedule:
        logger.error("No games specified. Use --games easy:TOKEN medium:TOKEN ...")
        return

    runner.run_schedule(schedule)


if __name__ == "__main__":
    main()
