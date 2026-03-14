"""
Leaderboard monitor.

Polls the leaderboard on a schedule, logs score changes, and alerts when
the team has been overtaken.

Usage:
  python leaderboard_monitor.py --team "YourTeamName" --interval 120

Output:
  - Prints alerts when scores change
  - Writes leaderboard history to leaderboard_history.jsonl
  - Logs score velocity (rate of improvement) per team
"""
from __future__ import annotations
import asyncio
import json
import logging
import argparse
import time
from datetime import datetime
from pathlib import Path

import aiohttp

logger = logging.getLogger("leaderboard")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

LEADERBOARD_URL = "https://app.ainm.no/api/leaderboard"  # adjust if different
HISTORY_FILE = Path(__file__).parent / "leaderboard_history.jsonl"


class LeaderboardMonitor:
    def __init__(self, our_team: str, interval_seconds: int = 120):
        self.our_team = our_team
        self.interval = interval_seconds
        self.last_scores: dict[str, int] = {}  # team → total_score
        self.our_best: int = 0

    async def run(self):
        logger.info(f"Monitoring leaderboard for team: {self.our_team}")
        logger.info(f"Poll interval: {self.interval}s")

        async with aiohttp.ClientSession() as session:
            while True:
                try:
                    await self._poll(session)
                except Exception as e:
                    logger.error(f"Poll failed: {e}")
                await asyncio.sleep(self.interval)

    async def _poll(self, session: aiohttp.ClientSession):
        async with session.get(LEADERBOARD_URL, timeout=aiohttp.ClientTimeout(total=10)) as resp:
            if resp.status != 200:
                logger.warning(f"Leaderboard returned HTTP {resp.status}")
                return
            data = await resp.json()

        # Expected format: list of {"team": str, "score": int, "scores": {...}}
        # Adjust based on actual API response structure
        teams = self._parse_leaderboard(data)
        self._process_update(teams)

    def _parse_leaderboard(self, data) -> list[dict]:
        """
        Parse the leaderboard API response.
        Adjust this method once you've seen the actual response format
        by running: curl https://app.ainm.no/api/leaderboard
        """
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and "teams" in data:
            return data["teams"]
        if isinstance(data, dict) and "leaderboard" in data:
            return data["leaderboard"]
        logger.warning(f"Unknown leaderboard format: {type(data)}")
        return []

    def _process_update(self, teams: list[dict]):
        timestamp = datetime.now().isoformat()
        new_scores: dict[str, int] = {}

        for team in teams:
            name = team.get("team") or team.get("name") or team.get("team_name", "unknown")
            score = team.get("score") or team.get("total_score") or 0
            new_scores[name] = score

        # Check if we've been overtaken
        our_score = new_scores.get(self.our_team, 0)
        if our_score > self.our_best:
            self.our_best = our_score

        # Find our rank
        sorted_teams = sorted(new_scores.items(), key=lambda x: -x[1])
        our_rank = next(
            (i + 1 for i, (name, _) in enumerate(sorted_teams) if name == self.our_team),
            None,
        )

        # Detect score changes
        changed = []
        for team_name, score in new_scores.items():
            old_score = self.last_scores.get(team_name, 0)
            if score != old_score:
                delta = score - old_score
                changed.append((team_name, old_score, score, delta))

        if changed:
            logger.info(f"=== Leaderboard update at {timestamp} ===")
            for team_name, old, new, delta in sorted(changed, key=lambda x: -x[3]):
                marker = " ← US" if team_name == self.our_team else ""
                logger.info(f"  {team_name}: {old} → {new} (+{delta}){marker}")

        # Alert if overtaken
        if our_rank and our_rank > 1:
            leader_name, leader_score = sorted_teams[0]
            gap = leader_score - our_score
            logger.warning(
                f"⚠️  We are rank #{our_rank} | "
                f"Leader: {leader_name} ({leader_score}) | "
                f"Gap: {gap} points"
            )
        elif our_rank == 1:
            logger.info(f"✅ We are #1 with {our_score} points!")

        # Log velocity for top teams (score/hour estimate)
        if changed:
            logger.info(f"Top 5: {sorted_teams[:5]}")

        # Append to history file
        HISTORY_FILE.parent.mkdir(exist_ok=True)
        with HISTORY_FILE.open("a") as f:
            f.write(json.dumps({
                "timestamp": timestamp,
                "our_rank": our_rank,
                "our_score": our_score,
                "top_teams": sorted_teams[:10],
                "changes": [(n, o, s, d) for n, o, s, d in changed],
            }) + "\n")

        self.last_scores = new_scores


async def main():
    parser = argparse.ArgumentParser(description="NM i AI 2026 — Leaderboard Monitor")
    parser.add_argument("--team", required=True, help="Your team name on the leaderboard")
    parser.add_argument("--interval", type=int, default=120, help="Poll interval in seconds")
    args = parser.parse_args()

    monitor = LeaderboardMonitor(our_team=args.team, interval_seconds=args.interval)
    await monitor.run()


if __name__ == "__main__":
    asyncio.run(main())
