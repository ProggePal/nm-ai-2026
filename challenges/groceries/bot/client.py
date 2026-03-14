"""
WebSocket game client.

Usage:
  python client.py --token <JWT_TOKEN> [--difficulty easy] [--save-map]

The client:
  1. Connects to wss://game.ainm.no/ws?token=<token>
  2. On first game_state: precomputes drop-off distance matrix + saves map
  3. Each round: calls coordinator.plan_round() and sends actions
  4. 1.5s safety guard: sends wait-all if planning overruns
  5. Saves replay JSON to ../replays/
"""
from __future__ import annotations
import asyncio
import json
import logging
import time
import argparse
from datetime import datetime
from pathlib import Path

import websockets

from state import GameState
from pathfinding import precompute_drop_distances
from coordinator import plan_round

logger = logging.getLogger("grocery_bot")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

WS_URL = "wss://game.ainm.no/ws"
REPLAYS_DIR = Path(__file__).parent.parent / "replays"
MAPS_DIR = Path(__file__).parent.parent / "maps"
PLANNING_TIMEOUT = 1.5  # seconds — send wait-all if planning exceeds this


class GroceryBotClient:
    def __init__(self, token: str, difficulty: str = "unknown", save_map: bool = True):
        self.token = token
        self.difficulty = difficulty
        self.save_map = save_map
        self.drop_dist: dict = {}
        self.replay: list[dict] = []
        self.final_score: int = 0

    async def run(self):
        url = f"{WS_URL}?token={self.token}"
        logger.info(f"Connecting to {url}")

        REPLAYS_DIR.mkdir(exist_ok=True)
        MAPS_DIR.mkdir(exist_ok=True)

        async with websockets.connect(url) as ws:
            async for raw_message in ws:
                data = json.loads(raw_message)
                msg_type = data.get("type")

                if msg_type == "game_state":
                    state = GameState.from_json(data)
                    self.final_score = state.score

                    # Precompute drop distances on round 0
                    if state.round == 0 or not self.drop_dist:
                        self.drop_dist = precompute_drop_distances(
                            state.drop_off_zones,
                            state.grid.walls,
                            state.grid.width,
                            state.grid.height,
                        )
                        logger.info(
                            f"Round 0: grid={state.grid.width}x{state.grid.height}, "
                            f"bots={len(state.bots)}, "
                            f"drop_zones={state.drop_off_zones}"
                        )
                        if self.save_map:
                            self._save_map(data)

                    # Plan with timeout guard
                    t0 = time.perf_counter()
                    actions = plan_round(state, self.drop_dist)
                    elapsed = time.perf_counter() - t0

                    if elapsed > PLANNING_TIMEOUT:
                        logger.warning(
                            f"Round {state.round}: planning took {elapsed:.2f}s — "
                            "sending wait-all as safety measure"
                        )
                        actions = [{"bot": b.id, "action": "wait"} for b in state.bots]

                    response = {"actions": actions}
                    await ws.send(json.dumps(response))

                    # Save replay frame
                    self.replay.append({
                        "round": state.round,
                        "score": state.score,
                        "actions": actions,
                        "state_summary": {
                            "bots": [(b.id, b.position, b.inventory) for b in state.bots],
                            "active_order": state.active_order.id if state.active_order else None,
                            "truly_needed": state.truly_needed(),
                            "items_on_map": len(state.items),
                        },
                        "planning_ms": round(elapsed * 1000),
                    })

                    if state.round % 50 == 0:
                        logger.info(
                            f"Round {state.round}/{state.max_rounds} | "
                            f"Score: {state.score} | "
                            f"Planning: {elapsed*1000:.1f}ms"
                        )

                elif msg_type == "game_over":
                    self.final_score = data.get("score", self.final_score)
                    logger.info(f"Game over! Final score: {self.final_score}")
                    self._save_replay()
                    break

                elif msg_type == "error":
                    logger.error(f"Server error: {data}")

    def _save_map(self, game_state_data: dict):
        """Save the map layout for use in the local simulator."""
        map_data = {
            "difficulty": self.difficulty,
            "grid": game_state_data["grid"],
            "drop_off_zones": game_state_data.get(
                "drop_off_zones",
                [game_state_data.get("drop_off")]
            ),
            "captured_at": datetime.now().isoformat(),
        }
        path = MAPS_DIR / f"{self.difficulty}.json"
        path.write_text(json.dumps(map_data, indent=2))
        logger.info(f"Map saved to {path}")

    def _save_replay(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = REPLAYS_DIR / f"{self.difficulty}_{timestamp}_score{self.final_score}.json"
        path.write_text(json.dumps(self.replay, indent=2))
        logger.info(f"Replay saved to {path}")


async def main():
    parser = argparse.ArgumentParser(description="NM i AI 2026 — Grocery Bot")
    parser.add_argument("--token", required=True, help="JWT token from app.ainm.no/challenge")
    parser.add_argument("--difficulty", default="unknown", help="Difficulty label for replay/map saving")
    parser.add_argument("--no-save-map", action="store_true", help="Skip saving map data")
    args = parser.parse_args()

    client = GroceryBotClient(
        token=args.token,
        difficulty=args.difficulty,
        save_map=not args.no_save_map,
    )
    await client.run()
    print(f"\nFinal score: {client.final_score}")


if __name__ == "__main__":
    asyncio.run(main())
