"""Core data types for Astar Island simulation."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Settlement:
    """A Norse settlement on the grid."""

    x: int
    y: int
    population: float = 1.0
    food: float = 1.0
    wealth: float = 0.0
    defense: float = 0.5
    tech_level: float = 0.0
    has_port: bool = False
    has_longship: bool = False
    owner_id: int = 0
    alive: bool = True
    raid_damage_taken: float = 0.0  # accumulated raid damage this turn

    def copy(self) -> Settlement:
        return Settlement(
            x=self.x,
            y=self.y,
            population=self.population,
            food=self.food,
            wealth=self.wealth,
            defense=self.defense,
            tech_level=self.tech_level,
            has_port=self.has_port,
            has_longship=self.has_longship,
            owner_id=self.owner_id,
            alive=self.alive,
            raid_damage_taken=self.raid_damage_taken,
        )


@dataclass
class WorldState:
    """Full state of the simulation world."""

    grid: np.ndarray  # shape (height, width), dtype int
    settlements: list[Settlement]
    width: int
    height: int
    war_pairs: set[tuple[int, int]] = field(default_factory=set)  # faction pairs at war this turn

    def copy(self) -> WorldState:
        return WorldState(
            grid=self.grid.copy(),
            settlements=[s.copy() for s in self.settlements],
            width=self.width,
            height=self.height,
            war_pairs=set(self.war_pairs),
        )

    @classmethod
    def from_api(cls, initial_state: dict, width: int, height: int) -> WorldState:
        """Create WorldState from API initial_state dict."""
        grid = np.array(initial_state["grid"], dtype=np.int32)
        settlements = []
        for idx, sdata in enumerate(initial_state["settlements"]):
            settlements.append(Settlement(
                x=sdata["x"],
                y=sdata["y"],
                has_port=sdata.get("has_port", False),
                alive=sdata.get("alive", True),
                owner_id=idx,  # Each settlement starts as its own faction
            ))
        return cls(grid=grid, settlements=settlements, width=width, height=height)


@dataclass
class Observation:
    """An observation from a simulate query."""

    seed_index: int
    viewport_x: int
    viewport_y: int
    viewport_w: int
    viewport_h: int
    grid: np.ndarray  # shape (viewport_h, viewport_w), dtype int
    settlements: list[Settlement]

    @classmethod
    def from_api(cls, seed_index: int, response: dict) -> Observation:
        """Create Observation from API simulate response."""
        vp = response["viewport"]
        grid = np.array(response["grid"], dtype=np.int32)
        assert grid.shape == (vp["h"], vp["w"]), (
            f"Grid shape {grid.shape} doesn't match viewport ({vp['h']}, {vp['w']})"
        )
        settlements = []
        for sdata in response.get("settlements", []):
            settlements.append(Settlement(
                x=sdata["x"],
                y=sdata["y"],
                population=sdata.get("population", 1.0),
                food=sdata.get("food", 1.0),
                wealth=sdata.get("wealth", 0.0),
                defense=sdata.get("defense", 0.5),
                has_port=sdata.get("has_port", False),
                alive=sdata.get("alive", True),
                owner_id=sdata.get("owner_id", 0),
            ))
        return cls(
            seed_index=seed_index,
            viewport_x=vp["x"],
            viewport_y=vp["y"],
            viewport_w=vp["w"],
            viewport_h=vp["h"],
            grid=grid,
            settlements=settlements,
        )

    @classmethod
    def from_dict(cls, data: dict) -> Observation:
        """Create Observation from a serialized dict (saved to disk)."""
        grid = np.array(data["grid"], dtype=np.int32)
        settlements = []
        for sdata in data.get("settlements", []):
            settlements.append(Settlement(
                x=sdata["x"],
                y=sdata["y"],
                population=sdata.get("population", 1.0),
                food=sdata.get("food", 1.0),
                wealth=sdata.get("wealth", 0.0),
                defense=sdata.get("defense", 0.5),
                has_port=sdata.get("has_port", False),
                alive=sdata.get("alive", True),
                owner_id=sdata.get("owner_id", 0),
            ))
        return cls(
            seed_index=data["seed_index"],
            viewport_x=data["viewport_x"],
            viewport_y=data["viewport_y"],
            viewport_w=data["viewport_w"],
            viewport_h=data["viewport_h"],
            grid=grid,
            settlements=settlements,
        )
