"""Simulation parameters with bounds for inference."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import ClassVar

import numpy as np


@dataclass
class SimParams:
    """~30 hidden parameters governing the simulation, grouped by phase."""

    # --- Growth ---
    base_food_production: float = 0.4
    forest_food_bonus: float = 0.25
    plains_food_bonus: float = 0.08
    food_consumption_rate: float = 0.2       # food consumed per unit population per turn
    pop_growth_rate: float = 0.15
    pop_growth_food_threshold: float = 0.6
    carrying_capacity_per_food: float = 6.0  # max pop = this * food_production
    port_development_threshold: float = 2.5
    longship_build_threshold: float = 1.5
    expansion_threshold: float = 2.0
    expansion_range: float = 3.0
    expansion_pop_transfer: float = 0.35

    # --- Conflict ---
    raid_range_base: float = 2.0
    raid_range_longship: float = 5.0
    raid_desperation_threshold: float = 0.3
    raid_strength_factor: float = 0.5
    raid_loot_fraction: float = 0.3
    raid_damage_factor: float = 0.25
    conquest_threshold: float = 0.3
    raid_kill_threshold: float = 0.15        # defender dies if pop drops below this after raid

    # --- Trade ---
    trade_range: float = 4.0
    trade_food_gain: float = 0.15
    trade_wealth_gain: float = 0.1
    tech_diffusion_rate: float = 0.1

    # --- Winter ---
    winter_severity_mean: float = 0.5
    winter_severity_variance: float = 0.3
    collapse_food_threshold: float = 0.4
    collapse_probability: float = 0.4
    survival_bonus: float = 0.3           # large settlements resist collapse (higher = stickier)
    dispersal_range: float = 3.0
    dispersal_fraction: float = 0.5

    # --- Environment ---
    forest_reclaim_probability: float = 0.1
    ruin_rebuild_range: float = 4.0
    ruin_rebuild_threshold: float = 2.5
    ruin_rebuild_fraction: float = 0.2
    ruin_to_plains_probability: float = 0.05

    # --- Additional collapse triggers (appended to preserve indices) ---
    harsh_winter_collapse_factor: float = 1.5   # collapse check fires when severity > this * mean
    raid_collapse_threshold: float = 0.3         # accumulated raid damage that triggers collapse

    # Bounds: (min, max) for each parameter
    BOUNDS: ClassVar[dict[str, tuple[float, float]]] = {
        # Growth
        "base_food_production": (0.05, 1.5),
        "forest_food_bonus": (0.02, 1.0),
        "plains_food_bonus": (0.0, 0.5),
        "food_consumption_rate": (0.1, 1.5),
        "pop_growth_rate": (0.01, 0.3),
        "pop_growth_food_threshold": (0.2, 3.0),
        "carrying_capacity_per_food": (1.0, 10.0),
        "port_development_threshold": (1.0, 8.0),
        "longship_build_threshold": (0.5, 5.0),
        "expansion_threshold": (2.0, 12.0),
        "expansion_range": (1.0, 6.0),
        "expansion_pop_transfer": (0.1, 0.6),
        # Conflict
        "raid_range_base": (1.0, 5.0),
        "raid_range_longship": (2.0, 10.0),
        "raid_desperation_threshold": (0.05, 1.0),
        "raid_strength_factor": (0.1, 1.5),
        "raid_loot_fraction": (0.05, 0.7),
        "raid_damage_factor": (0.05, 0.6),
        "conquest_threshold": (0.05, 0.7),
        "raid_kill_threshold": (0.05, 0.5),
        # Trade
        "trade_range": (1.0, 8.0),
        "trade_food_gain": (0.02, 0.6),
        "trade_wealth_gain": (0.02, 0.5),
        "tech_diffusion_rate": (0.01, 0.4),
        # Winter
        "winter_severity_mean": (0.1, 2.0),
        "winter_severity_variance": (0.01, 0.8),
        "collapse_food_threshold": (0.05, 1.5),
        "collapse_probability": (0.05, 0.9),
        "survival_bonus": (0.0, 1.0),
        "dispersal_range": (1.0, 6.0),
        "dispersal_fraction": (0.1, 0.8),
        # Environment
        "forest_reclaim_probability": (0.01, 0.5),
        "ruin_rebuild_range": (1.0, 6.0),
        "ruin_rebuild_threshold": (1.0, 8.0),
        "ruin_rebuild_fraction": (0.05, 0.5),
        "ruin_to_plains_probability": (0.01, 0.3),
        # Additional collapse triggers
        "harsh_winter_collapse_factor": (1.0, 3.0),
        "raid_collapse_threshold": (0.1, 0.8),
    }

    @classmethod
    def param_names(cls) -> list[str]:
        """Return ordered list of parameter names."""
        return [f.name for f in fields(cls)]

    def to_array(self) -> np.ndarray:
        """Convert params to a flat numpy array."""
        return np.array([getattr(self, name) for name in self.param_names()])

    @classmethod
    def from_array(cls, arr: np.ndarray) -> SimParams:
        """Create SimParams from a flat numpy array, clamping to bounds."""
        names = cls.param_names()
        kwargs = {}
        for idx, name in enumerate(names):
            lo, hi = cls.BOUNDS[name]
            kwargs[name] = float(np.clip(arr[idx], lo, hi))
        return cls(**kwargs)

    @classmethod
    def default(cls) -> SimParams:
        """Return default parameters."""
        return cls()

    def to_dict(self) -> dict[str, float]:
        """Convert params to a named dictionary (insertion-safe serialization)."""
        return {name: getattr(self, name) for name in self.param_names()}

    @classmethod
    def from_dict(cls, named: dict[str, float]) -> SimParams:
        """Create SimParams from a named dictionary, clamping to bounds."""
        kwargs = {}
        for name in cls.param_names():
            if name in named:
                lo, hi = cls.BOUNDS[name]
                kwargs[name] = float(np.clip(named[name], lo, hi))
            # Missing params use dataclass defaults
        return cls(**kwargs)

    @classmethod
    def bounds_arrays(cls) -> tuple[np.ndarray, np.ndarray]:
        """Return (lower_bounds, upper_bounds) as numpy arrays."""
        names = cls.param_names()
        lower = np.array([cls.BOUNDS[n][0] for n in names])
        upper = np.array([cls.BOUNDS[n][1] for n in names])
        return lower, upper

    @classmethod
    def ndim(cls) -> int:
        """Number of parameters."""
        return len(fields(cls))

    @classmethod
    def random(cls, rng: np.random.Generator | None = None) -> SimParams:
        """Generate random params within bounds."""
        if rng is None:
            rng = np.random.default_rng()
        lower, upper = cls.bounds_arrays()
        arr = rng.uniform(lower, upper)
        return cls.from_array(arr)
