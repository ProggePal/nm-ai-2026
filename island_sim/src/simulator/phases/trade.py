"""Trade phase: port trade and tech diffusion."""

from __future__ import annotations

import numpy as np

from ...types import WorldState
from ..params import SimParams
from ..world import distance


def run_trade(state: WorldState, params: SimParams, rng: np.random.Generator) -> None:
    """Execute the trade phase between ports."""
    # Find all alive ports
    ports = [s for s in state.settlements if s.alive and s.has_port]

    # Track which pairs already traded this turn
    traded: set[tuple[int, int]] = set()

    for idx_a in range(len(ports)):
        for idx_b in range(idx_a + 1, len(ports)):
            port_a = ports[idx_a]
            port_b = ports[idx_b]

            # Skip if same faction already has internal trade handled by growth
            # But different factions CAN trade — that's the point
            # Check they're not at war (different factions can still trade if not recently raided)
            # Simplified: ports of different factions trade if in range
            dist = distance(port_a.x, port_a.y, port_b.x, port_b.y)
            if dist > params.trade_range:
                continue

            pair_key = (
                min(id(port_a), id(port_b)),
                max(id(port_a), id(port_b)),
            )
            if pair_key in traded:
                continue
            traded.add(pair_key)

            _execute_trade(port_a, port_b, params)
            _diffuse_tech(port_a, port_b, params)


def _execute_trade(port_a, port_b, params: SimParams) -> None:
    """Exchange food and wealth between two ports."""
    # Both parties benefit from trade
    port_a.food += params.trade_food_gain
    port_b.food += params.trade_food_gain
    port_a.wealth += params.trade_wealth_gain
    port_b.wealth += params.trade_wealth_gain


def _diffuse_tech(port_a, port_b, params: SimParams) -> None:
    """Technology diffuses from higher-tech to lower-tech partner."""
    if port_a.tech_level > port_b.tech_level:
        diff = port_a.tech_level - port_b.tech_level
        port_b.tech_level += diff * params.tech_diffusion_rate
    elif port_b.tech_level > port_a.tech_level:
        diff = port_b.tech_level - port_a.tech_level
        port_a.tech_level += diff * params.tech_diffusion_rate
