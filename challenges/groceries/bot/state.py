"""
GameState parser and derived properties.
All bot logic reads from a GameState instance — never from raw JSON directly.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import json


@dataclass
class Bot:
    id: int
    position: tuple[int, int]
    inventory: list[str]


@dataclass
class Item:
    id: str
    type: str
    position: tuple[int, int]


@dataclass
class Order:
    id: str
    items_required: list[str]
    items_delivered: list[str]
    complete: bool
    status: str  # "active" | "preview"

    @property
    def still_needed(self) -> list[str]:
        """Items required but not yet delivered (does NOT account for in-transit)."""
        delivered = list(self.items_delivered)
        needed = []
        for item in self.items_required:
            if item in delivered:
                delivered.remove(item)
            else:
                needed.append(item)
        return needed


@dataclass
class Grid:
    width: int
    height: int
    walls: frozenset[tuple[int, int]]


@dataclass
class GameState:
    round: int
    max_rounds: int
    grid: Grid
    bots: list[Bot]
    items: list[Item]
    orders: list[Order]
    drop_off_zones: list[tuple[int, int]]
    score: int

    # ── Derived properties ──────────────────────────────────────────────────

    @property
    def active_order(self) -> Optional[Order]:
        for o in self.orders:
            if o.status == "active":
                return o
        return None

    @property
    def preview_order(self) -> Optional[Order]:
        for o in self.orders:
            if o.status == "preview":
                return o
        return None

    @property
    def all_bot_inventory(self) -> list[str]:
        """Flat list of all item types currently held by all bots."""
        return [item for bot in self.bots for item in bot.inventory]

    def truly_needed(self, order: Optional[Order] = None) -> list[str]:
        """
        Items still needed for the order that are NOT already in any bot's inventory.
        This is the true remaining work to be done.
        """
        if order is None:
            order = self.active_order
        if order is None:
            return []

        needed = order.still_needed
        in_transit = list(self.all_bot_inventory)

        result = []
        for item_type in needed:
            if item_type in in_transit:
                in_transit.remove(item_type)  # claim this one as covered
            else:
                result.append(item_type)
        return result

    def available_items_for(self, item_types: list[str]) -> list[Item]:
        """Items on the map that match any of the given item types."""
        return [i for i in self.items if i.type in item_types]

    def bot_by_id(self, bot_id: int) -> Optional[Bot]:
        for b in self.bots:
            if b.id == bot_id:
                return b
        return None

    @property
    def end_game(self) -> bool:
        """True when < 10% of rounds remain — triggers deliver-immediately mode."""
        return self.round > self.max_rounds * 0.9

    # ── Parser ──────────────────────────────────────────────────────────────

    @classmethod
    def from_json(cls, data: dict) -> "GameState":
        grid = Grid(
            width=data["grid"]["width"],
            height=data["grid"]["height"],
            walls=frozenset(tuple(w) for w in data["grid"]["walls"]),
        )

        bots = [
            Bot(
                id=b["id"],
                position=tuple(b["position"]),
                inventory=list(b["inventory"]),
            )
            for b in data["bots"]
        ]

        items = [
            Item(id=i["id"], type=i["type"], position=tuple(i["position"]))
            for i in data["items"]
        ]

        orders = [
            Order(
                id=o["id"],
                items_required=list(o["items_required"]),
                items_delivered=list(o["items_delivered"]),
                complete=o["complete"],
                status=o["status"],
            )
            for o in data["orders"]
        ]

        # Handle both "drop_off_zones" (array) and legacy "drop_off" (single)
        if "drop_off_zones" in data:
            drop_offs = [tuple(z) for z in data["drop_off_zones"]]
        elif "drop_off" in data:
            drop_offs = [tuple(data["drop_off"])]
        else:
            drop_offs = []

        return cls(
            round=data["round"],
            max_rounds=data["max_rounds"],
            grid=grid,
            bots=bots,
            items=items,
            orders=orders,
            drop_off_zones=drop_offs,
            score=data.get("score", 0),
        )
