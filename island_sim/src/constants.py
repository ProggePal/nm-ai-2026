"""Terrain codes and class mappings for Astar Island."""

# Internal terrain codes (as returned by the API)
OCEAN = 10
PLAINS = 11
EMPTY = 0
SETTLEMENT = 1
PORT = 2
RUIN = 3
FOREST = 4
MOUNTAIN = 5

# Prediction class indices
NUM_CLASSES = 6
CLASS_EMPTY = 0
CLASS_SETTLEMENT = 1
CLASS_PORT = 2
CLASS_RUIN = 3
CLASS_FOREST = 4
CLASS_MOUNTAIN = 5

# Map internal terrain codes → prediction class index
TERRAIN_TO_CLASS: dict[int, int] = {
    OCEAN: CLASS_EMPTY,
    PLAINS: CLASS_EMPTY,
    EMPTY: CLASS_EMPTY,
    SETTLEMENT: CLASS_SETTLEMENT,
    PORT: CLASS_PORT,
    RUIN: CLASS_RUIN,
    FOREST: CLASS_FOREST,
    MOUNTAIN: CLASS_MOUNTAIN,
}

# Terrain codes that never change during simulation
STATIC_TERRAINS = {OCEAN, MOUNTAIN}

# All valid terrain codes
ALL_TERRAIN_CODES = {OCEAN, PLAINS, EMPTY, SETTLEMENT, PORT, RUIN, FOREST, MOUNTAIN}

# Terrains that count as "land" (settlements can be placed on these)
LAND_TERRAINS = {PLAINS, EMPTY, FOREST}

# Terrains that are buildable (new settlement can be founded here)
# Includes FOREST (settlements clear forest) and RUIN (rebuild on ruins)
BUILDABLE_TERRAINS = {PLAINS, EMPTY, FOREST, RUIN}
