import { describe, it, expect } from "vitest";
import { posKey, parsePos, manhattanDistance, posEquals, positionsInclude } from "../utils.ts";
import type { Position } from "../types.ts";

describe("posKey", () => {
  it("converts a position to a string key", () => {
    expect(posKey([3, 7])).toBe("3,7");
  });

  it("handles zero coordinates", () => {
    expect(posKey([0, 0])).toBe("0,0");
  });
});

describe("parsePos", () => {
  it("converts a string key back to a position", () => {
    expect(parsePos("3,7")).toEqual([3, 7]);
  });

  it("roundtrips with posKey", () => {
    const pos: Position = [12, 5];
    expect(parsePos(posKey(pos))).toEqual(pos);
  });
});

describe("manhattanDistance", () => {
  it("returns 0 for same position", () => {
    expect(manhattanDistance([2, 3], [2, 3])).toBe(0);
  });

  it("returns correct distance for adjacent cells", () => {
    expect(manhattanDistance([0, 0], [1, 0])).toBe(1);
    expect(manhattanDistance([0, 0], [0, 1])).toBe(1);
  });

  it("returns correct distance for diagonal cells", () => {
    expect(manhattanDistance([0, 0], [3, 4])).toBe(7);
  });
});

describe("posEquals", () => {
  it("returns true for equal positions", () => {
    expect(posEquals([1, 2], [1, 2])).toBe(true);
  });

  it("returns false for different positions", () => {
    expect(posEquals([1, 2], [1, 3])).toBe(false);
    expect(posEquals([1, 2], [2, 2])).toBe(false);
  });
});

describe("positionsInclude", () => {
  it("returns true when target is in the list", () => {
    const positions: Position[] = [[1, 2], [3, 4], [5, 6]];
    expect(positionsInclude(positions, [3, 4])).toBe(true);
  });

  it("returns false when target is not in the list", () => {
    const positions: Position[] = [[1, 2], [3, 4]];
    expect(positionsInclude(positions, [5, 6])).toBe(false);
  });

  it("returns false for empty list", () => {
    expect(positionsInclude([], [1, 2])).toBe(false);
  });
});
