# Code Review Report: Island Simulator Codebase

**Date:** 2026-03-21
**Scope:** Full codebase — Python simulator, Rust backend, orchestration, inference, prediction, scripts, configuration
**Review type:** Architecture, Code Quality, Tests, Performance

## Context

Full code review of the NM i AI 2026 island simulation pipeline for the Astar Island competition. The system predicts probability distributions of terrain types on a 40×40 Norse civilization grid after 50 simulated years, using 50 viewport queries per round across 5 seeds.

**Pipeline:** API observation → CMA-ES parameter inference → Monte Carlo prediction (Rust-accelerated) → hybrid blending → submission.

**Simulator implementations:**
- Python (`src/simulator/`) — fallback, used by CMA-ES inference
- Rust (`rust/src/`) — production MC predictions + grid search
- Rust on GCP (`~/island_sim/rust/`) — high-core remote grid search

---

## Section 1: Architecture

### A1. Triple-Implementation Sync Risk — HIGH RISK

**Problem:** The simulator exists in three places that must produce identical results. DEVELOPMENT.md documents a 6-step manual checklist for adding a parameter, but there is no automated verification. A missed step in `rust/src/params.rs` `from_array()` index mapping causes **silent misalignment** — the grid search optimizes for different behavior than what the orchestrator submits.

**Current state:** All 36 params verified to match across `params.py` (line 12-59), `params.rs`, and `grid_search.rs`. Bounds match. Defaults match. Phase logic is functionally identical (verified by comparing all 5 phases line-by-line). The only minor inconsistency is the trade phase RNG signature — Python's `run_trade()` accepts an `rng` parameter that is unused; Rust omits it entirely.

**Recommendation:** Add a pytest cross-validation test (1-year, 50-year, and MC prediction levels) that runs both Python and Rust simulators with identical params/seed and asserts convergent output.

**Files involved:**
- New: `tests/test_sync.py`
- References: `src/simulator/engine.py`, `src/simulator/params.py`, `rust/src/lib.rs` (via `island_sim_core`)
- Template: DEVELOPMENT.md lines 88-116

---

### A2. Parameter Management Fragility — HIGH RISK

**Problem:** 36 parameters are identified by **array index position**, not by name. `data/rounds/best_params.json` stores a bare `"params": [0.18, 0.12, ...]` array. A parameter insertion shifts every subsequent parameter silently.

`SimParams.from_array()` (`src/simulator/params.py:116-123`) deserializes by index:
```python
for idx, name in enumerate(names):
    lo, hi = cls.BOUNDS[name]
    kwargs[name] = float(np.clip(arr[idx], lo, hi))
```

**Recommendation:** Store named params in `best_params.json` as `{"base_food_production": 0.18, ...}` with name-based deserialization. The `to_array()`/`from_array()` boundary stays the same (Rust needs arrays), but the serialized JSON format becomes insertion-safe and self-documenting.

**Files involved:**
- `data/rounds/best_params.json` — restructure from array to named dict
- `src/simulator/params.py` — add `to_dict()`/`from_dict()` methods
- `scripts/grid_search_rust.py` — update save logic
- `scripts/smart_search.py` — update save/load logic
- `src/orchestrator.py` — update load logic

---

### A3. No Automated Validation Pipeline — MEDIUM RISK

**Problem:** DEVELOPMENT.md includes a manual verification script (lines 88-116) that checks imports, Rust build, cross-implementation compatibility, and param count matching. But it's a copy-paste shell snippet, not an automated test.

**Recommendation:** Convert to pytest suite in `tests/`. Combine with the cross-validation test from A1.

**Files involved:**
- New: `tests/test_sync.py` (combined with A1)
- New: `tests/conftest.py` for shared fixtures

---

## Section 2: Code Quality

### Q1. DRY Violation: Observation Smoothing in Two Places — LOW

**Problem:** Observation-to-probability transformation exists in two places with different smoothing:
- `src/prediction/observation_model.py:51-75` — Laplace smoothing with explicit `alpha=0.5` (Jeffreys prior)
- `src/inference/loss.py:47-48` — uses Rust `generate_prediction` output with a bare `1e-10` floor

These serve legitimately different purposes: the observation model aggregates raw viewport data, while the loss function evaluates MC predictions against observations.

**Recommendation:** Document the intentional difference with comments in both files explaining why smoothing differs.

---

### Q2. Trade Phase Pair Tracking Uses `id()` — LOW, EASY FIX

**Problem:** `src/simulator/phases/trade.py:33-36` uses Python `id()` for pair deduplication:
```python
pair_key = (
    min(id(port_a), id(port_b)),
    max(id(port_a), id(port_b)),
)
```

`id()` returns memory addresses, which break if settlements are ever deep-copied. The Rust implementation uses index-based tracking instead.

**Recommendation:** Use position as pair key:
```python
pair_key = (
    min((port_a.x, port_a.y), (port_b.x, port_b.y)),
    max((port_a.x, port_a.y), (port_b.x, port_b.y)),
)
```

Note: Given the current iteration pattern (nested loop over `idx_a < idx_b`), pairs are already unique and the `traded` set is technically redundant. But keeping it is harmless and defensive.

---

### Q3. smart_search.py Issues — NOTED ONLY

- `evaluate_params()` (line 67-72) is **dead code** — never called
- The actual `objective()` closure (line 118-169) has **in-loop imports** (lines 141-142, 147) and a **dead expression** `round_scores` (line 152)
- These are cosmetic, not functional

---

### Q4. calibrate.py Dual Prediction Computation — NOTED ONLY

`scripts/calibrate.py` generates predictions twice:
- Lines 86-89: Scoring pass with `num_runs=num_runs, base_seed=seed_idx * 100000`
- Lines 203-206: Bias analysis with `num_runs=20, base_seed=seed_idx * 100000 + 999`

The different seed is likely intentional (independent samples for unbiased bias estimation). The hardcoded `num_runs=20` may be a bug — doesn't scale with `--mc-runs` argument.

---

## Section 3: Tests

The project currently has **zero tests**. `pyproject.toml` lists `pytest` as a dev dependency, but there's no `tests/` directory.

### T1. Python↔Rust Cross-Validation Test — HIGH PRIORITY

**What to test (3 levels):**

1. **1-year simulation:** Run both Python and Rust engines for 1 year with identical params, seed, and initial state. Compare deterministic aspects (grid topology, static cells).
2. **50-year simulation:** Full simulation run. Compare MC prediction distributions (statistical convergence) — with enough runs (e.g., 500), distributions should be close (within ~0.05 per cell).
3. **MC prediction tensor:** Run `generate_prediction` in both Python and Rust with same params and N runs. Assert probability tensors are within tolerance.

**Key challenge:** Python uses `np.random.default_rng(seed)`, Rust uses `ChaCha8Rng::seed_from_u64(seed)` — different random sequences. Exact grid match is NOT expected. Instead, compare converged probability distributions.

**File:** `tests/test_sync.py`

---

### T2. Golden-File Snapshot Tests — MEDIUM PRIORITY

Run full simulation with fixed seed and params, save output, assert future runs produce identical results. Separate snapshots for Python and Rust (each must be self-consistent across runs).

**What to snapshot:**
- Final grid state for seed 0, round 001, default params
- Settlement positions and alive/dead status
- MC prediction tensor (2 runs, low for speed)

**File:** `tests/test_regression.py`

---

### T3. Scoring/Postprocessing Tests — HIGH PRIORITY

**Test cases for `src/scoring.py`:**
- Perfect prediction (identical distributions) → score = 100
- Uniform prediction → known score
- Zero probability where GT has mass → high KL
- Entropy weighting: static cells excluded, dynamic cells weighted proportionally

**Test cases for `src/prediction/postprocess.py`:**
- Ocean cells → `[1.0, 0, 0, 0, 0, 0]` (class 0 only)
- Mountain cells → `[0, 0, 0, 0, 0, 1.0]` (class 5 only)
- Non-coastal cells → port probability (class 2) forced to 0
- Coastal cells → port probability preserved (with floor)
- Non-mountain cells → mountain probability (class 5) forced to 0
- Probability floor → every non-zero class ≥ 0.01
- Normalization → all rows sum to 1.0

**Files:** `tests/test_scoring.py`, `tests/test_postprocess.py`

---

### T4. Parameter Round-Trip Test — LOW PRIORITY, QUICK

**Test cases:**
- `SimParams.from_array(SimParams.default().to_array())` equals `SimParams.default()`
- Random params round-trip preserves all values
- `len(SimParams.param_names())` == `SimParams.ndim()` == number of BOUNDS entries
- Named JSON round-trip: `from_dict(params.to_dict())` preserves all values
- Rust param count matches Python

**File:** `tests/test_params.py`

---

## Section 4: Performance

All 4 performance items evaluated as **negligible for the current 40×40 map size**. No action needed.

### P1. postprocess.py Coastal Mask — O(H×W) Python Loop

`src/prediction/postprocess.py:36-46` — 4-level nested loop, 14,400 inner iterations per call on 40×40. ~1ms. Called 5 times per round.

### P2. observer.py Greedy Viewport Search — O(H×W) per Query

`src/inference/observer.py:53-66` — ~81 positions searched per query, ~6 queries per seed. ~109K total operations. Negligible.

### P3. observation_model.py Python Loop Aggregation — O(N_obs × V_area)

`src/prediction/observation_model.py:37-46` — 6 observations × 225 cells = 1,350 iterations. Negligible.

### P4. Rust WorldState Cloning in MC — 2000 × ~6.4KB

`rust/src/engine.rs` `generate_prediction()` clones WorldState per MC run. ~12.8MB total memcpy. Trivial vs. simulation cost.

---

## Summary

| # | Issue | Recommendation | Effort | Priority |
|---|-------|----------------|--------|----------|
| **Architecture** | | | | |
| A1 | Triple-impl sync risk | Cross-validation pytest | ~1 hr | HIGH |
| A2 | Parameter fragility | Named params in JSON | ~1 hr | HIGH |
| A3 | No validation pipeline | Pytest suite (combined with A1) | ~1 hr | HIGH |
| **Code Quality** | | | | |
| Q1 | Smoothing DRY violation | Document with comments | ~10 min | LOW |
| Q2 | Trade pair `id()` | Use position as key | ~5 min | LOW |
| Q3 | smart_search.py | Note only (dead code, cosmetic) | — | — |
| Q4 | calibrate.py duplicate | Note only (hardcoded num_runs) | — | — |
| **Tests** | | | | |
| T1 | Python↔Rust cross-validation | 3-level test | ~1 hr | HIGH |
| T2 | Golden-file snapshots | Regression tests | ~30 min | MEDIUM |
| T3 | Scoring/postprocess tests | Known-value assertions | ~1 hr | HIGH |
| T4 | Param round-trip test | Identity + JSON round-trip | ~20 min | LOW |
| **Performance** | | | | |
| P1-P4 | All items | No action needed | — | — |

## Recommended Implementation Order

1. **Q2** — Fix trade pair key (5 min, trivial, no dependencies)
2. **A2** — Named params in `best_params.json` + `to_dict()`/`from_dict()` (needed before tests)
3. **T4** — Param round-trip test (quick, validates A2)
4. **T3** — Scoring/postprocess tests (high ROI, validates submission pipeline)
5. **T1 + A1 + A3** — Cross-validation pytest (the main test, combines architecture issues)
6. **T2** — Golden-file snapshots (regression safety net)
7. **Q1** — Document smoothing difference (final cleanup)

## Verification

After all changes:

```bash
# Run full test suite
pytest tests/ -v

# Verify Rust still builds
cd rust && cargo check && cd ..

# Verify orchestrator still works end-to-end (dry run — DO NOT call /simulate)
python -c "
from src.simulator.params import SimParams
import json
d = json.loads(open('data/rounds/best_params.json').read())
p = SimParams.from_dict(d['params'])  # New named format
print(f'Loaded {SimParams.ndim()} params: {p.base_food_production=}')
assert p.to_array().shape == (SimParams.ndim(),)
print('OK')
"
```
