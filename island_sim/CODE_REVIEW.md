# Code Review: Settlement Stat Matching + Supporting Changes

**Date:** 2026-03-22
**Scope:** Settlement stat matching in CMA-ES loss, postprocess no-zero-probability fix, multi-expansion growth, orchestrator cleanup, observer viewport clipping, bounds widening, new postprocess tests.

---

## 1. Architecture

### 1.1 — Stat matching design is clean (Good)

The combined objective `avg_nll + stat_weight * avg_stat_loss` is the right approach. NLL captures terrain distribution fidelity, stat MSE captures settlement dynamics that NLL misses (population levels, wealth, faction diversity). These are complementary signals — a set of params can produce the right terrain layout but wrong settlement stats, and vice versa.

The `ViewportSettlementStats` struct with 6 dimensions (num_settlements, num_ports, mean_population, mean_food, mean_wealth, num_owners) covers the key observable fingerprints of hidden params.

### 1.2 — `generate_prediction_with_stats()` is efficient (Good)

`engine.rs:69-170`: Stats are collected in the same MC loop as terrain counts, avoiding a second pass. Per-viewport settlement filtering uses simple bounds checking. The `HashSet<i32>` for owner counting per run is collected into a `Vec<HashSet>` per viewport, then averaged — slightly unusual but correct. Memory: `num_viewports × num_runs` HashSets, each small. Fine for 50 viewports × 100 runs.

### 1.3 — Backward compatibility handled correctly (Good)

`lib.rs:279-280`: `parse_settlement_stats()` returns `None` when the "settlements" key is missing. `cmaes_inference.rs:122-125`: stat MSE is only computed when `observed_stats` is `Some`. `stat_weight=0.0` short-circuits via the multiplication. Three tests verify this.

### 1.4 — `seed_index` preserved on SeedState (Good fix)

`cmaes_inference.rs:42`: `pub seed_index: usize` now carries the original Python seed index through to `base_seed = MC_BASE_SEED + seed_state.seed_index * 10000`. Previously this used the enumeration index, which was wrong when some seeds had no observations (the filtered `seed_states` vec would have different indices than the original seed numbers). This is a correctness fix.

### 1.5 — Query concentration reverted in orchestrator

`orchestrator.py:265-266`: Phase 1 now uses `plan_viewports(initial_states, budget=systematic_budget)` without `focus_seeds`. Phase 2 also doesn't pass `focus_seeds`. The concentrated query strategy we implemented earlier has been removed. The `select_focus_seeds` import is also gone from line 24.

This may be intentional (perhaps spreading evenly works better with the stat matching loss since it gives stats for all seeds). Worth confirming this was a deliberate choice rather than an accidental revert.

---

## 2. Code Quality

### 2.1 — Normalization constants in `compute_stat_mse` are hardcoded magic numbers

`cmaes_inference.rs:51-58`:
```rust
(obs.num_settlements - sim.num_settlements) / 30.0,
(obs.num_ports - sim.num_ports) / 5.0,
(obs.mean_population - sim.mean_population) / 2.0,
(obs.mean_food - sim.mean_food) / 0.5,
(obs.mean_wealth - sim.mean_wealth) / 0.05,
(obs.num_owners - sim.num_owners) / 5.0,
```

These normalize each dimension to roughly unit scale. The values (30, 5, 2, 0.5, 0.05, 5) appear reasonable for a 15×15 viewport on a 40×40 map, but they're assumptions about typical ranges. If the optimizer finds params that produce very different ranges (e.g., R18 with 656 settlements), these normalizations may over/under-weight certain dimensions.

Not a bug — but worth documenting what these constants represent (e.g., "30.0 = approximate max settlements in a 15×15 viewport") so they can be revisited if stat MSE doesn't converge well.

### 2.2 — `mean_wealth` normalization constant (0.05) may be too small

With the new `wealth_production_rate` param (grid search found 0.17), settlements accumulate wealth of ~0.75 by year 10. By year 50, wealthy settlements could have wealth > 1.0. The normalization constant of 0.05 means a wealth difference of 0.05 produces a normalized error of 1.0. This makes wealth the most sensitive dimension — a small absolute error dominates the MSE. Consider bumping to 0.2-0.5 to match the actual wealth scale with the new mechanics.

### 2.3 — Python fallback doesn't support stat matching

`optimizer.py:50-53`: `_infer_python` doesn't accept `stat_weight` — the parameter is silently dropped:
```python
return _infer_python(
    observations, initial_states, num_runs_per_eval,
    max_evaluations, population_size, sigma0, warm_start,
)
```

This is documented by omission (Python path is terrain-only), but could surprise someone debugging with `HAS_RUST=False`. A comment would help: `# Note: Python fallback uses terrain-only NLL (no stat matching)`.

### 2.4 — Multi-expansion while loop in growth is good (Good)

`growth.py:122`, `growth.rs:94`: `while settlement.population >= params.expansion_threshold` allows a single large settlement to spawn multiple children per turn. Each expansion costs population, so the loop naturally terminates. This better models the docs' "prosperous settlements expand" — a settlement with 4x the threshold can found 2-3 children, matching explosive growth periods.

`MAX_ALIVE_SETTLEMENTS` bumped to 800 (from 150) in both Python and Rust. This accommodates the multi-expansion behavior and the fact that grid search found `expansion_threshold = 2.1` (very aggressive expansion).

### 2.5 — Orchestrator cleanup (Good)

Constants extracted to named variables (`PHASE1_BUDGET_FRACTION`, `MC_RUNS_PER_EVAL_PHASE1`, etc.). `_query_with_retry` adds resilience. Both good improvements.

### 2.6 — Postprocess no-zero fix (Good)

`postprocess.py:102-110` and `postprocess.rs:107-117`: `_make_exact_static` now uses `PROB_FLOOR_REMOTE` for non-dominant classes instead of 0.0, then renormalizes. This prevents the infinite KL divergence edge case where the ground truth has any nonzero probability for a class we predicted as exactly 0. The scoring docs explicitly warn about this. Clean fix.

### 2.7 — Observer viewport clipping (Good)

`observer.py:111-112,220-221`: Viewport width/height now clipped to map bounds before being sent to the API. Prevents partial observations at map edges. This was a latent bug — if a viewport was placed at x=30 on a 40-wide map, the API would clamp to w=10 but our tracking assumed w=15.

### 2.8 — Bounds widening

`params.py:78-85`: Several bounds widened:
- `pop_growth_rate`: (0.01, 0.3) → (0.01, 0.4)
- `carrying_capacity_per_food`: (1.0, 10.0) → (1.0, 12.0)
- `expansion_threshold`: (2.0, 12.0) → (1.0, 12.0)
- `expansion_range`: (1.0, 6.0) → (1.0, 7.0)
- `expansion_pop_transfer`: (0.1, 0.6) → (0.05, 0.6)

These are all in the growth section. The lower bound for `expansion_threshold` dropped from 2.0 to 1.0, and `expansion_pop_transfer` from 0.1 to 0.05. Both allow more aggressive expansion, which matches the grid search finding (expansion_threshold = 2.1, near the old lower bound).

**Important:** These bounds must also be updated in `rust/src/params.rs` `bounds_lower()` and `bounds_upper()`. Verify they match — bounds desync is the #1 bug source per DEVELOPMENT.md. The `TestBoundsSync.test_bounds_match` test should catch this.

---

## 3. Tests

### 3.1 — Stat matching tests cover the critical paths (Good)

Three new tests:
- `test_no_settlements_stat_loss_zero` — backward compat (no settlements key → stat loss = 0)
- `test_stat_weight_zero_matches_terrain_only` — stat_weight=0 disables stat matching
- `test_cmaes_with_stats_returns_valid_params` — end-to-end with settlements and stat_weight=0.5

These cover the three main contract guarantees: backward compat, disabling, and basic functionality.

### 3.2 — Postprocess tests are high-value (Good)

`TestPostprocess` with 4 tests:
- `test_no_zero_probabilities` — the core invariant, any zero = potential infinite KL
- `test_ocean_cell_no_zeros` — static cell still has floor
- `test_mountain_cell_no_zeros` — same for mountains
- `test_rust_python_postprocess_sync` — cross-implementation check

These directly validate the scoring-critical guarantee. Good additions.

### 3.3 — NLL cross-validation tolerance tightened (Good)

`test_nll_matches_python:303`: Tolerance tightened from 50% to 15%. With both paths using the same Rust MC engine and tiny sigma, this is reasonable. The tighter bound gives more confidence in loss equivalence.

### 3.4 — Missing: stat MSE cross-validation test

No test verifies that the Rust stat MSE matches what a Python implementation would produce. The normalization constants and mean computation could diverge between the observation parsing (lib.rs) and the MC accumulation (engine.rs). A test that constructs known stats and asserts the MSE value would catch this.

---

## 4. Performance

### 4.1 — `generate_prediction_with_stats` overhead is minimal

The per-viewport settlement filtering is O(num_settlements) per viewport per run. With ~50 settlements, 50 viewports, 100 runs: 250K settlement checks. Each is a bounds comparison — fast. The `HashSet` allocation per viewport per run is the main overhead, but HashSets with <10 entries are trivially small.

### 4.2 — `MAX_ALIVE_SETTLEMENTS` at 800 increases simulation cost

With multi-expansion and threshold=2.1, settlement counts can reach 600+ (per R18 observation). The conflict phase is O(n²) for target finding — 800 settlements means ~640K spatial checks per conflict phase per year, per MC run. With 100 MC runs × 50 years: ~3.2 billion checks. This is manageable with Rust but worth monitoring. The previous cap of 150 was a safety valve that's now relaxed.

---

## Summary

| # | Finding | Severity | Action |
|---|---------|----------|--------|
| 2.1 | Stat MSE normalization constants are hardcoded magic numbers | Low | Add comments documenting what they represent |
| 2.2 | `mean_wealth` normalization (0.05) may be too small for new wealth mechanics | Low | Consider bumping to 0.2-0.5 |
| 2.3 | Python fallback silently drops `stat_weight` | Low | Add comment |
| 2.8 | Bounds widened in Python — verify Rust bounds match | Medium | Run `test_bounds_match` to confirm |
| 1.5 | Query concentration removed from orchestrator | Info | Confirm intentional |
| 3.4 | No stat MSE cross-validation test | Low | Optional |

**Overall: Solid implementation.** The stat matching addition is well-designed — correct backward compat, clean separation of terrain NLL and stat MSE, efficient MC stat collection. The postprocess no-zero fix addresses a real scoring edge case. The multi-expansion and bounds widening align with empirical findings. Main thing to verify: Rust bounds match the Python changes.
