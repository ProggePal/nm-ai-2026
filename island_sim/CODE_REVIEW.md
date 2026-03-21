# Code Review: Rust CMA-ES Inference + Recent Changes

**Date:** 2026-03-21
**Scope:** New `cmaes_inference.rs`, refactored `optimizer.py`, PyO3 bindings, `params.rs` dedup, `grid_search.rs` cleanup, tests.

---

## 1. Architecture

### 1.1 — NLL loss path correctly differs from grid search (Good)

`cmaes_inference.rs:1-7` clearly documents: CMA-ES uses `generate_prediction → Laplace smoothing → NLL` while grid search uses `generate_prediction → postprocess → KL divergence`. This matches `loss.py` exactly. Good documentation of a non-obvious design choice.

### 1.2 — Sequential seed evaluation within population members (Info)

`cmaes_inference.rs:138`: Objective iterates seeds sequentially. The `cmaes` crate's `run_parallel()` parallelizes across population members. With default popsize ~15 for 40 dims, this should keep cores busy. No action needed. Worth revisiting if profiling shows idle cores on the 384-vCPU GCP VMs (popsize << cores).

### 1.3 — Python fallback preserved cleanly (Good)

`optimizer.py:43-51`: Clean routing — Rust when available, Python `_infer_python` fallback. Both paths share the same interface and warm-start behavior.

---

## 2. Code Quality

### 2.1 — `num_runs_per_eval` reverted to 50 in orchestrator

**`orchestrator.py:271,313`**: Both `infer_params` calls use `num_runs_per_eval=50`. We previously agreed to bump to 100 for more accurate loss evaluation. The Rust backend makes this affordable — 100 MC runs per eval adds negligible wall-clock time.

**Recommendation:** Bump back to 100 in both calls.

### 2.2 — `params.rs` dedup eliminates maintenance hazard (Good)

`grid_search.rs:10` now imports `default_params`, `bounds_lower`, `bounds_upper` from `crate::params`. Previously these were duplicated — the #1 bug source per DEVELOPMENT.md. Clean fix.

### 2.3 — Terrain-to-class conversion at PyO3 boundary (Good)

`lib.rs:267-273`: Terrain codes converted to class indices during parsing, before passing to Rust CMA-ES. The inference code only sees class indices — clean separation.

### 2.4 — Fixed CMA-ES seed = 42 (Fine)

`optimizer.py:95`, `cmaes_inference.rs:32`: Both hardcode seed 42. Deterministic MC + CMA-ES means identical runs with same observations. Good for reproducibility. If you ever want exploration diversity across re-runs, expose as parameter.

### 2.5 — Soft boundary handling via clamping (Correct)

`cmaes_inference.rs:129`: Out-of-bounds samples clamped to [0,1] inside the objective. The `cmaes` crate doesn't natively support bounds, so this is the right approach. Clamping creates a "sticky" boundary (many points map to the same corner), which CMA-ES handles well in practice.

---

## 3. Tests

### 3.1 — Bounds sync tests are high-value (Good)

`TestBoundsSync` (3 tests): Verifies Rust/Python bounds, defaults, and NDIM match exactly. This is the single most error-prone area. `get_param_bounds()` and `get_default_params()` are clean test-only PyO3 exports.

### 3.2 — NLL cross-validation is clever (Good)

`test_nll_matches_python:251-301`: Uses tiny sigma (1e-10) to force CMA-ES to evaluate near the warm start, then compares against Python NLL. 50% tolerance accounts for different RNGs. The approach is sound and the tolerance is documented.

### 3.3 — Missing: Python fallback warm-start test (Low risk)

No test verifies `_infer_python` uses `warm_start` as x0 instead of defaults. Low risk since Rust path is primary, but the Python fallback had a bug here before (`best_params = default_params`). A one-liner assertion in a test would prevent regression.

### 3.4 — Missing: Routing integration test (Low risk)

No test verifies `infer_params()` routes to Rust vs Python correctly. The routing is 3 lines, so low risk.

---

## 4. Performance

### 4.1 — Big win: Full Rust CMA-ES loop

Previously each CMA-ES evaluation crossed Python→Rust for MC prediction, then computed NLL in Python. Now the entire loop runs in Rust. `py.allow_threads()` at `lib.rs:308` correctly releases the GIL. Expected speedup: 10-20x.

### 4.2 — `run_parallel()` correct for this workload

`cmaes_inference.rs:165`: Population-level parallelism via Rayon. Correct choice — nested parallelism (seeds within members) would cause thread pool contention.

### 4.3 — MC base seed matches Python (Correct)

`cmaes_inference.rs:139`: `base_seed = 42 + seed_idx * 10000` matches `loss.py:44`. Ensures the Rust and Python NLL functions evaluate equivalent stochastic landscapes. Important for warm-start continuity.

---

## Summary

| # | Finding | Severity | Action |
|---|---------|----------|--------|
| 2.1 | `num_runs_per_eval` reverted to 50 in orchestrator | Low | Bump to 100 |
| 3.3 | No Python fallback warm-start test | Low | Optional |
| 3.4 | No routing integration test | Low | Optional |

**Overall: Clean implementation.** The Rust CMA-ES module correctly mirrors the Python NLL loss, PyO3 bindings handle marshalling well, params dedup eliminates a maintenance hazard, and tests cover the critical sync points. One actionable item: bump `num_runs_per_eval` back to 100.
