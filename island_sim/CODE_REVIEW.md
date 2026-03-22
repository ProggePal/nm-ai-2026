# Code Review: Multi-round Warm-starting + Resubmission Loop

**Date:** 2026-03-22
**Scope:** `orchestrator.py` — warm-start chaining, `load_best_warm_start()`, `resubmit()`, argparse CLI.

---

## 1. Architecture

### 1.1 — Warm-start priority chain is well-designed (Good)

`load_best_warm_start()` (lines 214-239): Priority order `latest CMA-ES > grid search > None` makes sense. CMA-ES results are per-round refinements that should be closer to the current round's hidden params than the global grid search optimum. The fallback chain ensures we always have something to start from.

### 1.2 — Phase 2 warm-starting from Phase 1 result (Good fix)

Line 354: `warm_start=params` now passes the Phase 1 CMA-ES result as warm start for Phase 2, instead of the original grid search params. This means Phase 2 builds on Phase 1's work rather than starting over. Correct — Phase 1 already adapted the params to this round's observations.

### 1.3 — Resubmission loop is simple and correct (Good)

`resubmit()` (lines 399-488): Clean design — load observations, warm-start from latest, run CMA-ES with more budget, submit, repeat. The round-still-active check at line 443 prevents wasting compute after the window closes. No new queries spent.

---

## 2. Code Quality

### 2.1 — Issue: `latest_cmaes_params.json` persists across rounds

`save_params()` (line 211) always overwrites `latest_cmaes_params.json`. `load_best_warm_start()` loads it without checking which round it's from. If round N's CMA-ES result is saved, then round N+1 starts, it will warm-start from round N's params (line 233 prints the round_id but doesn't filter by it).

This is **probably fine** — round N's CMA-ES params are a better starting point than grid search defaults for round N+1 (they're at least adapted to recent observations). But it's worth being explicit: the docstring says "CMA-ES results are round-specific refinements" which implies they should only be used for the same round. In practice, using them cross-round is beneficial since hidden params don't change drastically between rounds.

**Recommendation:** Add a comment in `load_best_warm_start()` making the cross-round behavior explicit and intentional.

### 2.2 — Resubmission loop uses same `mc_base_seed` every attempt

`resubmit()` line 473: `mc_base_seed=seed_idx * 100000` is the same every attempt. Combined with the deterministic CMA-ES seed (42), if the warm-start params don't change much between attempts, the resubmitted predictions may be nearly identical. This limits the value of multiple attempts.

**Options:**
- **A) Leave as-is** — the warm-start DOES change each attempt (because `save_params` at line 462 updates `latest_cmaes_params.json`, and line 449 loads it). So each attempt starts from a different point and CMA-ES explores different directions. The MC seed being the same is fine — we want deterministic predictions given the params.
- **B) Vary mc_base_seed per attempt** — add `attempt * 1000` to the seed. Gives different MC noise, which slightly changes predictions.

**Recommendation: A (leave as-is).** The determinism is a feature — same params should produce same predictions. The variation comes from CMA-ES finding different params each attempt.

### 2.3 — `resubmit()` doesn't call `sync_history()` (Fine)

The resubmission loop skips history sync. This is correct — we don't want to spend time fetching completed rounds when we're racing to resubmit before the window closes.

### 2.4 — No sleep between resubmission attempts

`resubmit()` runs attempts back-to-back with no delay. Each attempt runs CMA-ES (800 evals × 100 MC runs = 80K sims, maybe ~30-60s with Rust) then generates predictions (5 seeds × 4000 MC runs = 20K sims, ~10s). So each attempt takes ~1-2 minutes. With 3 attempts, that's ~5 minutes total. Fine for a 2h45m window — no sleep needed.

### 2.5 — Good: argparse CLI (Good)

Lines 491-506: Clean argument parsing with `--resubmit`, `--max-attempts`, `--eval-budget`. Defaults are sensible (3 attempts, 800 evals). The `main()` path is unchanged for backwards compatibility.

---

## 3. Tests

### 3.1 — No tests for the new features

No tests for `load_best_warm_start()`, `save_params` writing `latest_cmaes_params.json`, or `resubmit()`. These are orchestration functions that depend on API calls, so unit testing is awkward. The critical logic (CMA-ES warm-starting, param loading) is already tested elsewhere.

Not a blocker — these are integration-level concerns better tested via a dry-run.

---

## 4. Performance

### 4.1 — Resubmission loop compute budget is reasonable

800 evals × 100 MC runs per eval = 80K simulations per attempt. With Rust on an M-series Mac, each sim takes ~0.1-0.2ms, so ~8-16 seconds per attempt for CMA-ES, plus ~10s for 20K prediction sims. 3 attempts ≈ 1-2 minutes. Well within the competition time window.

---

## Summary

| # | Finding | Severity | Action |
|---|---------|----------|--------|
| 2.1 | `latest_cmaes_params.json` persists across rounds — intentional? | Info | Add comment documenting cross-round warm-start is intentional |

**Overall: Clean, focused implementation.** Both features are well-scoped — warm-start chaining addresses the CMA-ES starting point issue, and the resubmission loop lets you extract more value from the time window without spending additional queries. No bugs found.
