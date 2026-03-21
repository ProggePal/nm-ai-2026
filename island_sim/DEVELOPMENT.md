# Development Guide — Keeping Things in Sync

The simulator exists in **three places** that must stay synchronized:

1. **Python** (`src/simulator/`) — used by CMA-ES inference, fallback if Rust not built
2. **Rust library** (`rust/src/`) — used by orchestrator for MC predictions, and locally for grid search
3. **Rust on GCP** (`~/island_sim/rust/`) — used by the GCP grid search

## What needs updating where

### Simulator mechanics (phases)

If you change how growth, conflict, trade, winter, or environment works:

| Change | Files to update |
|--------|----------------|
| Phase logic | `src/simulator/phases/*.py` AND `rust/src/phases/*.rs` |
| New/changed param | `src/simulator/params.py` AND `rust/src/params.rs` AND `rust/src/grid_search.rs` (defaults, bounds, to_vec) |
| Terrain constants | `src/constants.py` AND `rust/src/types.rs` |

After changing Rust code:
```bash
cd rust && maturin develop --release && cd ..
```

### Post-processing

If you change how predictions are post-processed:

| File | Purpose |
|------|---------|
| `src/prediction/postprocess.py` | Used by orchestrator's hybrid prediction |
| `rust/src/postprocess.rs` | Used by grid search scoring |

**Both must match exactly**, or the grid search optimizes for different scoring than what the orchestrator submits.

### Adding a new parameter

This is the most error-prone change. Full checklist:

1. **`src/simulator/params.py`** — add field with default, add to BOUNDS dict
2. **`rust/src/params.rs`** — add field to struct, update `NDIM`, update `from_array` indices (everything after the new param shifts by 1!)
3. **`rust/src/grid_search.rs`** — update `default_params()`, `bounds_lower()`, `bounds_upper()`, and `to_vec()`
4. **The phase that uses it** — both Python (`src/simulator/phases/*.py`) and Rust (`rust/src/phases/*.rs`)
5. **`data/rounds/best_params.json`** — old params have wrong length. Must insert the new param's default value at the correct index
6. **Rebuild Rust** — `cd rust && maturin develop --release`

### Deploying to GCP (cluster)

The cluster setup (`scripts/gcp_cluster.sh`) handles code upload and Rust build automatically.
For the legacy single-VM setup, see README.md.

**Do NOT rebuild while a grid search is running** — it will crash the search.

### Deploying to GCP (single VM, legacy)

```bash
# Upload changed files
gcloud compute scp --recurse rust/src/ island-sim-search:~/island_sim/rust/src/ --zone=$ZONE --project=ai-nm26osl-1886
gcloud compute scp rust/Cargo.toml island-sim-search:~/island_sim/rust/Cargo.toml --zone=$ZONE --project=ai-nm26osl-1886
gcloud compute scp src/prediction/postprocess.py island-sim-search:~/island_sim/src/prediction/postprocess.py --zone=$ZONE --project=ai-nm26osl-1886
gcloud compute scp scripts/grid_search_rust.py island-sim-search:~/island_sim/scripts/grid_search_rust.py --zone=$ZONE --project=ai-nm26osl-1886

# SSH in and rebuild
gcloud compute ssh island-sim-search --zone=$ZONE --project=ai-nm26osl-1886
source ~/.cargo/env
export VIRTUAL_ENV=~/env
export PATH="$VIRTUAL_ENV/bin:$PATH"
cd ~/island_sim/rust && maturin develop --release
```

### Uploading new round data to GCP

```bash
gcloud compute ssh island-sim-search --zone=$ZONE --project=ai-nm26osl-1886 -- "mkdir -p ~/island_sim/data/rounds/round_NNN"
gcloud compute scp data/rounds/round_NNN/* island-sim-search:~/island_sim/data/rounds/round_NNN/ --zone=$ZONE --project=ai-nm26osl-1886
```

## Verification

After any change, run these checks:

```bash
# 1. All imports work
python -c "from src.orchestrator import main; from src.simulator.params import SimParams; print(f'OK, {SimParams.ndim()} params')"

# 2. Rust builds
cd rust && cargo check && cd ..

# 3. Rust and Python produce identical results
python -c "
import island_sim_core, numpy as np
from src.types import WorldState, Settlement
from src.simulator.params import SimParams
from src.prediction.postprocess import postprocess
from src.scoring import score_prediction
import json
from pathlib import Path

detail = json.loads(Path('data/rounds/round_001/detail.json').read_text())
gt = np.array(json.loads(Path('data/rounds/round_001/analysis_seed_0.json').read_text())['ground_truth'])
state = WorldState.from_api(detail['initial_states'][0], 40, 40)
params = SimParams.default()

pred = island_sim_core.generate_prediction(state.grid, state.settlements, params.to_array(), 50, 0)
pred_p = postprocess(np.array(pred), state)
score = score_prediction(gt, pred_p)
print(f'Score: {score[\"score\"]:.1f} — if this runs without error, things are in sync')
"

# 4. Param count matches everywhere
python -c "
from src.simulator.params import SimParams
import json
d = json.loads(open('data/rounds/best_params.json').read())
assert len(d['params']) == SimParams.ndim(), f'best_params has {len(d[\"params\"])} but SimParams expects {SimParams.ndim()}'
print(f'Params: {SimParams.ndim()} — best_params.json matches')
"
```

## Common mistakes

- **Forgetting to update Rust `from_array` indices** after adding a param — everything after the new param reads the wrong value silently
- **Forgetting to update `grid_search.rs` bounds** — the grid search uses hardcoded bounds, not Python's
- **Not rebuilding Rust after changes** — the old `.so` stays loaded
- **Updating GCP code while a search is running** — kills the process
- **Not migrating `best_params.json`** after adding a param — wrong param count causes crashes or silent misalignment
