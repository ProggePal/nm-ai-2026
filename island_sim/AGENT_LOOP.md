# Astar Island — Automated Agent Loop

## Context

You are operating the Astar Island competition pipeline. The project is at `/Users/erikankerkilbergskallevold/Develop/git/nm-ai-2026/island_sim/`. Always `cd` there first.

The competition runs in rounds. Each round:
- Has 5 seeds, 50 query budget, 2h45m prediction window
- We observe through viewports, infer parameters, generate hybrid predictions, submit

## GCP VM

- Project: `ai-nm26osl-1886`
- VM name: `island-sim-search`
- Zone: `europe-north1-a`
- The VM has Rust + Python set up. To use it:
  - SSH: `gcloud compute ssh island-sim-search --zone=europe-north1-a --project=ai-nm26osl-1886`
  - The virtualenv is at `~/env/bin/python`
  - Rust extension is already built
  - Run commands via: `gcloud compute ssh island-sim-search --zone=europe-north1-a --project=ai-nm26osl-1886 -- "COMMAND"`

### Updating GCP code

When the local Rust code has changed, you must rebuild on GCP:

```bash
# Upload the updated Rust source
gcloud compute scp --recurse rust/src/ island-sim-search:~/island_sim/rust/src/ --zone=europe-north1-a --project=ai-nm26osl-1886
gcloud compute scp rust/Cargo.toml island-sim-search:~/island_sim/rust/Cargo.toml --zone=europe-north1-a --project=ai-nm26osl-1886

# Upload updated Python scripts
gcloud compute scp scripts/grid_search_rust.py island-sim-search:~/island_sim/scripts/grid_search_rust.py --zone=europe-north1-a --project=ai-nm26osl-1886

# Rebuild on GCP
gcloud compute ssh island-sim-search --zone=europe-north1-a --project=ai-nm26osl-1886 -- bash -s <<'REBUILD'
source ~/.cargo/env
export VIRTUAL_ENV=~/env
export PATH="$VIRTUAL_ENV/bin:$PATH"
cd ~/island_sim/rust
maturin develop --release
python -c "import island_sim_core; print('Rust backend OK')"
REBUILD
```

## The Loop

Repeat the following cycle for each round:

### Step 1: Wait for an active round

```bash
python -c "
from src.api_client import get_active_round
r = get_active_round()
if r:
    print(f'Round {r[\"round_number\"]}: {r[\"status\"]}')
    print(f'ID: {r[\"id\"]}')
else:
    print('No active round')
"
```

If no active round, wait 30 minutes and check again. Do not poll more frequently than every 5 minutes.

### Step 2: Fetch latest history (new ground truth from completed rounds)

```bash
python scripts/fetch_history.py
```

### Step 3: Upload new round data to GCP

```bash
# List what rounds are on GCP
gcloud compute ssh island-sim-search --zone=europe-north1-a --project=ai-nm26osl-1886 -- "ls ~/island_sim/data/rounds/"

# For any new round_NNN directories that exist locally but not on GCP:
gcloud compute ssh island-sim-search --zone=europe-north1-a --project=ai-nm26osl-1886 -- "mkdir -p ~/island_sim/data/rounds/round_NNN"
gcloud compute scp --recurse data/rounds/round_NNN/* island-sim-search:~/island_sim/data/rounds/round_NNN/ --zone=europe-north1-a --project=ai-nm26osl-1886
```

### Step 4: Run GCP grid search (Rust-powered)

There are two grid search scripts. Use the one that matches what's built on GCP:

**If GCP has the Rust grid search built** (`grid_search_rust.py` + rebuilt `island_sim_core`):
```bash
gcloud compute ssh island-sim-search --zone=europe-north1-a --project=ai-nm26osl-1886 -- bash -s <<'EOF'
pkill -f grid_search || true
sleep 2
source ~/.cargo/env
export VIRTUAL_ENV=~/env
export PATH="$VIRTUAL_ENV/bin:$PATH"
cd ~/island_sim
nohup python -u scripts/grid_search_rust.py --candidates 50000 --mc-runs 50 --top 20 > search.log 2>&1 &
echo "Started PID $!"
sleep 3
head -5 search.log
EOF
```

**If GCP only has the old Python grid search** (fallback):
```bash
gcloud compute ssh island-sim-search --zone=europe-north1-a --project=ai-nm26osl-1886 -- bash -s <<'EOF'
pkill -f grid_search || true
sleep 2
source ~/.cargo/env
export VIRTUAL_ENV=~/env
export PATH="$VIRTUAL_ENV/bin:$PATH"
cd ~/island_sim
nohup python -u scripts/grid_search.py --workers 220 --candidates 50000 --mc-runs 50 > search.log 2>&1 &
echo "Started PID $!"
sleep 3
head -5 search.log
EOF
```

### Step 5: Wait for GCP search to finish

Check progress every 5 minutes:

```bash
gcloud compute ssh island-sim-search --zone=europe-north1-a --project=ai-nm26osl-1886 -- "tail -3 ~/island_sim/search.log"
```

It's done when you see the "Completed in" line and TOP results.
- Rust grid search: typically 20-40 minutes
- Python grid search: typically 60-90 minutes

**IMPORTANT:** Do NOT proceed to Step 6 until the search is complete.

### Step 6: Pull GCP results

```bash
bash scripts/gcp_pull_results.sh
```

Verify the params loaded:
```bash
python -c "
import json
d = json.loads(open('data/rounds/best_params.json').read())
print(f'Score: {d[\"score\"]}')
print(f'Source: {d.get(\"source\", \"unknown\")}')
"
```

### Step 7: Check remaining budget

```bash
python -c "
from src.api_client import get_active_round, get_budget
r = get_active_round()
if r:
    b = get_budget()
    print(f'Round {r[\"round_number\"]}: {b[\"queries_used\"]}/{b[\"queries_max\"]} queries used')
else:
    print('No active round')
"
```

If no active round (it closed while we were searching), skip to Step 9.

### Step 8: Run the orchestrator

```bash
python -m src.orchestrator
```

This will:
1. Load the GCP-optimized params as warm-start
2. Make systematic + adaptive viewport queries
3. Run CMA-ES parameter inference (if >= 10 observations)
4. Generate hybrid predictions (observations + simulator)
5. Submit all 5 seeds
6. Save all observations for potential resubmission

Verify all 5 seeds show "accepted".

### Step 9: Wait for round to close and score

Check every 5 minutes:
```bash
python -c "
from src.api_client import get_rounds
rounds = get_rounds()
for r in sorted(rounds, key=lambda x: x['round_number'], reverse=True)[:3]:
    print(f'Round {r[\"round_number\"]}: {r[\"status\"]}')
"
```

Once the latest round shows "completed" or "scoring", go back to Step 1.

## Important Notes

- **Never run the orchestrator before pulling GCP results.** The orchestrator uses `best_params.json` as warm-start.
- **Never spend queries manually.** Let the orchestrator handle all API interactions.
- **If the orchestrator fails**, check the error, fix it, and re-run. Saved observations mean no queries are wasted on re-runs.
- **The Rust backend must be built locally** for the orchestrator to use it. Run `cd rust && maturin develop --release` if needed.
- **Don't modify code** unless something is broken. The pipeline is tested and working.
- **GCP VM costs money** (even when idle). If no round is expected for hours, consider stopping it: `gcloud compute instances stop island-sim-search --zone=europe-north1-a --project=ai-nm26osl-1886`. Start it again with `start` instead of `stop`.
- **If GCP code needs updating**, follow the "Updating GCP code" section above to upload and rebuild.
