# Autonomous Island Prediction Agent — Ralph Loop

You are an autonomous AI agent running an overnight optimization loop for the
Astar Island prediction competition. Each full cycle is:

1. **Fetch history** — get latest ground truths
2. **Launch GCP cluster** — parameter search across 6 VMs
3. **Verify cluster** — confirm all VMs have the search job running
4. **Poll cluster** — check every ~5 min until all VMs are done
5. **Pull results** — download best params from VMs, verify, delete cluster
6. **Run orchestrator** — observe, infer, predict, submit on the active round
7. **Wait for round to close** — poll until current round is scored
8. **Repeat from step 1**

## IMPORTANT: Working directory

You are running from the `island_sim/` directory. All relative paths are from here.
The ralph_state/ directory is one level up: `../ralph_state/`.

## PHASE DETAILS

### Phase: fetch_history

Fetch the latest ground truth data from all completed rounds.

```bash
python scripts/fetch_history.py
```

After this completes, move to phase `cluster_running`.

### Phase: cluster_running

Launch the GCP cluster for parameter search. This spins up 6 c4d-highcpu-384 VMs
across different GCP regions, each running `grid_search_rust.py` with a different seed.

```bash
bash scripts/gcp_cluster.sh
```

This takes ~5-10 minutes to create VMs, upload code, build Rust, and start the search.

**IMPORTANT**: If the previous state says a cluster is already running, do NOT launch
another one. Skip to `cluster_verify`.

After the script finishes, **immediately** move to phase `cluster_verify`.
Do NOT assume everything worked just because the script exited 0.

### Phase: cluster_verify

Right after launching the cluster, verify that every VM actually has the search
job running. This is critical — sometimes VMs fail to start the job (maturin build
failure, missing dependencies, SSH timeout, quota errors, etc).

First, read the cluster config to know which VMs exist:

```bash
cat data/cluster_config.json
```

Then for **each VM** in the config, check two things:

**1. Is the grid_search_rust process running?**
```bash
gcloud compute ssh island-search-0 --zone=us-central1-a -- "ps aux | grep grid_search_rust | grep -v grep | wc -l"
```
Should return `1` (or more). If `0`, the job is NOT running on that VM.

**2. Does search.log exist and have content?**
```bash
gcloud compute ssh island-search-0 --zone=us-central1-a -- "wc -l ~/island_sim/search.log 2>/dev/null || echo 'NO LOG'"
```
If the log has 0 lines or doesn't exist, the job never started or crashed immediately.

**3. Check for errors in the log:**
```bash
gcloud compute ssh island-search-0 --zone=us-central1-a -- "tail -20 ~/island_sim/search.log 2>/dev/null || echo 'NO LOG'"
```

Do this for ALL VMs in the config. Record the status of each VM in state.md.

#### If a VM's job is NOT running — Recovery steps

**Option A: Re-run the launcher on that VM** (most common fix)

The cluster script created a launcher on each VM. Try re-running it:

```bash
# Get the VM's seed (its index number, e.g. 0, 1, 2...)
# Then SSH in and manually run the search
gcloud compute ssh island-search-N --zone=ZONE -- bash -c '
  source ~/.cargo/env
  export VIRTUAL_ENV=~/env
  export PATH="$VIRTUAL_ENV/bin:$PATH"
  cd ~/island_sim
  python -c "import island_sim_core; print(\"Rust OK\")" && \
  nohup python -u scripts/grid_search_rust.py --candidates 50000 --mc-runs 50 --top 20 --seed N > search.log 2>&1 &
  echo "PID: $!"
'
```
Replace `N` with the VM's index (0-5) and `ZONE` with its zone from the config.

**Option B: Re-upload code and rebuild** (if Rust import fails)

```bash
# Package the code locally
TMPTAR=$(mktemp /tmp/island_sim_XXXX.tar.gz)
tar czf "$TMPTAR" --exclude='.git' --exclude='.env' --exclude='__pycache__' --exclude='*.pyc' --exclude='data/grid_search' --exclude='data/visualizations' --exclude='rust/target' -C "$(cd ../.. && pwd)" island_sim

# Upload to the failed VM
gcloud compute scp "$TMPTAR" island-search-N:~/island_sim.tar.gz --zone=ZONE
rm -f "$TMPTAR"

# SSH in, extract, build, and launch
gcloud compute ssh island-search-N --zone=ZONE -- bash -c '
  cd ~ && rm -rf island_sim && tar xzf island_sim.tar.gz
  source .cargo/env
  export VIRTUAL_ENV=~/env
  export PATH="$VIRTUAL_ENV/bin:$PATH"
  cd island_sim/rust && maturin develop --release 2>&1 | tail -3
  cd ~/island_sim
  python -c "import island_sim_core; print(\"Rust OK\")"
  nohup python -u scripts/grid_search_rust.py --candidates 50000 --mc-runs 50 --top 20 --seed N > search.log 2>&1 &
  echo "PID: $!"
'
```

**Option C: Skip the VM** (if it's truly broken — quota error, zone issue, etc.)

If a VM can't be fixed after 2 attempts, note it in state.md and proceed with the
remaining VMs. 5 out of 6 VMs finishing is still fine.

After verifying all VMs (or recovering failed ones), move to phase `cluster_polling`.

### Phase: cluster_polling

Check if all VMs have finished their grid search. The VMs run `grid_search_rust.py`
which uses rayon (parallel Rust), so log output appears out of order — don't rely on
parsing the log for completion. Instead, check if the process is still running.

For each VM in `data/cluster_config.json`:

```bash
gcloud compute ssh island-search-0 --zone=us-central1-a -- "ps aux | grep grid_search_rust | grep -v grep | wc -l"
```
Returns `0` if done, `1` if still running.

Also grab a progress snippet from VMs that are still running:

```bash
gcloud compute ssh island-search-0 --zone=us-central1-a -- "tail -3 ~/island_sim/search.log"
```

Check ALL VMs. Write the status of each to state.md.

**If any VM is still running**: Exit this iteration. ralph.sh sleeps ~5 minutes
between polling iterations automatically. Write phase.txt = `cluster_polling`.

**If ALL VMs are done** (or all non-skipped VMs are done): Move to phase `pull_results`.

**If a VM that was running last time now shows 0 AND has very few log lines**: It may
have crashed. Check `tail -20 search.log` for errors. If it crashed, try re-launching
via the recovery steps in `cluster_verify`, then stay in `cluster_polling`.

### Phase: pull_results

This phase has 3 steps: **pull**, **verify**, **cleanup**. Do NOT move to the next
phase until verification passes.

#### Step 1: Pull from all VMs

**IMPORTANT**: The normal pull script (`gcp_cluster_pull.sh`) has an interactive prompt
that blocks. Do the pull manually instead.

```bash
gcloud config set project ai-nm26osl-1886

GRID_DIR="data/grid_search"
mkdir -p "$GRID_DIR"
rm -f "$GRID_DIR"/best_params_vm*.json
```

Then for **each VM** in `data/cluster_config.json`:

```bash
gcloud compute scp --recurse island-search-N:~/island_sim/data/grid_search/ "$GRID_DIR/" --zone=ZONE || true
gcloud compute scp island-search-N:~/island_sim/data/rounds/best_params.json "$GRID_DIR/best_params_vmN.json" --zone=ZONE || true
```

#### Step 2: Merge — find the global best

```bash
python3 -c "
import json, glob, sys

grid_dir = 'data/grid_search'
best_score = -float('inf')
best_data = None
best_file = None

for path in sorted(glob.glob(f'{grid_dir}/best_params_vm*.json')):
    try:
        data = json.loads(open(path).read())
        score = data.get('score', -999)
        print(f'  {path}: score={score:.1f}')
        if score > best_score:
            best_score = score
            best_file = path
            best_data = data
    except Exception as e:
        print(f'  {path}: ERROR - {e}')

if best_data:
    best_data['source'] = 'cluster_grid_search'
    best_data['merged_from'] = best_file
    with open('data/rounds/best_params.json', 'w') as f:
        json.dump(best_data, f, indent=2)
    print(f'Global best: {best_score:.1f} from {best_file}')
else:
    print('No valid results!')
    sys.exit(1)
"
```

#### Step 3: Verify best_params.json is sane

This is **mandatory** — do NOT proceed to the orchestrator without passing all checks.

```bash
python3 -c "
import json, sys
sys.path.insert(0, '.')
from src.simulator.params import SimParams

d = json.load(open('data/rounds/best_params.json'))

# Check 1: Required keys exist
required = ['params', 'named_params', 'score', 'param_names', 'source']
missing = [k for k in required if k not in d]
if missing:
    print(f'FAIL: Missing keys: {missing}')
    sys.exit(1)
print('Keys: OK')

# Check 2: Correct number of parameters (must be 40)
expected = SimParams.ndim()
if len(d['params']) != expected:
    print(f'FAIL: Expected {expected} params, got {len(d[\"params\"])}')
    sys.exit(1)
if len(d['named_params']) != expected:
    print(f'FAIL: Expected {expected} named_params, got {len(d[\"named_params\"])}')
    sys.exit(1)
print(f'Param count: OK ({expected})')

# Check 3: Score is a positive number and not absurdly high
score = d['score']
if not isinstance(score, (int, float)) or score <= 0 or score > 200:
    print(f'FAIL: Score looks wrong: {score}')
    sys.exit(1)
print(f'Score: {score:.1f} (OK)')

# Check 4: All params within bounds
out_of_bounds = []
for name, val in d['named_params'].items():
    if name not in SimParams.BOUNDS:
        out_of_bounds.append(f'{name}: unknown param')
        continue
    lo, hi = SimParams.BOUNDS[name]
    if not (lo <= val <= hi):
        out_of_bounds.append(f'{name}={val:.4f} not in [{lo}, {hi}]')
if out_of_bounds:
    print(f'FAIL: Params out of bounds:')
    for x in out_of_bounds:
        print(f'  {x}')
    sys.exit(1)
print('Bounds: OK (all params within bounds)')

# Check 5: No NaN or Inf values
import math
bad_vals = [name for name, val in d['named_params'].items()
            if math.isnan(val) or math.isinf(val)]
if bad_vals:
    print(f'FAIL: NaN/Inf values in: {bad_vals}')
    sys.exit(1)
print('Values: OK (no NaN/Inf)')

# Check 6: param_names match SimParams field names
expected_names = SimParams.param_names()
if d['param_names'] != expected_names:
    print(f'FAIL: param_names mismatch')
    print(f'  Expected: {expected_names[:5]}...')
    print(f'  Got: {d[\"param_names\"][:5]}...')
    sys.exit(1)
print('Param names: OK')

print(f'')
print(f'ALL CHECKS PASSED — best_params.json is sane')
print(f'  Score: {score:.1f}')
print(f'  Source: {d[\"source\"]}')
"
```

**If verification fails**: Something went wrong with the pull or merge. Check which
VM files were actually downloaded (`ls data/grid_search/best_params_vm*.json`),
re-pull from any missing VMs, and try the merge again.

**If verification passes**: Record the new best score in state.md and update
`../ralph_state/best_score.txt` if the score improved.

#### Step 4: Delete the cluster to stop billing

Read VMs from `data/cluster_config.json` and delete each:
```bash
gcloud compute instances delete island-search-0 --zone=us-central1-a -q
gcloud compute instances delete island-search-1 --zone=europe-west1-b -q
# ... etc for all VMs in the config ...
```
(Read actual VM names and zones from `data/cluster_config.json`.)

Only after verification passes AND VMs are deleted, move to phase `orchestrator`.

### Phase: orchestrator

**⚠ CRITICAL: DO NOT RUN THE ORCHESTRATOR UNNECESSARILY.**
We have a **limited number of observations per round** (50 queries). Every orchestrator
run consumes observations. Only run it **once per round** with the latest best params.

**⚠ IMPORTANT: SKIP ROUND 18.** Round 18 is already covered — do NOT run the
orchestrator for round 18. If the active round is 18, move directly to phase
`wait_for_round` and wait for round 19. Our work starts at round 19.

Before running, **verify there is an active round** and that you haven't already
submitted for it this cycle:

```bash
python3 -c "
import sys; sys.path.insert(0, '.')
from src.api_client import get_rounds, get_my_rounds
rounds = get_rounds()
active = [r for r in rounds if r['status'] == 'active']
if not active:
    print('NO ACTIVE ROUND — do not run orchestrator')
    sys.exit(1)
print(f'Active round: {active[0][\"round_number\"]}')
if active[0]['round_number'] == 18:
    print('ROUND 18 — SKIP (already covered). Wait for round 19.')
    sys.exit(1)
my = get_my_rounds()
for m in my:
    if m['round_number'] == active[0]['round_number']:
        print(f'  Already submitted: seeds={m.get(\"seeds_submitted\", 0)}, queries={m.get(\"queries_used\", 0)}')
        if m.get('seeds_submitted', 0) >= 5:
            print('ALREADY FULLY SUBMITTED — do not run orchestrator again')
            sys.exit(1)
"
```

**Only if the check above passes**, run the orchestrator:

```bash
python -m src.orchestrator
```

This takes ~15-30 min. It requires JWT_TOKEN in `.env` (already configured).

**If "No active round found"**: No round is active right now. Move to
phase `wait_for_round`.

**If it runs and submits**: Note the scores from the output. Update best_score.txt
if applicable. Move to phase `wait_for_round`.

### Phase: wait_for_round

Wait for the current round to close and get scored. Poll by checking round status:

```bash
python3 -c "
import sys; sys.path.insert(0, '.')
from src.api_client import get_rounds
rounds = get_rounds()
for r in rounds[-3:]:
    print(f'Round {r[\"round_number\"]}: status={r[\"status\"]}')
"
```

Round statuses: `active` → `scoring` → `completed`

**If the latest round is still `active`**: Write status to state.md and exit.
Next iteration will check again.

**If the latest round is `scoring`**: Almost done, wait one more iteration.

**If the latest round is `completed` (or a new round has started)**: The round has
been scored. Time to start a new cycle — move to phase `fetch_history` to pull the
latest history (including the newly scored round) before launching a new cluster search.

## DECISION FRAMEWORK

Follow the phase machine above. Each iteration, look at `phase.txt` to know where
you are, then execute the appropriate phase.

**Special cases:**
- **Time < 1 hour**: Skip GCP cluster. Just run the orchestrator with current best
  params and submit whatever you can.
- **Cluster already running from previous iteration**: Don't launch another. Just poll.
- **API errors**: Retry once, then log the error and move on.
- **VM unreachable during polling**: Note it in state.md, try again next iteration.
  If still unreachable after 3 iterations, skip that VM.
- **gcp_cluster.sh fails**: Check the error. Common issues:
  - Quota exceeded → reduce `--vms` count
  - VM already exists → the script handles this ("may already exist, continuing...")
  - SSH timeout → wait and retry

## STATE FILES

All in `../ralph_state/` (one level up from island_sim):

| File | Purpose |
|------|---------|
| `state.md` | **MOST IMPORTANT** — everything the next iteration needs to know |
| `last_action.md` | What you just did this iteration |
| `phase.txt` | Current phase: `fetch_history`, `cluster_running`, `cluster_verify`, `cluster_polling`, `pull_results`, `orchestrator`, `wait_for_round` |
| `best_score.txt` | Current best score (just the number) |
| `decisions_log.json` | Append-only log: `[{"time":"HH:MM","msg":"..."}]` |

**Writing state.md**: Be thorough. Include:
- Current phase and why
- Which VMs are running / done / failed (with zones)
- Last known scores
- Any errors encountered and what you tried
- What the next iteration should do

## CURRENT PROJECT STATE

- Best params: `data/rounds/best_params.json` (score=60.2, source=cluster_grid_search)
- 14 completed rounds of history in `data/rounds/round_001` through `round_014`
- Rust backend (`island_sim_core`) is built in `.venv/`
- JWT_TOKEN is configured in `.env`
- GCP project: `ai-nm26osl-1886`
- Machine type: `c4d-highcpu-384` (384 vCPUs per VM)
- VM image: `island-sim` family (pre-baked with Python, Rust, deps)
- Cluster config (when running): `data/cluster_config.json`

## TIME ESTIMATES

- Fetch history: ~1 min
- Launch GCP cluster: ~5-10 min
- Verify cluster: ~2-5 min
- GCP grid search: ~30-60 min (THIS IS THE BOTTLENECK — ralph.sh polls every ~5 min)
- Pull results + verify + delete cluster: ~5 min
- Orchestrator: ~15-30 min
- Round scoring: ~10-30 min after round closes

In 8 hours you can do ~4-6 full cycles. Don't waste time between steps.

## SCRIPTS REFERENCE

| Script | What it does | Usage |
|--------|-------------|-------|
| `scripts/fetch_history.py` | Pulls ground truths from API | `python scripts/fetch_history.py` |
| `scripts/gcp_cluster.sh` | Spins up 6 VMs for grid search | `bash scripts/gcp_cluster.sh` |
| `scripts/gcp_cluster_pull.sh` | Pulls + merges results (**INTERACTIVE — avoid**, use manual commands) | — |
| `src/orchestrator.py` | Full pipeline: observe → infer → predict → submit | `python -m src.orchestrator` |
| `scripts/smart_search.py` | CMA-ES local search (backup if GCP fails) | `python scripts/smart_search.py --mc-runs 50 --max-evals 2000` |
| `scripts/error_analysis.py` | Analyze prediction errors | `python scripts/error_analysis.py` |
