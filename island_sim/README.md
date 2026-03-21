# Astar Island — NM i AI 2026

Predict probability distributions of terrain types on a 40x40 Norse civilization grid after 50 simulated years.

## Setup

```bash
cd island_sim
python -m venv .venv && source .venv/bin/activate
pip install numpy requests python-dotenv cma pytest maturin
```

Build the Rust simulator backend (36x faster than Python):

```bash
cd rust && maturin develop --release && cd ..
```

Create `.env` with your JWT token:

```
JWT_TOKEN=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJkNmFmNDU4NC0xYmNmLTQxZGMtYWU3Yy04NzEyNmVmMmVmMTIiLCJlbWFpbCI6ImVyaWtAYXBwZmFybS5pbyIsImlzX2FkbWluIjpmYWxzZSwiZXhwIjoxNzc0NjI2NjAwfQ.bsUzF162S2DQUbWg_3W8C1ohAj9kI8j1qxrt_ko95cY
```

## Usage

### Run the full pipeline (live round)

```bash
python -m src.orchestrator
```

This will:

1. Sync completed round history for warm-starting
2. Load best params from grid search (`data/rounds/best_params.json`)
3. Query the simulator (50 viewport observations)
4. Run CMA-ES parameter inference to fine-tune for this round (Rust-accelerated)
5. Generate hybrid predictions (observations + 2000 Monte Carlo sims)
6. Submit predictions for all 5 seeds
7. Save observations for potential resubmission

### Fetch historical round data

```bash
python scripts/fetch_history.py
```

### Grid search for optimal parameters

**Local (Rust-powered):**

```bash
python scripts/grid_search_rust.py --candidates 5000 --mc-runs 50 --top 10
python scripts/grid_search_rust.py --candidates 5000 --mc-runs 50 --top 10 --seed 7  # Different search region
```

**GCP cluster (5 × c4d-highcpu-384 = 1,920 Zen 5 vCPUs):**

```bash
bash scripts/gcp_cluster.sh                         # 5 VMs, 50k candidates each
bash scripts/gcp_cluster.sh --vms 3 --candidates 100000  # Custom
```

Best params are saved to `data/rounds/best_params.json` and automatically loaded by the orchestrator.

### Calibrate simulator against ground truth

```bash
python scripts/calibrate.py            # Sim-only
python scripts/calibrate.py --hybrid   # Hybrid (observation + sim)
```

### Visualize predictions vs ground truth

```bash
python scripts/visualize.py                       # Latest scored round
python scripts/visualize.py --round 14            # Specific round
python scripts/visualize.py --round 14 --seed 0   # Specific seed
python scripts/visualize.py --no-show             # Save only, no popup
```

Output: `data/visualizations/` — overview, per-class errors, probability bars.

## GCP Cluster (5 × 384 vCPUs)

- Project: `ai-nm26osl-1886`
- Machine: `c4d-highcpu-384` (AMD EPYC 9B45 Turin, Zen 5, 4.1 GHz)
- Each VM explores a different region of parameter space via `--seed`

### Launch cluster

```bash
bash scripts/gcp_cluster.sh
```

Creates 5 VMs, installs Python + Rust, uploads code, builds, and starts grid search on each.

### Monitor progress

```bash
# Quick status check (reads zones from data/cluster_config.json)
CONFIG=data/cluster_config.json
for i in $(seq 0 4); do
    zone=$(python3 -c "import json; print(json.load(open('$CONFIG'))[$i]['zone'])")
    echo "=== VM $i ($zone) ==="
    gcloud compute ssh island-search-$i --zone=$zone -- 'tail -3 ~/island_sim/search.log; uptime'
done
```

### Pull results (merges all VMs, keeps global best)

```bash
bash scripts/gcp_cluster_pull.sh
```

### Delete cluster

```bash
CONFIG=data/cluster_config.json
for i in $(seq 0 4); do
    zone=$(python3 -c "import json; print(json.load(open('$CONFIG'))[$i]['zone'])")
    gcloud compute instances delete island-search-$i --zone=$zone -q
done
```

## GCP Single VM (legacy)

For smaller runs, the single-VM setup still works:

- VM: `island-sim-search` in `europe-north1-a`
- Machine: `n2d-highcpu-224` (224 vCPUs)

```bash
bash scripts/gcp_setup.sh        # Create VM
bash scripts/gcp_pull_results.sh  # Pull results
```

## Automated Agent Loop

See `AGENT_LOOP.md` for instructions on running the pipeline overnight in a Claude Code loop.

## Project Structure

```
src/
├── constants.py              # Terrain codes, class mappings
├── types.py                  # Settlement, WorldState, Observation
├── api_client.py             # API wrapper with rate limiting
├── scoring.py                # Local scoring (entropy-weighted KL divergence)
├── orchestrator.py           # Main pipeline
├── simulator/
│   ├── params.py             # SimParams (40 hidden params with bounds)
│   ├── world.py              # Grid utilities
│   ├── engine.py             # simulate() — Rust or Python backend
│   └── phases/               # growth, conflict, trade, winter, environment
├── inference/
│   ├── observer.py           # Viewport placement strategy
│   ├── loss.py               # NLL loss for parameter inference
│   └── optimizer.py          # CMA-ES parameter search (Rust backend with Python fallback)
└── prediction/
    ├── monte_carlo.py        # Run N sims → probability tensor (Rust-accelerated)
    ├── observation_model.py  # Aggregate viewport observations
    ├── hybrid.py             # Blend observations + simulator
    └── postprocess.py        # Static overrides, probability floor, normalize

rust/                          # Rust simulator backend (PyO3)
├── Cargo.toml
└── src/
    ├── lib.rs                # PyO3 bindings (simulate, generate_prediction, run_grid_search, run_cmaes_inference)
    ├── engine.rs             # Main simulation loop
    ├── params.rs             # SimParams + shared bounds/defaults (must match Python exactly)
    ├── types.rs              # Settlement, WorldState, terrain codes
    ├── world.rs              # Grid utilities
    ├── scoring.rs            # Entropy-weighted KL divergence
    ├── postprocess.rs        # Static overrides + probability floor
    ├── grid_search.rs        # Parallel grid search with Rayon
    ├── cmaes_inference.rs    # CMA-ES parameter inference (NLL loss, parallel population eval)
    └── phases/               # growth, conflict, trade, winter, environment

scripts/
├── fetch_history.py          # Download completed round data
├── calibrate.py              # Score simulator against ground truth
├── visualize.py              # Visualize predictions vs ground truth
├── grid_search.py            # Python multiprocessing grid search (legacy)
├── grid_search_rust.py       # Rust-powered grid search (preferred, --seed for parallel runs)
├── smart_search.py           # CMA-ES powered parameter search
├── gcp_setup.sh              # Create single GCP VM (legacy)
├── gcp_pull_results.sh       # Pull results from single VM (legacy)
├── gcp_cluster.sh            # Launch cluster of c4d-highcpu-384 VMs
└── gcp_cluster_pull.sh       # Pull + merge results from all cluster VMs
```
