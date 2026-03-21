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
4. Run CMA-ES parameter inference to fine-tune for this round
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
```

**GCP (224 cores, Rust-powered):**

```bash
python scripts/grid_search_rust.py --candidates 50000 --mc-runs 50 --top 20
```

Best params are saved to `data/rounds/best_params.json` and automatically loaded by the orchestrator.

### Calibrate simulator against ground truth

```bash
python scripts/calibrate.py            # Sim-only
python scripts/calibrate.py --hybrid   # Hybrid (observation + sim)
```

## GCP VM (224 cores)

- Project: `ai-nm26osl-1886`
- VM: `island-sim-search` in `europe-north1-a`
- Machine: `n2d-highcpu-224` (224 vCPUs)

### First-time setup

```bash
bash scripts/gcp_setup.sh
```

Creates the VM, installs Python + Rust + dependencies, uploads code, and builds the Rust extension.

### Updating code on GCP

```bash
# Upload Rust source + scripts
gcloud compute scp --recurse rust/src/ island-sim-search:~/island_sim/rust/src/ --zone=europe-north1-a --project=ai-nm26osl-1886
gcloud compute scp rust/Cargo.toml island-sim-search:~/island_sim/rust/Cargo.toml --zone=europe-north1-a --project=ai-nm26osl-1886
gcloud compute scp scripts/grid_search_rust.py island-sim-search:~/island_sim/scripts/grid_search_rust.py --zone=europe-north1-a --project=ai-nm26osl-1886
gcloud compute scp src/prediction/postprocess.py island-sim-search:~/island_sim/src/prediction/postprocess.py --zone=europe-north1-a --project=ai-nm26osl-1886

# SSH in and rebuild
gcloud compute ssh island-sim-search --zone=europe-north1-a --project=ai-nm26osl-1886
source ~/.cargo/env
export VIRTUAL_ENV=~/env
export PATH="$VIRTUAL_ENV/bin:$PATH"
cd ~/island_sim/rust && maturin develop --release
```

### Uploading new round data

```bash
gcloud compute ssh island-sim-search --zone=europe-north1-a --project=ai-nm26osl-1886 -- "mkdir -p ~/island_sim/data/rounds/round_NNN"
gcloud compute scp data/rounds/round_NNN/* island-sim-search:~/island_sim/data/rounds/round_NNN/ --zone=europe-north1-a --project=ai-nm26osl-1886
```

### Running grid search on GCP

```bash
gcloud compute ssh island-sim-search --zone=europe-north1-a --project=ai-nm26osl-1886
source ~/.cargo/env
export VIRTUAL_ENV=~/env
export PATH="$VIRTUAL_ENV/bin:$PATH"
cd ~/island_sim
nohup python -u scripts/grid_search_rust.py --candidates 50000 --mc-runs 50 --top 20 > search.log 2>&1 &
tail -f search.log
```

### Pull results

```bash
bash scripts/gcp_pull_results.sh
```

### VM management

```bash
# Stop (saves money when idle)
gcloud compute instances stop island-sim-search --zone=europe-north1-a --project=ai-nm26osl-1886

# Start
gcloud compute instances start island-sim-search --zone=europe-north1-a --project=ai-nm26osl-1886

# Delete
gcloud compute instances delete island-sim-search --zone=europe-north1-a --project=ai-nm26osl-1886
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
│   ├── params.py             # SimParams (36 hidden params with bounds)
│   ├── world.py              # Grid utilities
│   ├── engine.py             # simulate() — Rust or Python backend
│   └── phases/               # growth, conflict, trade, winter, environment
├── inference/
│   ├── observer.py           # Viewport placement strategy
│   ├── loss.py               # NLL loss for parameter inference
│   └── optimizer.py          # CMA-ES parameter search
└── prediction/
    ├── monte_carlo.py        # Run N sims → probability tensor (Rust-accelerated)
    ├── observation_model.py  # Aggregate viewport observations
    ├── hybrid.py             # Blend observations + simulator
    └── postprocess.py        # Static overrides, probability floor, normalize

rust/                          # Rust simulator backend (PyO3)
├── Cargo.toml
└── src/
    ├── lib.rs                # PyO3 bindings (simulate, generate_prediction, run_grid_search)
    ├── engine.rs             # Main simulation loop
    ├── params.rs             # SimParams (must match Python exactly)
    ├── types.rs              # Settlement, WorldState, terrain codes
    ├── world.rs              # Grid utilities
    ├── scoring.rs            # Entropy-weighted KL divergence
    ├── postprocess.rs        # Static overrides + probability floor
    ├── grid_search.rs        # Parallel grid search with Rayon
    └── phases/               # growth, conflict, trade, winter, environment

scripts/
├── fetch_history.py          # Download completed round data
├── calibrate.py              # Score simulator against ground truth
├── grid_search.py            # Python multiprocessing grid search (legacy)
├── grid_search_rust.py       # Rust-powered grid search (preferred)
├── gcp_setup.sh              # Create and configure GCP VM
└── gcp_pull_results.sh       # Pull grid search results from GCP
```
