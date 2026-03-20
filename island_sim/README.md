# Astar Island — NM i AI 2026

Predict probability distributions of terrain types on a 40x40 Norse civilization grid after 50 simulated years.

## Setup

```bash
cd island_sim
python -m venv .venv && source .venv/bin/activate
pip install numpy requests python-dotenv cma pytest
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
2. Find the active round and query the simulator
3. Infer parameters if enough observations (>= 10)
4. Generate hybrid predictions (observations + Monte Carlo)
5. Submit predictions for all 5 seeds

### Fetch historical round data

```bash
python scripts/fetch_history.py
```

Downloads ground truth, initial states, and scores for all completed rounds into `data/rounds/`.

### Calibrate simulator against ground truth

```bash
python scripts/calibrate.py            # Sim-only calibration
python scripts/calibrate.py --hybrid   # Hybrid (observation + sim) calibration
```

### Grid search for optimal parameters (local)

```bash
python scripts/grid_search.py --workers 8 --candidates 200 --mc-runs 30
```

Best params are saved to `data/rounds/best_params.json` and automatically loaded by the orchestrator.

## GCP Grid Search (96 cores)

### Prerequisites

- `gcloud` CLI installed
- Authenticated with your `@gcplab.me` account: `gcloud auth login`

### Steps

```bash
# 1. Create VM, install deps, upload code + data
bash scripts/gcp_setup.sh

# 2. SSH into the VM
gcloud compute ssh island-sim-search --zone=europe-north1-a

# 3. Run the grid search with nohup (survives SSH disconnect, ~60-90 min)
cd ~/island_sim
nohup ~/env/bin/python scripts/grid_search.py --workers 90 --candidates 2000 --mc-runs 50 > search.log 2>&1 &

# 4. Check progress (can disconnect and reconnect anytime)
tail -f search.log

# 5. When done, exit SSH and pull results back to your laptop
bash scripts/gcp_pull_results.sh

# 6. Delete the VM
gcloud compute instances delete island-sim-search --zone=europe-north1-a
```

## Project Structure

```
src/
├── constants.py          # Terrain codes, class mappings
├── types.py              # Settlement, WorldState, Observation
├── api_client.py         # API wrapper with rate limiting
├── scoring.py            # Local scoring (entropy-weighted KL divergence)
├── orchestrator.py       # Main pipeline
├── simulator/
│   ├── params.py         # SimParams (~35 hidden params with bounds)
│   ├── world.py          # Grid utilities
│   ├── engine.py         # simulate() — 50 years × 5 phases
│   └── phases/           # growth, conflict, trade, winter, environment
├── inference/
│   ├── observer.py       # Viewport placement strategy
│   ├── loss.py           # NLL loss for parameter inference
│   └── optimizer.py      # CMA-ES parameter search
└── prediction/
    ├── monte_carlo.py    # Run N sims → probability tensor
    ├── observation_model.py  # Aggregate viewport observations
    ├── hybrid.py         # Blend observations + simulator
    └── postprocess.py    # Floor clamping, static overrides, normalize
```
