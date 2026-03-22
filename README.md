# Astar Island — Terrain Prediction Engine

Probabilistic terrain prediction agent for the Astar Island challenge. Built for NM i AI 2026, Task 2.

## How it works

Given a partially revealed grid, the agent queries viewports to observe terrain, builds a statistical model of cell distributions, and submits probability predictions for all six terrain classes across all seeds.

**Query strategy — two-phase breadth-first:**
- Phase 1 (10 queries): 2 per seed — ensures all seeds are on the board immediately
- Phase 2 (40 queries): 8 per seed — refines predictions with higher-value viewpoints ranked by volatility

Static classes (ocean, mountain) are locked after first observation. Dynamic cells (settlements, ports, ruins, forest) are modelled with Laplace-smoothed frequency counts.

## Usage

```bash
python daemon.py <JWT_TOKEN>             # auto-submit daemon, runs continuously
python astar.py <JWT_TOKEN> <ROUND_ID>  # single round, manual
```

## Stack

Python · NumPy · requests

## License

[MIT](./LICENSE)
