# Astar Island — Round 1 Plan

## Timeline tonight

| Time (Oslo) | What happens | What to run |
|---|---|---|
| 22:42 | Round closes | (automatic) |
| ~22:45 | Scoring in progress | wait |
| ~22:50 | Round "completed", ground truth available | `python analyze.py TOKEN` |
| 22:55 | Read results, tune priors | see "After analysis" below |
| 23:00+ | Daemon running for round 2 | `python daemon.py TOKEN` |

**Round closes at: 20:42 UTC = 22:42 Oslo time**

---

## Option A — Leave daemon running (recommended)

Just run this and leave it:
```bash
cd /Users/pal/dev_pals/nmai
source .venv/bin/activate
python daemon.py TOKEN
```

The daemon will:
1. Notice the round is no longer active
2. Detect it completed
3. Automatically fetch ground truth for all 5 seeds → saves to `data/`
4. Print training data summary
5. Keep polling for the next round — when it opens, auto-submit

---

## Option B — Cron job (if you close your laptop)

Add this to crontab (`crontab -e`):

```cron
# Fetch ground truth at 22:55 Oslo (after round closes + scoring)
55 22 * * * cd /Users/pal/dev_pals/nmai && .venv/bin/python post_round.py TOKEN >> logs/cron.log 2>&1

# Then run analysis at 23:00
00 23 * * * cd /Users/pal/dev_pals/nmai && .venv/bin/python analyze.py TOKEN >> logs/cron.log 2>&1

# Daemon restarts at 23:05 to catch next round
05 23 * * * cd /Users/pal/dev_pals/nmai && .venv/bin/python daemon.py TOKEN >> logs/daemon.log 2>&1
```

---

## What "analyze.py" tells you

After it runs you'll see per-seed:
- **Estimated score** (0-100)
- **Worst 5 cells** — exactly which cells we got most wrong
- **Per-class breakdown** — are we failing on Ruins? Forests? Settlements?

Example output to look for:
```
Seed 0:
  Estimated score: 45.2/100
  Per-class breakdown:
    Forest      347 cells   KL: 0.12   Accuracy: 91%  ← good
    Settlement   29 cells   KL: 1.40   Accuracy: 48%  ← bad, needs work
    Ruin          4 cells   KL: 2.10   Accuracy: 20%  ← very bad
    Empty      1186 cells   KL: 0.05   Accuracy: 97%  ← great
```

---

## After analysis — what to update in the model

### If Ruins score badly:
→ Our prior for Settlement→Ruin transition is wrong
→ Edit `resubmit_prior.py`, increase P(Ruin) for Settlement cells
```python
# Current
p = [0.08, 0.42, 0.08, 0.30, 0.10, 0.02]  # Settlement prior
# Try
p = [0.08, 0.35, 0.08, 0.38, 0.09, 0.02]  # more weight on Ruin
```

### If Plains-near-settlement scores badly:
→ Settlements expanded more (or less) than expected
→ Adjust the distance thresholds in `terrain_prior()`

### If Forest scores badly:
→ Forest is less stable than assumed (P=0.85 too high)
→ Reduce to 0.70-0.75

---

## Next round strategy

The new `astar.py` saves everything automatically:
- Every simulate response → `data/{round_id}/seed_{i}/queries.jsonl`
- Every prediction → `data/{round_id}/predictions/seed_{i}_v{n}.npy`
- Ground truth (fetched by daemon) → `data/{round_id}/seed_{i}/ground_truth.npy`

**Biggest improvement available for round 2:**
Use settlement `alive` status from simulate responses to boost Ruin probability:
```python
# If a settlement we observed at t=50 is alive=False → it collapsed
# → that cell has high P(Ruin), boost it before submitting
signals = get_settlement_signals(round_id, seed_idx)
for (sx, sy), stats in signals.items():
    if not stats["alive"]:
        prediction[sy, sx] = ruin_boosted_prior()
```
This is already implemented in `store.py` (`get_settlement_signals`) — just needs wiring into the prediction builder.

---

## File map

| File | Purpose |
|---|---|
| `astar.py` | Core observation + prediction + submission |
| `daemon.py` | Runs forever: auto-submits rounds, fetches ground truth |
| `analyze.py` | Post-round analysis: score breakdown, worst cells |
| `resubmit_prior.py` | Resubmit with better spatial prior (no queries needed) |
| `store.py` | All persistence: queries, predictions, ground truth |
| `data/` | All saved data (grows across rounds) |

---

## Commands cheat sheet

```bash
# Activate environment
source .venv/bin/activate

# Run full pipeline for active round
python astar.py TOKEN

# Resubmit with spatial prior only (no queries used)
python resubmit_prior.py TOKEN

# Analyze last completed round
python analyze.py TOKEN

# Leave running overnight (handles everything)
python daemon.py TOKEN

# Check what training data we have
python -c "from store import training_data_summary; training_data_summary()"
```
