# Phase 3 — Run the Overnight Loop

**When:** Night of March 14–15 (before anyone sleeps)
**Owner:** Person C sets it up, runs unattended overnight
**Done when:** Loop has run at least 10 iterations, score has improved or held steady

---

## Prerequisites

Before starting the overnight loop:
- [ ] Simulator validated (Phase 2 complete)
- [ ] At least 10 replay files in `challenges/groceries/replays/`
- [ ] `ANTHROPIC_API_KEY` set in environment
- [ ] Slack message posted: "Loop starting, baseline score X on medium"

---

## How to Start the Loop

```bash
export ANTHROPIC_API_KEY=your_key_here

node --experimental-transform-types shared/agent/improve-loop.ts \
  --bot-dir challenges/groceries/bot \
  --sim-script challenges/groceries/sim/run_sim.ts \
  --replay-dir challenges/groceries/replays \
  --games-per-eval 20 \
  --max-iterations 50 \
  --min-improvement 2 \
  --difficulty medium
```

Run this in a `screen` or `tmux` session so it survives if your terminal closes:

```bash
screen -S improve_loop
# [paste the command above]
# Ctrl+A, D to detach
# screen -r improve_loop to reattach
```

---

## What to Check in the Morning

1. Open `challenges/groceries/improve_history.json`
2. Post a summary in Slack:

```
Overnight results:
- Iterations run: X
- Starting median: Y
- Final median: Z
- Fixes accepted: N
- Best fix: [describe the pattern it fixed]
```

3. If the loop ran into errors, paste the last 50 lines of terminal output into Claude and ask what went wrong.

---

## While the Loop Runs: Real Server Submissions

The loop improves the bot on the local sim. But you should also be pushing improved bots to the real server during the day.

**Rule: Any time the sim score improves by 10+ points, run a real server game.**

This keeps our leaderboard score up-to-date and locks in the tie-breaker timestamp on each improvement milestone.

Use the runner:
```bash
node --experimental-transform-types shared/runner/runner.ts \
  --bot-dir challenges/groceries/bot \
  --games medium:TOKEN_HERE hard:TOKEN_HERE
```

---

## Tuning the Loop

If the loop isn't finding improvements after 10+ iterations:

**Problem: improvements are too small**
→ Lower `--min-improvement` to `1`

**Problem: loop keeps accepting then reverting**
→ Increase `--games-per-eval` to `30` (more games = more reliable eval)

**Problem: Claude keeps proposing the same fix**
→ Ask Claude directly: "Here is `improve_history.json`. The loop has been fixing the same pattern 5 times. What different failure mode should we look at next?"

**Problem: scores are going down**
→ Check `git log` — the loop commits fixes. Run `git diff HEAD~3 HEAD` and paste into Claude: "These changes went into the bot overnight. Scores dropped. What's wrong?"

---

## End of Phase 3 — What We Should Have

- [ ] Loop ran all night (10+ iterations minimum)
- [ ] Bot has improved from baseline
- [ ] `improve_history.json` shows accepted fixes
- [ ] Morning Slack summary posted
- [ ] If improved: new real server submission made
