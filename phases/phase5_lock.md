# Phase 5 — Lock In Final Scores

**When:** March 15, 18:00 → March 16, 00:00 (hard deadline)
**Owner:** All three
**Done when:** Best possible bot has been submitted on all 5 difficulties, confirmed on leaderboard

---

## The Rule

> Submit your best scores by **midnight March 15**. Do not wait for "one more improvement."

The deadline is 03:00 March 16. But if something goes wrong at 02:45 (token expired, network issue, server glitch), you have no time to fix it. Midnight gives you a 3-hour buffer.

The tie-breaker was already locked in Phase 1. There's no benefit to waiting.

---

## Final Submission Checklist

For each difficulty, in this order:

**Nightmare (do first — most tokens needed, most important)**
- [ ] Run 3 real server games → pick best score, confirm on leaderboard
- [ ] Leaderboard shows our Nightmare score updated

**Expert**
- [ ] Run 2 real server games → confirm best score on leaderboard

**Hard**
- [ ] Run 2 real server games → confirm best score on leaderboard

**Medium**
- [ ] Run 1–2 real server games → confirm best score on leaderboard

**Easy**
- [ ] Run 1 real server game → confirm best score on leaderboard

---

## Leaderboard Sanity Check

After all submissions, go to `app.ainm.no` and confirm manually:
- Our team name appears
- All 5 difficulty scores are showing
- Total score matches what we expect
- Timestamp is before March 16 03:00 CET

Screenshot the leaderboard and post in Slack. Keep the screenshot.

---

## If We're Behind the Leader

If we're not in first place at midnight March 15, evaluate:

**Gap ≤ 20 points:**
→ One more Nightmare game. The variance in Nightmare is high enough that a lucky run could close the gap.

**Gap 20–100 points:**
→ Look at which difficulty has the biggest gap. Is it Nightmare? Run 2 more Nightmare games.

**Gap > 100 points:**
→ Accept the position. Document what we'd do differently. This is all learning for the main competition.

**If we ARE in first place:**
→ Stop submitting. Don't burn rate limits. Watch the leaderboard until 03:00. If someone passes us in the last hour, evaluate one more submission.

---

## Post-Competition: What We Take Into the Main Event

After the deadline, spend 30 minutes documenting:

Create `session_notes.md` in the repo root:

```markdown
# Pre-comp Post-Mortem

## Final Scores
- Easy: X
- Medium: X
- Hard: X
- Expert: X
- Nightmare: X
- Total: X
- Final rank: #X

## What Worked
- [list 3 things]

## What Didn't Work
- [list 3 things]

## Infrastructure Status
- Local sim: working / not working / partial
- Improve loop: working / not working / needs X
- Leaderboard monitor: working / not working

## For the Main Competition
- Biggest bot improvement opportunity: [describe]
- Simulator needs: [describe]
- Team process improvement: [describe]
```

This becomes the starting point for the main competition.

---

## The Bigger Picture

The main competition is the real prize — 1,000,000 NOK.

Everything we've built (sim, improve loop, leaderboard monitor, bot framework, team workflow) carries over. The challenge format might be different but the infrastructure pattern — build locally, iterate overnight with Claude, validate on real server, watch competitors — applies to any AI competition.

**We are not building a grocery bot. We are building a competition-winning machine.**
