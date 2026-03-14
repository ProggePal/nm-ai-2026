# Phase 4 — Monitor Competitors and Iterate

**When:** March 15, all day
**Owner:** Person C (monitor), Person A (iterate), Person B (maintain sim)
**Done when:** We're confident we have competitive scores or we've run out of rate limits

---

## Morning Briefing (do this first thing)

1. **Check leaderboard** — What is our rank? What's the gap to #1?
2. **Check improve_history.json** — What did the overnight loop find?
3. **Check replays** — Did the bot regress anywhere?
4. **Post in Slack** (keep it short):
   > "Morning status: Rank #X, score Y. Overnight loop improved medium by Z points. Plan today: [one line]"

---

## Competitor Intelligence Routine (Person C)

Do this once in the morning and once in the evening.

### Step 1 — Find the top scorers

From `leaderboard_history.jsonl`, identify:
- Who has the highest total score?
- Who improved the most overnight?
- What difficulty is the gap largest on?

### Step 2 — Watch their replays

Go to `app.ainm.no` → find replay viewer → find the top team's most recent games.

For each difficulty where we're behind by > 20 points:
1. Watch the replay (or export it)
2. Paste a summary into Claude:

```
Our bot scores [X] on [difficulty].
Top competitor scores [Y] on [difficulty].

Here are 10 sample rounds from their replay:
[paste rounds]

Here are 10 sample rounds from our replay:
[paste rounds]

What specific differences do you see? What should we change?
```

### Step 3 — Act on what you learn

Post findings in Slack:
> "Competitor on Hard: bots pre-load preview items while delivering. We don't. Person A — is this in our bot?"

Person A evaluates and decides whether to implement.

---

## Iteration Rhythm (Person A)

Each iteration cycle:
1. Run 10 sim games → check median score
2. If there's an obvious pattern: fix it, test, commit
3. If stuck: ask Claude with a replay
4. Every 10+ point improvement: run real server game and submit

**Focus shift by difficulty:**
- Morning: Medium and Hard (most room to improve quickly)
- Afternoon: Expert and Nightmare (highest ceiling, needs more effort)
- Evening: Easy (clean up any remaining points, should be near-perfect)

**Nightmare is the priority in Phase 4.**
It has 500 rounds, 20 bots, 3 drop zones. The score ceiling is 3–5× other difficulties. If the leaderboard competition is close, Nightmare is where it's won or lost.

---

## Rate Limit Budget for March 15

You have 300 games/day. Suggested allocation:

| Difficulty | Games budgeted | Reason |
|------------|---------------|--------|
| Easy | 20 | Near-perfect by now |
| Medium | 40 | Validate sim improvements |
| Hard | 50 | Active improvement zone |
| Expert | 60 | Active improvement zone |
| Nightmare | 100 | Highest ceiling, most effort |
| Buffer | 30 | Unexpected issues |

Use `shared/runner/runner.ts` to stay within limits automatically.

---

## When to Stop Iterating and Lock In

Stop iterating on a difficulty and submit final when ANY of these are true:

1. Score hasn't improved in 3 consecutive real server games
2. You've hit > 90% of the theoretical max (hard to estimate but ask Claude)
3. It's past 18:00 on March 15 — lock in and focus on Nightmare

---

## End of Phase 4 — What We Should Have

- [ ] Competitor analysis done (at least 2 difficulties compared)
- [ ] Nightmare bot meaningfully improved from baseline
- [ ] All 5 difficulties have competitive scores
- [ ] Rate limit budget not exhausted (saving 50+ games for final submissions)
- [ ] No score regression from Phase 1 (check all difficulties)
