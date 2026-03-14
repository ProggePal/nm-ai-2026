# Phase 1 — Get on the Leaderboard

**When:** March 14, hours 2–8
**Owner:** All three, but Person A leads
**Done when:** At least one score on all 5 difficulties on the real leaderboard

---

## Why This Is Urgent

The tie-breaker is timestamp. If we and another team end up with the same total score, whoever submitted first wins. A mediocre score submitted today beats a perfect score submitted Sunday night in a tie.

---

## Step-by-Step

### Step 1 — Get a working bot running (Person A)

The bot code is in `challenges/groceries/bot/`. It is ready to run.

```
cd challenges/groceries/bot
node --experimental-transform-types client.ts --token YOUR_TOKEN --difficulty easy
```

Get a token from `app.ainm.no/challenge` → click Play on the Easy map.

**If the bot errors:** Paste the error and `client.ts` into Claude. Say: "Fix this."

### Step 2 — Extract map intelligence (Person A, during first game)

The client automatically saves the map on the first run to `challenges/groceries/maps/`.

After the first game on each difficulty, answer this question and post it in Slack:

> **Are the maps fixed or random?**
> Play the same difficulty twice. Open both saved map files. Compare the `walls` arrays.
> - If identical → **fixed maps** (huge advantage — post this in Slack)
> - If different → **random maps** (post this in Slack so Person B knows)

This single answer changes how Person B builds the simulator.

### Step 3 — Submit a score on all 5 difficulties

For each difficulty, get a token and run the bot. Even a score of 10 is enough — we just need the timestamp.

**Order of priority:**
1. Easy (fastest to score, gets timestamp locked)
2. Medium
3. Hard
4. Expert
5. Nightmare (do this last — it's the most complex)

Post each score in Slack as you get them:
> "Easy: 47 ✅"
> "Medium: 83 ✅"

### Step 4 — Note the game URL pattern

While running games, note:
- Does the token come from a URL parameter on `app.ainm.no`?
- Is there any API endpoint visible in browser devtools?
- What does the WebSocket `game_over` message look like?

Post your observations in Slack for Person C (leaderboard monitor setup).

---

## Parallel Track: Person C starts leaderboard monitor

While Person A is playing games, Person C should:
1. Open `app.ainm.no` in browser, open devtools → Network tab
2. Find the leaderboard API call (look for `/api/leaderboard` or similar)
3. Update the URL in `shared/eval/leaderboard-monitor.ts`
4. Run it: `node --experimental-transform-types shared/eval/leaderboard-monitor.ts --team "YourTeamName" --interval 120`
5. Confirm it's logging to `shared/eval/leaderboard_history.jsonl`

---

## End of Phase 1 — What We Should Know

- [ ] Scores on all 5 difficulties (timestamps locked)
- [ ] Maps: fixed or random?
- [ ] Leaderboard monitor running
- [ ] Our current rank posted in Slack
- [ ] Best replay files saved for Person C to start analysis
