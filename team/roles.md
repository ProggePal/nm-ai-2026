# Team Roles

3 people. Each owns a track. All tracks converge at end of Day 1.

---

## Person A — Bot Logic

**You own:** Making the bot smart enough to score well.

**Your track:**
1. Get the bot running against the real server (Phase 1)
2. Understand the game mechanics hands-on — play a few games yourself first
3. Improve the bot's decision-making: pathfinding, when to deliver, multi-bot coordination
4. Own the Nightmare difficulty (highest score ceiling — this is where the competition is won)

**Your input to the team:**
- "The bot is getting stuck in aisle 3 because X"
- "Nightmare is scoring Y — here's what I think is wrong"
- Replay files that show specific failure modes

**Your output:**
- A bot that scores competitively on all 5 difficulties
- Replay files saved to `challenges/groceries/replays/`

**How to use Claude:**
- Paste a replay summary and ask "why is the bot waiting so much in rounds 150-200?"
- Ask Claude to review a specific function and suggest improvements
- Ask Claude to implement a specific algorithm (e.g. "implement A* pathfinding as a replacement for BFS in pathfinding.py")

---

## Person B — Infrastructure

**You own:** The local simulator and the game runner.

**Your track:**
1. After Person A plays the first real game — capture the map data that comes out
2. Build the local simulator so we can play unlimited games without burning rate limits
3. Validate the simulator matches real server scores within 15%
4. Set up the rate-limit-aware runner for automated real-server submissions

**Why this matters:**
We have 300 games/day on the real server. With a local sim, we can run 10,000 games overnight. This is our biggest structural advantage.

**Your output:**
- `challenges/groceries/sim/` — working local game simulator
- Simulator validated: same bot scores within 15% on sim vs real server
- Runner script working: `python shared/runner/runner.py --games easy:TOKEN ...`

**How to use Claude:**
- Share the game spec (`challenges-groceries.md`) and ask Claude to implement the simulator
- Ask Claude to identify why the sim and real server scores diverge
- Ask Claude to add specific mechanics you're missing (order transitions, collision rules)

**Critical question for Day 1:**
> Are the maps fixed per difficulty, or randomized each game?

Play the same difficulty twice. Compare the `grid.walls` from both `game_state` messages. If identical → maps are fixed. Save them. If different → simulator uses random generation per spec.

---

## Person C — Intelligence

**You own:** The leaderboard monitor and the overnight improvement loop.

**Your track:**
1. Get the leaderboard monitor running — figure out the actual API URL by inspecting `app.ainm.no` in browser devtools
2. Set it to poll every 2 minutes and alert when we're overtaken
3. Watch competitor replays daily — public replays are free intelligence
4. Set up the Claude improvement loop to run overnight

**Why this matters:**
The leaderboard monitor tells us when to push harder and when we're safe. Competitor replays tell us what strategies are working. The overnight loop multiplies our iteration speed by 10×.

**Your output:**
- Leaderboard monitor running and logging to `shared/eval/leaderboard_history.jsonl`
- Competitor replay analysis: a short doc on what top-scoring teams are doing differently
- Overnight loop running: score improving or holding steady by morning

**How to use Claude:**
- Share `shared/eval/leaderboard_history.jsonl` and ask "which teams are improving fastest and what does that suggest about their strategy?"
- Share a competitor's replay and ask "what is this bot doing that ours isn't?"
- Ask Claude to adjust the improvement loop parameters if it's not finding improvements

---

## Shared Responsibilities (All Three)

| Task | Who does it |
|------|------------|
| Getting a JWT token for each game | Whoever is running that game |
| Noting whether maps are fixed or random | Person A (Day 1, first thing) |
| Submitting final scores before deadline | Everyone checks at midnight March 15 |
| Watching the leaderboard in the last 6 hours | Person C primary, others secondary |

---

## Decision Authority

| Decision | Who decides |
|----------|------------|
| Bot strategy changes | Person A |
| Sim architecture | Person B |
| Whether to watch a competitor replay | Person C |
| Whether to submit a new score | Anyone — submit early, submit often |
| Whether to change difficulty focus | Group discussion, < 5 min |

---

## Communication Rhythm

- **Start of each session:** 2-line status in Slack: "I'm working on X, current score Y"
- **Blockers:** Post immediately — don't sit on a blocker for more than 30 min
- **Score updates:** Post every time you beat your own best score
- **Overnight:** Person C sets up the loop and posts "loop running" in Slack before sleeping
