# Phase 2 — Build the Local Simulator

**When:** March 14–15
**Owner:** Person B
**Done when:** Local sim scores within 15% of real server on same bot

---

## Why This Is the Most Important Infrastructure Piece

- Real server: 300 games/day, 60s cooldown between games
- Local sim: unlimited, instant

Without the simulator, the overnight improve loop cannot run. Every hour without a sim is an hour of lost iteration.

**Target: simulator running before midnight March 14.**

---

## What the Simulator Must Do

The simulator re-implements the server game loop locally. It receives the same game state format as the real server and accepts the same action format.

Minimum viable simulator:
1. Load a map (from `challenges/groceries/maps/easy.json` saved in Phase 1)
2. Generate items on shelves at the start of each game
3. Generate orders (active + preview) matching the spec
4. Accept `{actions: [...]}` each round and update state
5. Handle: movement, collision, pick-up, drop-off, order completion, scoring
6. Output final score

That's it. It does not need a visual interface. It does not need to be perfectly faithful — just close enough that a bot that improves on the sim also improves on the real server.

---

## How to Build It

**Give Claude the full game spec and ask it to build the simulator:**

```
Read the game specification in Pre-comp/challange-groceries.md.
Build a TypeScript local game simulator in challenges/groceries/sim/simulator.ts.

The simulator should:
- Accept a map file path (from challenges/groceries/maps/)
- Run N games with a given bot directory
- Output one score per line to stdout (for the improve loop to parse)
- CLI: node --experimental-transform-types simulator.ts --map-file maps/easy.json --bot-dir bot/ --games 20

Implement these game mechanics exactly per the spec:
- Bot movement with wall collision
- Pick-up rule: bot must be adjacent (Manhattan distance 1) to item, inventory < 3
- Drop-off rule: bot must stand ON drop-off cell, only active-order items count
- Order transition: when active order completes, preview becomes active
- Scoring: +1 per item delivered, +5 per complete order
- End condition: max_rounds reached OR 120s wall-clock (simulate as max_rounds only)
```

---

## Validation Protocol

Once Person B has a simulator running, validate it:

1. Run the same bot on the real server 5 times on Easy → note scores
2. Run the same bot on the sim 20 times on Easy → note median score
3. Calculate: `|real_median - sim_median| / real_median`
4. If < 15% → sim is valid, post "Sim validated ✅" in Slack
5. If ≥ 15% → paste the divergent results into Claude and ask what's wrong

**Common divergence causes:**
- Items re-spawning incorrectly after pickup
- Order transition not firing immediately when active order completes
- Collision detection allowing two bots on the same cell
- Drop-off accepting items for the preview order (it shouldn't)

---

## The Simulator's Interface Contract

The improve loop expects to call the simulator like this:

```
node --experimental-transform-types challenges/groceries/sim/run_sim.ts \
  --bot-dir challenges/groceries/bot \
  --games 20 \
  --difficulty medium
```

And get one integer score per line on stdout:

```
84
91
78
...
```

Person B owns making this interface work. The improve loop depends on it.

---

## If Maps Are Random (per Phase 1 finding)

If maps are randomized each game, the simulator generates random maps per the spec:
- Place walls as aisles (2 cells wide, evenly spaced)
- Place items randomly on wall cells (shelves)
- Randomize drop-off zone positions (within spec constraints)

Ask Claude: "Generate a random map matching the Medium difficulty spec: 16×12 grid, 3 aisles, 8 item types, 1 drop zone."

## If Maps Are Fixed (per Phase 1 finding)

If maps are fixed, the simulator loads the saved map file directly. This is simpler and more accurate.

---

## End of Phase 2 — What We Should Have

- [ ] `challenges/groceries/sim/run_sim.ts` working
- [ ] Simulator validated (< 15% score divergence from real server)
- [ ] "Sim validated ✅" posted in Slack with divergence number
- [ ] Person A can now iterate the bot locally without touching rate limits
