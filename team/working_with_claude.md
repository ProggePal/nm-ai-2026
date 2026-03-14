# Working with Claude

Claude is the fourth team member. Use it aggressively throughout.

---

## The Core Pattern

Every time you're about to spend more than 20 minutes figuring something out, ask Claude first.

**Don't ask:** "How do I do X in Python?"
**Do ask:** "Here is my `planner.py` file and here is a replay showing the bot waiting for 40 rounds doing nothing. What's the most likely cause and what's the minimal fix?"

Give Claude context. Get specific answers.

---

## How to Start Every Claude Session in This Project

Open Claude Code in the `nm-ai-2026` directory. It will read `CLAUDE.md` automatically and know the project context.

Or paste this at the start of any conversation:

```
I'm working on NM i AI 2026 — a competition where we build bots for a WebSocket-based grocery store game.
The game spec is in Pre-comp/challange-groceries.md.
The bot code is in challenges/groceries/bot/.
[Then describe your specific question or task]
```

---

## What Claude Is Best At in This Project

### Bot improvement
Give Claude a replay file (or summary) + the relevant bot file and ask:
> "The bot scores 47 in this replay. Looking at rounds 120–180, it seems to be waiting a lot. What's causing it and what's the fix?"

### Writing specific functions
> "In pathfinding.py, the `bfs` function doesn't handle the case where the goal cell is a wall. Fix this."

### Debugging
> "This error occurs when running client.py: [paste traceback]. Here's the file: [paste code]"

### Implementing new algorithms
> "Replace the greedy task assignment in assignment.py with the Hungarian algorithm using scipy."

### Analyzing competitor replays
> "Here's a competitor's replay JSON. Our bot scores 80 on medium, theirs scores 140. What are they doing differently?"

---

## Overnight Improve Loop — How Claude Runs It

The loop in `shared/agent/improve_loop.py` runs Claude automatically. You just start it before sleeping.

**What it does each iteration:**
1. Reads recent replay files from `challenges/groceries/replays/`
2. Reads all bot Python files
3. Asks Claude: "What's the single most impactful failure pattern? Propose a minimal fix."
4. Applies the fix, runs eval games in the local sim
5. If score improves: commits the change
6. If not: reverts, tries again

**Before you sleep, post in Slack:**
> "Loop running. Starting score: X on medium. Bot dir: challenges/groceries/bot. Check improve_history.json in the morning."

**In the morning:**
> "Loop ran 23 iterations overnight. Best score went from 80 → 112. Here's the improve_history.json: [paste]. What were the most impactful fixes?"

---

## Prompts That Work Well

### Understanding a failure
```
Here is a replay summary from our grocery bot game.
The bot scored [X]. The theoretical max for this difficulty is roughly [Y].

[paste replay summary or key rounds]

What are the top 3 reasons the bot is underperforming and which one is easiest to fix?
```

### Writing a fix
```
Here is the current [file.py]:
[paste file]

The problem: [describe problem from replay analysis]

Write a minimal fix — change as few lines as possible. Only touch [function name].
```

### Reviewing a fix before committing
```
I'm about to commit this change to [file.py]:
[paste diff]

Does this look correct? Are there any edge cases I'm missing, particularly:
- What happens when inventory is full?
- What happens when there are no items on the map?
- What happens in end-game (round > 270/500)?
```

### Competitor analysis
```
Our bot scores [X] on [difficulty]. A competitor scores [Y].
Here is their public replay: [paste key rounds or summary]
Here is our bot's replay on the same difficulty: [paste key rounds]

What are they doing that we aren't? Give me 2-3 specific differences.
```

---

## What NOT to Ask Claude

- Don't ask Claude to write the entire bot from scratch mid-competition — it'll take too long to review and validate
- Don't ask Claude to redesign the architecture during the competition — iterate on what exists
- Don't ask vague questions — always include the actual code or replay data

---

## Keeping Claude's Context Fresh

Between sessions, Claude forgets what you discussed. Make it fast to catch up:

Keep a running `session_notes.md` in the repo root with:
- Current best scores per difficulty
- What we tried that didn't work
- Known bot weaknesses
- What the overnight loop found

Start each Claude session: "Read session_notes.md first."
