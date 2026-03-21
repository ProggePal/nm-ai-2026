#!/bin/bash
# ralph.sh — Self-steering overnight loop via Gemini CLI
#
# The loop: fetch history → GCP cluster search → poll until done → pull results
#           → run orchestrator → wait for round to close → repeat
#
# Usage:
#   GEMINI_API_KEY=... bash ralph.sh
#   GEMINI_API_KEY=... bash ralph.sh 5    # resume from iteration 5

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOOP_DIR="$ROOT_DIR/ralph_state"
AGENT_PROMPT="$ROOT_DIR/OVERNIGHT_AGENT.md"
ISLAND_DIR="$ROOT_DIR/island_sim"

MAX_ITER=50
TOTAL_HOURS=8

echo ""
echo "  ╔══════════════════════════════════════╗"
echo "  ║    RALPH LOOP — Island Prediction    ║"
echo "  ║      Autonomous Overnight Agent      ║"
echo "  ╚══════════════════════════════════════╝"
echo ""

# Check gemini CLI
if ! command -v gemini &> /dev/null; then
  echo "ERROR: gemini CLI not found"
  echo "Install: npm install -g @google/gemini-cli"
  exit 1
fi

# Activate venv
if [ -f "$ISLAND_DIR/.venv/bin/activate" ]; then
  source "$ISLAND_DIR/.venv/bin/activate"
  echo "Activated venv: $(which python)"
else
  echo "WARNING: No venv found at $ISLAND_DIR/.venv"
fi

# Init state
mkdir -p "$LOOP_DIR"

ITER=${1:-1}
if [ "$ITER" -eq 1 ] && [ ! -f "$LOOP_DIR/deadline.txt" ]; then
  if [[ "$OSTYPE" == "darwin"* ]]; then
    DEADLINE=$(date -v+${TOTAL_HOURS}H +%s)
  else
    DEADLINE=$(date -d "+${TOTAL_HOURS} hours" +%s)
  fi
  echo "$DEADLINE" > "$LOOP_DIR/deadline.txt"
  echo "$ITER" > "$LOOP_DIR/iteration.txt"
  echo "initializing" > "$LOOP_DIR/phase.txt"
  echo "60.2" > "$LOOP_DIR/best_score.txt"
  echo "[]" > "$LOOP_DIR/decisions_log.json"
else
  DEADLINE=$(cat "$LOOP_DIR/deadline.txt")
fi

# ============================================================
# MAIN LOOP
# ============================================================
while [ $ITER -le $MAX_ITER ]; do
  NOW=$(date +%s)
  REMAINING=$((DEADLINE - NOW))
  [ $REMAINING -le 0 ] && echo "TIME'S UP" && break

  HOURS_LEFT=$((REMAINING / 3600))
  MINS_LEFT=$(( (REMAINING % 3600) / 60 ))

  echo ""
  echo "═══════════════════════════════════════"
  echo "  RALPH ITERATION $ITER  (${HOURS_LEFT}h ${MINS_LEFT}m left)"
  echo "═══════════════════════════════════════"

  echo "$ITER" > "$LOOP_DIR/iteration.txt"

  # Read state files
  CURRENT_STATE=$(cat "$LOOP_DIR/state.md" 2>/dev/null || echo "First iteration — no prior state.")
  BEST_SCORE=$(cat "$LOOP_DIR/best_score.txt" 2>/dev/null || echo "60.2")
  LAST_ACTION=$(cat "$LOOP_DIR/last_action.md" 2>/dev/null || echo "No previous actions.")
  CURRENT_PHASE=$(cat "$LOOP_DIR/phase.txt" 2>/dev/null || echo "initializing")

  # Build prompt with fresh context
  PROMPT="$(cat "$AGENT_PROMPT")

---

## CURRENT STATE (auto-injected by ralph.sh)

**Iteration:** $ITER
**Hours remaining:** ${HOURS_LEFT}h ${MINS_LEFT}m
**Best score so far:** $BEST_SCORE
**Current phase:** $CURRENT_PHASE

### Previous State
$CURRENT_STATE

### Last Action
$LAST_ACTION

---

## YOUR TASK THIS ITERATION

You are in iteration $ITER. You have ${HOURS_LEFT}h ${MINS_LEFT}m remaining.

1. Read the previous state and current phase above
2. Determine which step of the pipeline you are in and what to do next
3. Execute it — run the commands, check status, etc.
4. Write updated state to ralph_state/state.md (be thorough — next iteration reads this)
5. Write what you did to ralph_state/last_action.md
6. Write the current phase to ralph_state/phase.txt (one of: fetch_history, cluster_running, cluster_verify, cluster_polling, pull_results, orchestrator, wait_for_round)
7. Update ralph_state/best_score.txt if you have a new best
8. Log every decision to ralph_state/decisions_log.json (append, format: [{\"time\":\"HH:MM\",\"msg\":\"...\"}])

Remember:
- Write state.md THOROUGHLY — the next iteration of yourself reads it
- If time < 1 hour, skip GCP cluster and just run the orchestrator with current best params
- If a GCP cluster is still running, just poll and exit — don't start another one
"

  # Run Gemini from the island_sim directory
  echo "Launching Gemini agent..."
  cd "$ISLAND_DIR"
  echo "$PROMPT" | gemini 2>&1 | tee "$LOOP_DIR/iteration_${ITER}_output.log"

  echo "Iteration $ITER complete."
  ITER=$((ITER + 1))

  # Sleep between iterations — longer during polling phases to avoid wasting Gemini tokens
  CURRENT_PHASE=$(cat "$LOOP_DIR/phase.txt" 2>/dev/null || echo "unknown")
  if [ "$CURRENT_PHASE" = "cluster_polling" ] || [ "$CURRENT_PHASE" = "wait_for_round" ]; then
    echo "Phase is $CURRENT_PHASE — sleeping 5 minutes before next check..."
    sleep 300
  else
    sleep 10
  fi
done

echo ""
echo "Ralph Loop complete after $((ITER-1)) iterations"
echo "Best score: $(cat "$LOOP_DIR/best_score.txt" 2>/dev/null || echo 'unknown')"
