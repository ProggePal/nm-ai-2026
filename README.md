# NM i AI 2026 — ProggePal

Team submission for [NM i AI 2026](https://app.ainm.no) — Norway's national AI championship.

## Team

**ProggePal**

## Tasks

### Task 1 — Tripletex (AI Accounting Agent)
**Folder:** [`task1-tripletex/`](./task1-tripletex/)

An LLM-powered agent (Claude Opus 4.6) that automates accounting tasks in Tripletex via natural language. Exposes a `/solve` HTTP endpoint that receives task prompts and executes Tripletex API calls autonomously using tool use and multi-step reasoning.

**Stack:** TypeScript, Bun, Hono

**Run:**
```bash
cd task1-tripletex
bun install
bun run start
```

---

### Task 2 — Astar Island (Prediction / Game Agent)
**Folder:** [`task2-astar/`](./task2-astar/)

An AI agent for the Astar Island grocery logistics simulation. Controls bots to pick and deliver grocery orders across multiple difficulty levels (Easy → Nightmare).

**Stack:** Python / TypeScript

See [`task2-astar/README.md`](./task2-astar/README.md) for run instructions.

---

### Task 3 — NorgesGruppen Data (Object Detection)
**Folder:** [`task3-norgesgruppen/`](./task3-norgesgruppen/)

Two-stage ML pipeline for retail product detection and classification:
1. **YOLOv11x** for bounding-box detection (70% of points)
2. **ResNet50** with cosine-similarity matching for product classification (30% of points)

**Stack:** Python, PyTorch, Ultralytics

See [`task3-norgesgruppen/README.md`](./task3-norgesgruppen/README.md) for run instructions.

---

## License

MIT — see [LICENSE](./LICENSE)
