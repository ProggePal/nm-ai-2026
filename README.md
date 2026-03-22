# NM i AI 2026 — Appfarm IO

Team **Appfarm IO**'s submission for [NM i AI 2026](https://app.ainm.no) — Norway's national AI championship, March 19–22, 2026.

We competed across all three tasks: an AI accounting agent, a grocery logistics game bot, and a retail object detection pipeline.

**Team:** [@hansjto](https://github.com/hansjto) · [@ErikAnkerKilbergSkallevold](https://github.com/ErikAnkerKilbergSkallevold) · [@ProggePal](https://github.com/ProggePal)

---

## Task 1 — Tripletex: AI Accounting Agent

**Folder:** [`task1-tripletex/`](./task1-tripletex/)

An autonomous LLM agent that solves accounting tasks in [Tripletex](https://www.tripletex.no/) via natural language. Receives a task prompt at `/solve`, reasons over it using Claude Opus 4.6 with tool use, and executes the necessary Tripletex API calls to complete the task.

Features multi-step planning, compound tool execution, sandbox verification, and a replay dashboard for debugging agent traces.

**Stack:** TypeScript, Bun, Hono, Claude Opus 4.6

```bash
cd task1-tripletex
bun install
bun run start
```

---

## Task 2 — Astar Island: Grocery Logistics Agent

**Folder:** [`task2-astar/`](./task2-astar/)

An AI agent that controls warehouse robots in the Astar Island grocery simulation. The agent picks and delivers grocery orders across five difficulty levels (Easy → Nightmare), optimizing for throughput and delivery speed.

**Stack:** Python, TypeScript

See [`task2-astar/README.md`](./task2-astar/README.md) for run instructions.

---

## Task 3 — NorgesGruppen: Retail Product Detection

**Folder:** [`task3-norgesgruppen/`](./task3-norgesgruppen/)

A two-stage computer vision pipeline for retail shelf analysis:
1. **YOLOv11x** — detects product bounding boxes (70% of score)
2. **ResNet50 + cosine similarity** — classifies detected products against a reference catalog (30% of score)

Built to run within the competition sandbox (8 GB RAM, no OpenCV).

**Stack:** Python, PyTorch, Ultralytics YOLOv11

```bash
cd task3-norgesgruppen
pip install -r requirements.txt
python run.py
```

---

## License

[MIT](./LICENSE)
