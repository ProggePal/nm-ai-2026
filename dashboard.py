"""
Astar Island — Command Center Dashboard

Reads from data/ directory (no API calls — daemon populates it).
Auto-refreshes every 30 seconds.

Run:
    streamlit run dashboard.py
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import streamlit as st
from pathlib import Path
from datetime import datetime, timezone

DATA_DIR = Path("data")

# ── Constants ──────────────────────────────────────────────────────────────────

CLASS_COLORS = [
    "#D2B48C",  # 0: Empty/Ocean/Plains (tan)
    "#FF8C00",  # 1: Settlement (orange)
    "#00CED1",  # 2: Port (teal)
    "#8B0000",  # 3: Ruin (dark red)
    "#228B22",  # 4: Forest (forest green)
    "#808080",  # 5: Mountain (gray)
]
CLASS_NAMES = ["Empty/Ocean/Plains", "Settlement", "Port", "Ruin", "Forest", "Mountain"]
FOG_COLOR = "#1a1a2e"

TERRAIN_TO_CLASS = {
    0: 0, 10: 0, 11: 0,   # Empty / Ocean / Plains → 0
    1: 1, 2: 2, 3: 3, 4: 4, 5: 5,
}


def _hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) / 255 for i in (0, 2, 4))


COLORS_RGB = [_hex_to_rgb(c) for c in CLASS_COLORS]
FOG_RGB = _hex_to_rgb(FOG_COLOR)


# ── Data loading ───────────────────────────────────────────────────────────────

def load_round_state():
    """Find the most recently modified round in data/."""
    if not DATA_DIR.exists():
        return None, None
    round_dirs = sorted(
        [d for d in DATA_DIR.iterdir()
         if d.is_dir() and (d / "round_detail.json").exists()],
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )
    if not round_dirs:
        return None, None
    d = round_dirs[0]
    detail = json.loads((d / "round_detail.json").read_text())
    return d.name, detail


def load_seed_state(round_id: str, seed_idx: int):
    """Return (queries, prediction, ground_truth) for one seed."""
    seed_dir = DATA_DIR / round_id / f"seed_{seed_idx}"
    pred_dir = DATA_DIR / round_id / "predictions"

    queries = []
    qpath = seed_dir / "queries.jsonl"
    if qpath.exists():
        for line in qpath.read_text().strip().split("\n"):
            if line.strip():
                try:
                    queries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    prediction = None
    if pred_dir.exists():
        files = sorted(pred_dir.glob(f"seed_{seed_idx}_v*.npy"))
        if files:
            prediction = np.load(str(files[-1]))

    ground_truth = None
    gt_path = seed_dir / "ground_truth.npy"
    if gt_path.exists():
        ground_truth = np.load(str(gt_path))

    return queries, prediction, ground_truth


# ── Map builders ───────────────────────────────────────────────────────────────

def initial_class_map(state: dict) -> np.ndarray:
    """H×W class index array from raw initial_states entry."""
    raw = np.array(state["grid"], dtype=int)
    vfunc = np.vectorize(lambda x: TERRAIN_TO_CLASS.get(int(x), 0))
    return vfunc(raw)


def observed_class_map(queries, height, width) -> np.ndarray:
    """H×W class map from simulate results; −1 = fog of war."""
    obs = np.full((height, width), -1, dtype=int)
    for q in queries:
        vx, vy = q["vx"], q["vy"]
        for ri, row in enumerate(q["result"]["grid"]):
            for ci, code in enumerate(row):
                obs[vy + ri][vx + ci] = TERRAIN_TO_CLASS.get(int(code), 0)
    return obs


def viewport_rects(queries):
    """List of (vx, vy, vw, vh) for red overlay rectangles."""
    rects = []
    for q in queries:
        vx, vy = q["vx"], q["vy"]
        grid = q["result"].get("grid", [])
        vh = len(grid)
        vw = len(grid[0]) if grid else 15
        rects.append((vx, vy, vw, vh))
    return rects


def settlement_signals(queries) -> dict:
    """Latest observed health for each settlement, keyed by (x, y)."""
    signals = {}
    for q in queries:
        for s in q["result"].get("settlements", []):
            signals[(s["x"], s["y"])] = {
                "alive": s.get("alive", True),
                "food":  round(s.get("food", 1.0), 2),
                "pop":   round(s.get("population", 1.0), 2),
                "wealth": round(s.get("wealth", 0.5), 2),
                "has_port": s.get("has_port", False),
            }
    return signals


# ── Rendering ──────────────────────────────────────────────────────────────────

def class_map_to_img(class_map: np.ndarray) -> np.ndarray:
    """Convert H×W class index (−1=fog) to H×W×3 RGB float."""
    H, W = class_map.shape
    img = np.zeros((H, W, 3), dtype=float)
    for cls, rgb in enumerate(COLORS_RGB):
        img[class_map == cls] = rgb
    img[class_map == -1] = FOG_RGB
    return img


def render_terrain_fig(class_map: np.ndarray, overlays=None, title="", figsize=4):
    fig, ax = plt.subplots(figsize=(figsize, figsize), facecolor="#0e1117")
    ax.set_facecolor("#0e1117")
    ax.imshow(class_map_to_img(class_map), interpolation="nearest", aspect="equal")
    if overlays:
        for vx, vy, vw, vh in overlays:
            rect = patches.Rectangle(
                (vx - 0.5, vy - 0.5), vw, vh,
                linewidth=1.5, edgecolor="red", facecolor="none", alpha=0.7,
            )
            ax.add_patch(rect)
    ax.set_title(title, color="white", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout(pad=0.2)
    return fig


def render_prediction_fig(prediction: np.ndarray, figsize=4):
    argmax = prediction.argmax(axis=2)
    confidence = prediction.max(axis=2)
    H, W = argmax.shape

    img = np.zeros((H, W, 4), dtype=float)
    for cls, rgb in enumerate(COLORS_RGB):
        mask = argmax == cls
        img[mask, :3] = rgb
        img[mask, 3] = confidence[mask]

    fig, ax = plt.subplots(figsize=(figsize, figsize), facecolor="#0e1117")
    ax.set_facecolor("#0e1117")
    ax.imshow(np.zeros((H, W, 3)), interpolation="nearest", aspect="equal")
    ax.imshow(img, interpolation="nearest", aspect="equal")
    ax.set_title("Prediction (argmax + confidence)", color="white", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout(pad=0.2)
    return fig


def render_comparison_fig(prediction: np.ndarray, ground_truth: np.ndarray, figsize=4):
    pred_cls = prediction.argmax(axis=2)
    gt_cls = ground_truth.argmax(axis=2) if ground_truth.ndim == 3 else ground_truth.astype(int)
    correct = pred_cls == gt_cls
    accuracy = correct.mean()
    H, W = pred_cls.shape

    def make_cls_img(cmap):
        img = np.zeros((H, W, 3), dtype=float)
        for cls, rgb in enumerate(COLORS_RGB):
            img[cmap == cls] = rgb
        return img

    diff_img = np.zeros((H, W, 3), dtype=float)
    diff_img[correct] = (0.1, 0.8, 0.1)
    diff_img[~correct] = (0.8, 0.1, 0.1)

    fig, axes = plt.subplots(1, 3, figsize=(figsize * 3, figsize), facecolor="#0e1117")
    for ax, img, title in [
        (axes[0], make_cls_img(pred_cls), "Prediction"),
        (axes[1], make_cls_img(gt_cls), "Ground Truth"),
        (axes[2], diff_img, f"Accuracy: {accuracy:.1%}"),
    ]:
        ax.set_facecolor("#0e1117")
        ax.imshow(img, interpolation="nearest", aspect="equal")
        ax.set_title(title, color="white", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout(pad=0.2)
    return fig, accuracy


# ── Main app ───────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="Astar Command Center",
        page_icon="⚔️",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    # Auto-refresh every 30 seconds via HTML meta tag
    st.markdown('<meta http-equiv="refresh" content="30">', unsafe_allow_html=True)

    st.markdown("""
    <style>
    .main .block-container { padding-top: 1rem; }
    .stTabs [data-baseweb="tab"] { color: #ccc; }
    .stTabs [aria-selected="true"] { color: white; border-bottom: 2px solid #FF8C00; }
    </style>
    """, unsafe_allow_html=True)

    # ── Load round state ───────────────────────────────────────────────────────
    round_id, detail = load_round_state()

    if round_id is None:
        st.title("⚔️ ASTAR COMMAND CENTER")
        st.info("No round data yet. Start the daemon to populate data/.")
        return

    width = detail["map_width"]
    height = detail["map_height"]
    seeds_count = detail["seeds_count"]
    round_num = detail.get("round_number", "?")
    closes_at_str = detail.get("closes_at", "")

    # Time remaining
    time_str = "—"
    if closes_at_str:
        try:
            closes = datetime.fromisoformat(closes_at_str.replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            remaining = (closes - now).total_seconds()
            if remaining > 0:
                m, s = divmod(int(remaining), 60)
                time_str = f"{m:02d}:{s:02d}"
            else:
                time_str = "CLOSED"
        except Exception:
            time_str = closes_at_str[:16]

    # Seeds submitted count
    pred_dir = DATA_DIR / round_id / "predictions"
    submitted = sum(
        1 for i in range(seeds_count)
        if pred_dir.exists() and list(pred_dir.glob(f"seed_{i}_v*.npy"))
    )

    # Total queries used
    total_queries = 0
    for i in range(seeds_count):
        qpath = DATA_DIR / round_id / f"seed_{i}" / "queries.jsonl"
        if qpath.exists():
            total_queries += sum(
                1 for line in qpath.read_text().strip().split("\n") if line.strip()
            )

    # ── Header row ────────────────────────────────────────────────────────────
    h1, h2, h3, h4 = st.columns([3, 2, 2, 2])
    with h1:
        st.markdown(f"## ⚔️ ASTAR COMMAND CENTER &nbsp; `Round {round_num}`")
    with h2:
        st.metric("Queries Used", f"{total_queries} / 50")
        st.progress(min(total_queries / 50, 1.0))
    with h3:
        seeds_icons = " ".join("✓" if i < submitted else "○" for i in range(seeds_count))
        st.metric("Seeds Submitted", f"{submitted} / {seeds_count}")
        st.markdown(f"`{seeds_icons}`")
    with h4:
        st.metric("Closes In", time_str)

    st.divider()

    # ── Color legend ──────────────────────────────────────────────────────────
    legend_html = " ".join(
        f'<span style="background:{color};padding:2px 10px;margin:2px;'
        f'border-radius:4px;color:white;font-size:11px;">{name}</span>'
        for color, name in zip(CLASS_COLORS + [FOG_COLOR], CLASS_NAMES + ["Fog of war"])
    )
    st.markdown(legend_html, unsafe_allow_html=True)
    st.markdown("")

    # ── Per-seed tabs ──────────────────────────────────────────────────────────
    tabs = st.tabs([f"Seed {i}" for i in range(seeds_count)])
    initial_states = detail.get("initial_states", [])

    for seed_idx, tab in enumerate(tabs):
        with tab:
            queries, prediction, ground_truth = load_seed_state(round_id, seed_idx)

            if not queries and prediction is None:
                st.info(f"No data yet for Seed {seed_idx}. Daemon will populate this.")
                continue

            init_cls = initial_class_map(initial_states[seed_idx]) \
                if seed_idx < len(initial_states) else None
            obs_cls = observed_class_map(queries, height, width) if queries else None
            rects = viewport_rects(queries) if queries else []
            signals = settlement_signals(queries) if queries else {}

            # ── Three map panels ──────────────────────────────────────────────
            mc1, mc2, mc3 = st.columns(3)

            with mc1:
                if init_cls is not None:
                    fig = render_terrain_fig(init_cls, title=f"Initial Map — Seed {seed_idx}")
                    st.pyplot(fig)
                    plt.close(fig)
                else:
                    st.write("Initial map unavailable")

            with mc2:
                if obs_cls is not None:
                    n_obs = (obs_cls != -1).sum()
                    fig = render_terrain_fig(
                        obs_cls, overlays=rects,
                        title=f"Observed t50 — {len(queries)} queries, {n_obs} cells",
                    )
                    st.pyplot(fig)
                    plt.close(fig)
                else:
                    st.write("No observations yet")

            with mc3:
                if prediction is not None:
                    fig = render_prediction_fig(prediction)
                    st.pyplot(fig)
                    plt.close(fig)
                else:
                    st.write("No prediction yet")

            # ── Settlement health table ────────────────────────────────────────
            if signals:
                st.subheader("Settlement Health")
                rows = []
                for (x, y), s in sorted(signals.items()):
                    rows.append({
                        "Position": f"({x}, {y})",
                        "Alive": "✓ alive" if s["alive"] else "✗ RUIN",
                        "Food": s["food"],
                        "Population": s["pop"],
                        "Wealth": s["wealth"],
                        "Port": "⚓" if s["has_port"] else "—",
                    })
                df = pd.DataFrame(rows)

                def style_alive(val):
                    if "RUIN" in str(val):
                        return "background-color: #5a0000; color: #ff9999"
                    return "background-color: #0a2e0a; color: #99ff99"

                styled = df.style.map(style_alive, subset=["Alive"])
                st.dataframe(styled, use_container_width=True, hide_index=True)

            # ── Ground truth comparison / coverage ────────────────────────────
            if ground_truth is not None and prediction is not None:
                st.subheader("Post-Round Analysis")
                fig, accuracy = render_comparison_fig(prediction, ground_truth)
                st.pyplot(fig)
                plt.close(fig)
                n_correct = int(accuracy * height * width)
                st.success(
                    f"Prediction accuracy: **{accuracy:.1%}** "
                    f"({n_correct} / {height * width} cells correct)"
                )
            else:
                if obs_cls is not None:
                    n_obs = int((obs_cls != -1).sum())
                    coverage = n_obs / (height * width)
                    st.info(
                        f"Coverage: {n_obs} / {height * width} cells observed "
                        f"({coverage:.1%}) — ground truth available after round closes."
                    )

    # ── Footer ────────────────────────────────────────────────────────────────
    st.divider()
    st.caption(
        f"Round ID: `{round_id}` · Map: {width}×{height} · "
        f"Auto-refreshes every 30s · "
        f"Last loaded: {datetime.now().strftime('%H:%M:%S')}"
    )


if __name__ == "__main__":
    main()
