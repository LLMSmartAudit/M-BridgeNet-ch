"""BGE-assisted bridge pair annotation tool.

Embeds all posts with BGE, ranks cross-platform pairs by cosine similarity,
and presents the top-K for manual labeling — making annotation tractable even
for large events (1000+ posts).

Usage:
    python scripts/smart_annotate.py \\
        --event data/cphot/raw/us_israel_iran_001.json \\
        --top-k 200

Controls:
    y / 1   → bridge pair
    n / 0   → not a bridge
    s       → skip (excluded from training)
    q       → quit and save progress
    p       → print stats
"""
from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mbridgenet.config import load_config
from mbridgenet.stage1.embedder import BGEEmbedder
from mbridgenet.stage2.lifecycle import assign_phases

WIDTH = 88


def _hr(char: str = "─") -> str:
    return char * WIDTH


def _show_pair(idx: int, total: int, pa: Dict, pb: Dict, sim: float) -> None:
    print(_hr("═"))
    print(f"  Pair {idx}/{total}   cosine_sim = {sim:.4f}")
    print(_hr())
    print(f"  [A]  {pa['platform'].upper():<10} {pa['timestamp'][:19]}  id={pa['post_id']}")
    for line in textwrap.wrap(pa["text"][:400], width=WIDTH - 6):
        print(f"       {line}")
    print()
    print(f"  [B]  {pb['platform'].upper():<10} {pb['timestamp'][:19]}  id={pb['post_id']}")
    for line in textwrap.wrap(pb["text"][:400], width=WIDTH - 6):
        print(f"       {line}")
    print(_hr())


def _prompt() -> str:
    return input("  Bridge pair? [y/n/s=skip/q=quit/p=stats] → ").strip().lower()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--event",  required=True)
    parser.add_argument("--top-k",  type=int, default=200,
                        help="Number of top similar pairs to annotate (default: 200)")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--resume", action="store_true",
                        help="Skip pairs already in scored_pairs")
    args = parser.parse_args()

    event_path = Path(args.event)
    with open(event_path, encoding="utf-8") as f:
        event = json.load(f)

    posts = event["posts"]
    cfg = load_config(args.config)

    # ── Embed all posts ───────────────────────────────────────────────────────
    print(f"\nEvent: {event['event_id']}  |  Posts: {len(posts)}")
    print(f"Loading BGE embedder from {cfg.stage1.embedding_model} ...")
    embedder = BGEEmbedder(cfg.stage1.embedding_model)
    texts = [p["text"] for p in posts]
    print(f"Embedding {len(texts)} posts ...")
    vecs = embedder.encode_batch(texts)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    mat = vecs / norms
    sim_matrix = mat @ mat.T
    print("Embeddings done.")

    already_labeled = {
        (sp["post_a_id"], sp["post_b_id"])
        for sp in event.get("scored_pairs", [])
    }

    # ── Compute lifecycle phases ──────────────────────────────────────────────
    from datetime import datetime, timezone
    timestamps = []
    for p in posts:
        ts = datetime.fromisoformat(p["timestamp"])
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        timestamps.append(ts)
    hourly_volumes = event.get("hourly_volumes", [])
    phases = assign_phases(timestamps, hourly_volumes,
                           thresholds=cfg.stage2.lifecycle_thresholds)
    phase_map = {p["post_id"]: ph for p, ph in zip(posts, phases)}

    # ── Rank cross-platform pairs by similarity ───────────────────────────────
    scored: List[Tuple[float, int, int]] = []
    for i in range(len(posts)):
        for j in range(i + 1, len(posts)):
            if posts[i]["platform"] == posts[j]["platform"]:
                continue
            key = (posts[i]["post_id"], posts[j]["post_id"])
            if args.resume and key in already_labeled:
                continue
            scored.append((float(sim_matrix[i, j]), i, j))

    scored.sort(key=lambda x: x[0], reverse=True)
    candidates = scored[: args.top_k]

    print(f"Top-{args.top_k} pairs selected from {len(scored)} cross-platform pairs.")
    print("\nInstructions: y=bridge  n=not bridge  s=skip  q=quit  p=stats\n")

    # ── Annotation loop ───────────────────────────────────────────────────────
    scored_pairs: List[Dict] = list(event.get("scored_pairs", []))
    bridge_pairs = list(event.get("bridge_pairs", []))
    bridge_set = {tuple(bp) for bp in bridge_pairs}

    n_yes = n_no = n_skip = 0

    def _save() -> None:
        event["scored_pairs"] = scored_pairs
        event["bridge_pairs"] = [[a, b] for a, b in bridge_set]
        with open(event_path, "w", encoding="utf-8") as fout:
            json.dump(event, fout, ensure_ascii=False, indent=2)
        print(f"\n  ✓ Saved → {event_path}")

    try:
        for rank, (sim, i, j) in enumerate(candidates, start=1):
            pa, pb = posts[i], posts[j]
            _show_pair(rank, len(candidates), pa, pb, sim)

            while True:
                answer = _prompt()
                if answer in ("y", "1"):
                    label = 1
                    bridge_set.add((pa["post_id"], pb["post_id"]))
                    n_yes += 1
                    break
                elif answer in ("n", "0"):
                    label = 0
                    n_no += 1
                    break
                elif answer == "s":
                    label = -1
                    n_skip += 1
                    break
                elif answer == "q":
                    print(f"\n  Quit after {rank - 1} pairs.")
                    _save()
                    return
                elif answer == "p":
                    print(f"\n  Stats: yes={n_yes}  no={n_no}  skip={n_skip}\n")
                else:
                    print("  ← type y / n / s / q / p")
                    continue

            scored_pairs.append({
                "post_a_id": pa["post_id"],
                "post_b_id": pb["post_id"],
                "s1": round(sim, 6),
                "s2": 0.0,
                "s3": 0.0,
                "s4": 0.0,
                "phase": phase_map.get(pa["post_id"], "emergence"),
                "label": label,
            })

    except (KeyboardInterrupt, EOFError):
        print("\n  Interrupted.")

    _save()
    print(f"\n  Done: {n_yes} bridge  {n_no} non-bridge  {n_skip} skipped")
    print(f"  Total bridge pairs saved: {len(bridge_set)}")


if __name__ == "__main__":
    main()
