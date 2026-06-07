"""Terminal-based bridge pair annotation tool.

After collecting posts with collect_data.py, run this script to manually
label which cross-platform post pairs are bridge pairs.

Usage:
    python scripts/annotate_pairs.py --event data/cphot/raw/e001.json

Controls:
    y / 1   → bridge pair  (label = 1)
    n / 0   → not a bridge (label = 0)
    s       → skip this pair (label = -1, excluded from training)
    q       → quit and save progress so far
    p       → print current stats

Bridge pair definition:
    Post A on platform X triggers or is echoed by Post B on platform Y,
    where B arrives within 72 h of A and discusses the same sub-event.
"""
from __future__ import annotations

import argparse
import json
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple


# ── Display helpers ───────────────────────────────────────────────────────────

WIDTH = 88


def _hr(char: str = "─") -> str:
    return char * WIDTH


def _show_pair(
    idx: int,
    total: int,
    pa: Dict,
    pb: Dict,
    delta_h: float,
) -> None:
    print(_hr("═"))
    print(f"  Pair {idx}/{total}   Δt = {delta_h:.1f} h")
    print(_hr())

    # Post A
    print(f"  [A]  {pa['platform'].upper():<10} {pa['timestamp'][:19]}  id={pa['post_id']}")
    for line in textwrap.wrap(pa["text"], width=WIDTH - 6):
        print(f"       {line}")
    print()

    # Post B
    print(f"  [B]  {pb['platform'].upper():<10} {pb['timestamp'][:19]}  id={pb['post_id']}")
    for line in textwrap.wrap(pb["text"], width=WIDTH - 6):
        print(f"       {line}")
    print(_hr())


def _prompt() -> str:
    return input("  Bridge pair? [y/n/s=skip/q=quit/p=stats] → ").strip().lower()


# ── Candidate generation ──────────────────────────────────────────────────────

def _candidates(posts: List[Dict], max_gap_hours: float = 72.0) -> List[Tuple[Dict, Dict]]:
    """All cross-platform ordered pairs within max_gap_hours."""
    pairs = []
    for i, pa in enumerate(posts):
        for pb in posts[i + 1:]:
            if pa["platform"] == pb["platform"]:
                continue
            ts_a = datetime.fromisoformat(pa["timestamp"])
            ts_b = datetime.fromisoformat(pb["timestamp"])
            delta = (ts_b - ts_a).total_seconds() / 3600
            if 0 < delta <= max_gap_hours:
                pairs.append((pa, pb))
    return pairs


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Manually label bridge pairs in a collected event JSON"
    )
    parser.add_argument("--event",     required=True,
                        help="Path to CPHot event JSON (output of collect_data.py)")
    parser.add_argument("--max-gap",   type=float, default=72.0,
                        help="Max time gap in hours for candidate pairs (default: 72)")
    parser.add_argument("--resume",    action="store_true",
                        help="Skip pairs already present in scored_pairs")
    args = parser.parse_args()

    event_path = Path(args.event)
    with open(event_path, encoding="utf-8") as f:
        event = json.load(f)

    posts    = event["posts"]
    existing = {
        (sp["post_a_id"], sp["post_b_id"])
        for sp in event.get("scored_pairs", [])
    }

    candidates = _candidates(posts, args.max_gap)
    print(f"\nEvent: {event['event_id']}  |  "
          f"Posts: {len(posts)}  |  Candidate pairs: {len(candidates)}")

    if args.resume and existing:
        candidates = [(a, b) for a, b in candidates
                      if (a["post_id"], b["post_id"]) not in existing]
        print(f"Resuming — {len(candidates)} pairs remaining after skipping already-labeled.")

    if not candidates:
        print("No pairs to annotate.")
        return

    print("\nInstructions: y=bridge  n=not bridge  s=skip  q=quit  p=stats\n")

    scored_pairs: List[Dict] = list(event.get("scored_pairs", []))
    bridge_pairs: List[List[str]] = list(event.get("bridge_pairs", []))
    bridge_set   = {tuple(bp) for bp in bridge_pairs}

    n_yes = n_no = n_skip = 0

    def _save() -> None:
        event["scored_pairs"] = scored_pairs
        event["bridge_pairs"] = [[a, b] for a, b in bridge_set]
        with open(event_path, "w", encoding="utf-8") as fout:
            json.dump(event, fout, ensure_ascii=False, indent=2)
        print(f"\n  ✓ Saved → {event_path}")

    try:
        for i, (pa, pb) in enumerate(candidates, start=1):
            ts_a = datetime.fromisoformat(pa["timestamp"])
            ts_b = datetime.fromisoformat(pb["timestamp"])
            delta_h = (ts_b - ts_a).total_seconds() / 3600

            _show_pair(i, len(candidates), pa, pb, delta_h)

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
                    label = -1      # excluded from MLP training
                    n_skip += 1
                    break
                elif answer == "q":
                    print(f"\n  Quit after {i - 1} pairs.")
                    _save()
                    return
                elif answer == "p":
                    print(f"\n  Stats so far: yes={n_yes}  no={n_no}  skip={n_skip}\n")
                else:
                    print("  ← type y / n / s / q / p")
                    continue

            scored_pairs.append({
                "post_a_id": pa["post_id"],
                "post_b_id": pb["post_id"],
                "s1": 0.0,   # recomputed by train.py / generate_synthetic_data.py
                "s2": 0.0,
                "s3": 0.0,
                "s4": 0.0,
                "phase": "unknown",
                "label": label,
            })

    except (KeyboardInterrupt, EOFError):
        print("\n  Interrupted.")

    _save()
    print(f"\n  Annotation complete: {n_yes} bridge  {n_no} non-bridge  {n_skip} skipped")
    print(f"  Total bridge pairs: {len(bridge_set)}")


if __name__ == "__main__":
    main()
