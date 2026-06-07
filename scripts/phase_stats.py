#!/usr/bin/env python3
"""
Compute per-event lifecycle phase distribution from scored_pairs.
Reports dominant-phase concentration (fraction of bridge pairs in max phase).

Run from M-BridgeNet/:
    .venv/bin/python scripts/phase_stats.py \
        --data data/cphot/processed/test_real
"""
from __future__ import annotations
import argparse
import json
from collections import Counter
from pathlib import Path


ACTIVE_PHASES = {"emergence", "diffusion", "peak", "decline"}


def phase_stats(data_dir: Path) -> list[dict]:
    rows = []
    for f in sorted(data_dir.glob("*.json")):
        data = json.loads(f.read_text(encoding="utf-8"))
        event_id = data["event_id"]
        sp = [p for p in data.get("scored_pairs", [])
              if p.get("label", -1) != -1
              and p.get("phase", "?") in ACTIVE_PHASES]
        bridges = [p for p in sp if p.get("label") == 1]

        phase_counts = Counter(p["phase"] for p in sp)
        bridge_phases = Counter(p["phase"] for p in bridges)
        total = sum(phase_counts.values())
        total_b = sum(bridge_phases.values())

        dominant = max(phase_counts, key=phase_counts.get) if phase_counts else "N/A"
        conc = phase_counts[dominant] / total if total else 0.0
        bridge_conc = bridge_phases[dominant] / total_b if total_b else 0.0

        rows.append({
            "event_id": event_id,
            "total_pairs": total,
            "total_bridges": total_b,
            "phase_counts": dict(phase_counts),
            "bridge_phases": dict(bridge_phases),
            "dominant_phase": dominant,
            "pair_concentration": round(conc, 3),
            "bridge_concentration": round(bridge_conc, 3),
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    args = parser.parse_args()

    rows = phase_stats(Path(args.data))

    print(f"\n{'Event':<38} {'Dominant':12} {'Pair conc.':>10} "
          f"{'Bridge conc.':>12} {'#Bridges':>8}")
    print("-" * 88)
    for r in rows:
        print(f"{r['event_id']:<38} {r['dominant_phase']:12} "
              f"{r['pair_concentration']:>10.1%} "
              f"{r['bridge_concentration']:>12.1%} "
              f"{r['total_bridges']:>8}")
    print()
    print("Bridge phase detail (event → {phase: count}):")
    for r in rows:
        print(f"  {r['event_id']}: {r['bridge_phases']}")


if __name__ == "__main__":
    main()
