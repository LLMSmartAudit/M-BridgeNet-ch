"""Offline MABD routing-ceiling analysis (NO LLM calls).

For every test event, runs Stage 1+2 only and records, per candidate:
  s1, s2, s3, s5, score_S, route, is_true_bridge (post_a in ground truth).

Then it answers the decisive question for improvements E/C/D:
  "Where do the true-bridge SOURCE NODES land — BRIDGE, MABD, or DISCARD zone?"

A routing change can only help if it moves true bridges that are currently
mis-routed (in DISCARD, or buried) into the MABD zone where debate can rescue
them. If 0 true bridges are recoverable, no debate-quality change matters.

Usage:
  MBRIDGENET_NO_FAISS=1 .venv/bin/python scripts/analyze_mabd_routing.py \
      --data data/cphot/processed/test_real \
      --checkpoint checkpoints/mlp_v25_fold2.pt \
      --low-s2-simonly 0.20
"""
from __future__ import annotations
import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

from mbridgenet.config import load_config
from mbridgenet.data.dataset import CPHotDataset
from mbridgenet.pipeline import MBridgeNetPipeline
from mbridgenet.schemas import Route

logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger(__name__)

# v25 calibrated routing band (config defaults)
TAU_HIGH = 0.72
TAU_LOW = 0.55


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--low-s2-simonly", type=float, default=0.0, dest="low_s2")
    ap.add_argument("--out", default="logs/mabd_routing_analysis.json")
    args = ap.parse_args()

    cfg = load_config(None)
    pipeline = MBridgeNetPipeline(
        config=cfg, mlp_checkpoint=args.checkpoint, llm_client=None,
    )
    dataset = CPHotDataset(Path(args.data))
    data_dir = Path(args.data)

    # Aggregate counters
    agg = {
        "events": 0,
        "total_bridge_nodes": 0,        # distinct true-bridge source nodes per event, summed
        "bridge_in_BRIDGE": 0,          # best candidate of a true bridge node lands in BRIDGE
        "bridge_in_MABD": 0,
        "bridge_in_DISCARD": 0,
        "bridge_no_candidate": 0,       # bridge node never produced a Stage-1 candidate
    }
    per_event = []

    for event in dataset:
        event_id = event.get("event_id", "")
        ground_truth = {pair[0] for pair in event["bridge_pairs"]}  # source nodes
        if not ground_truth:
            continue

        # load s5 sidecar
        s5_map = {}
        s5_path = data_dir / f"{event_id}_s5.json"
        if s5_path.exists():
            s5_map = json.loads(s5_path.read_text())

        candidates = pipeline.run_stage1_stage2(
            event["posts"], event["hourly_volumes"],
            low_s2_simonly_threshold=args.low_s2,
            s5_map=s5_map if s5_map else None,
        )

        # best candidate per source node (highest score_S)
        best: dict = {}
        for c in candidates:
            pid = c.post_a.post_id
            if pid not in best or c.score_S > best[pid].score_S:
                best[pid] = c

        # zone tallies for this event's true-bridge source nodes
        ev = {
            "event_id": event_id,
            "n_bridge_nodes": len(ground_truth),
            "in_BRIDGE": 0, "in_MABD": 0, "in_DISCARD": 0, "no_cand": 0,
            # for improvement-C/D/E diagnostics on DISCARD bridges:
            "discard_bridge_details": [],   # [{score_S, s1, s2, s3, s5}]
            "mabd_bridge_details": [],
        }
        for node in ground_truth:
            if node not in best:
                ev["no_cand"] += 1
                continue
            c = best[node]
            d = {"score_S": round(c.score_S, 4), "s1": round(c.s1, 4),
                 "s2": round(c.s2, 4), "s3": round(c.s3, 4),
                 "s5": round(getattr(c, "s5", 0.0), 4)}
            if c.route == Route.BRIDGE:
                ev["in_BRIDGE"] += 1
            elif c.route == Route.MABD:
                ev["in_MABD"] += 1
                ev["mabd_bridge_details"].append(d)
            else:  # DISCARD
                ev["in_DISCARD"] += 1
                ev["discard_bridge_details"].append(d)

        agg["events"] += 1
        agg["total_bridge_nodes"] += ev["n_bridge_nodes"]
        agg["bridge_in_BRIDGE"] += ev["in_BRIDGE"]
        agg["bridge_in_MABD"] += ev["in_MABD"]
        agg["bridge_in_DISCARD"] += ev["in_DISCARD"]
        agg["bridge_no_candidate"] += ev["no_cand"]
        per_event.append(ev)

        print(f"{event_id:42s} bridges={ev['n_bridge_nodes']:3d}  "
              f"BRIDGE={ev['in_BRIDGE']:3d}  MABD={ev['in_MABD']:2d}  "
              f"DISCARD={ev['in_DISCARD']:3d}  no_cand={ev['no_cand']:2d}")

    print("\n" + "=" * 70)
    print("AGGREGATE (67-event test_real)")
    print("=" * 70)
    tb = agg["total_bridge_nodes"]
    print(f"  Total true-bridge source nodes : {tb}")
    print(f"  → in BRIDGE zone  (S≥{TAU_HIGH}) : {agg['bridge_in_BRIDGE']:4d}  "
          f"({100*agg['bridge_in_BRIDGE']/tb:.1f}%)")
    print(f"  → in MABD  zone  ({TAU_LOW}–{TAU_HIGH}) : {agg['bridge_in_MABD']:4d}  "
          f"({100*agg['bridge_in_MABD']/tb:.1f}%)  ← current MABD opportunity")
    print(f"  → in DISCARD     (S<{TAU_LOW})  : {agg['bridge_in_DISCARD']:4d}  "
          f"({100*agg['bridge_in_DISCARD']/tb:.1f}%)  ← E/C target pool")
    print(f"  → no Stage-1 candidate          : {agg['bridge_no_candidate']:4d}  "
          f"({100*agg['bridge_no_candidate']/tb:.1f}%)  ← unrecoverable")

    # Improvement E ceiling: DISCARD bridges with high cosine (rescuable by cosine top-N)
    discard_high_s1 = sum(
        1 for ev in per_event for d in ev["discard_bridge_details"] if d["s1"] >= 0.85
    )
    discard_total = sum(len(ev["discard_bridge_details"]) for ev in per_event)
    print(f"\n  [E ceiling] DISCARD bridges with s1≥0.85 (cosine-rescuable): "
          f"{discard_high_s1}/{discard_total}")

    # Improvement C ceiling: DISCARD bridges just below tau_low (within 0.10)
    near_band = sum(
        1 for ev in per_event for d in ev["discard_bridge_details"]
        if TAU_LOW - 0.10 <= d["score_S"] < TAU_LOW
    )
    print(f"  [C ceiling] DISCARD bridges in [{TAU_LOW-0.10:.2f},{TAU_LOW}) "
          f"(widen-band rescuable): {near_band}/{discard_total}")

    # Improvement D diagnostic: among MABD-zone bridges, |s5 - score_S| distribution
    mabd_disagreements = [
        abs(d["s5"] - d["score_S"]) for ev in per_event for d in ev["mabd_bridge_details"]
    ]
    if mabd_disagreements:
        avg_dis = sum(mabd_disagreements) / len(mabd_disagreements)
        print(f"  [D diagnostic] MABD-zone bridges avg |s5-S| = {avg_dis:.3f} "
              f"(n={len(mabd_disagreements)})")
    else:
        print(f"  [D diagnostic] No true bridges in MABD zone — D has no signal to route on")

    out = {"aggregate": agg, "per_event": per_event,
           "ceilings": {"E_cosine_rescuable": discard_high_s1,
                        "C_band_rescuable": near_band,
                        "discard_total": discard_total}}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\nSaved → {args.out}")


if __name__ == "__main__":
    main()
