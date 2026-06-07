"""Test the claim: MABD recovers bridges that are 'crucial yet difficult to uncover'.

'Difficult to uncover' = lands in the borderline MABD zone (Stage 1+2 uncertain),
   with LOWER semantic similarity s1 than the easy direct-BRIDGE bridges.
'Crucial'             = HIGHER structural pivotality — betweenness centrality s4
   and/or cross-platform route rarity s3 — than the easy bridges.

We characterise every TRUE-bridge source node by the zone its best candidate
lands in (BRIDGE / MABD / DISCARD) and compare signal distributions across zones.
NO LLM calls — purely the Stage 1+2 signals that define the routing.

Usage:
  MBRIDGENET_NO_FAISS=1 .venv/bin/python scripts/analyze_mabd_crucial.py \
      --data data/cphot/processed/test_real --checkpoint checkpoints/mlp_v25_fold2.pt
"""
from __future__ import annotations
import argparse
import json
import logging
import statistics as st
from pathlib import Path

from mbridgenet.config import load_config
from mbridgenet.data.dataset import CPHotDataset
from mbridgenet.pipeline import MBridgeNetPipeline
from mbridgenet.schemas import Route

logging.getLogger().setLevel(logging.ERROR)


def summ(xs: list[float]) -> str:
    if not xs:
        return "   n=0"
    return (f"n={len(xs):4d}  mean={st.mean(xs):.3f}  "
            f"median={st.median(xs):.3f}  "
            f"p25={sorted(xs)[len(xs)//4]:.3f}  p75={sorted(xs)[3*len(xs)//4]:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--low-s2-simonly", type=float, default=0.0, dest="low_s2")
    ap.add_argument("--out", default="logs/mabd_crucial_analysis.json")
    args = ap.parse_args()

    cfg = load_config(None)
    pipe = MBridgeNetPipeline(config=cfg, mlp_checkpoint=args.checkpoint, llm_client=None)
    ds = CPHotDataset(Path(args.data))
    data_dir = Path(args.data)

    # signal samples per zone (one entry per true-bridge source node)
    zones = {z: {"s1": [], "s3": [], "s4": [], "score_S": []}
             for z in ("BRIDGE", "MABD", "DISCARD")}

    for event in ds:
        eid = event.get("event_id", "")
        gt = {p[0] for p in event["bridge_pairs"]}
        if not gt:
            continue
        s5_map = {}
        p5 = data_dir / f"{eid}_s5.json"
        if p5.exists():
            s5_map = json.loads(p5.read_text())
        cands = pipe.run_stage1_stage2(
            event["posts"], event["hourly_volumes"],
            low_s2_simonly_threshold=args.low_s2,
            s5_map=s5_map if s5_map else None)
        # best candidate per true-bridge source node
        best: dict = {}
        for c in cands:
            if c.post_a.post_id in gt:
                pid = c.post_a.post_id
                if pid not in best or c.score_S > best[pid].score_S:
                    best[pid] = c
        for c in best.values():
            z = ("BRIDGE" if c.route == Route.BRIDGE
                 else "MABD" if c.route == Route.MABD else "DISCARD")
            zones[z]["s1"].append(c.s1)
            zones[z]["s3"].append(c.s3)
            zones[z]["s4"].append(c.s4)
            zones[z]["score_S"].append(c.score_S)

    print("=" * 78)
    print("TRUE-BRIDGE SIGNAL PROFILE BY ROUTING ZONE (67-event test_real)")
    print("  BRIDGE  = easy / direct  (S >= tau_high)")
    print("  MABD    = overlooked, debate-targeted (tau_low <= S < tau_high)")
    print("  DISCARD = mis-routed     (S < tau_low)")
    print("=" * 78)
    for sig, label in [("s1", "s1  cosine similarity   (LOWER = harder to uncover)"),
                       ("s4", "s4  betweenness central. (HIGHER = more pivotal)"),
                       ("s3", "s3  migration rarity     (HIGHER = rarer route)"),
                       ("score_S", "S   composite MLP score")]:
        print(f"\n{label}")
        for z in ("BRIDGE", "MABD", "DISCARD"):
            print(f"  {z:8s} {summ(zones[z][sig])}")

    # Decisive contrasts: MABD vs BRIDGE
    def delta(sig):
        b, m = zones["BRIDGE"][sig], zones["MABD"][sig]
        if not b or not m:
            return None
        return st.mean(m) - st.mean(b)

    print("\n" + "=" * 78)
    print("DECISIVE CONTRAST  (MABD-zone bridges vs direct-BRIDGE bridges)")
    print("=" * 78)
    d_s1, d_s4, d_s3 = delta("s1"), delta("s4"), delta("s3")
    print(f"  Δs1 (cosine)      = {d_s1:+.3f}   "
          f"{'✓ overlooked bridges ARE harder (lower cosine)' if d_s1 < 0 else '✗ not lower'}")
    print(f"  Δs4 (betweenness) = {d_s4:+.3f}   "
          f"{'✓ overlooked bridges ARE more pivotal (higher BC)' if d_s4 > 0 else '✗ not higher'}")
    print(f"  Δs3 (rarity)      = {d_s3:+.3f}   "
          f"{'✓ overlooked bridges cross rarer routes' if d_s3 > 0 else '✗ not rarer'}")

    out = {z: {k: {"n": len(v), "mean": (st.mean(v) if v else None),
                   "median": (st.median(v) if v else None)}
               for k, v in d.items()} for z, d in zones.items()}
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nSaved → {args.out}")


if __name__ == "__main__":
    main()
