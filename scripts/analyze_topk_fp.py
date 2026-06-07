"""Validate the LLM-as-top-K-causal-verifier opportunity.

AP@K is decided by the top-K ranked source nodes. We characterise the
FALSE POSITIVES that appear in the top-K (non-bridge source nodes ranked high)
and ask the decisive question:

  Are these FPs high-similarity / high-s5 coincidental co-occurrences that
  CrossEncoder ALSO ranks high (and therefore cannot demote) — yet have weak
  temporal precedence (near-zero / wrong-direction time gap) that an LLM
  reasoning about causality COULD exploit?

If yes  → an LLM top-K verifier can beat CrossEncoder (fixes its blind spot).
If FPs already have low s5 → no CrossEncoder-beating room.

Compares top-K FPs vs top-K TPs on s1, s5, s2, |Δt| (hours).  NO LLM calls.
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

logging.getLogger().setLevel(logging.ERROR)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--low-s2-simonly", type=float, default=0.20, dest="low_s2")
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--out", default="logs/topk_fp_analysis.json")
    args = ap.parse_args()

    cfg = load_config(None)
    pipe = MBridgeNetPipeline(config=cfg, mlp_checkpoint=args.checkpoint, llm_client=None)
    ds = CPHotDataset(Path(args.data))
    data_dir = Path(args.data)

    tp = {"s1": [], "s5": [], "s2": [], "dt": []}
    fp = {"s1": [], "s5": [], "s2": [], "dt": []}
    n_events = 0
    events_with_topk_fp = 0
    total_topk_fp = 0

    for event in ds:
        eid = event.get("event_id", "")
        gt = {p[0] for p in event["bridge_pairs"]}
        if not gt:
            continue
        s5_map = {}
        p5 = data_dir / f"{eid}_s5.json"
        if p5.exists():
            s5_map = json.loads(p5.read_text())
        out_pairs = pipe.run(
            event, low_s2_simonly_threshold=args.low_s2,
            s5_map=s5_map if s5_map else None)
        # dedup by source node, keep best score_final (mirrors evaluate.py)
        best = {}
        for c in out_pairs:
            pid = c.post_a.post_id
            if pid not in best or c.score_final > best[pid].score_final:
                best[pid] = c
        ranked = sorted(best.values(), key=lambda c: c.score_final, reverse=True)
        topk = ranked[:args.K]
        n_events += 1
        ev_fp = 0
        for c in topk:
            dt = abs((c.post_b.timestamp - c.post_a.timestamp).total_seconds() / 3600)
            bucket = tp if c.post_a.post_id in gt else fp
            bucket["s1"].append(c.s1)
            bucket["s5"].append(getattr(c, "s5", 0.0))
            bucket["s2"].append(c.s2)
            bucket["dt"].append(dt)
            if c.post_a.post_id not in gt:
                ev_fp += 1
        total_topk_fp += ev_fp
        if ev_fp:
            events_with_topk_fp += 1

    def line(d, k):
        xs = d[k]
        if not xs:
            return "n=0"
        return f"n={len(xs):4d} mean={st.mean(xs):.3f} median={st.median(xs):.3f}"

    print("=" * 76)
    print(f"TOP-{args.K} RANKING: true positives vs false positives (67-event test_real)")
    print(f"  {events_with_topk_fp}/{n_events} events have ≥1 FP in top-{args.K}; "
          f"{total_topk_fp} FP nodes total")
    print("=" * 76)
    for k, lbl in [("s1", "s1  cosine similarity"),
                   ("s5", "s5  CrossEncoder score  (HIGH on FP = CrossEncoder blind)"),
                   ("s2", "s2  temporal-gap score"),
                   ("dt", "|Δt| hours between A and B")]:
        print(f"\n{lbl}")
        print(f"  TP  {line(tp, k)}")
        print(f"  FP  {line(fp, k)}")

    print("\n" + "=" * 76)
    print("DECISIVE TEST")
    print("=" * 76)
    if fp["s5"] and tp["s5"]:
        fp_s5, tp_s5 = st.mean(fp["s5"]), st.mean(tp["s5"])
        print(f"  FP mean s5 = {fp_s5:.3f}  vs  TP mean s5 = {tp_s5:.3f}")
        if fp_s5 > 0.5:
            print("  ✓ Top-K FPs have HIGH s5 → CrossEncoder also ranks them high "
                  "→ LLM causal verifier can beat CrossEncoder by demoting them")
        else:
            print("  ✗ Top-K FPs have LOW s5 → CrossEncoder/s5 already handles them; "
                  "limited CrossEncoder-beating room")
    if fp["dt"] and tp["dt"]:
        print(f"  FP median |Δt| = {st.median(fp['dt']):.1f}h  "
              f"vs TP median |Δt| = {st.median(tp['dt']):.1f}h  "
              f"(temporal-precedence tell for the LLM)")

    Path(args.out).write_text(json.dumps(
        {"K": args.K, "events_with_topk_fp": events_with_topk_fp, "n_events": n_events,
         "total_topk_fp": total_topk_fp,
         "TP": {k: (st.mean(v) if v else None) for k, v in tp.items()},
         "FP": {k: (st.mean(v) if v else None) for k, v in fp.items()}}, indent=2))
    print(f"\nSaved → {args.out}")


if __name__ == "__main__":
    main()
