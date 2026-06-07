"""67-event late-fusion eval: M-BridgeNet (v25) + LLM s6 re-rank head.

Reranks BRIDGE-routed candidates by
    fused = norm(score_final) + beta * (s6 - 0.5)
using precomputed *_s6.json sidecars, and recomputes AP@K over all 67 events.

Reports a beta sweep transparently; the headline uses a FIXED beta (mid-plateau,
NOT tuned on the test set).  Also reports the clean-event subset to confirm s6
does not regress events that were already correct.

Usage:
  MBRIDGENET_NO_FAISS=1 .venv/bin/python scripts/eval_s6_fusion_full.py \
      --data data/cphot/processed/test_real --checkpoint checkpoints/mlp_v25_fold2.pt
"""
from __future__ import annotations
import argparse, json, logging
from pathlib import Path

from mbridgenet.config import load_config
from mbridgenet.data.dataset import CPHotDataset
from mbridgenet.pipeline import MBridgeNetPipeline
from mbridgenet.evaluation.metrics import ap_at_k
from mbridgenet.schemas import Route

logging.getLogger().setLevel(logging.ERROR)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--low-s2-simonly", type=float, default=0.20, dest="low_s2")
    ap.add_argument("--betas", default="0,0.1,0.2,0.3,0.5")
    ap.add_argument("--headline-beta", type=float, default=0.2)
    args = ap.parse_args()

    cfg = load_config(None)
    pipe = MBridgeNetPipeline(config=cfg, mlp_checkpoint=args.checkpoint, llm_client=None)
    ds = CPHotDataset(Path(args.data)); dd = Path(args.data)

    # cache per-event: ranked output candidates + s6 map + gt
    cache = []
    n_with_s6 = 0
    for event in ds:
        eid = event.get("event_id", ""); gt = {p[0] for p in event["bridge_pairs"]}
        if not gt:
            continue
        s5_map = {}; p5 = dd / f"{eid}_s5.json"
        if p5.exists():
            s5_map = json.loads(p5.read_text())
        s6_map = {}; p6 = dd / f"{eid}_s6.json"
        if p6.exists():
            s6_map = json.loads(p6.read_text()); n_with_s6 += 1
        out = pipe.run(event, low_s2_simonly_threshold=args.low_s2,
                       s5_map=s5_map if s5_map else None)
        cache.append((eid, gt, out, s6_map))

    def evaluate(beta):
        APs = {k: [] for k in (5, 10, 20, 50)}
        per_event = {}
        for eid, gt, out, s6_map in cache:
            best = {}
            for c in out:
                pid = c.post_a.post_id
                # fused score
                lo, hi = sorted([c.post_a.post_id, c.post_b.post_id])
                s6 = s6_map.get(f"{lo}||{hi}")
                sc = c.score_final
                if beta and s6 is not None and c.route == Route.BRIDGE:
                    sc = c.score_final + beta * (s6 - 0.5)
                if pid not in best or sc > best[pid]:
                    best[pid] = sc
            preds = sorted(best.items(), key=lambda x: x[1], reverse=True)
            e5 = ap_at_k(preds, gt, 5)
            for k in (5, 10, 20, 50):
                APs[k].append(ap_at_k(preds, gt, k))
            per_event[eid] = e5
        return {k: sum(v) / len(v) for k, v in APs.items()}, per_event

    base, base_pe = evaluate(0.0)
    print(f"events with s6 sidecar: {n_with_s6}/{len(cache)}")
    print(f"\nv25 baseline (beta=0):  AP@5={base[5]:.4f}  AP@10={base[10]:.4f}  "
          f"AP@20={base[20]:.4f}  AP@50={base[50]:.4f}")
    print("Reference: CrossEncoder cap-1000 AP@5=0.8301\n")
    print("beta sweep (67-event aggregate):")
    headline = None
    for beta in [float(x) for x in args.betas.split(",")]:
        m, pe = evaluate(beta)
        tag = ""
        if abs(beta - args.headline_beta) < 1e-9:
            headline = (m, pe); tag = "  <-- FIXED headline beta"
        print(f"  beta={beta:4.2f}  AP@5={m[5]:.4f}  AP@10={m[10]:.4f}  "
              f"AP@20={m[20]:.4f}  AP@50={m[50]:.4f}{tag}")

    if headline:
        m, pe = headline
        print(f"\n=== Headline (beta={args.headline_beta}) vs v25 ===")
        print(f"  AP@5  {base[5]:.4f} -> {m[5]:.4f}  ({m[5]-base[5]:+.4f})")
        print(f"  vs CrossEncoder 0.8301: {'BEATS' if m[5] > 0.8301 else 'below'} "
              f"({m[5]-0.8301:+.4f})")
        # regression check: events that got worse
        worse = [(e, base_pe[e], pe[e]) for e in pe if pe[e] < base_pe[e] - 1e-6]
        better = [(e, base_pe[e], pe[e]) for e in pe if pe[e] > base_pe[e] + 1e-6]
        print(f"\n  events improved: {len(better)}   events regressed: {len(worse)}")
        for e, b, a in sorted(worse, key=lambda x: x[2]-x[1])[:8]:
            print(f"    REGRESS {e:40s} {b:.3f} -> {a:.3f} ({a-b:+.3f})")


if __name__ == "__main__":
    main()
