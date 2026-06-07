"""Oracle late-fusion test for s6 (zero new LLM calls; reuses logs/probe_s6.json).

If fusing s6 into the ranking — even with the weight tuned to MAXIMISE AP@5 on
the test events (an oracle upper bound) — cannot beat the baseline, then s6 has
no ranking value and the full MLP retrain is futile.

For each of the 26 FP-heavy events: re-run Stage 1+2 to get the ranked source
nodes (by score_final) + labels, attach probed s6 to the top-10 (matched by
rank order, verified via s1), then re-rank the top-10 by
  fused = score_final_norm + beta * (s6 - 0.5)
and recompute mean AP@5 / AP@10 over a grid of beta. Report the best.
"""
from __future__ import annotations
import json, logging
from pathlib import Path

from mbridgenet.config import load_config
from mbridgenet.data.dataset import CPHotDataset
from mbridgenet.pipeline import MBridgeNetPipeline
from mbridgenet.evaluation.metrics import ap_at_k

logging.getLogger().setLevel(logging.ERROR)

probe = json.load(open("logs/probe_s6.json"))
by_event = {}
for r in probe:
    by_event.setdefault(r["event"], []).append(r)   # preserves top-K order

cfg = load_config(None)
pipe = MBridgeNetPipeline(config=cfg, mlp_checkpoint="checkpoints/mlp_v25_fold2.pt",
                          llm_client=None)
ds = CPHotDataset(Path("data/cphot/processed/test_topk_subset"))
dd = Path("data/cphot/processed/test_topk_subset")

# Build per-event: ranked (post_a_id, score_final, is_tp), and s6 for top-10
events = []
for event in ds:
    eid = event.get("event_id", ""); gt = {p[0] for p in event["bridge_pairs"]}
    if eid not in by_event:
        continue
    s5_map = {}; p5 = dd / f"{eid}_s5.json"
    if p5.exists():
        s5_map = json.loads(p5.read_text())
    out = pipe.run(event, low_s2_simonly_threshold=0.20, s5_map=s5_map if s5_map else None)
    best = {}
    for c in out:
        pid = c.post_a.post_id
        if pid not in best or c.score_final > best[pid].score_final:
            best[pid] = c
    ranked = sorted(best.values(), key=lambda c: c.score_final, reverse=True)
    rows = by_event[eid]
    # attach s6 to top-len(rows) by rank, verify s1 alignment
    s6_by_pid = {}
    for i, r in enumerate(rows):
        if i < len(ranked) and abs(ranked[i].s1 - r["s1"]) < 1e-3:
            s6_by_pid[ranked[i].post_a.post_id] = r["s6"]
    events.append((eid, gt, ranked, s6_by_pid))

# normalise score_final per event to [0,1] for stable fusion
def ap5_ap10(beta):
    a5 = []; a10 = []
    for eid, gt, ranked, s6map in events:
        scores = [c.score_final for c in ranked]
        lo, hi = min(scores), max(scores)
        rng = (hi - lo) or 1.0
        fused = []
        for c in ranked:
            base = (c.score_final - lo) / rng
            s6 = s6map.get(c.post_a.post_id)
            f = base + beta * (s6 - 0.5) if s6 is not None else base
            fused.append((c.post_a.post_id, f))
        fused.sort(key=lambda x: x[1], reverse=True)
        a5.append(ap_at_k(fused, gt, 5))
        a10.append(ap_at_k(fused, gt, 10))
    return sum(a5) / len(a5), sum(a10) / len(a10)

base5, base10 = ap5_ap10(0.0)
print(f"baseline (beta=0):   AP@5={base5:.4f}  AP@10={base10:.4f}")
print("\nbeta sweep (oracle upper bound — beta tuned on the test set itself):")
best = (base5, 0.0, base10)
for beta in [0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0]:
    a5, a10 = ap5_ap10(beta)
    flag = "  <-- best" if a5 > best[0] else ""
    if a5 > best[0]:
        best = (a5, beta, a10)
    print(f"  beta={beta:4.2f}   AP@5={a5:.4f}  AP@10={a10:.4f}{flag}")
print(f"\nBEST oracle fusion: beta={best[1]}  AP@5={best[0]:.4f} "
      f"(baseline {base5:.4f}, delta {best[0]-base5:+.4f})")
print("If best oracle delta <= ~0 → s6 has no ranking value → skip the retrain.")
