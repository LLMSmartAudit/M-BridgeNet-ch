"""Oracle event-router ceiling — pre-check for Direction B (LLM event router).

v25 routes each event to one of two scoring regimes by a heuristic
(median-s2 of top-100 candidates < 0.20  ->  s5-fallback; else MLP/s1xs3).

This script computes, per event, AP@5 under BOTH regimes, then compares:
  - always-MLP        : every event uses MLP/s1xs3
  - always-s5         : every event ranked by s5 (CrossEncoder) else s1
  - heuristic (v25)   : median-s2 rule picks the regime
  - ORACLE            : pick the better regime per event (upper bound)

oracle - heuristic = the most any router (LLM included) could gain.  NO LLM calls.
"""
from __future__ import annotations
import json, logging
from pathlib import Path

from mbridgenet.config import load_config
from mbridgenet.data.dataset import CPHotDataset
from mbridgenet.pipeline import MBridgeNetPipeline
from mbridgenet.evaluation.metrics import ap_at_k
from mbridgenet.schemas import Route

logging.getLogger().setLevel(logging.ERROR)
LOW_S2 = 0.20


def ap5_from_candidates(cands, gt, mode):
    best = {}
    for c in cands:
        if mode == "s5":
            s5 = getattr(c, "s5", 0.0)
            sc = s5 if s5 > 0.0 else c.s1
        else:  # "mlp" → use score_final from the v25 MLP/s1xs3 path
            sc = c.score_final
        pid = c.post_a.post_id
        if pid not in best or sc > best[pid]:
            best[pid] = sc
    preds = sorted(best.items(), key=lambda x: x[1], reverse=True)
    return ap_at_k(preds, gt, 5)


def main():
    cfg = load_config(None)
    pipe = MBridgeNetPipeline(config=cfg, mlp_checkpoint="checkpoints/mlp_v25_fold2.pt",
                              llm_client=None)
    ds = CPHotDataset(Path("data/cphot/processed/test_real"))
    dd = Path("data/cphot/processed/test_real")

    a_mlp = []; a_s5 = []; a_heur = []; a_oracle = []
    heur_wrong = 0
    rows = []
    for event in ds:
        eid = event.get("event_id", ""); gt = {p[0] for p in event["bridge_pairs"]}
        if not gt:
            continue
        s5_map = {}; p5 = dd / f"{eid}_s5.json"
        if p5.exists():
            s5_map = json.loads(p5.read_text())
        # MLP/s1xs3 regime: run with fallback OFF so score_final is the MLP path
        cands = pipe.run(event, low_s2_simonly_threshold=0.0,
                         s5_map=s5_map if s5_map else None)
        if not cands:
            continue
        ap_mlp = ap5_from_candidates(cands, gt, "mlp")
        ap_s5 = ap5_from_candidates(cands, gt, "s5")
        # heuristic decision (median-s2 top-100)
        top100 = sorted(cands, key=lambda c: c.s1, reverse=True)[:100]
        med = sorted([c.s2 for c in top100])[len(top100) // 2]
        pick_s5 = med < LOW_S2
        ap_heur = ap_s5 if pick_s5 else ap_mlp
        ap_orc = max(ap_mlp, ap_s5)
        a_mlp.append(ap_mlp); a_s5.append(ap_s5); a_heur.append(ap_heur); a_oracle.append(ap_orc)
        if abs(ap_orc - ap_heur) > 1e-6:
            heur_wrong += 1
            rows.append((eid, med, pick_s5, ap_mlp, ap_s5, ap_orc - ap_heur))

    n = len(a_mlp)
    print(f"events: {n}")
    print(f"  always-MLP       AP@5 = {sum(a_mlp)/n:.4f}")
    print(f"  always-s5        AP@5 = {sum(a_s5)/n:.4f}")
    print(f"  heuristic (v25)  AP@5 = {sum(a_heur)/n:.4f}")
    print(f"  ORACLE router    AP@5 = {sum(a_oracle)/n:.4f}")
    print(f"\n  oracle - heuristic = +{sum(a_oracle)/n - sum(a_heur)/n:.4f}  "
          f"(max any router incl. LLM could gain)")
    print(f"  heuristic mis-routes {heur_wrong}/{n} events")
    print(f"\n  events the LLM router would need to fix (event, med_s2, heur_picked_s5, "
          f"ap_mlp, ap_s5, gain):")
    for eid, med, ps5, am, a5, g in sorted(rows, key=lambda x: -x[5])[:15]:
        print(f"    {eid:38s} med_s2={med:.3f} heur_s5={ps5}  "
              f"mlp={am:.2f} s5={a5:.2f}  +{g:.3f}")


if __name__ == "__main__":
    main()
