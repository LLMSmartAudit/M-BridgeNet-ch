"""67-event per-phase AP@K for tab:phase27 (v25, 5 phases incl. pre_event).

Methodology (matches the original 27-event table):
  - run the full v25 pipeline per event (score_final = adaptive s1xs3 / s1 / fallback);
  - group candidates by their assigned lifecycle phase (c.phase);
  - within each phase, dedup by source post (max score_final) and rank;
  - per-phase ground truth = bridge source posts whose annotated pair phase = that phase;
  - AP@K per phase = mean over events that contain >=1 such GT bridge;
  - Macro avg = mean of the per-phase AP values.
"""
from __future__ import annotations
import json, logging
from collections import defaultdict
from pathlib import Path

from mbridgenet.config import load_config
from mbridgenet.data.dataset import CPHotDataset
from mbridgenet.pipeline import MBridgeNetPipeline
from mbridgenet.evaluation.metrics import ap_at_k

logging.getLogger().setLevel(logging.ERROR)
PHASES = ["pre_event", "emergence", "diffusion", "peak", "decline"]
KS = [5, 10, 20, 50]


def main():
    cfg = load_config(None)
    pipe = MBridgeNetPipeline(config=cfg, mlp_checkpoint="checkpoints/mlp_v25_fold2.pt",
                              llm_client=None)
    dd = Path("data/cphot/processed/test_real")
    ds = CPHotDataset(dd)

    # per phase: list of per-event AP@K dicts
    phase_ap = {ph: {k: [] for k in KS} for ph in PHASES}
    phase_ev = defaultdict(int)

    for event in ds:
        eid = event.get("event_id", "")
        # source-post phase from annotated scored_pairs (label==1)
        src_phase = {}
        for p in event.get("scored_pairs", []):
            if p.get("label") == 1:
                src_phase.setdefault(p["post_a_id"], p.get("phase", "pre_event"))
        gt_all = {p[0] for p in event["bridge_pairs"]}
        if not gt_all:
            continue
        s5_map = {}; p5 = dd / f"{eid}_s5.json"
        if p5.exists():
            s5_map = json.loads(p5.read_text())
        cands = pipe.run(event, low_s2_simonly_threshold=0.20,
                         s5_map=s5_map if s5_map else None)
        if not cands:
            continue
        # group candidates by phase, dedup by source (max score_final)
        by_phase = defaultdict(dict)
        for c in cands:
            ph = c.phase or "pre_event"
            pid = c.post_a.post_id
            sc = c.score_final
            if pid not in by_phase[ph] or sc > by_phase[ph][pid]:
                by_phase[ph][pid] = sc
        for ph in PHASES:
            gt_ph = {s for s in gt_all if src_phase.get(s) == ph}
            if not gt_ph:
                continue
            phase_ev[ph] += 1
            preds = sorted(by_phase.get(ph, {}).items(), key=lambda x: x[1], reverse=True)
            for k in KS:
                phase_ap[ph][k].append(ap_at_k(preds, gt_ph, k))

    print(f"{'Phase':<11}" + "".join(f"{'AP@'+str(k):>8}" for k in KS) + f"{'#Ev':>6}")
    macro = {k: [] for k in KS}
    for ph in PHASES:
        n = phase_ev[ph]
        if n == 0:
            print(f"{ph:<11}" + "".join(f"{'—':>8}" for k in KS) + f"{0:>6}")
            continue
        vals = {k: 100*sum(phase_ap[ph][k])/len(phase_ap[ph][k]) for k in KS}
        for k in KS: macro[k].append(vals[k])
        print(f"{ph:<11}" + "".join(f"{vals[k]:>8.1f}" for k in KS) + f"{n:>6}")
    print(f"{'Macro avg':<11}" + "".join(f"{sum(macro[k])/len(macro[k]):>8.1f}" for k in KS)
          + f"{len(list(__import__('glob').glob(str(dd/'*.json'))))}")  # placeholder ev count
    # accurate total event count
    tot = sum(1 for e in CPHotDataset(dd) if {p[0] for p in e['bridge_pairs']})
    print(f"(total events with GT: {tot})")


if __name__ == "__main__":
    main()
