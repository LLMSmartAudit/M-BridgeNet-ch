"""Full 67-event per-event regime breakdown: AP@5 under MLP vs s5, oracle pick.

Emits a LaTeX-ready TSV: event, #bridges, ap_mlp, ap_s5, oracle_regime, oracle_ap.
NO LLM calls.
"""
from __future__ import annotations
import json, logging
from pathlib import Path

from mbridgenet.config import load_config
from mbridgenet.data.dataset import CPHotDataset
from mbridgenet.pipeline import MBridgeNetPipeline
from mbridgenet.evaluation.metrics import ap_at_k

logging.getLogger().setLevel(logging.ERROR)


def ap5(cands, gt, mode):
    best = {}
    for c in cands:
        if mode == "s5":
            s5 = getattr(c, "s5", 0.0); sc = s5 if s5 > 0.0 else c.s1
        else:
            sc = c.score_final
        pid = c.post_a.post_id
        if pid not in best or sc > best[pid]:
            best[pid] = sc
    return ap_at_k(sorted(best.items(), key=lambda x: x[1], reverse=True), gt, 5)


def main():
    cfg = load_config(None)
    pipe = MBridgeNetPipeline(config=cfg, mlp_checkpoint="checkpoints/mlp_v25_fold2.pt",
                              llm_client=None)
    dd = Path("data/cphot/processed/test_real")
    ds = CPHotDataset(dd)

    rows = []
    for event in ds:
        eid = event.get("event_id", ""); gt = {p[0] for p in event["bridge_pairs"]}
        if not gt:
            continue
        s5_map = {}; p5 = dd / f"{eid}_s5.json"
        if p5.exists():
            s5_map = json.loads(p5.read_text())
        cands = pipe.run(event, low_s2_simonly_threshold=0.0,
                         s5_map=s5_map if s5_map else None)
        if not cands:
            continue
        am = ap5(cands, gt, "mlp"); a5 = ap5(cands, gt, "s5")
        reg = "s5" if a5 > am else ("MLP" if am > a5 else "=")
        rows.append((eid, len(gt), am, a5, reg, max(am, a5)))

    rows.sort(key=lambda r: -(r[5] - min(r[2], r[3])))  # sort by oracle gain desc
    import numpy as np
    am = np.array([r[2] for r in rows]); a5 = np.array([r[3] for r in rows])
    orc = np.array([r[5] for r in rows])
    print(f"events={len(rows)}  mean: MLP={am.mean()*100:.2f}  s5={a5.mean()*100:.2f}  "
          f"oracle={orc.mean()*100:.2f}")
    print(f"s5-wins={sum(1 for r in rows if r[4]=='s5')}  "
          f"MLP-wins={sum(1 for r in rows if r[4]=='MLP')}  "
          f"ties={sum(1 for r in rows if r[4]=='=')}")
    print("\nTSV (event\tbridges\tap_mlp\tap_s5\toracle_regime\toracle_ap):")
    for eid, nb, m, s, reg, o in rows:
        print(f"{eid}\t{nb}\t{m*100:.1f}\t{s*100:.1f}\t{reg}\t{o*100:.1f}")


if __name__ == "__main__":
    main()
