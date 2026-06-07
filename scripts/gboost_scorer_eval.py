"""Train a gradient-boosted Stage-2 scorer on (s1,s2,s3,s5) and eval on 67 events.

Mirrors v25's pipeline EXACTLY except the scorer for non-fallback events:
  - candidate generation: run_stage1_stage2 (low_s2=0 so no internal fallback)
  - low-s2 fallback (top-100 median s2 < 0.20): rank by s5 if available else s1
    (identical to v25)
  - otherwise: rank by GBoost P(bridge | s1,s2,s3,s5)   [v25 uses the MLP here]
  - dedup by post_a (max score), then AP@K

This isolates "GBoost scorer vs LifecycleMLP scorer" on identical features and
the identical fallback, for a fair comparison to v25 (82.29%) and CrossEncoder (83.01%).
"""
from __future__ import annotations
import glob, json, logging
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

from mbridgenet.config import load_config
from mbridgenet.data.dataset import CPHotDataset
from mbridgenet.pipeline import MBridgeNetPipeline
from mbridgenet.evaluation.metrics import ap_at_k

logging.getLogger().setLevel(logging.ERROR)
LOW_S2 = 0.20


def load_train(dirs):
    X, y = [], []
    for d in dirs:
        for f in glob.glob(f"{d}/*.json"):
            if any(s in f for s in ("_s5", "_s6", "_emb")):
                continue
            ev = json.load(open(f)); eid = ev.get("event_id", "")
            s5 = {}; p5 = Path(d) / f"{eid}_s5.json"
            if p5.exists():
                s5 = json.loads(p5.read_text())
            for p in ev.get("scored_pairs", []):
                if p.get("label", -1) == -1:
                    continue
                lo, hi = sorted([p["post_a_id"], p["post_b_id"]])
                X.append([p["s1"], p["s2"], p["s3"], float(s5.get(f"{lo}||{hi}", 0.0))])
                y.append(int(p["label"]))
    return np.array(X), np.array(y)


def main():
    cfg = load_config(None)
    # Train GBoost on the 43-event training split only (honest — no test leakage)
    Xtr, ytr = load_train(["data/cphot/processed/train_all"])
    print(f"Train: {len(ytr)} pairs ({ytr.sum()} pos)")
    gb = GradientBoostingClassifier(random_state=42, n_estimators=200, max_depth=3,
                                    learning_rate=0.05)
    # Upweight the HARD negatives (high-s1 non-bridges) so the global objective
    # cannot ignore the 2.6% hard region. Tests whether forcing attention there
    # lifts ranking. NEG_W set via env (default 1 = no upweight).
    import os
    neg_w = float(os.environ.get("HARD_NEG_W", "1.0"))
    sw = np.ones(len(ytr))
    if neg_w != 1.0:
        hard_neg = (ytr == 0) & (Xtr[:, 0] >= 0.85)
        sw[hard_neg] = neg_w
        print(f"Upweighting {hard_neg.sum()} high-s1 negatives x{neg_w}")
    gb.fit(Xtr, ytr, sample_weight=sw)
    print("GBoost trained. Feature importances (s1,s2,s3,s5):",
          [round(v, 3) for v in gb.feature_importances_])

    pipe = MBridgeNetPipeline(config=cfg, mlp_checkpoint="checkpoints/mlp_v25_fold2.pt",
                              llm_client=None)
    ds = CPHotDataset(Path("data/cphot/processed/test_real"))
    dd = Path("data/cphot/processed/test_real")

    APs = {k: [] for k in (5, 10, 20, 50)}
    n_fallback = 0
    for event in ds:
        eid = event.get("event_id", ""); gt = {p[0] for p in event["bridge_pairs"]}
        if not gt:
            continue
        s5_map = {}; p5 = dd / f"{eid}_s5.json"
        if p5.exists():
            s5_map = json.loads(p5.read_text())
        cands = pipe.run_stage1_stage2(
            event["posts"], event["hourly_volumes"],
            low_s2_simonly_threshold=0.0,  # disable internal fallback; we replicate it
            s5_map=s5_map if s5_map else None)
        if not cands:
            continue
        # replicate v25 low-s2 fallback (top-100 median s2)
        top100 = sorted(cands, key=lambda c: c.s1, reverse=True)[:100]
        med_s2 = sorted([c.s2 for c in top100])[len(top100) // 2]
        if med_s2 < LOW_S2:
            n_fallback += 1
            for c in cands:
                s5 = getattr(c, "s5", 0.0)
                c.score_final = s5 if s5 > 0.0 else c.s1   # v25 fallback ranking
        else:
            feats = np.array([[c.s1, c.s2, c.s3, getattr(c, "s5", 0.0)] for c in cands])
            proba = gb.predict_proba(feats)[:, 1]
            for c, p in zip(cands, proba):
                c.score_final = float(p)
        best = {}
        for c in cands:
            pid = c.post_a.post_id
            if pid not in best or c.score_final > best[pid]:
                best[pid] = c.score_final
        preds = sorted(best.items(), key=lambda x: x[1], reverse=True)
        for k in (5, 10, 20, 50):
            APs[k].append(ap_at_k(preds, gt, k))

    m = {k: sum(v) / len(v) for k, v in APs.items()}
    print(f"\nfallback events: {n_fallback}/{len(APs[5])}")
    print("\n=== GBoost Stage-2 scorer (67-event test_real) ===")
    print(f"  AP@5={m[5]:.4f}  AP@10={m[10]:.4f}  AP@20={m[20]:.4f}  AP@50={m[50]:.4f}")
    print(f"\n  v25 (LifecycleMLP):   AP@5=0.8229")
    print(f"  CrossEncoder cap1000: AP@5=0.8301")
    print(f"  GBoost:               AP@5={m[5]:.4f}  "
          f"({'BEATS CrossEncoder' if m[5] > 0.8301 else 'beats v25' if m[5] > 0.8229 else 'below v25'})")


if __name__ == "__main__":
    main()
