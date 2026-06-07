"""Can the winning scoring-regime be predicted from event-level features?

For each event: compute AP@5 under MLP/s1xs3 vs s5-fallback, label = argmax,
and event-level features available at inference. Then 5-fold CV a classifier
and measure the AP@5 obtained by applying the PREDICTED regime per event,
versus the v25 heuristic and the oracle.

If predicted-router AP@5 approaches oracle (and beats CrossEncoder 0.8301),
a cheap event-feature router suffices — no LLM needed. NO LLM calls.
"""
from __future__ import annotations
import json, logging
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from mbridgenet.config import load_config
from mbridgenet.data.dataset import CPHotDataset
from mbridgenet.pipeline import MBridgeNetPipeline
from mbridgenet.evaluation.metrics import ap_at_k

logging.getLogger().setLevel(logging.ERROR)
LOW_S2 = 0.20


def ap5(cands, gt, mode):
    best = {}
    for c in cands:
        if mode == "s5":
            s5 = getattr(c, "s5", 0.0); sc = s5 if s5 > 0 else c.s1
        else:
            sc = c.score_final
        pid = c.post_a.post_id
        if pid not in best or sc > best[pid]:
            best[pid] = sc
    return ap_at_k(sorted(best.items(), key=lambda x: x[1], reverse=True), gt, 5)


def event_features(event, cands):
    posts = event["posts"]
    plats = {p.platform for p in posts}
    top = sorted(cands, key=lambda c: c.s1, reverse=True)[:100]
    s1 = np.array([c.s1 for c in top]); s2 = np.array([c.s2 for c in top])
    s5 = np.array([getattr(c, "s5", 0.0) for c in top])
    # rank disagreement between s1 and s5 over top-50
    t50 = top[:50]
    import scipy.stats as ss
    a = [c.s1 for c in t50]; b = [getattr(c, "s5", 0.0) for c in t50]
    try:
        tau = ss.kendalltau(a, b).correlation
        if tau != tau:  # nan
            tau = 0.0
    except Exception:
        tau = 0.0
    n_weibo = sum(1 for p in posts if p.platform == "weibo")
    return [
        len(posts), len(plats), n_weibo / max(len(posts), 1),
        float(np.median(s2)), float(s1.mean()), float(s1.std()),
        float(s5.mean()), float(s5.std()), float((s5 > 0).mean()), float(tau),
    ]


def main():
    cfg = load_config(None)
    pipe = MBridgeNetPipeline(config=cfg, mlp_checkpoint="checkpoints/mlp_v25_fold2.pt",
                              llm_client=None)
    ds = CPHotDataset(Path("data/cphot/processed/test_real"))
    dd = Path("data/cphot/processed/test_real")

    feats, labels, ap_mlp_l, ap_s5_l, heur_pick = [], [], [], [], []
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
        feats.append(event_features(event, cands))
        labels.append(1 if a5 > am else 0)   # 1 = s5 regime wins
        ap_mlp_l.append(am); ap_s5_l.append(a5)
        top100 = sorted(cands, key=lambda c: c.s1, reverse=True)[:100]
        med = sorted([c.s2 for c in top100])[len(top100) // 2]
        heur_pick.append(1 if med < LOW_S2 else 0)

    X = np.array(feats); y = np.array(labels)
    am = np.array(ap_mlp_l); a5 = np.array(ap_s5_l); hp = np.array(heur_pick)
    n = len(y)
    def route_ap(picks):
        return np.mean(np.where(picks == 1, a5, am))
    print(f"events={n}  (s5-wins={y.sum()}, mlp-wins={n-y.sum()})")
    print(f"  heuristic AP@5 = {route_ap(hp):.4f}")
    print(f"  oracle    AP@5 = {route_ap(y):.4f}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    pred = cross_val_predict(
        GradientBoostingClassifier(random_state=42, n_estimators=150, max_depth=2),
        X, y, cv=skf)
    acc = (pred == y).mean()
    print(f"\n  CV router accuracy = {acc:.3f}  ({(pred==y).sum()}/{n})")
    print(f"  CV-router AP@5     = {route_ap(pred):.4f}")
    print(f"\n  Reference: CrossEncoder 0.8301")
    rap = route_ap(pred)
    print(f"  CV-router {'BEATS CrossEncoder' if rap>0.8301 else 'below CrossEncoder'} "
          f"(routerAP={rap:.4f})")


if __name__ == "__main__":
    main()
