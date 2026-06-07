from __future__ import annotations
import json, logging
from pathlib import Path
import numpy as np
from mbridgenet.config import load_config
from mbridgenet.data.dataset import CPHotDataset
from mbridgenet.pipeline import MBridgeNetPipeline
from mbridgenet.evaluation.metrics import ap_at_k
logging.getLogger().setLevel(logging.ERROR)
KS = [5, 10, 20, 50]
def ap(cands, gt, key, k):
    best = {}
    for c in cands:
        sc = key(c); pid = c.post_a.post_id
        if pid not in best or sc > best[pid]:
            best[pid] = sc
    return ap_at_k(sorted(best.items(), key=lambda x: x[1], reverse=True), gt, k)
def main():
    cfg = load_config(None)
    pipe = MBridgeNetPipeline(config=cfg, mlp_checkpoint="checkpoints/mlp_v25_fold2.pt", llm_client=None)
    dd = Path("data/cphot/processed/test_real"); ds = CPHotDataset(dd)
    rows = []
    for event in ds:
        eid = event.get("event_id", ""); gt = {p[0] for p in event["bridge_pairs"]}
        if not gt: continue
        s5_map = {}; p5 = dd / f"{eid}_s5.json"
        if p5.exists(): s5_map = json.loads(p5.read_text())
        cands = pipe.run(event, low_s2_simonly_threshold=0.20, s5_map=s5_map if s5_map else None)
        if not cands: continue
        m = {k: ap(cands, gt, lambda c: c.score_final, k) for k in KS}
        s = {k: ap(cands, gt, lambda c: c.s1, k) for k in KS}
        rows.append((eid, len(gt), m, s))
    rows.sort(key=lambda r: -r[2][5]); n = len(rows)
    m5 = np.array([r[2][5] for r in rows]); m50 = np.array([r[2][50] for r in rows])
    s5 = np.array([r[3][5] for r in rows]); s50 = np.array([r[3][50] for r in rows])
    rng = np.random.default_rng(42)
    def ci(a):
        bs = [a[rng.integers(0, n, n)].mean() for _ in range(10000)]
        return round(np.percentile(bs,2.5)*100,1), round(np.percentile(bs,97.5)*100,1)
    print(f"events={n}")
    print(f"MBN mean AP@5={m5.mean()*100:.2f} AP@50={m50.mean()*100:.2f} (canon 82.29/66.79)")
    print(f"Sim mean AP@5={s5.mean()*100:.2f} AP@50={s50.mean()*100:.2f} (canon 79.58/64.52)")
    print(f"CI MBN AP@5={ci(m5)} AP@50={ci(m50)} | Sim AP@5={ci(s5)} AP@50={ci(s50)}")
    print("TSV")
    for eid, nb, m, s in rows:
        print(f"{eid}\t{nb}\t{m[5]*100:.1f}\t{m[10]*100:.1f}\t{m[20]*100:.1f}\t{m[50]*100:.1f}\t{s[5]*100:.1f}\t{s[50]*100:.1f}")
if __name__ == "__main__": main()
