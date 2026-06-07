"""Test improvement D: does ranking the MABD queue by |s5 - score_S| surface
true bridges better than the current top-N-by-score_S selection?

For each event, collect ALL MABD-zone candidates (bridge + non-bridge) and,
for mabd-limit N in {10, 20}, count how many TRUE bridges fall in the top-N
under (a) score_S ordering [current] vs (b) |s5-score_S| ordering [improv D].
"""
from __future__ import annotations
import json
from pathlib import Path

from mbridgenet.config import load_config
from mbridgenet.data.dataset import CPHotDataset
from mbridgenet.pipeline import MBridgeNetPipeline
from mbridgenet.schemas import Route

DATA = "data/cphot/processed/test_real"
CKPT = "checkpoints/mlp_v25_fold2.pt"

cfg = load_config(None)
pipe = MBridgeNetPipeline(config=cfg, mlp_checkpoint=CKPT, llm_client=None)
ds = CPHotDataset(Path(DATA))
data_dir = Path(DATA)

import logging
logging.getLogger().setLevel(logging.ERROR)

totals = {10: {"cur": 0, "D": 0}, 20: {"cur": 0, "D": 0}}
total_mabd_bridges = 0

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
        low_s2_simonly_threshold=0.20, s5_map=s5_map if s5_map else None)
    mabd = [c for c in cands if c.route == Route.MABD]
    if not mabd:
        continue
    n_br = sum(1 for c in mabd if c.post_a.post_id in gt)
    total_mabd_bridges += n_br
    by_score = sorted(mabd, key=lambda c: c.score_S, reverse=True)
    by_dis = sorted(mabd, key=lambda c: abs(getattr(c, "s5", 0.0) - c.score_S),
                    reverse=True)
    for N in (10, 20):
        cur = sum(1 for c in by_score[:N] if c.post_a.post_id in gt)
        dd = sum(1 for c in by_dis[:N] if c.post_a.post_id in gt)
        totals[N]["cur"] += cur
        totals[N]["D"] += dd

print(f"Total true bridges in MABD zone (across events): {total_mabd_bridges}")
print(f"\nTrue bridges captured in top-N of MABD queue:")
print(f"  {'N':>4} {'by score_S (current)':>22} {'by |s5-S| (improv D)':>22}")
for N in (10, 20):
    print(f"  {N:>4} {totals[N]['cur']:>22} {totals[N]['D']:>22}")
