#!/usr/bin/env python3
"""
Phase-label sensitivity analysis.
Randomly flips X% of phase labels in scored_pairs and measures
how much F1-Strict@20 degrades, averaged over 5 seeds.

Run from M-BridgeNet/:
    .venv/bin/python scripts/phase_sensitivity.py \
        --data data/cphot/processed/test_real \
        --checkpoint checkpoints/mlp_v25_fold2.pt \
        --noise-levels 0.0 0.1 0.2 0.3 0.5
"""
from __future__ import annotations
import argparse
import json
import random
from pathlib import Path

import torch
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from mbridgenet.stage2.scorer import LifecycleMLP, PHASE2IDX

ACTIVE_PHASES = list(PHASE2IDX.keys())  # ['pre_event','emergence','diffusion','peak','decline']


def f1_at_k(pids: list[str], scores: list[float], gt: set[str], k: int) -> float:
    ranked = sorted(zip(pids, scores), key=lambda x: x[1], reverse=True)
    top_k = {pid for pid, _ in ranked[:k]}
    tp = len(top_k & gt)
    fp = len(top_k - gt)
    fn = len(gt - top_k)
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    return 2 * p * r / (p + r) if (p + r) else 0.0


def run_eval(dataset_dir: Path, mlp: LifecycleMLP,
             noise: float, seed: int, use_s1s3: bool = True) -> float:
    """Return macro-avg F1@20 across events with `noise` fraction of phases flipped."""
    rng = random.Random(seed)
    f1_vals = []
    for f in sorted(dataset_dir.glob("*.json")):
        data = json.loads(f.read_text(encoding="utf-8"))
        sp = [p for p in data.get("scored_pairs", []) if p.get("label", -1) != -1]
        if not sp:
            continue
        gt = {pair[0] for pair in data["bridge_pairs"]}

        phases = []
        for p in sp:
            orig = p.get("phase", "pre_event")
            if rng.random() < noise:
                orig = rng.choice(ACTIVE_PHASES)  # random flip
            phases.append(PHASE2IDX.get(orig, 0))

        signals = torch.tensor([[p["s1"], p["s2"], p["s3"]] for p in sp],
                                dtype=torch.float32)
        phase_t = torch.tensor(phases, dtype=torch.long)

        with torch.no_grad():
            scores = mlp(signals, phase_t).numpy()

        if use_s1s3:
            final_scores = [float(sc) * p["s1"] * p["s3"]
                            for sc, p in zip(scores, sp)]
        else:
            final_scores = [float(sc) for sc in scores]

        pids = [p["post_a_id"] for p in sp]
        f1_vals.append(f1_at_k(pids, final_scores, gt, 20))

    return 100 * sum(f1_vals) / len(f1_vals) if f1_vals else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--noise-levels", nargs="+", type=float,
                        default=[0.0, 0.1, 0.2, 0.3, 0.5])
    parser.add_argument("--n-seeds", type=int, default=5)
    args = parser.parse_args()

    mlp = LifecycleMLP()
    mlp.load_state_dict(torch.load(args.checkpoint, map_location="cpu",
                                   weights_only=True))
    mlp.eval()

    print(f"\n{'Noise':>8}  {'F1@20 (mean)':>13}  {'±std':>6}  {'Delta':>7}")
    print("-" * 45)
    baseline = None
    for noise in args.noise_levels:
        vals = [run_eval(Path(args.data), mlp, noise, seed=s)
                for s in range(args.n_seeds)]
        mean = sum(vals) / len(vals)
        std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
        if baseline is None:
            baseline = mean
        delta = mean - baseline
        sign = "+" if delta >= 0 else ""
        print(f"{noise:>8.0%}  {mean:>13.2f}%  {std:>6.2f}  {sign}{delta:>6.2f}")


if __name__ == "__main__":
    main()
