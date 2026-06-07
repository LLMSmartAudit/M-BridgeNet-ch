#!/usr/bin/env python3
"""
Compute per-phase F1-Strict@20 using precomputed scored_pairs.
Runs both the full model and the lifecycle ablation checkpoint
to produce a comparison table.

Run from M-BridgeNet/:
    .venv/bin/python scripts/per_phase_baseline.py \
        --data data/cphot/processed/test_real \
        --checkpoint checkpoints/mlp_v25_fold2.pt \
        --baseline-checkpoint checkpoints/mlp_abl_lifecycle_fold5.pt
"""
from __future__ import annotations
import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from mbridgenet.stage2.scorer import LifecycleMLP, PHASE2IDX


ACTIVE_PHASES = ["emergence", "diffusion", "peak", "decline"]


def f1_at_k(pids: list[str], scores: list[float],
            gt: set[str], k: int) -> float:
    ranked = sorted(zip(pids, scores), key=lambda x: x[1], reverse=True)
    top_k = {pid for pid, _ in ranked[:k]}
    tp = len(top_k & gt)
    fp = len(top_k - gt)
    fn = len(gt - top_k)
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    return 2 * p * r / (p + r) if (p + r) else 0.0


def eval_per_phase(data_dir: Path, mlp: LifecycleMLP,
                   use_s1s3: bool = True) -> tuple[dict[str, float], dict[str, int]]:
    phase_f1: dict[str, list[float]] = defaultdict(list)
    phase_gt_counts: dict[str, int] = defaultdict(int)

    for f in sorted(data_dir.glob("*.json")):
        data = json.loads(f.read_text(encoding="utf-8"))
        sp = [p for p in data.get("scored_pairs", [])
              if p.get("label", -1) != -1]
        if not sp:
            continue

        signals = torch.tensor([[p["s1"], p["s2"], p["s3"]] for p in sp],
                                dtype=torch.float32)
        phases = torch.tensor([PHASE2IDX.get(p.get("phase", "pre_event"), 0)
                                for p in sp], dtype=torch.long)

        with torch.no_grad():
            scores = mlp(signals, phases).numpy()

        if use_s1s3:
            final_scores = [float(sc) * p["s1"] * p["s3"]
                            for sc, p in zip(scores, sp)]
        else:
            final_scores = [float(sc) for sc in scores]

        pids = [p["post_a_id"] for p in sp]

        # Phase-specific GT vs global top-20
        phase_gt: dict[str, set[str]] = defaultdict(set)
        for pair in data["bridge_pairs"]:
            pa_id = pair[0]
            phase = next(
                (p["phase"] for p in sp if p["post_a_id"] == pa_id), "unknown"
            )
            if phase in ACTIVE_PHASES:
                phase_gt[phase].add(pa_id)

        for phase in ACTIVE_PHASES:
            gt_ph = phase_gt.get(phase, set())
            if gt_ph:
                f1 = f1_at_k(pids, final_scores, gt_ph, 20)
                phase_f1[phase].append(f1)
                phase_gt_counts[phase] += len(gt_ph)

    results = {
        phase: round(100 * sum(vals) / len(vals), 1) if vals else 0.0
        for phase, vals in phase_f1.items()
    }
    return results, dict(phase_gt_counts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--baseline-checkpoint", required=True)
    args = parser.parse_args()

    mlp_full = LifecycleMLP()
    mlp_full.load_state_dict(torch.load(args.checkpoint, map_location="cpu",
                                        weights_only=True))
    mlp_full.eval()

    mlp_abl = LifecycleMLP()
    mlp_abl.load_state_dict(torch.load(args.baseline_checkpoint,
                                       map_location="cpu", weights_only=True))
    mlp_abl.eval()

    full_results, gt_counts = eval_per_phase(Path(args.data), mlp_full)
    abl_results, _ = eval_per_phase(Path(args.data), mlp_abl)

    print(f"\n{'Phase':<12} {'#GT':>6} {'M-BridgeNet':>12} "
          f"{'w/o Lifecycle':>14} {'Delta':>7}")
    print("-" * 55)

    total_gt = 0
    weighted_full = 0.0
    weighted_abl = 0.0

    for phase in ACTIVE_PHASES:
        full = full_results.get(phase, 0.0)
        abl = abl_results.get(phase, 0.0)
        cnt = gt_counts.get(phase, 0)
        delta = full - abl
        sign = "+" if delta >= 0 else ""
        print(f"{phase:<12} {cnt:>6} {full:>12.1f} {abl:>14.1f} "
              f"{sign}{delta:>6.1f}")
        total_gt += cnt
        weighted_full += full * cnt
        weighted_abl += abl * cnt

    if total_gt:
        ov_full = weighted_full / total_gt
        ov_abl = weighted_abl / total_gt
        print("-" * 55)
        print(f"{'Overall':<12} {total_gt:>6} {ov_full:>12.1f} "
              f"{ov_abl:>14.1f} {ov_full-ov_abl:>+7.1f}")


if __name__ == "__main__":
    main()
