"""tau_high sensitivity sweep for M-BridgeNet.

Sweeps tau_high ∈ {0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70} on the full
27-event test_real split using mlp_v18_fold5.pt + adaptive s1×s3 scoring.
tau_low is kept at tau_high - 0.15 (fixed-width MABD band).
No LLM calls — pure Stage 1+2 evaluation.

Usage:
    cd /path/to/M-BridgeNet
    MBRIDGENET_NO_FAISS=1 .venv/bin/python scripts/tau_high_sweep.py \\
        --data data/cphot/processed/test_real \\
        --checkpoint checkpoints/mlp_v18_fold5.pt \\
        --k 5 20 50
"""

import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

TAU_HIGH_VALUES = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
PAPER_TAU_HIGH  = 0.50   # current paper value


def run_event_with_tau(pipeline, event: dict, tau_high: float,
                       k_values: List[int]) -> Dict[str, float]:
    """Run one event with patched tau_high/tau_low, return AP@k dict."""
    from mbridgenet.pipeline import MBridgeNetPipeline

    tau_low = max(0.0, tau_high - 0.15)

    # Monkey-patch the config on the pipeline's stage2 config
    original_th = pipeline.cfg.stage2.tau_high
    original_tl = pipeline.cfg.stage2.tau_low
    pipeline.cfg.stage2.tau_high = tau_high
    pipeline.cfg.stage2.tau_low  = tau_low
    try:
        output_pairs = pipeline.run(event, use_s1s3_scoring=True)
    finally:
        pipeline.cfg.stage2.tau_high = original_th
        pipeline.cfg.stage2.tau_low  = original_tl

    ground_truth = {pair[0] for pair in event["bridge_pairs"]}
    seen_ids: dict = {}
    for cp in output_pairs:
        pid = cp.post_a.post_id
        if pid not in seen_ids or cp.score_final > seen_ids[pid]:
            seen_ids[pid] = cp.score_final
    predictions = sorted(seen_ids.items(), key=lambda x: x[1], reverse=True)

    from mbridgenet.evaluation.metrics import ap_at_k
    return {f"AP@{k}": ap_at_k(predictions, ground_truth, k) for k in k_values}


def mean_ap(results: List[Dict[str, float]], k: int) -> float:
    vals = [r[f"AP@{k}"] for r in results]
    return sum(vals) / len(vals) if vals else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",       required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--k",          nargs="+", type=int, default=[5, 20, 50])
    args = parser.parse_args()

    from mbridgenet.config import load_config
    from mbridgenet.data.dataset import CPHotDataset
    from mbridgenet.pipeline import MBridgeNetPipeline

    cfg = load_config(None)
    pipeline = MBridgeNetPipeline(config=cfg, mlp_checkpoint=args.checkpoint)

    dataset = list(CPHotDataset(Path(args.data)))
    logger.info("Loaded %d events from %s", len(dataset), args.data)

    rows = []
    for tau_h in TAU_HIGH_VALUES:
        tau_l = max(0.0, tau_h - 0.15)
        event_results = []
        for event in dataset:
            r = run_event_with_tau(pipeline, event, tau_h, args.k)
            event_results.append(r)

        row = {"tau_high": tau_h, "tau_low": round(tau_l, 2)}
        for k in args.k:
            row[f"AP@{k}"] = round(mean_ap(event_results, k), 4)
        rows.append(row)

        marker = "  ← paper default" if abs(tau_h - PAPER_TAU_HIGH) < 1e-6 else ""
        logger.info(
            "tau_high=%.2f  tau_low=%.2f  "
            "AP@5=%.2f%%  AP@20=%.2f%%  AP@50=%.2f%%%s",
            tau_h, tau_l,
            row["AP@5"] * 100, row["AP@20"] * 100, row["AP@50"] * 100,
            marker,
        )

    # ── Pretty table ─────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"{'tau_high':>10}  {'tau_low':>8}  ", end="")
    for k in args.k:
        print(f"{'AP@'+str(k):>10}", end="")
    print()
    print("-" * 70)
    for row in rows:
        marker = " *" if abs(row["tau_high"] - PAPER_TAU_HIGH) < 1e-6 else "  "
        print(f"{row['tau_high']:>10.2f}  {row['tau_low']:>8.2f}  ", end="")
        for k in args.k:
            print(f"{row[f'AP@{k}']*100:>9.2f}%", end="")
        print(marker)
    print("  * = paper default (tau_high=0.50)")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    out_path = Path("logs/tau_high_sweep_results.json")
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(
        {"sweep": rows, "paper_tau_high": PAPER_TAU_HIGH, "k_values": args.k},
        indent=2
    ))
    logger.info("Results saved → %s", out_path)


if __name__ == "__main__":
    main()
