"""
Comprehensive MABD analysis script.
Runs the full pipeline with MABD and extracts:
  1. AP@K / F1-Strict@K (overall)
  2. F1-Strict@20 per lifecycle phase
  3. F1-Loose@20 (account-level)
  4. MABD debate quality stats (per debate_outcome)

Usage:
  MBRIDGENET_NO_FAISS=1 OPENAI_API_KEY=sk-... \\
    .venv/bin/python scripts/analyze_mabd.py \\
    --data data/cphot/processed/test_real \\
    --checkpoint checkpoints/mlp_v25_fold2.pt \\
    --mabd-limit 20 \\
    --use-s1s3-scoring
"""
import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import torch
from openai import OpenAI

from mbridgenet.config import load_config
from mbridgenet.data.dataset import CPHotDataset
from mbridgenet.evaluation.metrics import f1_strict_at_k, ap_at_k
from mbridgenet.pipeline import MBridgeNetPipeline, Route
from mbridgenet.schemas import CandidatePair

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


def f1_strict(preds, ground_truth, k):
    top_k = {pid for pid, _ in preds[:k]}
    tp = len(top_k & ground_truth)
    fp = len(top_k - ground_truth)
    fn = len(ground_truth - top_k)
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    return 2 * p * r / (p + r) if (p + r) else 0.0


def f1_loose(preds, ground_truth_accounts, k):
    """
    Account-level F1: a prediction is a TP if post_a.post_id matches any account
    in ground_truth_accounts (ground truth = set of bridge post_a IDs).
    This is already the same as F1-Strict in our setup since ground_truth
    is already the set of bridge post_a IDs.
    
    F1-Loose: A match if the predicted source account has ANY bridge pair,
    regardless of the specific cross-platform pair.
    """
    top_k_ids = {pid for pid, _ in preds[:k]}
    tp = len(top_k_ids & ground_truth_accounts)
    fp = len(top_k_ids - ground_truth_accounts)
    fn = len(ground_truth_accounts - top_k_ids)
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    return 2 * p * r / (p + r) if (p + r) else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mabd-limit", type=int, default=20)
    parser.add_argument("--use-s1s3-scoring", action="store_true", dest="use_s1s3_scoring")
    parser.add_argument("--k", nargs="+", type=int, default=[5, 10, 20, 50])
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_config(Path(args.config) if args.config else None)
    llm_client = OpenAI()

    pipeline = MBridgeNetPipeline(
        config=cfg,
        mlp_checkpoint=args.checkpoint,
        llm_client=llm_client,
    )

    dataset = CPHotDataset(Path(args.data))
    logger.info("Running analysis on %d events, mabd_limit=%d", len(dataset), args.mabd_limit)

    # ─── Global accumulators ───────────────────────────────────────────────
    results_ap = defaultdict(list)
    results_f1 = defaultdict(list)
    f1_loose_vals = []

    # Per-phase
    phase_results = defaultdict(lambda: {"preds": [], "gt": set(), "counts": 0})

    # MABD debate quality
    debate_stats = defaultdict(lambda: {"n": 0, "n_bridge_gt": 0, "confidences": [],
                                        "tp": 0, "fp": 0, "fn": 0, "tn": 0})

    for event in dataset:
        event_id = event.get("event_id", "?")
        logger.info("Processing event: %s", event_id)
        ground_truth = {pair[0] for pair in event["bridge_pairs"]}
        
        # Run pipeline — monkey-patch debater to capture debate records
        captured_debates = []
        # Pipeline stores debater as self.debater (no underscore)
        original_debate = pipeline.debater.debate if pipeline.debater else None

        if original_debate:
            def patched_debate(pair, _orig=original_debate):
                record = _orig(pair)
                if record is not None:
                    captured_debates.append((pair, record))
                return record
            pipeline.debater.debate = patched_debate

        output_pairs = pipeline.run(
            event,
            mabd_limit=args.mabd_limit,
            use_s1s3_scoring=args.use_s1s3_scoring,
        )

        # Restore
        if original_debate:
            pipeline.debater.debate = original_debate

        # Deduplicate by post_a_id (keep highest score_final)
        seen_ids: dict = {}
        for c in output_pairs:
            pid = c.post_a.post_id
            if pid not in seen_ids or c.score_final > seen_ids[pid]:
                seen_ids[pid] = c.score_final
        predictions = sorted(seen_ids.items(), key=lambda x: x[1], reverse=True)

        # Overall AP@K and F1@K
        for k in args.k:
            results_f1[f"F1@{k}"].append(f1_strict(predictions, ground_truth, k))
            results_ap[f"AP@{k}"].append(ap_at_k(predictions, ground_truth, k))
        f1_loose_vals.append(f1_loose(predictions, ground_truth, 20))

        # Per-phase breakdown: use GLOBAL top-K predictions, phase-specific GT
        # This matches the method used in the paper table (global top-20 sorted by
        # score_final; for each phase, compute F1 against that phase's bridge GT).
        phase_gt = defaultdict(set)

        # Build phase GT from scored_pairs
        scored_pairs_by_a = defaultdict(lambda: "unknown")
        for sp in event.get("scored_pairs", []):
            scored_pairs_by_a[sp["post_a_id"]] = sp.get("phase", "unknown")
        for bridge_pair in event["bridge_pairs"]:
            post_a_id = bridge_pair[0]
            phase = scored_pairs_by_a[post_a_id]
            phase_gt[phase].add(post_a_id)

        # Use global `predictions` (already deduped, sorted globally)
        for phase in ["emergence", "diffusion", "peak", "decline"]:
            gt_ph = phase_gt.get(phase, set())
            if gt_ph:  # only count if there's ground truth for this phase
                # Global top-K predictions vs phase-specific GT
                f1_ph = f1_strict(predictions, gt_ph, 20)
                phase_results[phase]["preds"].append(f1_ph)
                phase_results[phase]["counts"] += len(gt_ph)

        # MABD debate quality
        for pair, record in captured_debates:
            outcome = record.debate_outcome
            ds = debate_stats[outcome]
            ds["n"] += 1
            ds["confidences"].append(record.confidence)
            is_bridge_gt = pair.post_a.post_id in ground_truth
            if is_bridge_gt:
                ds["n_bridge_gt"] += 1
            if record.is_bridge and is_bridge_gt:
                ds["tp"] += 1
            elif record.is_bridge and not is_bridge_gt:
                ds["fp"] += 1
            elif not record.is_bridge and is_bridge_gt:
                ds["fn"] += 1
            else:
                ds["tn"] += 1

    # ─── Print results ────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("OVERALL METRICS")
    print("="*60)
    for k in args.k:
        ap_vals = results_ap[f"AP@{k}"]
        f1_vals = results_f1[f"F1@{k}"]
        print(f"AP@{k:2d}: {100*sum(ap_vals)/len(ap_vals):.2f}%  "
              f"F1@{k:2d}: {100*sum(f1_vals)/len(f1_vals):.2f}%")
    print(f"F1-Loose@20: {100*sum(f1_loose_vals)/len(f1_loose_vals):.2f}%")

    print("\n" + "="*60)
    print("PER-PHASE F1-Strict@20")
    print("="*60)
    total_f1 = []
    for phase in ["emergence", "diffusion", "peak", "decline"]:
        vals = phase_results[phase]["preds"]
        cnt = phase_results[phase]["counts"]
        mean_f1 = 100 * sum(vals) / len(vals) if vals else 0.0
        print(f"  {phase:12s}: {mean_f1:.2f}%  (#bridge_pairs_gt={cnt})")
        total_f1.extend(vals)

    print("\n" + "="*60)
    print("MABD DEBATE QUALITY")
    print("="*60)
    total_n = sum(v["n"] for v in debate_stats.values())
    for outcome in ["proposer_won", "challenger_won", "balanced"]:
        ds = debate_stats.get(outcome, {"n": 0, "n_bridge_gt": 0, "confidences": [],
                                         "tp": 0, "fp": 0, "fn": 0, "tn": 0})
        n = ds["n"]
        bridge_pct = 100 * ds["n_bridge_gt"] / n if n else 0
        avg_conf = sum(ds["confidences"]) / len(ds["confidences"]) if ds["confidences"] else 0
        tp, fp, fn = ds["tp"], ds["fp"], ds["fn"]
        p = tp / (tp + fp) if (tp + fp) else 0
        r = tp / (tp + fn) if (tp + fn) else 0
        f1 = 2*p*r/(p+r) if (p+r) else 0
        print(f"  {outcome:20s}: n={n:3d}  bridge%={bridge_pct:5.1f}%  "
              f"avg_conf={avg_conf:.3f}  F1={100*f1:.1f}%")
    print(f"  {'All MABD':20s}: n={total_n}")

    # Save as JSON
    output = {
        "overall": {f"AP@{k}": round(100*sum(results_ap[f"AP@{k}"])/max(len(results_ap[f"AP@{k}"]),1), 2)
                   for k in args.k} | {f"F1@{k}": round(100*sum(results_f1[f"F1@{k}"])/max(len(results_f1[f"F1@{k}"]),1), 2)
                   for k in args.k} | {"F1-Loose@20": round(100*sum(f1_loose_vals)/max(len(f1_loose_vals),1), 2)},
        "per_phase": {phase: round(100*sum(v["preds"])/max(len(v["preds"]),1), 2)
                     for phase, v in phase_results.items()},
        "debate_quality": {
            outcome: {
                "n": debate_stats[outcome]["n"],
                "bridge_pct": round(100*debate_stats[outcome]["n_bridge_gt"]/max(debate_stats[outcome]["n"],1), 1),
                "avg_confidence": round(sum(debate_stats[outcome]["confidences"])/max(len(debate_stats[outcome]["confidences"]),1), 3),
            }
            for outcome in ["proposer_won", "challenger_won", "balanced"]
        },
    }
    out_path = Path("logs/mabd_analysis.json")
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
