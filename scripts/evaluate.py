"""Evaluate M-BridgeNet on CPHot test split and print result table.

Usage (full pipeline, requires BGE):
    python scripts/evaluate.py \\
        --data data/cphot/processed \\
        --checkpoint checkpoints/mlp_fold1.pt \\
        --config configs/default.yaml

Usage (precomputed signals, no BGE needed):
    python scripts/evaluate.py \\
        --data data/cphot/processed \\
        --checkpoint checkpoints/mlp_fold1.pt \\
        --config configs/default.yaml \\
        --precomputed

Results are appended to logs/eval_results.jsonl automatically.
"""
import argparse
import json
import logging
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore

from mbridgenet.config import load_config
from mbridgenet.data.dataset import CPHotDataset
from mbridgenet.evaluation.metrics import f1_strict_at_k, ap_at_k
from mbridgenet.stage2.scorer import LifecycleMLP, PHASE2IDX

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

LOG_PATH = Path("logs/eval_results.jsonl")


def _save_log(record: dict) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info("Run logged → %s", LOG_PATH)


def _load_mlp(checkpoint: str | None) -> LifecycleMLP:
    mlp = LifecycleMLP()
    if checkpoint:
        mlp.load_state_dict(
            torch.load(checkpoint, map_location="cpu", weights_only=True)
        )
    mlp.eval()
    return mlp


def run_precomputed(args) -> None:
    """Evaluate MLP directly on precomputed scored_pairs — no BGE needed."""
    cfg = load_config(Path(args.config) if args.config else None)
    mlp = _load_mlp(args.checkpoint)
    dataset = CPHotDataset(Path(args.data))
    logger.info("Precomputed eval on %d events", len(dataset))

    all_tp = all_fp = all_fn = all_tn = 0

    for event in dataset:
        pairs = [p for p in event.get("scored_pairs", []) if p["label"] != -1]
        if not pairs:
            continue
        signals = torch.tensor(
            [[p["s1"], p["s2"], p["s3"], p.get("s4", 0.5)] for p in pairs],
            dtype=torch.float32,
        )
        phases = torch.tensor(
            [PHASE2IDX.get(p["phase"], 0) for p in pairs],
            dtype=torch.long,
        )
        with torch.no_grad():
            probs = mlp(signals, phases).squeeze()
        if probs.dim() == 0:
            probs = probs.unsqueeze(0)

        labels = torch.tensor([p["label"] for p in pairs], dtype=torch.float32)
        preds  = (probs >= 0.5).float()

        all_tp += int(((preds == 1) & (labels == 1)).sum())
        all_fp += int(((preds == 1) & (labels == 0)).sum())
        all_fn += int(((preds == 0) & (labels == 1)).sum())
        all_tn += int(((preds == 0) & (labels == 0)).sum())

    total = all_tp + all_fp + all_fn + all_tn
    precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) else 0.0
    recall    = all_tp / (all_tp + all_fn) if (all_tp + all_fn) else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) else 0.0)
    accuracy  = (all_tp + all_tn) / total if total else 0.0

    print(f"\nPrecomputed MLP evaluation (threshold=0.5, {total} pairs):")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1:        {f1:.4f}")
    print(f"  TP={all_tp}  FP={all_fp}  FN={all_fn}  TN={all_tn}")

    _save_log({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": "precomputed",
        "data": args.data,
        "checkpoint": args.checkpoint,
        "n_events": len(dataset),
        "n_pairs": total,
        "metrics": {
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        },
        "confusion": {"tp": all_tp, "fp": all_fp, "fn": all_fn, "tn": all_tn},
    })


def run_full_pipeline(args) -> None:
    """Full pipeline evaluation including BGE Stage-1 embedding."""
    from mbridgenet.pipeline import MBridgeNetPipeline

    llm_client = None
    if args.openai:
        if OpenAI is None:
            raise RuntimeError("openai package not installed — run: pip install openai")
        llm_client = OpenAI()
        logger.info("Stage-3 MABD enabled")

    cfg = load_config(Path(args.config) if args.config else None)
    pipeline = MBridgeNetPipeline(
        config=cfg,
        mlp_checkpoint=args.checkpoint,
        llm_client=llm_client,
        embedder_model=args.embedder,
        drop_signal=getattr(args, "drop_signal", ""),
        uniform_weights=getattr(args, "uniform_weights", False),
        uniform_phase_window=getattr(args, "uniform_phase_window", False),
    )
    if args.pair_encoder:
        pipeline.load_pair_encoder(args.pair_encoder)
        train_dir = args.train or "data/cphot/processed/train_all"
        logger.info("Calibrating PairEncoder thresholds from %s ...", train_dir)
        tau_high, tau_low = pipeline.calibrate_pair_encoder_thresholds(train_dir)
        logger.info("Using PairEncoder scorer  tau_high=%.3f  tau_low=%.3f", tau_high, tau_low)
    if getattr(args, "logreg", None):
        pipeline.load_logreg(args.logreg)
        train_dir = args.train or "data/cphot/processed/train_all"
        logger.info("Calibrating LR-NoPhase thresholds from %s ...", train_dir)
        tau_high, tau_low = pipeline.calibrate_logreg_thresholds(train_dir)
        logger.info("Using LR-NoPhase scorer  tau_high=%.3f  tau_low=%.3f", tau_high, tau_low)
    dataset = CPHotDataset(Path(args.data))
    logger.info("Full pipeline eval on %d events, k=%s", len(dataset), args.k)

    results = defaultdict(list)
    per_event_records: list[dict] = []
    data_dir = Path(args.data)
    for i, event in enumerate(dataset):
        ground_truth = {pair[0] for pair in event["bridge_pairs"]}

        # Load precomputed s5 (CrossEncoder) sidecar if present
        event_id = event.get("event_id", "")
        s5_map: dict = {}
        s5_path = data_dir / f"{event_id}_s5.json"
        if s5_path.exists():
            import json as _json
            s5_map = _json.loads(s5_path.read_text())

        output_pairs = pipeline.run(
            event,
            mabd_limit=args.mabd_limit,
            rerank_n=args.rerank_n,
            rerank_final_n=getattr(args, "rerank_final_n", None),
            rescue_discard_n=getattr(args, "rescue_discard_n", None),
            per_bucket_min_mabd=args.per_bucket_min_mabd,
            use_s1s3_scoring=args.use_s1s3_scoring,
            always_s1s3=getattr(args, "always_s1s3", False),
            simonly=getattr(args, "simonly", False),
            tco=getattr(args, "tco", False),
            s1_floor=getattr(args, "s1_floor", 0.0),
            low_s2_simonly_threshold=getattr(args, "low_s2_simonly_threshold", 0.0),
            score_bridge_by_s1=getattr(args, "score_bridge_by_s1", False),
            min_bridge_fallback=getattr(args, "min_bridge_fallback", 0),
            s5_map=s5_map if s5_map else None,
        )
        # Deduplicate by post_a_id — keep highest score_final per source node
        seen_ids: dict = {}
        for c in output_pairs:
            pid = c.post_a.post_id
            if pid not in seen_ids or c.score_final > seen_ids[pid]:
                seen_ids[pid] = c.score_final
        predictions = sorted(seen_ids.items(), key=lambda x: x[1], reverse=True)
        event_ap = {}
        for k in args.k:
            f1v = f1_strict_at_k(predictions, ground_truth, k)
            apv = ap_at_k(predictions, ground_truth, k)
            results[f"F1-Strict@{k}"].append(f1v)
            results[f"AP@{k}"].append(apv)
            event_ap[f"AP@{k}"] = round(apv, 4)
        eid = event.get("event_id", f"event_{i}")
        n_bridges = len(ground_truth)
        per_event_records.append({"event_id": eid, "n_bridges": n_bridges, **event_ap})
        logger.info("  [%d/%d] %-35s bridges=%-4d AP@5=%.3f AP@50=%.3f",
                    i + 1, len(dataset), eid, n_bridges,
                    event_ap.get("AP@5", 0), event_ap.get("AP@50", 0))
        if (i + 1) % 5 == 0:
            logger.info("  processed %d/%d events", i + 1, len(dataset))

    import os as _os
    if _os.environ.get("DUMP_PER_EVENT"):
        Path(_os.environ["DUMP_PER_EVENT"]).write_text(
            json.dumps(per_event_records, ensure_ascii=False, indent=0))

    print(f"\n{'Metric':<20} {'Mean':>8}  {'Min':>8}  {'Max':>8}")
    print("-" * 48)
    summary = {}
    for metric in sorted(results):
        vals = results[metric]
        mean = sum(vals) / len(vals) if vals else 0.0
        summary[metric] = {"mean": round(mean, 4), "min": round(min(vals), 4), "max": round(max(vals), 4)}
        print(f"{metric:<20} {mean:>8.4f}  {min(vals):>8.4f}  {max(vals):>8.4f}")

    _save_log({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": "full_pipeline",
        "stage3_mabd": args.openai,
        "mabd_limit": args.mabd_limit,
        "rerank_n": args.rerank_n,
        "rerank_final_n": getattr(args, "rerank_final_n", None),
        "rescue_discard_n": getattr(args, "rescue_discard_n", None),
        "per_bucket_min_mabd": args.per_bucket_min_mabd,
        "use_s1s3_scoring": args.use_s1s3_scoring,
        "always_s1s3": getattr(args, "always_s1s3", False),
        "simonly": getattr(args, "simonly", False),
        "tco": getattr(args, "tco", False),
        "s1_floor": getattr(args, "s1_floor", 0.0),
        "low_s2_simonly_threshold": getattr(args, "low_s2_simonly_threshold", 0.0),
        "score_bridge_by_s1": getattr(args, "score_bridge_by_s1", False),
        "min_bridge_fallback": getattr(args, "min_bridge_fallback", 0),
        "logreg": getattr(args, "logreg", None),
        "data": args.data,
        "checkpoint": args.checkpoint,
        "k_values": args.k,
        "n_events": len(dataset),
        "tau_high": pipeline._pe_tau_high if args.pair_encoder and pipeline._pe_tau_high else cfg.stage2.tau_high,
        "tau_low":  pipeline._pe_tau_low  if args.pair_encoder and pipeline._pe_tau_low  else cfg.stage2.tau_low,
        "tau_fine": cfg.stage2.tau_fine,
        "pair_encoder": args.pair_encoder,
        "llm_model": cfg.stage3.llm_model if args.openai else None,
        "metrics": summary,
        "per_event_metrics": per_event_records,
    })


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate M-BridgeNet on a CPHot split"
    )
    parser.add_argument("--data",        required=True)
    parser.add_argument("--checkpoint",  default=None)
    parser.add_argument("--embedder",    default=None)
    parser.add_argument("--k",           nargs="+", type=int, default=[5, 10, 20])
    parser.add_argument("--config",      default=None)
    parser.add_argument("--openai",      action="store_true")
    parser.add_argument("--mabd-limit",  type=int, default=None,
                        help="Max MABD pairs per event (top by score_S); None = no limit")
    parser.add_argument("--rerank-n",    type=int, default=None, dest="rerank_n",
                        help="Apply MABD to top-N candidates by s1 sim instead of MLP score zone. "
                             "Requires --openai. Typical values: 30-100.")
    parser.add_argument("--rerank-final-n", type=int, default=None, dest="rerank_final_n",
                        help="Plan A: Apply MABD to top-K candidates ranked by Stage-1+2 "
                             "score_final (adaptive s1×s3 scoring). Puts debate on the exact "
                             "candidates that determine AP@K. Requires --openai. "
                             "Typical values: 10-30.")
    parser.add_argument("--rescue-discard-n", type=int, default=None, dest="rescue_discard_n",
                        help="Plan B: Apply MABD to top-N DISCARD candidates by cosine sim. "
                             "Rescues true bridges the MLP wrongly rejected (structural null fix). "
                             "Requires --openai. Stacks with other MABD modes. "
                             "Typical values: 10-30.")
    parser.add_argument("--pair-encoder", default=None, dest="pair_encoder",
                        help="Path to PairEncoder checkpoint. If set, replaces LifecycleMLP in Stage 2.")
    parser.add_argument("--train", default=None,
                        help="Training data dir for PairEncoder tau auto-calibration. "
                             "Defaults to data/cphot/processed/train_all")
    parser.add_argument("--use-s1s3-scoring", action="store_true",
                        dest="use_s1s3_scoring",
                        help="Use s1*s3 (cosine_sim × rarity) as score_final for direct-BRIDGE "
                             "pairs instead of MLP score_S. Fixes minority-bucket events "
                             "(e.g. suzhou Weibo×Zhihu) where MLP over-rates dominant buckets.")
    parser.add_argument("--per-bucket-min-mabd", type=int, default=0,
                        dest="per_bucket_min_mabd",
                        help="Promote top-K DISCARD candidates per (platform_a, platform_b) "
                             "bucket to MABD. 0 = disabled (default). Requires --openai to have "
                             "effect. Recommended: 10–20 to fix minority-bucket events (suzhou).")
    parser.add_argument("--precomputed", action="store_true",
                        help="Use precomputed scored_pairs; skip BGE embedder")
    parser.add_argument("--drop-signal", default="", choices=["", "s1", "s2", "s3"],
                        dest="drop_signal",
                        help="Ablation: drop one signal from MLP input (requires matching checkpoint)")
    parser.add_argument("--uniform-weights", action="store_true",
                        dest="uniform_weights",
                        help="Ablation: replace MLP with mean(s1,s2,s3) scoring")
    parser.add_argument("--uniform-phase-window", action="store_true",
                        dest="uniform_phase_window",
                        help="Ablation: use W=72h for all lifecycle phases (no phase adaptation)")
    parser.add_argument("--always-s1s3", action="store_true",
                        dest="always_s1s3",
                        help="Ablation: always use s1×s3 (skip adaptive bucket-count check). "
                             "Requires --use-s1s3-scoring. Demonstrates adaptive gating is necessary.")
    parser.add_argument("--simonly", action="store_true",
                        help="Baseline: rank all Stage-1 candidates by cosine similarity (s1) only, "
                             "bypassing Stage-2 MLP routing entirely.")
    parser.add_argument("--tco", action="store_true",
                        help="Baseline: rank Stage-1 candidates by s1 * exp(-|Δt_h| / 72h). "
                             "Tests whether temporal proximity adds signal over cosine similarity.")
    parser.add_argument("--s1-floor", type=float, default=0.0, dest="s1_floor",
                        help="s1-floor threshold (0=disabled). Pairs with score_S < tau_low but "
                             "s1 >= this value are promoted DISCARD→BRIDGE (score_final=s1), "
                             "preventing the MLP from discarding high-cosine candidates in "
                             "diffusion/peak phases where learned weights collapse to pure-s2. "
                             "Recommended: 0.80")
    parser.add_argument("--min-bridge-fallback", type=int, default=0,
                        dest="min_bridge_fallback",
                        help="Sparse-BRIDGE SimOnly fallback threshold (0=disabled). "
                             "If after MLP routing fewer than this many pairs are in the "
                             "BRIDGE zone, fall back to SimOnly routing for that event. "
                             "Catches events where MLP over-discards (suzhou-pattern) "
                             "without incorrectly triggering on events with legitimately "
                             "sparse-but-correct BRIDGE sets. Recommended: 5")
    parser.add_argument("--score-bridge-by-s1", action="store_true",
                        dest="score_bridge_by_s1",
                        help="After MLP routing, rank all BRIDGE pairs by s1 (cosine) instead "
                             "of score_S. Turns MLP into a pure filter: routes DISCARD/MABD/BRIDGE "
                             "but uses cosine for final ranking within the BRIDGE set. "
                             "Can be combined with --low-s2-simonly.")
    parser.add_argument("--low-s2-simonly", type=float, default=0.0,
                        dest="low_s2_simonly_threshold",
                        help="Event-level low-s2 SimOnly fallback threshold (0=disabled). "
                             "If the MEDIAN s2 across all Stage-2 candidates in an event is "
                             "below this value, fall back to SimOnly (rank all by s1) for that "
                             "event.  Detects Weibo-censored events where temporal signal is "
                             "globally suppressed (suzhou: median_s2≈0.03). "
                             "Recommended: 0.08")
    parser.add_argument("--logreg", default=None,
                        help="Baseline: path to LR-NoPhase pkl model (trained by train_logreg.py). "
                             "Uses LogisticRegression on (s1,s2,s3) without phase conditioning.")
    args = parser.parse_args()

    if args.precomputed:
        run_precomputed(args)
    else:
        run_full_pipeline(args)


if __name__ == "__main__":
    main()
