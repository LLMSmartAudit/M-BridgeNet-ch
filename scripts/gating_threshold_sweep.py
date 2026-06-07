"""Adaptive gating threshold sweep for methodological validation.

Separates the 27 test events into:
  - val_events (3): xiao_fei_scandal_001, suzhou_realestate_arrest_001,
                    likeqiang_death_001  ← originally used for τ_high/τ_low tuning
  - held_out (24):  all other test_real events

Sweeps dom_s3_threshold × minority_fraction combinations, selects best on
val_events, then reports uncontaminated performance on held_out events.

Usage:
    cd /path/to/M-BridgeNet
    MBRIDGENET_NO_FAISS=1 .venv/bin/python scripts/gating_threshold_sweep.py \
        --data data/cphot/processed/test_real \
        --checkpoint checkpoints/mlp_v18_fold5.pt \
        --k 5 20 50
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from itertools import product
from typing import List, Dict, Any

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# The three events used for τ tuning (validation split)
VAL_EVENT_IDS = {
    "xiao_fei_scandal_001",
    "suzhou_realestate_arrest_001",
    "likeqiang_death_001",
}

# Threshold grid
DOM_S3_THRESHOLDS = [0.55, 0.60, 0.65, 0.70, 0.75, float("inf")]   # inf = condition disabled
MINORITY_FRACS    = [0.05, 0.10, 0.15, 0.20, float("inf")]          # inf = condition disabled


def ap_at_k(predictions, ground_truth: set, k: int) -> float:
    from mbridgenet.evaluation.metrics import ap_at_k as _ap
    return _ap(predictions, ground_truth, k)


def run_event(pipeline, event: dict, dom_s3_thresh: float, minority_frac: float,
              k_values: List[int]) -> Dict[str, float]:
    """Run one event with patched gating thresholds, return AP@k dict."""
    import mbridgenet.pipeline as _pmod

    # Monkey-patch the _DOM_S3_THRESHOLD default and minority fraction at call site.
    # We override pipeline._s1s3_score to inject the sweep values.
    from mbridgenet.pipeline import MBridgeNetPipeline, Route
    from collections import defaultdict as _dd

    def _patched_s1s3(all_candidates, c):
        bridge_candidates = [x for x in all_candidates if x.route == Route.BRIDGE]
        bucket_s3 = _dd(list)
        for cand in bridge_candidates:
            bucket = tuple(sorted([cand.post_a.platform, cand.post_b.platform]))
            bucket_s3[bucket].append(cand.s3)

        if len(bucket_s3) < 2:
            return c.s1

        dominant_bucket = max(bucket_s3, key=lambda b: len(bucket_s3[b]))
        dom_vals = sorted(bucket_s3[dominant_bucket])
        dom_s3_median = dom_vals[len(dom_vals) // 2]

        if dom_s3_median >= dom_s3_thresh:
            return c.s1

        total_bridge = len(bridge_candidates)
        largest_minority = max(
            (len(v) for k, v in bucket_s3.items() if k != dominant_bucket),
            default=0,
        )
        if total_bridge > 0 and largest_minority < minority_frac * total_bridge:
            return c.s1

        return c.s1 * c.s3

    # Temporarily replace the static method
    original = MBridgeNetPipeline._s1s3_score
    MBridgeNetPipeline._s1s3_score = staticmethod(
        lambda all_cands, c: _patched_s1s3(all_cands, c)
    )
    try:
        output_pairs = pipeline.run(event, use_s1s3_scoring=True)
    finally:
        MBridgeNetPipeline._s1s3_score = original

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
    parser.add_argument("--primary-k",  type=int, default=5,
                        help="K used for threshold selection on val split")
    args = parser.parse_args()

    from mbridgenet.config import load_config
    from mbridgenet.data.dataset import CPHotDataset
    from mbridgenet.pipeline import MBridgeNetPipeline

    cfg = load_config(None)
    pipeline = MBridgeNetPipeline(
        config=cfg,
        mlp_checkpoint=args.checkpoint,
    )

    dataset = CPHotDataset(Path(args.data))
    val_events   = [e for e in dataset if e.get("event_id", "") in VAL_EVENT_IDS]
    held_events  = [e for e in dataset if e.get("event_id", "") not in VAL_EVENT_IDS]

    logger.info("Val events  : %d (%s)", len(val_events),
                [e["event_id"] for e in val_events])
    logger.info("Held-out    : %d events", len(held_events))

    if not val_events:
        logger.error("No validation events found — check event_id field in JSONs")
        sys.exit(1)

    pk = args.primary_k
    best_val_ap = -1.0
    best_combo  = None
    sweep_rows  = []

    total_combos = len(DOM_S3_THRESHOLDS) * len(MINORITY_FRACS)
    logger.info("Sweeping %d threshold combinations on %d val events...",
                total_combos, len(val_events))

    for dom_s3, minority in product(DOM_S3_THRESHOLDS, MINORITY_FRACS):
        val_results = []
        for event in val_events:
            r = run_event(pipeline, event, dom_s3, minority, args.k)
            val_results.append(r)
        val_ap = mean_ap(val_results, pk)

        dom_s3_str   = f"{dom_s3:.2f}" if dom_s3 != float("inf") else "∞"
        minority_str = f"{minority:.2f}" if minority != float("inf") else "∞"
        sweep_rows.append((dom_s3, minority, val_ap))
        logger.info("  dom_s3_thresh=%-6s  minority_frac=%-6s  val AP@%d=%.4f",
                    dom_s3_str, minority_str, pk, val_ap)

        if val_ap > best_val_ap:
            best_val_ap = val_ap
            best_combo  = (dom_s3, minority)

    dom_str   = '∞' if best_combo[0] == float('inf') else f'{best_combo[0]:.2f}'
    minor_str = '∞' if best_combo[1] == float('inf') else f'{best_combo[1]:.2f}'
    print(f"\n{'='*60}")
    print(f"BEST on {len(val_events)}-event val split (AP@{pk}): "
          f"dom_s3_thresh={dom_str}  "
          f"minority_frac={minor_str}  "
          f"val AP@{pk}={best_val_ap:.4f}")

    # Current paper thresholds for reference
    paper_dom, paper_minor = 0.65, 0.10
    logger.info("Evaluating paper thresholds (%.2f, %.2f) on val split...",
                paper_dom, paper_minor)
    paper_val_results = [run_event(pipeline, e, paper_dom, paper_minor, args.k)
                         for e in val_events]
    paper_val_ap = mean_ap(paper_val_results, pk)

    print(f"\nPaper thresholds (0.65, 0.10):  val AP@{pk}={paper_val_ap:.4f}")
    print(f"Best val thresholds:            val AP@{pk}={best_val_ap:.4f}  "
          f"(Δ={best_val_ap-paper_val_ap:+.4f})")

    # Evaluate best thresholds on held-out events
    print(f"\n{'='*60}")
    print(f"HELD-OUT evaluation ({len(held_events)} events) with best val thresholds "
          f"(dom_s3={best_combo[0]}, minority={best_combo[1]}):")
    held_results = [run_event(pipeline, e, best_combo[0], best_combo[1], args.k)
                    for e in held_events]
    for k in args.k:
        print(f"  AP@{k} = {mean_ap(held_results, k)*100:.2f}%")

    # Also paper thresholds on held-out
    print(f"\nHeld-out with paper thresholds (dom_s3=0.65, minority=0.10):")
    paper_held_results = [run_event(pipeline, e, paper_dom, paper_minor, args.k)
                          for e in held_events]
    for k in args.k:
        print(f"  AP@{k} = {mean_ap(paper_held_results, k)*100:.2f}%")

    # Full 27-event comparison (val + held)
    print(f"\n{'='*60}")
    print("FULL 27-event results (val + held-out):")
    best_all = held_results + [run_event(pipeline, e, best_combo[0], best_combo[1], args.k)
                                for e in val_events]
    paper_all = paper_held_results + paper_val_results
    for k in args.k:
        ba = mean_ap(best_all, k) * 100
        pa = mean_ap(paper_all, k) * 100
        print(f"  AP@{k}: best_val_thresholds={ba:.2f}%  paper_thresholds={pa:.2f}%  "
              f"Δ={ba-pa:+.2f}pp")

    # Save results
    out = {
        "val_event_ids":     [e["event_id"] for e in val_events],
        "held_out_n":        len(held_events),
        "best_dom_s3":       best_combo[0],
        "best_minority":     best_combo[1],
        "best_val_ap_at_k":  {str(pk): round(best_val_ap, 4)},
        "paper_val_ap_at_k": {str(pk): round(paper_val_ap, 4)},
        "held_out_best": {f"AP@{k}": round(mean_ap(held_results, k), 4) for k in args.k},
        "held_out_paper":{f"AP@{k}": round(mean_ap(paper_held_results, k), 4) for k in args.k},
        "full27_best":   {f"AP@{k}": round(mean_ap(best_all,   k), 4) for k in args.k},
        "full27_paper":  {f"AP@{k}": round(mean_ap(paper_all,  k), 4) for k in args.k},
    }
    out_path = Path("logs/gating_sweep_results.json")
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    logger.info("Results saved → %s", out_path)


if __name__ == "__main__":
    main()
