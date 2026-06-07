"""BM25 baseline evaluation on CPHot test_real.

Uses the same Stage-1 candidate pool as SimOnly / M-BridgeNet
(pipeline.run with simonly=True to get all candidates), then
re-ranks each (post_a, post_b) pair by BM25 score.

Tokenisation: jieba word-level segmentation (Chinese-aware) with
a fallback to character unigrams if jieba is unavailable.

For each event:
  1. Run Stage-1 retrieval to get all cross-platform candidate pairs.
  2. Build a BM25 index over the unique post_b texts in the candidate pool.
  3. For each post_a, score every (post_a, post_b) pair by
     BM25_score(tokenize(post_a.text), post_b_index).
  4. Deduplicate by post_a_id (keep highest BM25 score), sort, compute
     AP@K and F1-Strict@K identical to other baselines.

Usage:
    cd /path/to/M-BridgeNet
    MBRIDGENET_NO_FAISS=1 .venv/bin/python scripts/evaluate_bm25.py \\
        --data data/cphot/processed/test_real \\
        --checkpoint checkpoints/mlp_v18_fold5.pt \\
        --k 5 10 20 50
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

LOG_PATH = Path("logs/eval_results.jsonl")

# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------

def _build_tokeniser():
    """Return a callable that tokenises a Chinese string into a list of tokens."""
    try:
        import jieba
        jieba.setLogLevel(logging.WARNING)  # suppress jieba init noise
        logger.info("Using jieba word-level tokenisation")
        def _tok(text: str):
            return list(jieba.cut(str(text or ""), cut_all=False))
        return _tok
    except ImportError:
        logger.warning("jieba not available — falling back to character unigrams")
        def _tok(text: str):
            return list(str(text or ""))
        return _tok


tokenise = _build_tokeniser()


# ---------------------------------------------------------------------------
# Per-event BM25 evaluation
# ---------------------------------------------------------------------------

def run_event_bm25(pipeline, event: dict, k_values) -> dict:
    """Evaluate BM25 ranking on one event, return AP@K / F1@K dict."""
    from rank_bm25 import BM25Okapi
    from mbridgenet.evaluation.metrics import ap_at_k, f1_strict_at_k

    ground_truth = {pair[0] for pair in event["bridge_pairs"]}

    # Get Stage-1 candidates via simonly=True (same pool as all other baselines)
    output_pairs = pipeline.run(event, simonly=True)
    if not output_pairs:
        logger.warning("  No candidates for event %s", event.get("event_id", "?"))
        return {f"AP@{k}": 0.0 for k in k_values} | {f"F1-Strict@{k}": 0.0 for k in k_values}

    # Collect unique post_b objects (indexed by post_id for dedup)
    pb_index: dict = {}   # post_id -> CandidatePair.post_b
    for cp in output_pairs:
        pid = cp.post_b.post_id
        if pid not in pb_index:
            pb_index[pid] = cp.post_b

    unique_pb_ids  = list(pb_index.keys())
    unique_pb_texts = [tokenise(pb_index[pid].text or "") for pid in unique_pb_ids]
    pb_pos = {pid: i for i, pid in enumerate(unique_pb_ids)}

    # Build BM25 index over post_b corpus
    bm25 = BM25Okapi(unique_pb_texts)

    # Score each (post_a, post_b) pair
    # Group candidates by post_a so we query BM25 once per unique post_a
    pa_groups: dict = defaultdict(list)   # post_a_id -> [CandidatePair, ...]
    for cp in output_pairs:
        pa_groups[cp.post_a.post_id].append(cp)

    # Build final scored list: (post_a_id, bm25_score)
    # For each post_a, get BM25 scores for its candidate post_b pool
    pa_best: dict = {}   # post_a_id -> best BM25 score across its post_b candidates
    for pa_id, cands in pa_groups.items():
        query_tokens = tokenise(cands[0].post_a.text or "")
        scores = bm25.get_scores(query_tokens)   # score for every doc in corpus
        for cp in cands:
            pb_idx = pb_pos.get(cp.post_b.post_id)
            if pb_idx is None:
                continue
            score = float(scores[pb_idx])
            if pa_id not in pa_best or score > pa_best[pa_id]:
                pa_best[pa_id] = score

    predictions = sorted(pa_best.items(), key=lambda x: x[1], reverse=True)

    result = {}
    for k in k_values:
        result[f"AP@{k}"]        = ap_at_k(predictions, ground_truth, k)
        result[f"F1-Strict@{k}"] = f1_strict_at_k(predictions, ground_truth, k)
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="BM25 baseline evaluation on CPHot test_real"
    )
    parser.add_argument("--data",       required=True, help="Path to test_real directory")
    parser.add_argument("--checkpoint", required=True, help="MLP checkpoint (for Stage-1 config)")
    parser.add_argument("--k",          nargs="+", type=int, default=[5, 10, 20, 50])
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
    logger.info("BM25 eval on %d events, k=%s", len(dataset), args.k)

    results = defaultdict(list)
    per_event_records = []

    for i, event in enumerate(dataset):
        eid = event.get("event_id", f"event_{i}")
        r = run_event_bm25(pipeline, event, args.k)
        for metric, val in r.items():
            results[metric].append(val)
        ap5  = r.get("AP@5",  0.0)
        ap50 = r.get("AP@50", 0.0)
        per_event_records.append({"event_id": eid, **{k: round(v, 4) for k, v in r.items()}})
        logger.info("  [%d/%d] %-35s AP@5=%.3f AP@50=%.3f",
                    i + 1, len(dataset), eid, ap5, ap50)

    # Summary table
    print(f"\n{'Metric':<20} {'Mean':>8}  {'Min':>8}  {'Max':>8}")
    print("-" * 48)
    summary = {}
    for metric in sorted(results):
        vals = results[metric]
        mean = sum(vals) / len(vals) if vals else 0.0
        summary[metric] = {
            "mean": round(mean, 4),
            "min":  round(min(vals), 4),
            "max":  round(max(vals), 4),
        }
        print(f"{metric:<20} {mean:>8.4f}  {min(vals):>8.4f}  {max(vals):>8.4f}")

    # Save to log
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode":      "bm25",
        "data":      args.data,
        "checkpoint": args.checkpoint,
        "k_values":  args.k,
        "n_events":  len(dataset),
        "tokeniser": "jieba" if "jieba" in str(tokenise) else "char_unigram",
        "metrics":   summary,
        "per_event_metrics": per_event_records,
    }
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info("Results saved → %s", LOG_PATH)


if __name__ == "__main__":
    main()
