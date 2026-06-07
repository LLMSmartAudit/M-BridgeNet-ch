"""Evaluate a CrossEncoder on test_real events.

Stage 1 candidate retrieval is shared with evaluate.py (BGE+cosine, numpy path).
The CrossEncoder then re-scores every Stage 1 candidate and returns the top-K.

Usage:
    MBRIDGENET_NO_FAISS=1 .venv/bin/python scripts/evaluate_crossencoder.py \\
        --data      data/cphot/processed/test_real \\
        --raw-dir   data/cphot/raw \\
        --checkpoint checkpoints/crossencoder_fold5 \\
        --k 5 10 20 50
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

try:
    import torch
    from transformers import AutoTokenizer, AutoModel
    import torch.nn as nn
except ImportError:
    raise SystemExit("Run: pip install torch transformers")

import numpy as np

from mbridgenet.config import load_config
from mbridgenet.schemas import Post
from mbridgenet.stage1.embedder import BGEEmbedder
from mbridgenet.evaluation.metrics import f1_strict_at_k, ap_at_k


# ── CrossEncoder model (mirror of train_crossencoder.py) ─────────────────────

class CrossEncoder(nn.Module):
    def __init__(self, bert_model):
        super().__init__()
        self.bert    = bert_model
        self.dropout = nn.Dropout(0.1)
        self.head    = nn.Linear(768, 1)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        cls = outputs.last_hidden_state[:, 0, :]
        return self.head(self.dropout(cls)).squeeze(-1)


def load_crossencoder(checkpoint_dir: Path, device: torch.device):
    from transformers import AutoModel as _AutoModel
    bert  = _AutoModel.from_pretrained(str(checkpoint_dir))
    model = CrossEncoder(bert).to(device)
    head_path = checkpoint_dir / "head.pt"
    state = torch.load(head_path, map_location=device)
    model.head.load_state_dict(state["head"])
    model.dropout.load_state_dict(state["dropout"])
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_dir))
    return model, tokenizer


# ── Scoring ───────────────────────────────────────────────────────────────────

def score_pairs(
    model,
    tokenizer,
    pairs: list[tuple[str, str]],   # (text_a, text_b)
    device: torch.device,
    batch_size: int = 32,
    max_length: int = 512,
) -> np.ndarray:
    """Return sigmoid scores for each (text_a, text_b) pair."""
    scores = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(pairs), batch_size):
            batch_pairs = pairs[i : i + batch_size]
            texts_a = [p[0] for p in batch_pairs]
            texts_b = [p[1] for p in batch_pairs]
            enc = tokenizer(
                texts_a,
                texts_b,
                truncation="longest_first",
                max_length=max_length,
                padding=True,
                return_tensors="pt",
            )
            logits = model(
                input_ids      = enc["input_ids"].to(device),
                attention_mask = enc["attention_mask"].to(device),
                token_type_ids = enc.get("token_type_ids",
                                         torch.zeros_like(enc["input_ids"])).to(device),
            )
            scores.extend(torch.sigmoid(logits).cpu().tolist())
    return np.array(scores, dtype=np.float32)


# ── Per-event evaluation ──────────────────────────────────────────────────────

def evaluate_event(
    event: dict,
    raw_text_map: dict[str, str],
    model,
    tokenizer,
    device: torch.device,
    cfg,
    ks: list[int],
    tau_coarse: float = 0.50,
    tau_fine:   float = 0.60,
    max_candidates: int | None = None,
) -> dict[str, float]:
    """Stage 1 candidate retrieval → CrossEncoder re-scoring → metrics."""
    posts_raw = event["posts"]
    bridge_pairs = event.get("bridge_pairs", [])
    ground_truth = set()
    for a, b in bridge_pairs:
        ground_truth.add(a)
        ground_truth.add(b)
    # Deduplicate: only post_a_id side (source) counts
    # (mirror of evaluate.py: ground truth is post_id of any bridge post)
    ground_truth = set(a for a, b in bridge_pairs) | set(b for a, b in bridge_pairs)

    # ── Build Post objects ────
    from datetime import datetime, timezone as _tz
    posts: list[Post] = []
    for rp in posts_raw:
        ts = datetime.fromisoformat(rp["timestamp"])
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=_tz.utc)
        posts.append(Post(
            post_id   = rp["post_id"],
            text      = rp["text"],
            timestamp = ts,
            platform  = rp["platform"],
            account_id= rp.get("account_id", ""),
            event_id  = rp.get("event_id", event["event_id"]),
        ))

    # ── Stage 1: BGE embeddings ────
    embedder = BGEEmbedder(cfg.stage1.embedding_model)
    texts    = [p.text for p in posts]
    post_ids = [p.post_id for p in posts]

    vecs = embedder.encode_batch(texts)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    mat = vecs / norms
    sim_matrix = mat @ mat.T

    # ── Filter: cross-platform, s1 ≥ tau_fine ────
    platform_map = {p.post_id: p.platform for p in posts}
    candidates: list[tuple[str, str, float]] = []   # (pid_a, pid_b, s1)
    for i in range(len(posts)):
        for j in range(i + 1, len(posts)):
            if posts[i].platform == posts[j].platform:
                continue
            s1 = float(sim_matrix[i, j])
            if s1 < tau_fine:
                continue
            candidates.append((post_ids[i], post_ids[j], s1))

    if not candidates:
        logger.warning("  No candidates after Stage 1 filtering")
        return {f"AP@{k}": 0.0 for k in ks} | {f"F1-Strict@{k}": 0.0 for k in ks}

    # Optionally cap to top-N by s1 score
    if max_candidates is not None and len(candidates) > max_candidates:
        candidates.sort(key=lambda x: -x[2])
        candidates = candidates[:max_candidates]
        logger.info("  Stage 1: %d candidates capped to %d (s1>=%.2f)",
                    len(candidates), max_candidates, tau_fine)
    else:
        logger.info("  Stage 1: %d candidates (s1>=%.2f)", len(candidates), tau_fine)

    # ── CrossEncoder scoring ────
    pair_texts = [
        (raw_text_map.get(a, posts_raw[0]["text"]),
         raw_text_map.get(b, posts_raw[0]["text"]))
        for a, b, _ in candidates
    ]
    # Build text map from this event's posts
    local_text_map = {p["post_id"]: p.get("text", "") for p in posts_raw}
    pair_texts = [
        (local_text_map.get(a, ""), local_text_map.get(b, ""))
        for a, b, _ in candidates
    ]

    ce_scores = score_pairs(model, tokenizer, pair_texts, device)

    # ── Build ranked prediction list (deduplicate by post_a_id) ────
    # For each candidate, both posts are potential bridge sources; use post_a_id side
    scored: list[tuple[str, float]] = []
    seen: dict[str, float] = {}
    for (pid_a, pid_b, _), score in zip(candidates, ce_scores):
        for pid in (pid_a, pid_b):
            if pid not in seen or score > seen[pid]:
                seen[pid] = float(score)
    scored = sorted(seen.items(), key=lambda x: -x[1])

    predictions = scored   # list of (pid, score) tuples — matches metrics API

    results = {}
    for k in ks:
        results[f"F1-Strict@{k}"] = f1_strict_at_k(predictions, ground_truth, k)
        results[f"AP@{k}"]         = ap_at_k(predictions, ground_truth, k)
    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CrossEncoder on test events")
    parser.add_argument("--data",        required=True, help="Processed test event dir")
    parser.add_argument("--raw-dir",     default="data/cphot/raw",
                        help="Raw event JSON dir (for post text)")
    parser.add_argument("--checkpoint",  required=True,
                        help="CrossEncoder checkpoint dir (output of train_crossencoder.py)")
    parser.add_argument("--config",      default="configs/default.yaml")
    parser.add_argument("--k",           nargs="+", type=int, default=[5, 10, 20, 50])
    parser.add_argument("--batch-size",  type=int, default=32)
    parser.add_argument("--tau-fine",    type=float, default=0.60,
                        help="Stage 1 cosine threshold; raise to 0.75–0.80 for large events")
    parser.add_argument("--max-candidates", type=int, default=None,
                        help="Cap Stage 1 candidates per event (top-N by s1); None = unlimited")
    args = parser.parse_args()

    cfg        = load_config(args.config)
    data_dir   = Path(args.data)
    raw_dir    = Path(args.raw_dir)
    ckpt_dir   = Path(args.checkpoint)
    ks         = sorted(args.k)

    device = (
        torch.device("mps")  if torch.backends.mps.is_available() else
        torch.device("cuda") if torch.cuda.is_available() else
        torch.device("cpu")
    )
    logger.info("Loading CrossEncoder from %s (device=%s) ...", ckpt_dir, device)
    model, tokenizer = load_crossencoder(ckpt_dir, device)

    event_files = sorted(data_dir.glob("*.json"))
    logger.info("Full pipeline eval on %d events, k=%s", len(event_files), ks)

    all_results: dict[str, dict] = {}
    for ef in event_files:
        with open(ef, encoding="utf-8") as f:
            event = json.load(f)
        event_id = event["event_id"]
        logger.info("Evaluating %s ...", event_id)
        metrics = evaluate_event(
            event          = event,
            raw_text_map   = {},   # unused; local_text_map built inside
            model          = model,
            tokenizer      = tokenizer,
            device         = device,
            cfg            = cfg,
            ks             = ks,
            tau_fine       = args.tau_fine,
            max_candidates = args.max_candidates,
        )
        all_results[event_id] = metrics
        for key, val in sorted(metrics.items()):
            logger.info("  %s = %.4f", key, val)

    # ── Aggregate ────────────────────────────────────────────────────────────
    import collections
    agg: dict[str, list] = collections.defaultdict(list)
    for event_id, metrics in all_results.items():
        for key, val in metrics.items():
            agg[key].append(val)

    print(f"\n{'Metric':<25} {'Mean':>8} {'Min':>8} {'Max':>8}")
    print("-" * 50)
    for key in sorted(agg):
        vals = agg[key]
        print(f"{key:<25} {sum(vals)/len(vals):>8.4f} {min(vals):>8.4f} {max(vals):>8.4f}")

    # ── Log to eval_results.jsonl ─────────────────────────────────────────────
    mean_metrics = {k: sum(v) / len(v) for k, v in agg.items()}
    log_record = {
        "timestamp":  datetime.now(timezone.utc).isoformat(),
        "mode":       "crossencoder",
        "checkpoint": str(ckpt_dir),
        "n_events":   len(event_files),
        "metrics":    {k: round(v, 4) for k, v in mean_metrics.items()},
        "per_event":  {eid: {k: round(v, 4) for k, v in m.items()}
                       for eid, m in all_results.items()},
    }
    log_path = Path("logs/eval_results.jsonl")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_record, ensure_ascii=False) + "\n")
    logger.info("Run logged → %s", log_path)


if __name__ == "__main__":
    main()
