"""Compute real s1–s4 signals for annotated pairs and sample negatives.

Takes a CPHot event JSON with bridge_pairs from annotate_pairs.py, runs
Stage-1/2 signal computation, adds negative examples, and saves the result
to a train-ready processed directory.

Usage:
    python scripts/prepare_training_data.py \\
        --event  data/cphot/raw/taiwan_tensions_001.json \\
        --output data/cphot/processed \\
        --neg-ratio 3
"""
from __future__ import annotations

import argparse
import json
import logging
import random
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from mbridgenet.config import load_config
from mbridgenet.schemas import Post, CandidatePair
from mbridgenet.stage1.embedder import BGEEmbedder
from mbridgenet.stage2.lifecycle import assign_phases
from mbridgenet.stage2.signals import compute_s2, compute_s3


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--event",     required=True, help="Path to annotated event JSON")
    parser.add_argument("--output",    default="data/cphot/processed")
    parser.add_argument("--config",    default="configs/default.yaml")
    parser.add_argument("--neg-ratio", type=int, default=3,
                        help="Negative samples per positive (default: 3)")
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument(
        "--stratify", action="store_true",
        help="Sample negatives equally per platform-pair bucket (default: random global)"
    )
    parser.add_argument(
        "--uniform-window", action="store_true", dest="uniform_window",
        help="Use flat 72h window for all lifecycle phases when computing s2."
    )
    args = parser.parse_args()

    random.seed(args.seed)
    cfg = load_config(args.config)

    with open(args.event, encoding="utf-8") as f:
        event = json.load(f)

    event_id       = event["event_id"]
    raw_posts      = event["posts"]
    bridge_pairs   = event.get("bridge_pairs", [])
    hourly_volumes = event.get("hourly_volumes", [])

    if not bridge_pairs:
        logger.error("No bridge_pairs found — run annotate_pairs.py first.")
        return

    logger.info("Event: %s  posts=%d  bridge_pairs=%d",
                event_id, len(raw_posts), len(bridge_pairs))

    pos_set = {(a, b) for a, b in bridge_pairs} | {(b, a) for a, b in bridge_pairs}

    # ── Build Post objects ────────────────────────────────────────────────────
    posts: list[Post] = []
    for rp in raw_posts:
        ts = datetime.fromisoformat(rp["timestamp"])
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        posts.append(Post(
            post_id=rp["post_id"],
            text=rp["text"],
            timestamp=ts,
            platform=rp["platform"],
            account_id=rp.get("account_id", ""),
            event_id=rp.get("event_id", event_id),
        ))

    # ── BGE embeddings → s1 ──────────────────────────────────────────────────
    model_path = cfg.stage1.embedding_model
    logger.info("Loading embedder from %s ...", model_path)
    embedder = BGEEmbedder(model_path)
    texts    = [p.text for p in posts]
    post_ids = [p.post_id for p in posts]
    logger.info("Embedding %d posts ...", len(texts))
    vecs = embedder.encode_batch(texts)
    # cosine similarity matrix (N x N)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    mat = vecs / norms
    sim_matrix = mat @ mat.T
    emb_dict = {post_ids[i]: mat[i] for i in range(len(post_ids))}
    logger.info("Embeddings done.")

    # Cross-platform pairs with a minimum similarity filter to keep memory manageable.
    # Always include bridge pairs regardless of threshold.
    MIN_S1 = 0.60
    pos_set = {(a, b) for a, b in event.get("bridge_pairs", [])}
    pos_set |= {(b, a) for a, b in pos_set}

    candidates: list[CandidatePair] = []
    for i, pa in enumerate(posts):
        for j, pb in enumerate(posts):
            if j <= i:
                continue
            if pa.platform == pb.platform:
                continue
            s1 = float(sim_matrix[i, j])
            is_bridge = (pa.post_id, pb.post_id) in pos_set
            if s1 < MIN_S1 and not is_bridge:
                continue
            candidates.append(CandidatePair(post_a=pa, post_b=pb, s1=s1))

    logger.info("Cross-platform pairs total (s1>=%.2f): %d", MIN_S1, len(candidates))

    cand_by_ids: dict = {}
    for c in candidates:
        key = tuple(sorted([c.post_a.post_id, c.post_b.post_id]))
        cand_by_ids[key] = c

    # ── Stage 2 signals ───────────────────────────────────────────────────────
    timestamps = [p.timestamp for p in posts]

    phase_map = dict(zip(
        post_ids,
        assign_phases(timestamps, hourly_volumes,
                      thresholds=cfg.stage2.lifecycle_thresholds),
    ))

    migration_counts: dict = defaultdict(int)
    post_platform = {p.post_id: p.platform for p in posts}
    for a, b in bridge_pairs:
        pa, pb = post_platform.get(a, ""), post_platform.get(b, "")
        if pa and pb:
            migration_counts[(pa, pb)] += 1
            migration_counts[(pb, pa)] += 1

    from mbridgenet.stage2.graph import build_event_graph, compute_betweenness
    from mbridgenet.stage2.signals import compute_s4_normalized

    cand_ids = [(c.post_a.post_id, c.post_b.post_id) for c in candidates]
    G = build_event_graph(posts, cand_ids)
    raw_bc = compute_betweenness(G)
    norm_bc = compute_s4_normalized(raw_bc)
    logger.info("Betweenness computed for %d nodes", len(norm_bc))

    def make_scored_pair(c, label: int) -> dict:
        phase  = phase_map.get(c.post_a.post_id, "emergence")
        t_a    = c.post_a.timestamp
        t_b    = c.post_b.timestamp
        delta_t = abs((t_a - t_b).total_seconds() / 3600)
        _phase_windows = (
            {"emergence": 72, "diffusion": 72, "peak": 72, "decline": 72}
            if args.uniform_window
            else cfg.stage2.phase_windows
        )
        s2 = compute_s2(delta_t, phase, _phase_windows)
        s3 = compute_s3(c.post_a.platform, c.post_b.platform, migration_counts)
        s4 = norm_bc.get(c.post_a.post_id, 0.0)
        return {
            "post_a_id": c.post_a.post_id,
            "post_b_id": c.post_b.post_id,
            "s1": round(float(c.s1), 6),
            "s2": round(s2, 6),
            "s3": round(s3, 6),
            "s4": round(s4, 6),
            "phase": phase,
            "label": label,
        }

    # ── Positives: bridge_pairs that are also Stage-1 candidates ─────────────
    scored = []
    missing_pos = 0
    for a, b in bridge_pairs:
        key = tuple(sorted([a, b]))
        c = cand_by_ids.get(key)
        if c is None:
            # Not in candidate pool — create with s1=0 using raw timestamps
            pa_obj = next((p for p in posts if p.post_id == a), None)
            pb_obj = next((p for p in posts if p.post_id == b), None)
            if pa_obj and pb_obj:
                c = CandidatePair(post_a=pa_obj, post_b=pb_obj, s1=0.0)
                missing_pos += 1
            else:
                continue
        scored.append(make_scored_pair(c, label=1))

    logger.info("Positives: %d  (not in candidate pool: %d)", len(scored), missing_pos)

    # ── Negatives: random or stratified candidates not in bridge_pairs ────────
    raw_negs = [
        c for key, c in cand_by_ids.items()
        if (c.post_a.post_id, c.post_b.post_id) not in pos_set
        and (c.post_b.post_id, c.post_a.post_id) not in pos_set
    ]

    if args.stratify:
        # Group by sorted platform pair so weibo×zhihu and zhihu×weibo merge
        neg_buckets: dict = defaultdict(list)
        for c in raw_negs:
            key = tuple(sorted([c.post_a.platform, c.post_b.platform]))
            neg_buckets[key].append(c)

        for bucket in neg_buckets.values():
            random.shuffle(bucket)

        n_total_budget = min(len(raw_negs), len(scored) * args.neg_ratio)
        n_buckets = len(neg_buckets)
        per_bucket = max(1, n_total_budget // n_buckets)

        neg_pool = []
        for bucket in neg_buckets.values():
            neg_pool.extend(bucket[:per_bucket])
        # Top-up if under budget (happens when some buckets are smaller than per_bucket)
        neg_pool_set = set(id(c) for c in neg_pool)
        remainder = [c for c in raw_negs if id(c) not in neg_pool_set]
        random.shuffle(remainder)
        neg_pool.extend(remainder[: n_total_budget - len(neg_pool)])
        random.shuffle(neg_pool)
        neg_pool = neg_pool[:n_total_budget]

        buckets_report = {str(k): len(v) for k, v in neg_buckets.items()}
        logger.info("Stratified neg buckets: %s", buckets_report)
    else:
        random.shuffle(raw_negs)
        neg_pool = raw_negs

    n_neg = min(len(neg_pool), len(scored) * args.neg_ratio)
    for c in neg_pool[:n_neg]:
        scored.append(make_scored_pair(c, label=0))

    logger.info("Negatives sampled: %d  (ratio 1:%d)", n_neg, args.neg_ratio)
    logger.info("Total scored_pairs: %d", len(scored))

    # ── Save processed event ──────────────────────────────────────────────────
    out_dir  = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{event_id}.json"

    out_event = {
        "event_id":       event_id,
        "posts":          raw_posts,
        "bridge_pairs":   bridge_pairs,
        "hourly_volumes": hourly_volumes,
        "scored_pairs":   scored,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_event, f, ensure_ascii=False, indent=2)

    logger.info("Saved → %s", out_path)
    pos_count = sum(1 for s in scored if s["label"] == 1)
    neg_count = sum(1 for s in scored if s["label"] == 0)
    logger.info("Labels — positive: %d  negative: %d", pos_count, neg_count)

    # Export per-post BGE embeddings for PairEncoder training
    emb_out = out_dir / f"{event_id}_emb.npz"
    np.savez_compressed(str(emb_out), **{pid: vec for pid, vec in emb_dict.items()})
    logger.info("Embeddings saved → %s", emb_out)


if __name__ == "__main__":
    main()
