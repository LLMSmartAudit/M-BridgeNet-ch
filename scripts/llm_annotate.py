"""LLM-assisted bridge pair annotation.

Embeds all posts with BGE, selects the top-K most similar cross-platform pairs,
then calls an LLM once per pair with strict bridge criteria to produce labels.
Replaces manual annotation (smart_annotate.py) for events where human labeling
is unreliable due to topic homogeneity.

Usage:
    OPENAI_API_KEY=sk-... .venv/bin/python scripts/llm_annotate.py \\
        --event data/cphot/raw/spacex_launch_opinion_001.json \\
        --top-k 200

Options:
    --top-k       Number of top cosine-similar cross-platform pairs to label (default: 200)
    --model       OpenAI model to use (default: from configs/default.yaml)
    --resume      Skip pairs already present in scored_pairs
    --dry-run     Show first 5 pairs and their LLM verdicts without saving
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

import openai
from mbridgenet.config import load_config
from mbridgenet.stage1.embedder import BGEEmbedder
from mbridgenet.stage2.lifecycle import assign_phases

ANNOTATION_SYSTEM = """\
You are an expert at identifying "bridge pairs" in Chinese social media data.

A bridge pair consists of two posts from DIFFERENT platforms that carry the SAME SPECIFIC \
NARRATIVE from one platform to another.

STRICT criteria — ALL must hold for a bridge pair:
1. SAME SPECIFIC SUB-EVENT: both posts refer to the exact same incident, announcement, \
statistic, or fact — not just the same broad topic.
2. SAME NARRATIVE ANGLE: both posts frame the event similarly (e.g. both report the same \
fact, both express the same sentiment about the same specific development).
3. CONTENT DEPENDENCY: the specific claim or narrative in one post is echoed or amplified \
in the other. If you removed Post A, Post B's specific claim would seem to appear from nowhere.

NOT a bridge pair if:
- Both discuss the same broad topic but focus on different aspects or sub-events
- They share vocabulary but make different claims
- One is general commentary, the other is specific reporting
- They have opposite viewpoints on the same event
- Either post is too generic (single word reactions, empty enthusiasm, hashtag-only)

Respond ONLY with valid JSON on a single line:
{"is_bridge": true/false, "confidence": 0.0-1.0, "reasoning": "one sentence"}
"""

ANNOTATION_USER = """\
Post A [{platform_a}] {ts_a}:
{text_a}

Post B [{platform_b}] {ts_b}:
{text_b}

Are these a bridge pair by the strict criteria above?
"""


def call_llm(
    client: openai.OpenAI,
    model: str,
    post_a: Dict,
    post_b: Dict,
    retries: int = 3,
) -> Optional[Dict]:
    prompt = ANNOTATION_USER.format(
        platform_a=post_a["platform"].upper(),
        ts_a=post_a["timestamp"][:19],
        text_a=post_a["text"][:600],
        platform_b=post_b["platform"].upper(),
        ts_b=post_b["timestamp"][:19],
        text_b=post_b["text"][:600],
    )
    delays = [5, 15, 30]
    for attempt in range(retries):
        try:
            kwargs = dict(
                model=model,
                messages=[
                    {"role": "system", "content": ANNOTATION_SYSTEM},
                    {"role": "user",   "content": prompt},
                ],
                max_completion_tokens=256,
            )
            # gpt-5.5 and o-series models don't accept temperature
            try:
                resp = client.chat.completions.create(**kwargs, temperature=0.0)
            except openai.BadRequestError:
                resp = client.chat.completions.create(**kwargs)
            raw = resp.choices[0].message.content.strip()
            # strip markdown code fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            result = json.loads(raw)
            if "is_bridge" not in result:
                raise ValueError("missing is_bridge key")
            result["confidence"] = float(result.get("confidence", 0.5))
            return result
        except openai.RateLimitError as e:
            retry_after = int(getattr(e, "retry_after", delays[min(attempt, 2)]))
            logger.warning("Rate limit hit — sleeping %ds", retry_after)
            time.sleep(retry_after)
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning("Parse error attempt %d: %s", attempt + 1, e)
            if attempt < retries - 1:
                time.sleep(delays[attempt])
        except Exception as e:
            logger.warning("API error attempt %d: %s", attempt + 1, e)
            if attempt < retries - 1:
                time.sleep(delays[attempt])
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--event",   required=True)
    parser.add_argument("--top-k",   type=int, default=None,
                        help="Fixed candidate count (overrides adaptive selection)")
    parser.add_argument("--tau",     type=float, default=0.85,
                        help="Adaptive: select all pairs with cosine >= tau (default: 0.85)")
    parser.add_argument("--max-k",   type=int, default=1000,
                        help="Adaptive: hard cap on candidates (default: 1000)")
    parser.add_argument("--config",  default="configs/default.yaml")
    parser.add_argument("--model",   default=None, help="Override LLM model from config")
    parser.add_argument("--resume",  action="store_true",
                        help="Skip pairs already in scored_pairs")
    parser.add_argument("--workers", type=int, default=24,
                        help="Concurrent LLM annotation workers (default: 24)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Label first 5 pairs only, do not save")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model = args.model or cfg.stage3.llm_model

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY not set")
        sys.exit(1)
    client = openai.OpenAI(api_key=api_key)

    event_path = Path(args.event)
    with open(event_path, encoding="utf-8") as f:
        event = json.load(f)

    posts = event["posts"]
    logger.info("Event: %s  posts=%d", event["event_id"], len(posts))

    # ── Embed ─────────────────────────────────────────────────────────────────
    embedder = BGEEmbedder(cfg.stage1.embedding_model)
    logger.info("Embedding %d posts ...", len(posts))
    vecs = embedder.encode_batch([p["text"] for p in posts])
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    mat = vecs / norms
    sim_matrix = mat @ mat.T
    logger.info("Embeddings done.")

    # ── Lifecycle phases ──────────────────────────────────────────────────────
    timestamps = []
    for p in posts:
        ts = datetime.fromisoformat(p["timestamp"])
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        timestamps.append(ts)
    phases = assign_phases(
        timestamps, event.get("hourly_volumes", []),
        thresholds=cfg.stage2.lifecycle_thresholds,
    )
    phase_map = {p["post_id"]: ph for p, ph in zip(posts, phases)}

    # ── Rank cross-platform pairs ─────────────────────────────────────────────
    already_labeled = {
        (sp["post_a_id"], sp["post_b_id"])
        for sp in event.get("scored_pairs", [])
    }
    scored: List[Tuple[float, int, int]] = []
    for i in range(len(posts)):
        for j in range(i + 1, len(posts)):
            if posts[i]["platform"] == posts[j]["platform"]:
                continue
            key = (posts[i]["post_id"], posts[j]["post_id"])
            if args.resume and key in already_labeled:
                continue
            scored.append((float(sim_matrix[i, j]), i, j))
    scored.sort(key=lambda x: x[0], reverse=True)

    if args.top_k is not None:
        # explicit fixed count
        candidates = scored[: args.top_k]
        logger.info("Selection: top-k=%d (fixed)", args.top_k)
    else:
        # adaptive: all pairs above tau, capped at max-k
        candidates = [t for t in scored if t[0] >= args.tau][: args.max_k]
        logger.info(
            "Selection: adaptive tau=%.2f → %d pairs (max-k=%d, actual cosine range %.4f–%.4f)",
            args.tau, len(candidates), args.max_k,
            candidates[-1][0] if candidates else 0.0,
            candidates[0][0] if candidates else 0.0,
        )

    if args.dry_run:
        candidates = candidates[:5]
    logger.info("Annotating %d pairs with %s ...", len(candidates), model)

    # ── Annotate ──────────────────────────────────────────────────────────────
    scored_pairs: List[Dict] = list(event.get("scored_pairs", []))
    bridge_set = {tuple(bp) for bp in event.get("bridge_pairs", [])}

    n_bridge = n_nonbridge = n_fail = 0

    # ── Concurrent annotation (thread pool) ───────────────────────────────────
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading
    lock = threading.Lock()
    total = len(candidates)
    completed = 0

    def _annotate_one(item):
        rank, (sim, i, j) = item
        pa, pb = posts[i], posts[j]
        return rank, sim, i, j, call_llm(client, model, pa, pb)

    logger.info("Annotating with %d concurrent workers ...", args.workers)
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_annotate_one, item)
                for item in enumerate(candidates, start=1)]
        for fut in as_completed(futs):
            rank, sim, i, j, result = fut.result()
            pa, pb = posts[i], posts[j]
            with lock:
                completed += 1
                if result is None:
                    n_fail += 1
                else:
                    label = 1 if result["is_bridge"] else 0
                    if label == 1:
                        bridge_set.add((pa["post_id"], pb["post_id"]))
                        n_bridge += 1
                    else:
                        n_nonbridge += 1
                    scored_pairs.append({
                        "post_a_id": pa["post_id"],
                        "post_b_id": pb["post_id"],
                        "s1": round(sim, 6),
                        "s2": 0.0, "s3": 0.0, "s4": 0.0,
                        "phase": phase_map.get(pa["post_id"], "emergence"),
                        "label": label,
                        "llm_confidence": round(result["confidence"], 3),
                        "llm_reasoning": result.get("reasoning", ""),
                    })
                # progress + crash-safe checkpoint every 100 completed
                if completed % 100 == 0 or completed == total:
                    logger.info("  progress %d/%d  (bridge=%d nonbridge=%d fail=%d)",
                                completed, total, n_bridge, n_nonbridge, n_fail)
                    if not args.dry_run:
                        _save(event, event_path, scored_pairs, bridge_set)

    bridge_rate = n_bridge / max(n_bridge + n_nonbridge, 1)
    logger.info(
        "Done: %d bridge  %d non-bridge  %d failed  bridge_rate=%.1f%%",
        n_bridge, n_nonbridge, n_fail, 100 * bridge_rate,
    )

    if args.dry_run:
        logger.info("Dry-run — not saving.")
        return

    _save(event, event_path, scored_pairs, bridge_set)
    logger.info("Saved → %s", event_path)


def _save(event: Dict, path: Path, scored_pairs: List[Dict], bridge_set) -> None:
    event["scored_pairs"] = scored_pairs
    event["bridge_pairs"] = [[a, b] for a, b in bridge_set]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(event, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
