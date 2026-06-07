"""Diagnose MABD decisions on a single test event.

Runs Stage 1+2, finds MABD-zone pairs, samples N of them, runs MABD on each,
and prints a decision table alongside the ground-truth label.

Usage:
    OPENAI_API_KEY=sk-... .venv/bin/python scripts/diagnose_mabd.py \
        --event data/cphot/processed/test/event_016.json \
        --sample 20
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import openai
from mbridgenet.config import load_config
from mbridgenet.pipeline import MBridgeNetPipeline
from mbridgenet.schemas import Post, Route
from mbridgenet.stage3.mabd import MABDDebater


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--event",   required=True)
    parser.add_argument("--config",  default="configs/default.yaml")
    parser.add_argument("--sample",  type=int, default=20, help="MABD pairs to inspect")
    parser.add_argument("--seed",    type=int, default=0)
    args = parser.parse_args()

    cfg = load_config(args.config)

    with open(args.event, encoding="utf-8") as f:
        event = json.load(f)

    # Ground-truth bridge pairs from annotations
    gt_pairs = {tuple(sorted(p)) for p in event.get("bridge_pairs", [])}
    scored_pairs = {
        tuple(sorted([sp["post_a_id"], sp["post_b_id"]])): sp["label"]
        for sp in event.get("scored_pairs", [])
    }

    logger.info("Event: %s  bridge_pairs=%d  scored_pairs=%d",
                event["event_id"], len(gt_pairs), len(scored_pairs))

    # Build pipeline (no debater — we run MABD manually for diagnostics)
    pipeline = MBridgeNetPipeline(cfg, mlp_checkpoint="checkpoints/mlp_v25_fold2.pt")

    posts_raw = event["posts"]
    posts = []
    for rp in posts_raw:
        ts = datetime.fromisoformat(rp["timestamp"])
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        posts.append(Post(
            post_id=rp["post_id"], text=rp["text"], timestamp=ts,
            platform=rp["platform"], account_id=rp.get("account_id", ""),
            event_id=rp.get("event_id", event["event_id"]),
        ))

    logger.info("Running Stage 1+2 ...")
    candidates = pipeline.run_stage1_stage2(posts, event.get("hourly_volumes", []))

    mabd_zone  = [c for c in candidates if c.route == Route.MABD]
    direct_bridge = [c for c in candidates if c.route == Route.BRIDGE]
    discard    = [c for c in candidates if c.route == Route.DISCARD]

    logger.info("Routing: BRIDGE=%d  MABD=%d  DISCARD=%d",
                len(direct_bridge), len(mabd_zone), len(discard))

    # Attach ground-truth labels to MABD pairs
    for c in mabd_zone:
        key = tuple(sorted([c.post_a.post_id, c.post_b.post_id]))
        c.label = scored_pairs.get(key, -1)

    true_pos_in_mabd  = sum(1 for c in mabd_zone if c.label == 1)
    true_pos_in_bridge = sum(1 for c in direct_bridge
                             if tuple(sorted([c.post_a.post_id, c.post_b.post_id])) in gt_pairs)
    logger.info("MABD zone: %d true bridges / %d total (%.0f%%)",
                true_pos_in_mabd, len(mabd_zone),
                100 * true_pos_in_mabd / max(len(mabd_zone), 1))
    logger.info("Direct-BRIDGE zone: %d confirmed bridges", true_pos_in_bridge)

    # Sample pairs to inspect — prioritise true bridges so we see MABD behaviour on positives
    positives   = [c for c in mabd_zone if c.label == 1]
    negatives   = [c for c in mabd_zone if c.label == 0]
    unknowns    = [c for c in mabd_zone if c.label == -1]

    import random
    random.seed(args.seed)
    sample = positives[:args.sample // 2]
    remaining = args.sample - len(sample)
    sample += random.sample(negatives + unknowns, min(remaining, len(negatives + unknowns)))
    random.shuffle(sample)

    logger.info("Sampling %d pairs for MABD diagnosis (pos=%d neg/unk=%d)",
                len(sample), sum(1 for c in sample if c.label == 1),
                sum(1 for c in sample if c.label != 1))

    # Run MABD on sample
    import os
    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    debater = MABDDebater(client, model=cfg.stage3.llm_model,
                          temperature=cfg.stage3.llm_temperature,
                          max_tokens=cfg.stage3.llm_max_tokens)

    rows = []
    for i, c in enumerate(sample):
        key = tuple(sorted([c.post_a.post_id, c.post_b.post_id]))
        gt = "BRIDGE" if key in gt_pairs else ("NON-BRIDGE" if c.label == 0 else "UNKNOWN")
        logger.info("[%d/%d] Debating pair %s×%s  score_S=%.3f  gt=%s",
                    i+1, len(sample), c.post_a.platform, c.post_b.platform, c.score_S, gt)

        record = debater.debate(c)
        if record is None:
            rows.append({"gt": gt, "score_S": c.score_S, "mabd": "PARSE_FAIL",
                         "confidence": None, "passed_threshold": False,
                         "relation": "-", "factor": "-", "outcome": "-"})
            continue

        passed = record.confidence >= cfg.stage3.min_confidence
        mabd_verdict = "BRIDGE" if record.is_bridge else "NON-BRIDGE"
        correct = (mabd_verdict == gt) if gt != "UNKNOWN" else None
        rows.append({
            "gt":               gt,
            "score_S":          round(c.score_S, 3),
            "mabd":             mabd_verdict,
            "confidence":       round(record.confidence, 3),
            "passed_threshold": passed,
            "correct":          correct,
            "relation":         record.narrative_relation,
            "factor":           record.deciding_factor,
            "outcome":          record.debate_outcome,
        })

    # Print summary table
    print("\n" + "="*110)
    print(f"{'GT':<12} {'score_S':<9} {'MABD':<12} {'conf':<7} {'pass?':<7} {'correct?':<10} {'relation':<14} {'factor':<12} {'outcome'}")
    print("-"*110)
    for r in rows:
        print(f"{r['gt']:<12} {r['score_S']:<9} {r['mabd']:<12} "
              f"{str(r['confidence']):<7} {str(r['passed_threshold']):<7} "
              f"{str(r.get('correct','?')):<10} {r['relation']:<14} {r['factor']:<12} {r['outcome']}")
    print("="*110)

    # Aggregate stats
    decided   = [r for r in rows if r["mabd"] not in ("PARSE_FAIL",)]
    passed    = [r for r in decided if r["passed_threshold"]]
    flipped_bridge = [r for r in passed if r["mabd"] == "BRIDGE"]
    flipped_non    = [r for r in passed if r["mabd"] == "NON-BRIDGE"]
    correct_on_gt  = [r for r in passed if r.get("correct") is True]
    wrong_on_gt    = [r for r in passed if r.get("correct") is False]

    print(f"\nSummary ({len(rows)} pairs):")
    print(f"  Parse failures:            {sum(1 for r in rows if r['mabd']=='PARSE_FAIL')}")
    print(f"  Passed min_confidence:     {len(passed)} / {len(decided)}")
    print(f"  → Flipped to BRIDGE:       {len(flipped_bridge)}")
    print(f"  → Flipped to NON-BRIDGE:   {len(flipped_non)}")
    print(f"  Correct decisions (vs GT): {len(correct_on_gt)} / {len([r for r in passed if r.get('correct') is not None])}")
    print(f"  Wrong decisions (vs GT):   {len(wrong_on_gt)}")
    if passed:
        confs = [r["confidence"] for r in passed]
        print(f"  Confidence range:          {min(confs):.3f} – {max(confs):.3f}  mean={sum(confs)/len(confs):.3f}")


if __name__ == "__main__":
    main()
