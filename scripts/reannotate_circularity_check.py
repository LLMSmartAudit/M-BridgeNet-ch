"""Circularity-check re-annotation study (Tier 3b).

Probes the W6 annotation-criterion circularity: the original criterion (ii)
uses cosine ≥ 0.70 (same BGE space as s1) for narrative non-redundancy.
This script re-annotates a stratified sample from 5 test events using
GPT-5.4-nano with a **purely causal prompt that contains no reference to
cosine similarity or embedding distance**.

If the two label sets agree highly (Cohen's κ ≥ 0.70), the circularity is
likely benign — BGE cosine similarity is a faithful proxy for the causal
criterion at this threshold. If agreement is low (κ < 0.50), the original
labels are co-influenced by the BGE representation, inflating AP@K estimates.

Selected events (5, diverse):
  - suzhou_realestate_arrest_001   (52 labeled pairs,  13 bridge)
  - deepseek_viral_001             (32 labeled pairs,   8 bridge)
  - sora_debut_001                 (56 labeled pairs,  14 bridge)
  - eileen_gu_identity_001         (36 labeled pairs,   9 bridge)
  - sam_altman_ousting_001         (312 labeled pairs, 78 bridge)
    → capped at 78 bridge + 78 non-bridge = 156 pairs (stratified)

Total pairs annotated: ~332.

Usage:
    cd /path/to/M-BridgeNet
    set -a && source .env && set +a
    .venv/bin/python scripts/reannotate_circularity_check.py \\
        --data data/cphot/processed/test_real \\
        --model gpt-5.4-nano \\
        --out  logs/circularity_check_results.json
"""

import argparse
import json
import logging
import os
import random
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Events to re-annotate (event_id -> max non-bridge cap; None = use all)
TARGET_EVENTS = {
    "suzhou_realestate_arrest_001":  None,   # 52 pairs total
    "deepseek_viral_001":            None,   # 32 pairs total
    "sora_debut_001":                None,   # 56 pairs total
    "eileen_gu_identity_001":        None,   # 36 pairs total
    "sam_altman_ousting_001":        78,     # cap non-bridge at 78 (= #bridge)
}

# ── Causal-only annotation prompt ──────────────────────────────────────────
# KEY DESIGN CHOICE: criterion (ii) is rewritten WITHOUT any cosine similarity
# or embedding-distance language. The prompt asks the LLM to make a purely
# causal / temporal inference about narrative origin.

SYSTEM_PROMPT = """You are an expert analyst of cross-platform social media propagation.
Your task is to judge whether a post on Platform A is a "bridge post" that causally
introduced a specific narrative to Platform B.

A post pair (post_A on Platform_A, post_B on Platform_B) is a BRIDGE PAIR if and only if
ALL THREE conditions hold:

(i)  TEMPORAL PRECEDENCE: post_A was published BEFORE post_B (check the timestamps).

(ii) NARRATIVE ORIGIN: The specific event narrative, framing, or key claim in post_B
     was likely introduced to Platform B by post_A — meaning post_A is a plausible
     proximate causal source of post_B's narrative on Platform B.
     This is TRUE when post_B's content closely follows post_A's narrative AND it is
     unlikely that post_B's author independently arrived at the same narrative without
     exposure to post_A or a post that itself came from post_A.
     This is FALSE when: (a) post_B's narrative is generic/widely-known on Platform B
     independently of post_A, or (b) post_B likely originates from a different source.
     DO NOT reference text similarity scores or embedding distances in your reasoning.
     Judge solely on the causal plausibility of narrative transfer.

(iii) CROSS-PLATFORM AUDIENCE SHIFT: post_B is primarily engaging an audience of
      Platform B users (not Platform A transplants). This is typically satisfied when
      post_B's replies/comments are from Platform B regulars.

Respond with EXACTLY this JSON object and nothing else:
{"label": <0 or 1>, "reason": "<one sentence>"}

label=1 means BRIDGE PAIR (all three conditions satisfied).
label=0 means NOT A BRIDGE PAIR (at least one condition fails).
"""

USER_TEMPLATE = """Platform A: {platform_a}
Post A (id={post_a_id}, published {time_a}):
\"\"\"{text_a}\"\"\"

Platform B: {platform_b}
Post B (id={post_b_id}, published {time_b}):
\"\"\"{text_b}\"\"\"

Is this a bridge pair? Respond with JSON only."""


def annotate_pair(client, pair: dict, event_posts: dict, model: str,
                  max_retries: int = 5) -> dict | None:
    """Call GPT-5.4-nano to label one pair. Returns {"label": 0|1, "reason": str}."""
    pa = event_posts.get(pair["post_a_id"], {})
    pb = event_posts.get(pair["post_b_id"], {})

    text_a = (pa.get("content") or pa.get("text") or "")[:800]
    text_b = (pb.get("content") or pb.get("text") or "")[:800]
    time_a = pa.get("created_at") or pa.get("timestamp") or "unknown"
    time_b = pb.get("created_at") or pb.get("timestamp") or "unknown"

    user_msg = USER_TEMPLATE.format(
        platform_a=pair.get("platform_a", pa.get("platform", "?")),
        platform_b=pair.get("platform_b", pb.get("platform", "?")),
        post_a_id=pair["post_a_id"],
        post_b_id=pair["post_b_id"],
        time_a=time_a,
        time_b=time_b,
        text_a=text_a or "(no text)",
        text_b=text_b or "(no text)",
    )

    delay = 2.0
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_msg},
                ],
                temperature=0,
                response_format={"type": "json_object"},
                max_completion_tokens=120,
            )
            raw = resp.choices[0].message.content or ""
            parsed = json.loads(raw)
            label = int(parsed.get("label", -1))
            if label not in (0, 1):
                raise ValueError(f"unexpected label {label}")
            return {"label": label, "reason": parsed.get("reason", "")}
        except Exception as e:
            logger.warning("Attempt %d failed for pair (%s, %s): %s",
                           attempt + 1, pair["post_a_id"], pair["post_b_id"], e)
            time.sleep(delay)
            delay = min(delay * 2, 60)
    return None


def cohen_kappa(y1, y2):
    """Compute Cohen's κ for two binary label lists."""
    n = len(y1)
    if n == 0:
        return float("nan")
    agree = sum(a == b for a, b in zip(y1, y2))
    po = agree / n
    p1 = sum(y1) / n
    p2 = sum(y2) / n
    pe = p1 * p2 + (1 - p1) * (1 - p2)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",  required=True)
    parser.add_argument("--model", default="gpt-5.4-nano")
    parser.add_argument("--out",   default="logs/circularity_check_results.json")
    parser.add_argument("--seed",  type=int, default=42)
    args = parser.parse_args()

    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    rng = random.Random(args.seed)
    data_dir = Path(args.data)

    all_results = []
    global_orig, global_new = [], []

    for event_id, nb_cap in TARGET_EVENTS.items():
        event_file = data_dir / f"{event_id}.json"
        if not event_file.exists():
            logger.warning("Event file not found: %s", event_file)
            continue

        event = json.loads(event_file.read_text())

        # Build post lookup
        posts = {p["post_id"]: p for p in event.get("posts", [])}

        # Stratified sample from scored_pairs
        labeled = [p for p in event.get("scored_pairs", []) if p.get("label", -1) != -1]
        bridges    = [p for p in labeled if p["label"] == 1]
        nonbridges = [p for p in labeled if p["label"] == 0]

        if nb_cap is not None and len(nonbridges) > nb_cap:
            rng.shuffle(nonbridges)
            nonbridges = nonbridges[:nb_cap]

        pairs_to_annotate = bridges + nonbridges
        rng.shuffle(pairs_to_annotate)

        logger.info("Event %-40s  %d pairs (%d bridge, %d non-bridge)",
                    event_id, len(pairs_to_annotate), len(bridges), len(nonbridges))

        event_results = []
        for i, pair in enumerate(pairs_to_annotate):
            result = annotate_pair(client, pair, posts, args.model)
            if result is None:
                logger.warning("Skipping pair %s/%s (all retries failed)",
                               pair["post_a_id"], pair["post_b_id"])
                continue
            rec = {
                "event_id":   event_id,
                "post_a_id":  pair["post_a_id"],
                "post_b_id":  pair["post_b_id"],
                "orig_label": int(pair["label"]),
                "new_label":  result["label"],
                "agree":      int(pair["label"]) == result["label"],
                "reason":     result["reason"],
                "s1":         pair.get("s1"),
            }
            event_results.append(rec)
            global_orig.append(rec["orig_label"])
            global_new.append(rec["new_label"])

            if (i + 1) % 20 == 0:
                logger.info("  [%d/%d] running agreement=%.3f",
                            i + 1, len(pairs_to_annotate),
                            sum(r["agree"] for r in event_results) / len(event_results))
            time.sleep(0.3)  # light rate-limit buffer

        # Per-event stats
        n = len(event_results)
        if n:
            agree_rate = sum(r["agree"] for r in event_results) / n
            kappa = cohen_kappa(
                [r["orig_label"] for r in event_results],
                [r["new_label"]  for r in event_results],
            )
            # Bridge-only agreement (most critical for W6)
            bridge_recs = [r for r in event_results if r["orig_label"] == 1]
            bridge_agree = (sum(r["agree"] for r in bridge_recs) / len(bridge_recs)
                            if bridge_recs else float("nan"))
            logger.info("  %-40s  n=%d  agree=%.3f  κ=%.3f  bridge_agree=%.3f",
                        event_id, n, agree_rate, kappa, bridge_agree)

        all_results.extend(event_results)

    # Global stats
    n_total = len(all_results)
    overall_agree = sum(r["agree"] for r in all_results) / n_total if n_total else 0
    kappa_global  = cohen_kappa(global_orig, global_new)
    bridge_recs   = [r for r in all_results if r["orig_label"] == 1]
    nb_recs       = [r for r in all_results if r["orig_label"] == 0]
    bridge_agree  = (sum(r["agree"] for r in bridge_recs) / len(bridge_recs)
                     if bridge_recs else float("nan"))
    nb_agree      = (sum(r["agree"] for r in nb_recs) / len(nb_recs)
                     if nb_recs else float("nan"))

    # False positive / false negative rates (orig=0 but new=1, and vice versa)
    fp_rate = (sum(1 for r in nb_recs   if r["new_label"] == 1) / len(nb_recs)
               if nb_recs else 0)
    fn_rate = (sum(1 for r in bridge_recs if r["new_label"] == 0) / len(bridge_recs)
               if bridge_recs else 0)

    print(f"\n{'='*60}")
    print(f"CIRCULARITY CHECK RESULTS  (model={args.model})")
    print(f"{'='*60}")
    print(f"  Total pairs annotated : {n_total}")
    print(f"  Bridge pairs          : {len(bridge_recs)}")
    print(f"  Non-bridge pairs      : {len(nb_recs)}")
    print(f"")
    print(f"  Overall agreement     : {overall_agree*100:.1f}%")
    print(f"  Cohen's κ             : {kappa_global:.3f}")
    print(f"  Bridge agreement      : {bridge_agree*100:.1f}%")
    print(f"  Non-bridge agreement  : {nb_agree*100:.1f}%")
    print(f"")
    print(f"  False-positive rate   : {fp_rate*100:.1f}%  (orig=0, new=1)")
    print(f"  False-negative rate   : {fn_rate*100:.1f}%  (orig=1, new=0)")
    print(f"")
    if kappa_global >= 0.70:
        verdict = "HIGH AGREEMENT — circularity appears BENIGN"
    elif kappa_global >= 0.50:
        verdict = "MODERATE AGREEMENT — circularity present but mild"
    else:
        verdict = "LOW AGREEMENT — circularity may be SUBSTANTIAL"
    print(f"  Verdict               : {verdict}")

    # Save
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "model":               args.model,
        "n_pairs":             n_total,
        "n_bridge":            len(bridge_recs),
        "n_nonbridge":         len(nb_recs),
        "overall_agreement":   round(overall_agree, 4),
        "cohen_kappa":         round(kappa_global, 4),
        "bridge_agreement":    round(bridge_agree, 4),
        "nonbridge_agreement": round(nb_agree, 4),
        "fp_rate":             round(fp_rate, 4),
        "fn_rate":             round(fn_rate, 4),
        "verdict":             verdict,
        "per_pair_results":    all_results,
    }
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    logger.info("Results saved → %s", out_path)


if __name__ == "__main__":
    main()
