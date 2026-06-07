"""Four-round MABD prompt templates.

Each function returns a complete messages list for the LLM API.
All prompts are in English; the LLM is expected to reason in English
while handling Chinese text snippets in the post content.
"""
from __future__ import annotations
import os
from typing import Any

# ── Prompt features (default ON; set env=0 to disable for ablation) ──────────
#   A: expose the CrossEncoder text score (s5) to all agents  → MABD_USE_S5=0 disables
#   B: prepend three few-shot exemplars to system prompts      → MABD_FEWSHOT=0 disables
# Rationale: A+B align the implementation with the paper's stated MABD design
# (the paper describes few-shot exemplars and a full signal tuple including the
# text signal).  On the 67-event CPHot test these have negligible aggregate AP
# effect — MABD's null is structural — but A+B give the best-behaved debate
# (B improves recall, A improves precision), which matters as CPHot grows toward
# sparser events whose borderline zone actually fills with true bridges.
_USE_S5 = os.environ.get("MABD_USE_S5", "1") == "1"
_FEWSHOT = os.environ.get("MABD_FEWSHOT", "1") == "1"

# Few-shot exemplars (B). Grounded in the bridge definition:
# temporal precedence + narrative non-redundancy + cross-platform audience shift.
# The two NON-BRIDGE cases target the empirically dominant false-positive
# signature: high semantic similarity but coincidental co-occurrence — both the
# same-hour wire-copy case and the LONG-GAP independent-recoverage case (true
# bridges transfer fast, median ~8h; coincidental pairs span ~28h+ with no
# narrative evolution).  High text similarity (s5≈1.0) does NOT imply a bridge.
_FEWSHOT_BLOCK = (
    "\n\nReference exemplars (for calibration):\n"
    "  [BRIDGE] A Weibo post breaks a leak 6h before a Zhihu thread reframes it "
    "as an analytical Q&A for a different audience — temporal precedence + "
    "narrative reframing + audience shift → is_bridge=true (deciding_factor=temporal).\n"
    "  [NON-BRIDGE] Two platforms post near-identical official wire copy within "
    "the same hour — high cosine but no precedence and no narrative evolution "
    "(parallel coverage) → is_bridge=false (narrative_relation=coincidental).\n"
    "  [NON-BRIDGE] Two highly similar posts about an ongoing event sit ~30h "
    "apart with no sign that one seeded the other — independent re-coverage of a "
    "still-developing story, not a narrative transfer; high s5 alone is not "
    "evidence of bridging → is_bridge=false (narrative_relation=coincidental).\n"
    "  [BRIDGE] A Bilibili explainer video summary is paraphrased into a Zhihu "
    "answer a day later, lowering cosine but preserving the causal claim and "
    "shifting platforms → is_bridge=true (narrative_relation=paraphrase).\n"
    "Weigh temporal plausibility: a true bridge usually transfers while the "
    "narrative is hot (short, directional gap); a large gap with parallel framing "
    "and no evolution signals coincidence, not bridging.\n"
)


def _sys(content: str) -> str:
    """Append few-shot exemplars to a system prompt (B; default on)."""
    return content + _FEWSHOT_BLOCK if _FEWSHOT else content


def proposer_prompt(pair_context: dict[str, Any]) -> list[dict]:
    """Round 1: Proposer builds the pro-bridge case."""
    return [
        {
            "role": "system",
            "content": _sys(
                "You are the Proposer in a structured debate about whether "
                "post A is a cross-platform bridge node that introduced a "
                "narrative to the platform of post B.\n"
                "Your task: present the strongest possible case FOR this being "
                "a bridge pair. Be specific. Use the provided signals.\n"
                "Output format (JSON):\n"
                '{"argument": "<your argument>", "key_evidence": ["<item1>", ...]}'
            ),
        },
        {
            "role": "user",
            "content": _pair_context_block(pair_context),
        },
    ]


def challenger_prompt(pair_context: dict[str, Any], proposer_output: str) -> list[dict]:
    """Round 2: Challenger mounts adversarial rebuttals."""
    return [
        {
            "role": "system",
            "content": _sys(
                "You are the Challenger in a structured debate.\n"
                "The Proposer argued this is a bridge pair. Your task: identify "
                "the weakest points in the Proposer's argument and present the "
                "strongest case AGAINST this being a bridge pair.\n"
                "Output format (JSON):\n"
                '{"rebuttal": "<your rebuttal>", "weak_points": ["<item1>", ...]}'
            ),
        },
        {
            "role": "user",
            "content": (
                f"{_pair_context_block(pair_context)}\n\n"
                f"Proposer's argument:\n{proposer_output}"
            ),
        },
    ]


def rebuttal_prompt(
    pair_context: dict[str, Any],
    proposer_output: str,
    challenger_output: str,
) -> list[dict]:
    """Round 3: Proposer defends against the challenge."""
    return [
        {
            "role": "system",
            "content": _sys(
                "You are the Proposer again.\n"
                "The Challenger raised objections. Address each weak point "
                "raised and reinforce your original argument.\n"
                "Output format (JSON):\n"
                '{"defense": "<your defense>", "addressed_points": ["<item1>", ...]}'
            ),
        },
        {
            "role": "user",
            "content": (
                f"{_pair_context_block(pair_context)}\n\n"
                f"Your original argument:\n{proposer_output}\n\n"
                f"Challenger's rebuttal:\n{challenger_output}"
            ),
        },
    ]


def judge_prompt(
    pair_context: dict[str, Any],
    proposer_output: str,
    challenger_output: str,
    rebuttal_output: str,
) -> list[dict]:
    """Round 4: Judge synthesizes the debate into a structured verdict."""
    return [
        {
            "role": "system",
            "content": _sys(
                "You are the Judge in a structured debate about bridge node detection.\n"
                "Review all arguments and produce a final structured verdict.\n\n"
                "Output ONLY valid JSON with exactly these fields:\n"
                "{\n"
                '  "is_bridge": true/false,\n'
                '  "confidence": <float 0.0-1.0>,\n'
                '  "narrative_relation": "paraphrase"|"elaboration"|"reaction"|"coincidental",\n'
                '  "deciding_factor": "temporal"|"semantic"|"rarity"|"structural"|"none",\n'
                '  "debate_outcome": "proposer_won"|"challenger_won"|"balanced"\n'
                "}"
            ),
        },
        {
            "role": "user",
            "content": (
                f"{_pair_context_block(pair_context)}\n\n"
                f"Proposer:\n{proposer_output}\n\n"
                f"Challenger:\n{challenger_output}\n\n"
                f"Proposer's defense:\n{rebuttal_output}\n\n"
                "Now render your verdict as JSON."
            ),
        },
    ]


def _pair_context_block(ctx: dict[str, Any]) -> str:
    """Format pair metadata into a readable context string."""
    return (
        f"Post A (bridge source candidate)\n"
        f"  ID: {ctx['post_a_id']}  Platform: {ctx['platform_a']}\n"
        f"  Time: {ctx['timestamp_a']}\n"
        f"  Text: {ctx['text_a']}\n\n"
        f"Post B (bridge target)\n"
        f"  ID: {ctx['post_b_id']}  Platform: {ctx['platform_b']}\n"
        f"  Time: {ctx['timestamp_b']}\n"
        f"  Text: {ctx['text_b']}\n\n"
        f"Signals:\n"
        f"  s1 (semantic similarity): {ctx['s1']:.3f}\n"
        f"  s2 (temporal gap score):  {ctx['s2']:.3f}\n"
        f"  s3 (migration rarity):    {ctx['s3']:.3f}\n"
        f"  s4 (graph BC):            {ctx['s4']:.3f}\n"
        + (f"  s5 (cross-encoder text):  {ctx.get('s5', 0.0):.3f}\n" if _USE_S5 else "")
        + f"  Composite score S:        {ctx['score_S']:.3f}\n"
        f"  Lifecycle phase:          {ctx['phase']}"
    )
