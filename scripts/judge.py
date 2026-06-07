"""Judger Agent — evaluates M-BridgeNet's methods, experiments, and results.

The agent runs five structured evaluation rounds (each a separate LLM call),
then synthesises a final verdict.  Output goes to logs/judge_report.md and
is also printed to stdout.

Usage:
    OPENAI_API_KEY=sk-... .venv/bin/python scripts/judge.py
    OPENAI_API_KEY=sk-... .venv/bin/python scripts/judge.py --model gpt-4o
    OPENAI_API_KEY=sk-... .venv/bin/python scripts/judge.py --model o3 --out logs/my_report.md
"""
import argparse
import json
import os
import textwrap
from datetime import datetime, timezone
from pathlib import Path

try:
    from openai import OpenAI
except ImportError:
    raise SystemExit("openai package not installed — run: pip install openai")

# ── paths ────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).parent.parent
SUMMARY_PATH   = ROOT / "EXPERIMENT_SUMMARY.md"
EVAL_LOG_PATH  = ROOT / "logs" / "eval_results.jsonl"
MABD_LOG_PATH  = ROOT / "logs" / "mabd_analysis.json"
DEFAULT_OUT    = ROOT / "logs" / "judge_report.md"

# ── helpers ──────────────────────────────────────────────────────────────────

def _load_text(path: Path, max_chars: int = 8000) -> str:
    if not path.exists():
        return f"[file not found: {path}]"
    text = path.read_text(encoding="utf-8")
    if len(text) > max_chars:
        text = text[:max_chars] + "\n...[truncated]"
    return text


def _load_eval_log(path: Path, last_n: int = 20) -> str:
    if not path.exists():
        return "[eval log not found]"
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    records = []
    for line in lines[-last_n:]:
        try:
            r = json.loads(line)
            records.append({
                "timestamp": r.get("timestamp", "")[:10],
                "mode":      r.get("mode", "?"),
                "checkpoint": Path(r.get("checkpoint", "?")).name,
                "metrics":   r.get("metrics", {}),
                "n_events":  r.get("n_events", "?"),
            })
        except json.JSONDecodeError:
            pass
    return json.dumps(records, indent=2, ensure_ascii=False)


def _call(client: OpenAI, model: str, system: str, user: str) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        temperature=0.3,
    )
    return resp.choices[0].message.content.strip()


# ── evaluation dimensions ────────────────────────────────────────────────────

SYSTEM_JUDGE = textwrap.dedent("""\
    You are a rigorous ML conference reviewer (NeurIPS / ACL / WWW level).
    Your job is to evaluate an academic system called M-BridgeNet,
    which detects cross-platform "bridge nodes" in social-media events.
    Be critical, specific, and constructive.  Where you identify a weakness,
    suggest how it could be addressed.  Score each dimension 1–5
    (1 = major flaw, 3 = acceptable, 5 = strong).
    Use Markdown formatting in your response.
""")


DIMENSIONS = [
    # (title, focus_prompt)
    (
        "Task Definition & Motivation",
        textwrap.dedent("""\
            Evaluate the task definition and motivation for bridge-node detection:
            1. Is "bridge node" clearly and operationally defined?
            2. Is the cross-platform bridge detection task well-motivated
               (real-world relevance, gap in prior work)?
            3. Are the three lifecycle phases (emergence/diffusion/peak/decline)
               principled or ad-hoc?
            4. Is CPHot a convincing benchmark dataset?
               Consider: size (3 test events), annotation method (LLM-assisted),
               bridge rate variance (2% – 88%), platform coverage.

            Score: Task & Motivation (1–5): X/5

            Provide bullet-point strengths and weaknesses, then specific suggestions.
        """),
    ),
    (
        "Method Design & Novelty",
        textwrap.dedent("""\
            Evaluate the three-stage pipeline design and its novelty:
            1. Stage 1 (BGE + FAISS-HNSW): Is it a reasonable candidate retrieval step?
            2. Stage 2 (LifecycleMLP + 3-way routing): Is the lifecycle phase
               conditioning principled? Is s1×s3 multiplicative scoring novel
               and well-motivated vs simpler alternatives?
            3. Stage 3 (MABD 4-round debate): Is a multi-agent debate necessary here,
               or would a single LLM call suffice?  Is the Proposer/Challenger/Rebuttal/Judge
               structure justified by the results?
            4. Is the overall system unnecessarily complex for the gains it achieves?

            Score: Method Design (1–5): X/5

            Strengths, weaknesses, suggestions.
        """),
    ),
    (
        "Experimental Setup & Baselines",
        textwrap.dedent("""\
            Evaluate the experimental methodology:
            1. Test set size: only 3 events (656–1,008 posts each).
               Is this sufficient to draw reliable conclusions?
            2. Are the baselines fair and comprehensive?
               (SimOnly, Stage 1+2 MLP, Stage 1+2+s1×s3, full +MABD)
               What important baselines are missing?
            3. Is F1-Strict@K the right primary metric for this retrieval task?
               Should AP@K be the headline metric instead?
            4. Is the 5-fold cross-validation on training data appropriate
               given the test set is held-out events (not random pairs)?
            5. LLM annotation at 15–25% bridge rate: is this a reliable gold standard?

            Score: Experimental Setup (1–5): X/5

            Strengths, weaknesses, suggestions.
        """),
    ),
    (
        "Results Analysis & Ablation Quality",
        textwrap.dedent("""\
            Evaluate the reported results and ablation study:
            1. AP@5 = 100% on 3 events — is this a meaningful result or
               a ceiling artifact of the small test set?
            2. Ablation rows D (w/o s2) and E (w/o s3) show positive Δ vs the
               MLP baseline when retrained — does this undermine the claim that
               s2 and s3 contribute positively?
            3. Ablation row A (w/o Lifecycle) is an inference-time approximation,
               not a properly retrained ablation.  How much does this weaken
               the lifecycle claim?
            4. Per-phase F1 (Diffusion = 6.41%) — is the breakdown informative
               or is the sample too small per phase to be meaningful?
            5. MABD analysis: only 60 pairs from 3 events.  Are the debate quality
               conclusions (83.3% bridge rate for proposer_won) reliable?

            Score: Results & Ablation (1–5): X/5

            Strengths, weaknesses, suggestions.
        """),
    ),
    (
        "Reproducibility & Completeness",
        textwrap.dedent("""\
            Evaluate reproducibility and completeness of the work:
            1. Are all hyperparameters (τ_high, τ_low, λ, phase windows) reported?
            2. Is the dataset (CPHot) publicly available, or only used internally?
               Does this limit reproducibility?
            3. Is the LLM annotation pipeline (GPT for bridge labeling) a
               reproducible gold standard, or does it vary with model versions?
            4. Are the inference-time ablations (rows A, B) clearly distinguished
               from retrained ablations (C, D, E) in the paper?
            5. Are known failure modes (censored events, Challenger conservatism,
               diffusion phase) adequately disclosed?

            Score: Reproducibility (1–5): X/5

            Strengths, weaknesses, suggestions.
        """),
    ),
]


SYNTHESIS_PROMPT = textwrap.dedent("""\
    You have now reviewed five dimensions of the M-BridgeNet system.
    Based on all five evaluations above, provide:

    ## Overall Assessment

    ### Summary Scores
    | Dimension | Score |
    |---|---|
    | Task & Motivation | X/5 |
    | Method Design | X/5 |
    | Experimental Setup | X/5 |
    | Results & Ablation | X/5 |
    | Reproducibility | X/5 |
    | **Overall** | **X/5** |

    ### Top 3 Strengths
    (Numbered list — specific and evidence-backed)

    ### Top 3 Critical Weaknesses
    (Numbered list — specific, with severity: Minor / Major / Fatal)

    ### Top 3 Actionable Recommendations
    (Numbered list — concrete, prioritised by impact)

    ### Accept / Weak-Accept / Weak-Reject / Reject
    (One-line verdict with one-sentence justification)
""")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Judger Agent for M-BridgeNet")
    parser.add_argument("--model", default="gpt-4.1",
                        help="OpenAI model to use (default: gpt-4.1)")
    parser.add_argument("--out", default=str(DEFAULT_OUT),
                        help="Output path for the judge report (Markdown)")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY not set")

    client = OpenAI(api_key=api_key)

    # ── load context ──────────────────────────────────────────────────────────
    summary   = _load_text(SUMMARY_PATH)
    eval_log  = _load_eval_log(EVAL_LOG_PATH)
    mabd_log  = _load_text(MABD_LOG_PATH, max_chars=2000)

    context = textwrap.dedent(f"""\
        ## Experiment Summary
        {summary}

        ## Recent Evaluation Log (last 20 runs, JSON)
        {eval_log}

        ## MABD Analysis
        {mabd_log}
    """)

    # ── run evaluation rounds ─────────────────────────────────────────────────
    print(f"Judger Agent — model: {args.model}")
    print("=" * 60)

    dimension_responses: list[tuple[str, str]] = []
    conversation_so_far = ""  # accumulate for synthesis

    for title, focus in DIMENSIONS:
        print(f"\n▶  Evaluating: {title} ...")
        user_msg = f"{context}\n\n---\n\n{focus}"
        response = _call(client, args.model, SYSTEM_JUDGE, user_msg)
        dimension_responses.append((title, response))
        conversation_so_far += f"\n\n### {title}\n{response}"
        # print first 3 lines as preview
        preview = "\n".join(response.splitlines()[:3])
        print(f"   {preview}")

    # ── synthesis ─────────────────────────────────────────────────────────────
    print("\n▶  Synthesising final verdict ...")
    synthesis_user = (
        f"{context}\n\n---\n\n"
        f"## Individual Dimension Reviews\n{conversation_so_far}\n\n"
        f"---\n\n{SYNTHESIS_PROMPT}"
    )
    synthesis = _call(client, args.model, SYSTEM_JUDGE, synthesis_user)
    print("\n" + synthesis)

    # ── write report ──────────────────────────────────────────────────────────
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    report_parts = [
        f"# M-BridgeNet — Judger Report",
        f"",
        f"> Generated: {datetime.now(timezone.utc).isoformat()}  ",
        f"> Model: `{args.model}`",
        f"",
        f"---",
        f"",
    ]
    for title, response in dimension_responses:
        report_parts.append(f"## {title}\n\n{response}\n")
        report_parts.append("---\n")
    report_parts.append(f"## Final Verdict\n\n{synthesis}\n")

    out_path.write_text("\n".join(report_parts), encoding="utf-8")
    print(f"\nReport saved → {out_path}")


if __name__ == "__main__":
    main()
