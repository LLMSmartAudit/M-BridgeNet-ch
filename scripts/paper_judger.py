"""Paper Judger Agent — three-phase hybrid reviewer for M-BridgeNet.

Phase 1 — Claim Extraction: regex-scan the paper .tex for numerical claims.
Phase 2 — Fact Check:       compare extracted claims against eval_results.jsonl.
Phase 3 — LLM Review:       5-dimension ICWSM/AAAI-style academic review + synthesis.

Usage:
    OPENAI_API_KEY=sk-... .venv/bin/python scripts/paper_judger.py
    OPENAI_API_KEY=sk-... .venv/bin/python scripts/paper_judger.py --model gpt-5.4-nano
    OPENAI_API_KEY=sk-... .venv/bin/python scripts/paper_judger.py \\
        --paper ../../06-papers/Arxiv_PRIME_AI_Style_Template/M-BridgeNet-paper.tex \\
        --out logs/my_report.md
    OPENAI_API_KEY=sk-... .venv/bin/python scripts/paper_judger.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import re
import textwrap
from datetime import datetime, timezone
from pathlib import Path

try:
    from openai import OpenAI
except ImportError:
    raise SystemExit("openai package not installed — run: pip install openai")

# ── paths ─────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).parent.parent
DEFAULT_PAPER = (
    ROOT.parent.parent
    / "06-papers"
    / "Arxiv_PRIME_AI_Style_Template"
    / "M-BridgeNet-paper.tex"
)
EVAL_LOG_PATH = ROOT / "logs" / "eval_results.jsonl"
DEFAULT_OUT   = ROOT / "logs" / "paper_judge_report.md"

PAPER_MAX_CHARS = 30_000   # fed to LLM; covers abstract→conclusion comfortably
EVAL_WINDOW     = 10       # most-recent eval records to summarise

# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — Claim Extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_claims(tex: str) -> dict[str, list[dict]]:
    """Return {claim_type → [{value, line_no, raw}]} from paper .tex."""

    claims: dict[str, list[dict]] = {
        "ap_values":        [],
        "event_counts":     [],
        "train_test_split": [],
        "bridge_pairs":     [],
        "checkpoint":       [],
        "phase_windows":    [],
        "baselines":        [],
    }

    lines = tex.splitlines()

    # ── AP@k values  e.g.  AP@5~=~70.3\%  or  AP@50~=~65.5\%
    ap_pat = re.compile(r"AP@(\d+)[~\s]*=~[~\s]*([\d.]+)\\?%", re.IGNORECASE)
    # ── also  AP@5~=~70.34\%  or  \textbf{70.34}  or inline AP@5=70.3
    ap_num_pat = re.compile(r"AP@(\d+)[~\s]*=~?[~\s]*(\d+(?:\.\d+)?)", re.IGNORECASE)
    # ── main results table row values: \textbf{70.34} or plain 68.42
    # We look for the headline M-BridgeNet row with 6 numbers after "Stage 1+2"
    ap10_pat = re.compile(r"Stage\s+1\+2.*?(\d+\.\d+)\s*&\s*(\d+\.\d+)\s*&\s*(\d+\.\d+)\s*&\s*\\textbf\{(\d+\.\d+)\}\s*&\s*\\textbf\{(\d+\.\d+)\}\s*&\s*\\textbf\{(\d+\.\d+)\}", re.DOTALL)

    # ── event counts  47~events  or  15-event  or  32~train
    ev_pat    = re.compile(r"(\d+)[~\s]*events?", re.IGNORECASE)
    ev2_pat   = re.compile(r"(\d+)-event", re.IGNORECASE)
    split_pat = re.compile(r"(\d+)[~\s]*train[^i]", re.IGNORECASE)
    test_pat  = re.compile(r"(\d+)[~\s]*test", re.IGNORECASE)

    # ── bridge pair counts  2,362  or  5,500 bridge pairs
    bp_pat = re.compile(r"([\d,]+)\s*(?:annotated\s+)?bridge\s+pairs", re.IGNORECASE)
    bp2_pat = re.compile(r"approx\}?([\d,]+)[~\s]*(?:annotated\s+)?bridge", re.IGNORECASE)

    # ── checkpoint   mlp_v18_fold5.pt  or  mlp\_v18\_fold5.pt (LaTeX)
    ck_pat = re.compile(r"mlp[_\\]+v(\d+)[_\\]+fold(\d+)\.pt", re.IGNORECASE)
    ck_tex_pat = re.compile(r"mlp\\_v(\d+)\\_fold(\d+)\\.pt", re.IGNORECASE)

    # ── phase windows from Table 1 rows  e.g.  & 12 h \\  or  & 240 h \\
    pw_pat = re.compile(r"&\s*(\d+)\s*h\s*\\\\", re.IGNORECASE)

    # ── baseline comparison numbers  +18.2\%~AP@20  or  +31.9 AP@20
    bl_pat = re.compile(r"\+([\d.]+)[~\\\\%\s]*AP@(\d+)", re.IGNORECASE)

    seen_ck: set[str] = set()

    for i, line in enumerate(lines, start=1):
        # AP values
        for m in ap_pat.finditer(line):
            claims["ap_values"].append({"k": int(m.group(1)), "value": float(m.group(2)), "line": i, "raw": line.strip()})
        for m in ap_num_pat.finditer(line):
            # avoid duplicates
            entry = {"k": int(m.group(1)), "value": float(m.group(2)), "line": i, "raw": line.strip()}
            if not any(e["k"] == entry["k"] and e["value"] == entry["value"] and e["line"] == i
                       for e in claims["ap_values"]):
                claims["ap_values"].append(entry)

        # event counts
        for m in ev_pat.finditer(line):
            claims["event_counts"].append({"value": int(m.group(1)), "line": i, "raw": line.strip()})
        for m in ev2_pat.finditer(line):
            claims["event_counts"].append({"value": int(m.group(1)), "type": "N-event", "line": i, "raw": line.strip()})

        # train/test split
        for m in split_pat.finditer(line):
            claims["train_test_split"].append({"value": int(m.group(1)), "type": "train", "line": i, "raw": line.strip()})
        for m in test_pat.finditer(line):
            claims["train_test_split"].append({"value": int(m.group(1)), "type": "test", "line": i, "raw": line.strip()})

        # bridge pairs
        for m in bp_pat.finditer(line):
            val_str = m.group(1).replace(",", "")
            claims["bridge_pairs"].append({"value": int(val_str), "line": i, "raw": line.strip()})
        for m in bp2_pat.finditer(line):
            val_str = m.group(1).replace(",", "")
            claims["bridge_pairs"].append({"value": int(val_str), "line": i, "raw": line.strip()})

        # checkpoint
        for pat in (ck_pat, ck_tex_pat):
            for m in pat.finditer(line):
                name = f"mlp_v{m.group(1)}_fold{m.group(2)}.pt"
                if name not in seen_ck:
                    seen_ck.add(name)
                    claims["checkpoint"].append({"value": name, "line": i, "raw": line.strip()})

        # phase windows
        for m in pw_pat.finditer(line):
            claims["phase_windows"].append({"value": int(m.group(1)), "line": i, "raw": line.strip()})

        # baselines
        for m in bl_pat.finditer(line):
            claims["baselines"].append({"delta": float(m.group(1)), "k": int(m.group(2)), "line": i, "raw": line.strip()})

    return claims


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — Fact Check
# ─────────────────────────────────────────────────────────────────────────────

def load_best_eval(path: Path) -> dict | None:
    """Return the most recent full-pipeline run on test_real with v18 checkpoint."""
    if not path.exists():
        return None
    records = []
    for line in path.read_text(encoding="utf-8").strip().splitlines():
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            pass

    # prefer: mode=full_pipeline, test_real, mlp_v18_fold5, NOT simonly/always_s1s3/tco
    for r in reversed(records):
        ck = r.get("checkpoint", "") or ""
        data = r.get("data", "") or ""
        mode = r.get("mode", "") or ""
        if (mode == "full_pipeline"
                and "test_real" in data
                and "v18" in ck
                and "fold5" in ck
                and not r.get("simonly", False)
                and not r.get("always_s1s3", False)
                and not r.get("tco", False)):
            return r

    # fallback: any full_pipeline on test_real (not ablations)
    for r in reversed(records):
        ck = r.get("checkpoint", "") or ""
        data = r.get("data", "") or ""
        mode = r.get("mode", "") or ""
        if (mode == "full_pipeline"
                and "test_real" in data
                and not r.get("simonly", False)
                and not r.get("always_s1s3", False)
                and not r.get("tco", False)):
            return r

    return None


def _pct(v: float) -> float:
    """Normalise: if value > 2 assume already %, else ×100."""
    return v if v > 2 else v * 100


def fact_check(claims: dict, eval_rec: dict | None) -> str:
    """Return a Markdown table of claim vs. ground-truth checks."""
    rows: list[str] = []

    def row(claim: str, paper: str, truth: str, status: str) -> None:
        rows.append(f"| {claim} | {paper} | {truth} | {status} |")

    if eval_rec is None:
        return (
            "**⚠️ No matching eval record found** in `logs/eval_results.jsonl`.\n"
            "Run `evaluate.py --data data/cphot/processed/test_real "
            "--checkpoint checkpoints/mlp_v18_fold5.pt --use-s1s3-scoring "
            "--k 5 10 20 50` first.\n"
        )

    metrics = eval_rec.get("metrics", {})
    n_events = eval_rec.get("n_events", "?")
    ck_used   = Path(eval_rec.get("checkpoint", "?")).name

    header = (
        "| Claim | Paper value | Eval-log value | Status |\n"
        "|-------|-------------|----------------|--------|\n"
    )

    def find_ap(k: int) -> float | None:
        """Pull AP@k mean from claims dict."""
        for e in claims["ap_values"]:
            if e["k"] == k:
                return _pct(e["value"])
        return None

    def log_ap(k: int) -> float | None:
        key = f"AP@{k}"
        m = metrics.get(key)
        if m is None:
            return None
        mean = m.get("mean") if isinstance(m, dict) else m
        return _pct(float(mean)) if mean is not None else None

    TOLERANCE = 0.6  # pp

    # AP@5, @10, @20, @50
    for k in (5, 10, 20, 50):
        paper_val = find_ap(k)
        log_val   = log_ap(k)
        if paper_val is None:
            row(f"AP@{k}", "—", f"{log_val:.2f}%" if log_val else "?", "⚠️ not found in paper")
        elif log_val is None:
            row(f"AP@{k}", f"{paper_val:.1f}%", "?", "⚠️ not in log")
        else:
            diff = abs(paper_val - log_val)
            status = "✅" if diff <= TOLERANCE else f"❌ Δ={diff:.1f}pp"
            row(f"AP@{k}", f"{paper_val:.1f}%", f"{log_val:.2f}%", status)

    # n_events
    test_ev = [e for e in claims["train_test_split"] if e["type"] == "test"]
    paper_nev = test_ev[0]["value"] if test_ev else None
    status = "✅" if paper_nev == n_events else (f"❌ paper={paper_nev} log={n_events}" if paper_nev else "⚠️")
    row("Test events (n_events)", str(paper_nev) if paper_nev else "—", str(n_events), status)

    # checkpoint
    ck_claims = [e["value"] for e in claims["checkpoint"]]
    ck_ok = ck_used in ck_claims
    paper_ck = ck_claims[0] if ck_claims else "—"
    row("Checkpoint", paper_ck, ck_used, "✅" if ck_ok else f"❌ log uses {ck_used}")

    # phase windows: expect [12, 72, 120, 240] from Table 1
    pw_vals = [e["value"] for e in claims["phase_windows"]]
    expected_pw = [12, 72, 120, 240]
    pw_ok = all(v in pw_vals for v in expected_pw)
    row("Phase windows (12/72/120/240 h)", str(pw_vals[:8]), str(expected_pw), "✅" if pw_ok else "❌ mismatch")

    return header + "\n".join(rows)


def load_eval_summary(path: Path, last_n: int = EVAL_WINDOW) -> str:
    """Return a compact JSON summary of the last N eval runs."""
    if not path.exists():
        return "[eval log not found]"
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    records = []
    for line in lines[-last_n:]:
        try:
            r = json.loads(line)
            records.append({
                "ts":         r.get("timestamp", "")[:10],
                "mode":       r.get("mode", "?"),
                "checkpoint": Path(r.get("checkpoint") or "?").name,
                "n_events":   r.get("n_events", "?"),
                "metrics":    r.get("metrics", {}),
            })
        except json.JSONDecodeError:
            pass
    return json.dumps(records, indent=2, ensure_ascii=False)


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3 — LLM Academic Review
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_JUDGE = textwrap.dedent("""\
    You are a rigorous ML/CS conference reviewer at the level of ICWSM, AAAI, or WWW.
    You are evaluating a paper called "M-BridgeNet: Cross-Platform Bridge Node Detection
    for Hot Event Propagation in Chinese Social Media".

    Be critical, specific, and constructive.
    Where you identify a weakness, suggest how it could be addressed.
    Score each dimension 1–5 (1 = major flaw, 3 = acceptable, 5 = strong).
    Use Markdown with clear bullet points and bold for key findings.
    Do NOT praise vaguely — ground every strength and weakness in specific evidence from
    the paper content or the evaluation data provided.
""")

DIMENSIONS: list[tuple[str, str]] = [
    (
        "Task Definition & Motivation",
        textwrap.dedent("""\
            Evaluate the task definition and motivation:

            1. Is "bridge node" clearly and operationally defined (post-pair level,
               temporal precedence, narrative non-redundancy)?
            2. Is the cross-platform bridge detection task well-motivated
               (real-world relevance, gap in prior work, censorship-driven migration pattern)?
            3. Are the lifecycle phases (Emergence/Diffusion/Peak/Decline) principled
               or arbitrary? Are the phase window values (12/72/120/240 h) justified?
            4. Is CPHot a convincing benchmark?
               - 15-event test set: is this adequate for the claimed generalizability?
               - LLM-assisted annotation (GPT-4.1-mini) with 10% spot-check: is this reliable?
               - Bridge rate variance 4%–88%: does this represent real diversity or noise?
            5. Does the paper adequately distinguish bridge-node detection from
               related tasks (source detection, cascade prediction, account-level bridging)?

            End with: **Score — Task & Motivation: X/5**
            Then: bullet strengths, bullet weaknesses, numbered suggestions.
        """),
    ),
    (
        "Method Design & Novelty",
        textwrap.dedent("""\
            Evaluate the three-stage pipeline design and novelty:

            1. Stage 1 (BGE-large-zh + FAISS-HNSW): Is 100% recall on the 15-event test
               a meaningful claim given FAISS is tuned to achieve it? What if τ_coarse changes?
            2. Stage 2 — LifecycleMLP:
               - Is lifecycle conditioning novel vs fixed-window temporal filtering?
               - Is s₁×s₃ (semantic × rarity) multiplicative scoring principled?
               - Is the adaptive formula (s₁ alone for single-bucket; s₁×s₃ for multi-bucket)
                 well-motivated, or is it over-engineering a heuristic?
            3. Stage 3 — MABD (4-round debate):
               - Is a Proposer/Challenger/Rebuttal/Judge structure necessary,
                 or would a single LLM classifier + calibration suffice?
               - Is MABD invoked only on the medium-confidence tier? Is 30% coverage justified?
            4. Is the overall 3-stage system unnecessarily complex for the performance gains?
               (AP@50 from +3.12pp over Stage 1+2 for MABD; +5.8pp AP@5 from adaptive s₁×s₃)
            5. Does the paper cite sufficient related work on multi-signal fusion,
               temporal co-occurrence scoring, and LLM debate mechanisms?

            End with: **Score — Method Design: X/5**
            Then: bullet strengths, bullet weaknesses, numbered suggestions.
        """),
    ),
    (
        "Experimental Setup & Baselines",
        textwrap.dedent("""\
            Evaluate the experimental methodology:

            1. Test set adequacy:
               - 15 events, 2,362 bridge pairs: sufficient for statistically meaningful claims?
               - AP@5 variance: 18% (us_election) to 100% (6 events) — does this imply
                 the metric is not discriminative enough?
            2. Baselines:
               - SimOnly (cosine only), PairEncoder (Siamese-BGE), CrossEncoder (MacBERT):
                 are these comprehensive? What about BM25, dense retrieval re-rankers,
                 or a simple temporal-window filter as baseline?
               - The baseline comparison is on the 7-event original subset while M-BridgeNet
                 is reported on the 15-event set. Is this methodologically fair?
            3. Primary metric:
               - AP@K is used as primary; is this the right choice over NDCG@K or Recall@K?
               - F1-Strict@K is reported but not primary — is the justification convincing?
            4. Training protocol:
               - 5-fold CV on training events (32 events): is event-level splitting correct?
               - Is there a risk of event-level data leakage (e.g., same narrative across events)?
            5. Is the LLM annotation pipeline (GPT-4.1-mini at τ=0.75–0.90 cosine)
               a reproducible gold standard? How sensitive are results to annotation threshold?

            End with: **Score — Experimental Setup: X/5**
            Then: bullet strengths, bullet weaknesses, numbered suggestions.
        """),
    ),
    (
        "Results, Ablation & Analysis Quality",
        textwrap.dedent("""\
            Evaluate the reported results and ablation study:

            1. Main results:
               - AP@5=70.3%, AP@20=65.7%, AP@50=65.5% on 15-event test:
                 are these headline numbers presented with sufficient context
                 (e.g., per-event breakdown, confidence intervals)?
               - us_election AP@5=18%, zhang_xuefeng AP@5=30%, sora AP@5=28%:
                 are these failures adequately analyzed?
            2. Adaptive s₁×s₃ ablation:
               - MLP-S-direct gives AP@5=64.5% vs adaptive s₁×s₃ gives 70.3% (+5.8pp):
                 is this gain from adaptive routing or from the rarity signal itself?
               - Is there an ablation for non-adaptive s₁×s₃ (always multiply, pangmao 5%→80%)?
               - The uniform phase window (+1.1pp AP@5) outperforms the full model on AP@5:
                 does this undermine the lifecycle contribution claim?
            3. Signal ablations:
               - s₁: −18.3pp AP@5; s₂: −2.6pp; s₃ in MLP: −0.8pp. Does −0.8pp for s₃
                 suggest s₃'s role in the MLP is negligible (its value is purely in score_final)?
               - Uniform weights: −6.9pp AP@5. Is this sufficient to validate the MLP's value?
            4. Phase-level analysis:
               - Emergence F1=58%, Diffusion F1=14.7%: is this breakdown informative on 7-event data?
               - Is there a per-event table showing where the model succeeds and fails?
            5. MABD analysis:
               - 140 pairs from 7 events: is this sufficient to draw conclusions about debate quality?
               - 87.9% challenger_won: does the conservative design bias explain the MABD AP trade-off?

            End with: **Score — Results & Ablation: X/5**
            Then: bullet strengths, bullet weaknesses, numbered suggestions.
        """),
    ),
    (
        "Reproducibility & Completeness",
        textwrap.dedent("""\
            Evaluate reproducibility and paper completeness:

            1. Hyperparameter transparency:
               - τ_high=0.50, τ_low=0.35, λ=0.5, τ_fine=0.60, τ_coarse=0.50 all reported?
               - Phase windows (12/72/120/240h), MLP architecture (64 hidden, 50 epochs) reported?
               - How were τ_high and τ_low tuned — on what validation split?
            2. Annotation–evaluation circularity:
               - GPT-4.1-mini annotates CPHot AND serves as MABD Judge.
               - Is this circularity adequately disclosed and mitigated?
               - Would using a different annotation model (e.g., GPT-4o) as the judge
                 change results significantly?
            3. Dataset release:
               - CPHot is anonymized for review; will it be publicly released?
               - If not, can results be reproduced from the described pipeline?
            4. Known failure modes:
               - Low-density events (≤8 bridges): are failure modes analyzed?
               - Censored platform events (suzhou, 13 bridges): adequately disclosed?
               - Single-bucket events (pangmao): adaptive mechanism disclosed clearly?
            5. Ethical considerations:
               - The paper states "retrospective analysis only, not real-time surveillance."
               - Is this sufficient given the platform coverage (Weibo censorship dynamics)?
               - Are there dual-use concerns (propaganda detection, narrative manipulation)?

            End with: **Score — Reproducibility: X/5**
            Then: bullet strengths, bullet weaknesses, numbered suggestions.
        """),
    ),
]

SYNTHESIS_PROMPT = textwrap.dedent("""\
    You have reviewed five dimensions of the M-BridgeNet paper.
    Synthesise all five reviews into a final verdict.

    ## Overall Assessment

    ### Score Summary
    | Dimension | Score (1-5) | One-line rationale |
    |---|---|---|
    | Task & Motivation | X/5 | ... |
    | Method Design | X/5 | ... |
    | Experimental Setup | X/5 | ... |
    | Results & Ablation | X/5 | ... |
    | Reproducibility | X/5 | ... |
    | **Weighted Overall** | **X/5** | |

    ### Top 3 Strengths
    1. **[Strength]** — evidence from the paper
    2. **[Strength]** — evidence from the paper
    3. **[Strength]** — evidence from the paper

    ### Top 3 Critical Weaknesses
    1. **[Weakness]** *(severity: Minor / Major / Fatal)* — specific concern
    2. **[Weakness]** *(severity: Minor / Major / Fatal)* — specific concern
    3. **[Weakness]** *(severity: Minor / Major / Fatal)* — specific concern

    ### Top 3 Actionable Recommendations
    1. **[Recommendation]** — concrete, prioritised by impact
    2. **[Recommendation]** — concrete, prioritised by impact
    3. **[Recommendation]** — concrete, prioritised by impact

    ### Questions for the Authors
    List 3 specific questions a reviewer would ask.

    ### Verdict
    **[Accept / Weak-Accept / Weak-Reject / Reject]**
    *One-sentence justification citing the paper's strongest result and its most critical weakness.*
""")


def _call(
    client: OpenAI,
    model: str,
    system: str,
    user: str,
    dry_run: bool = False,
) -> str:
    if dry_run:
        return "[DRY-RUN — LLM not called]"
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        temperature=0.3,
    )
    return resp.choices[0].message.content.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Report writer
# ─────────────────────────────────────────────────────────────────────────────

def _build_report(
    model: str,
    fact_check_md: str,
    dimension_responses: list[tuple[str, str]],
    synthesis: str,
    claims_summary: str,
) -> str:
    ts = datetime.now(timezone.utc).isoformat()
    parts = [
        "# M-BridgeNet — Paper Judge Report",
        "",
        f"> Generated: {ts}  ",
        f"> Model: `{model}`",
        "",
        "---",
        "",
        "## Phase 1 + 2 — Automated Fact Check",
        "",
        "### Extracted Claims (summary)",
        claims_summary,
        "",
        "### Claim vs. Eval-Log Verification",
        "",
        fact_check_md,
        "",
        "---",
        "",
        "## Phase 3 — Academic Review",
        "",
    ]
    for title, response in dimension_responses:
        parts.append(f"### {title}")
        parts.append("")
        parts.append(response)
        parts.append("")
        parts.append("---")
        parts.append("")

    parts += [
        "## Final Verdict",
        "",
        synthesis,
        "",
    ]
    return "\n".join(parts)


def _claims_summary(claims: dict) -> str:
    lines = []
    ap_vals = {e["k"]: e["value"] for e in claims["ap_values"]}
    if ap_vals:
        lines.append("**AP@k claimed in paper:** " +
                     ", ".join(f"AP@{k}={v:.1f}%" for k, v in sorted(ap_vals.items())))
    cks = [e["value"] for e in claims["checkpoint"]]
    if cks:
        lines.append(f"**Checkpoint(s):** {', '.join(cks)}")
    pws = [e["value"] for e in claims["phase_windows"]]
    if pws:
        lines.append(f"**Phase windows:** {pws[:8]} h")
    bps = [e["value"] for e in claims["bridge_pairs"]]
    if bps:
        lines.append(f"**Bridge pair counts:** {bps[:6]}")
    n_events = sorted({e["value"] for e in claims["event_counts"]})
    if n_events:
        lines.append(f"**Event count values found:** {n_events}")
    return "\n".join(f"- {l}" for l in lines) if lines else "*(no claims extracted)*"


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hybrid Paper Judger Agent for M-BridgeNet"
    )
    parser.add_argument(
        "--paper", default=str(DEFAULT_PAPER),
        help="Path to the paper .tex file",
    )
    parser.add_argument(
        "--model", default="gpt-5.4-nano",
        help="OpenAI model (default: gpt-5.4-nano)",
    )
    parser.add_argument(
        "--out", default=str(DEFAULT_OUT),
        help="Output path for the judge report (Markdown)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Skip LLM calls; only run Phases 1 and 2 (fact-check only)",
    )
    args = parser.parse_args()

    paper_path = Path(args.paper)
    if not paper_path.exists():
        raise SystemExit(
            f"Paper not found: {paper_path}\n"
            "Pass --paper <path> to specify the .tex file."
        )

    if not args.dry_run:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise SystemExit("OPENAI_API_KEY not set")
        client = OpenAI(api_key=api_key)
    else:
        client = None  # type: ignore

    # ── Phase 1: Claim Extraction ──────────────────────────────────────────
    print("Phase 1 — Extracting claims from paper .tex ...")
    tex = paper_path.read_text(encoding="utf-8")
    claims = extract_claims(tex)

    ap_found   = len(claims["ap_values"])
    ck_found   = len(claims["checkpoint"])
    pw_found   = len(claims["phase_windows"])
    print(f"  AP@k claims found: {ap_found}")
    print(f"  Checkpoint refs:   {ck_found} → {[e['value'] for e in claims['checkpoint']]}")
    print(f"  Phase window refs: {pw_found} → {[e['value'] for e in claims['phase_windows']]}")

    # ── Phase 2: Fact Check ───────────────────────────────────────────────
    print("\nPhase 2 — Fact-checking against eval log ...")
    eval_rec = load_best_eval(EVAL_LOG_PATH)
    if eval_rec:
        ck = Path(eval_rec.get("checkpoint", "?")).name
        ts = eval_rec.get("timestamp", "?")[:10]
        print(f"  Best matching record: {ck} / {ts} / n_events={eval_rec.get('n_events')}")
    else:
        print("  ⚠️  No matching eval record found — Phase 2 will note this.")

    fact_check_md  = fact_check(claims, eval_rec)
    eval_summary   = load_eval_summary(EVAL_LOG_PATH)
    claims_summary = _claims_summary(claims)

    print("\n" + fact_check_md)

    if args.dry_run:
        print("\n[DRY-RUN] Skipping Phase 3 LLM review.")
        report = _build_report(
            model=args.model + " (dry-run)",
            fact_check_md=fact_check_md,
            dimension_responses=[(d[0], "[DRY-RUN]") for d in DIMENSIONS],
            synthesis="[DRY-RUN]",
            claims_summary=claims_summary,
        )
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report, encoding="utf-8")
        print(f"\nDry-run report saved → {out_path}")
        return

    # ── Phase 3: LLM Academic Review ─────────────────────────────────────
    print(f"\nPhase 3 — LLM review ({args.model}) ...")

    # Build shared context block
    paper_excerpt = tex[:PAPER_MAX_CHARS]
    if len(tex) > PAPER_MAX_CHARS:
        paper_excerpt += "\n...[truncated at 30K chars — covers abstract through conclusion]"

    context_block = textwrap.dedent(f"""\
        ## Paper Content (M-BridgeNet-paper.tex, first 30K chars)

        {paper_excerpt}

        ---

        ## Automated Fact-Check Report (Phase 2)

        {fact_check_md}

        ---

        ## Recent Evaluation Log (last {EVAL_WINDOW} runs)

        {eval_summary}
    """)

    dimension_responses: list[tuple[str, str]] = []
    conversation_so_far = ""

    for title, focus in DIMENSIONS:
        print(f"  ▶  {title} ...")
        user_msg = f"{context_block}\n\n---\n\n## Your Task\n\n{focus}"
        response = _call(client, args.model, SYSTEM_JUDGE, user_msg)
        dimension_responses.append((title, response))
        conversation_so_far += f"\n\n### {title}\n{response}"
        preview = "\n".join(response.splitlines()[:3])
        print(f"     {preview[:120]}")

    # Synthesis
    print("  ▶  Synthesising final verdict ...")
    synthesis_user = (
        f"{context_block}\n\n---\n\n"
        f"## Individual Dimension Reviews\n{conversation_so_far}\n\n"
        f"---\n\n## Your Task\n\n{SYNTHESIS_PROMPT}"
    )
    synthesis = _call(client, args.model, SYSTEM_JUDGE, synthesis_user)
    print("\n" + synthesis[:600])

    # ── Write Report ──────────────────────────────────────────────────────
    report = _build_report(
        model=args.model,
        fact_check_md=fact_check_md,
        dimension_responses=dimension_responses,
        synthesis=synthesis,
        claims_summary=claims_summary,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    print(f"\nReport saved → {out_path}")


if __name__ == "__main__":
    main()
