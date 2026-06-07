"""Bootstrap 95% CI and Wilcoxon signed-rank test for M-BridgeNet vs SimOnly.

Reads per_event_metrics from logs/eval_results.jsonl, matches the M-BridgeNet run
and the SimOnly run by their timestamps (most-recent pair by default), then:

  1. Prints per-event AP table (side-by-side model vs baseline)
  2. Computes bootstrap 95% CI on mean AP@k (N=10,000 resamples)
  3. Runs one-sided Wilcoxon signed-rank test (H1: model > baseline)

Usage:
    python scripts/bootstrap_ci.py                     # use two most-recent runs
    python scripts/bootstrap_ci.py --k 5 20 50
    python scripts/bootstrap_ci.py \\
        --model-ts  2026-05-17T10:00:00+00:00 \\
        --simonly-ts 2026-05-17T09:55:00+00:00

Output is also saved to logs/bootstrap_ci_report.md.
"""
import argparse
import json
import random
import sys
from pathlib import Path

LOG_PATH = Path("logs/eval_results.jsonl")
REPORT_PATH = Path("logs/bootstrap_ci_report.md")
N_BOOTSTRAP = 10_000
SEED = 42

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def load_runs(log_path: Path) -> list[dict]:
    """Return all full-pipeline runs that have per_event_metrics."""
    runs = []
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("mode") == "full_pipeline" and "per_event_metrics" in rec:
                runs.append(rec)
    return runs


def pick_run(runs: list[dict], ts: str | None, simonly: bool) -> dict:
    """Pick a run by timestamp prefix or fallback to most-recent matching simonly flag."""
    if ts:
        for r in reversed(runs):
            if r["timestamp"].startswith(ts):
                return r
        raise ValueError(f"No run with timestamp starting '{ts}'")
    # most-recent that matches simonly flag
    for r in reversed(runs):
        if bool(r.get("simonly")) == simonly:
            return r
    raise ValueError(
        f"No {'simonly' if simonly else 'non-simonly'} full-pipeline run with "
        "per_event_metrics found in log."
    )


def ap_vector(run: dict, k: int) -> dict[str, float]:
    """Map event_id → AP@k for a run."""
    key = f"AP@{k}"
    return {
        rec["event_id"]: rec.get(key, 0.0)
        for rec in run["per_event_metrics"]
    }


def bootstrap_ci(values: list[float], n: int = N_BOOTSTRAP, seed: int = SEED,
                 alpha: float = 0.05) -> tuple[float, float, float]:
    """Return (mean, ci_low, ci_high) via percentile bootstrap."""
    rng = random.Random(seed)
    means = []
    for _ in range(n):
        sample = [rng.choice(values) for _ in values]
        means.append(sum(sample) / len(sample))
    means.sort()
    lo = int(alpha / 2 * n)
    hi = int((1 - alpha / 2) * n)
    return sum(values) / len(values), means[lo], means[hi]


def wilcoxon_signed_rank(x: list[float], y: list[float]) -> tuple[float, str]:
    """One-sided Wilcoxon signed-rank test: H1: x > y.

    Uses normal approximation (valid for n ≥ 10).
    Returns (p_value, significance_string).
    """
    import math

    diffs = [xi - yi for xi, yi in zip(x, y)]
    nonzero = [(abs(d), d > 0) for d in diffs if d != 0]
    if not nonzero:
        return 1.0, "n.s."
    n = len(nonzero)
    # rank ties: sort by abs diff, assign average ranks
    nonzero_sorted = sorted(enumerate(nonzero), key=lambda t: t[1][0])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n and nonzero_sorted[j][1][0] == nonzero_sorted[i][1][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[nonzero_sorted[k][0]] = avg_rank
        i = j
    w_plus = sum(r for r, (_, pos) in zip(ranks, nonzero) if pos)
    # normal approximation
    mu = n * (n + 1) / 4.0
    sigma = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    z = (w_plus - mu) / sigma
    # one-sided p-value (H1: x > y → large W+)
    # Φ(z) approximation (Zelen & Severo 1964)
    def phi(z: float) -> float:
        t = 1.0 / (1.0 + 0.2316419 * abs(z))
        poly = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937
               + t * (-1.821255978 + t * 1.330274429))))
        pdf = math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)
        cdf = 1.0 - pdf * poly
        return cdf if z >= 0 else 1.0 - cdf

    p_val = 1.0 - phi(z)  # one-sided upper tail

    if p_val < 0.001:
        sig = "***"
    elif p_val < 0.01:
        sig = "**"
    elif p_val < 0.05:
        sig = "*"
    else:
        sig = "n.s."
    return p_val, sig


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap CI + Wilcoxon for M-BridgeNet vs SimOnly")
    parser.add_argument("--k", nargs="+", type=int, default=[5, 20, 50])
    parser.add_argument("--model-ts", default=None, dest="model_ts",
                        help="Timestamp prefix of the M-BridgeNet run (e.g. 2026-05-17T10)")
    parser.add_argument("--simonly-ts", default=None, dest="simonly_ts",
                        help="Timestamp prefix of the SimOnly run")
    parser.add_argument("--log", default=str(LOG_PATH),
                        help="Path to eval_results.jsonl")
    parser.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP, dest="n_bootstrap")
    args = parser.parse_args()

    log_path = Path(args.log)
    if not log_path.exists():
        print(f"ERROR: log file not found: {log_path}", file=sys.stderr)
        sys.exit(1)

    all_runs = load_runs(log_path)
    if not all_runs:
        print("ERROR: no full-pipeline runs with per_event_metrics found.", file=sys.stderr)
        sys.exit(1)

    model_run  = pick_run(all_runs, args.model_ts,  simonly=False)
    simonly_run = pick_run(all_runs, args.simonly_ts, simonly=True)

    print(f"\nM-BridgeNet run : {model_run['timestamp']}  checkpoint={model_run.get('checkpoint')}")
    print(f"SimOnly run     : {simonly_run['timestamp']}  checkpoint={simonly_run.get('checkpoint')}")

    lines: list[str] = []
    lines.append("# Bootstrap CI + Wilcoxon: M-BridgeNet vs SimOnly\n")
    lines.append(f"- **M-BridgeNet**: `{model_run['timestamp']}` "
                 f"checkpoint=`{model_run.get('checkpoint')}`\n")
    lines.append(f"- **SimOnly**: `{simonly_run['timestamp']}` "
                 f"checkpoint=`{simonly_run.get('checkpoint')}`\n")
    lines.append(f"- Bootstrap N={args.n_bootstrap}, seed={SEED}\n\n")

    for k in args.k:
        model_ap  = ap_vector(model_run,  k)
        simonly_ap = ap_vector(simonly_run, k)

        # find common events
        common = sorted(set(model_ap) & set(simonly_ap))
        if not common:
            print(f"WARNING: no common events for AP@{k}")
            continue

        m_vals = [model_ap[e]  for e in common]
        s_vals = [simonly_ap[e] for e in common]

        # per-event table
        print(f"\n--- AP@{k} per event ({len(common)} events) ---")
        print(f"{'Event':<40} {'M-BridgeNet':>12} {'SimOnly':>10} {'Δ':>8}")
        print("-" * 72)
        tbl_rows = [f"| Event | M-BridgeNet | SimOnly | Δ |\n|---|---|---|---|\n"]
        for eid, mv, sv in zip(common, m_vals, s_vals):
            delta = mv - sv
            marker = " ↑" if delta > 0.01 else (" ↓" if delta < -0.01 else "")
            print(f"{eid:<40} {mv:>12.4f} {sv:>10.4f} {delta:>+8.4f}{marker}")
            tbl_rows.append(f"| {eid} | {mv:.4f} | {sv:.4f} | {delta:+.4f} |\n")

        # bootstrap CI
        m_mean, m_lo, m_hi = bootstrap_ci(m_vals, n=args.n_bootstrap)
        s_mean, s_lo, s_hi = bootstrap_ci(s_vals, n=args.n_bootstrap)

        print(f"\nBootstrap 95% CI (N={args.n_bootstrap}):")
        print(f"  M-BridgeNet : {m_mean:.4f}  [{m_lo:.4f}, {m_hi:.4f}]")
        print(f"  SimOnly     : {s_mean:.4f}  [{s_lo:.4f}, {s_hi:.4f}]")

        # Wilcoxon
        p_val, sig = wilcoxon_signed_rank(m_vals, s_vals)
        print(f"\nWilcoxon signed-rank (one-sided, H1: M-BridgeNet > SimOnly):")
        print(f"  p = {p_val:.4f}  {sig}")

        # write to report
        lines.append(f"## AP@{k} ({len(common)} events)\n\n")
        lines.append("".join(tbl_rows) + "\n")
        lines.append(f"**Bootstrap 95% CI** (N={args.n_bootstrap}, seed={SEED}):\n\n")
        lines.append(f"- M-BridgeNet: **{m_mean:.4f}** [{m_lo:.4f}, {m_hi:.4f}]\n")
        lines.append(f"- SimOnly:     {s_mean:.4f} [{s_lo:.4f}, {s_hi:.4f}]\n\n")
        lines.append(f"**Wilcoxon signed-rank** (one-sided, H₁: M-BridgeNet > SimOnly):\n\n")
        lines.append(f"- p = {p_val:.4f}  {sig}\n\n")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("".join(lines), encoding="utf-8")
    print(f"\nReport saved → {REPORT_PATH}")


if __name__ == "__main__":
    main()
