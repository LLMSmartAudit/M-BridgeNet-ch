"""Generate per-event AP@5 scatter plot: M-BridgeNet vs SimOnly, colored by dominant phase.

Usage
-----
python scripts/plot_per_event_scatter.py \\
    --mbnet   logs/eval_v23_fold3.log \\
    --simonly logs/eval_simonly_41events.log \\
    --data    data/cphot/processed/test_real \\
    --out     ../../06-papers/Arxiv_PRIME_AI_Style_Template/media/fig_per_event_scatter

Outputs <out>.pdf and <out>.png.

The script is fully self-contained: it derives dominant lifecycle phases from the
event JSON files in --data rather than relying on a hard-coded lookup table.
"""

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# ── Constants ────────────────────────────────────────────────────────────────

VALIDATION_EVENTS = {
    "likeqiang_death_001",
    "suzhou_realestate_arrest_001",
    "xiao_fei_scandal_001",
}

PHASE_COLORS = {
    "pre_event":  "#10B981",  # teal-green
    "emergence": "#E87D2B",   # orange
    "diffusion":  "#8B5CF6",  # purple
    "peak":       "#EF4444",  # red
    "decline":    "#3B82F6",  # blue
    "unknown":    "#9E9E9E",  # grey fallback
}

# Annotations are selected data-drivenly in build_figure(): the events whose
# |M-BridgeNet - SimOnly| AP@5 gap is largest (the informative off-diagonal
# points) are labelled automatically, so the figure stays correct as the
# evaluation data changes.
ANNOTATE_MIN_DELTA = 20.0   # pp: label events at least this far off-diagonal
ANNOTATE_MAX_N = 6          # cap number of labels to avoid clutter
# Short display names for annotated events (strip the _NNN suffix).
def _short_name(ev: str) -> str:
    import re as _re
    return _re.sub(r"_\d+$", "", ev)


# ── Parsing helpers ──────────────────────────────────────────────────────────

_LOG_PATTERN = re.compile(
    r"\[(\d+)/\d+\]\s+(\S+)\s+bridges=\d+\s+AP@5=([\d.]+)"
)


def parse_eval_log(log_path: Path) -> dict[str, float]:
    """Return {event_name: ap5_float} from an evaluate.py log file OR a
    per-event JSON dump (DUMP_PER_EVENT output: a list of records with
    keys ``event_id`` and ``AP@5``).

    Expected log line format (produced by evaluate.py):
        [3/41]  drone_show_event_001  bridges=22  AP@5=0.9000
    """
    if log_path.suffix == ".json":
        records = json.loads(log_path.read_text())
        scores = {r["event_id"]: float(r["AP@5"]) for r in records}
        if not scores:
            raise ValueError(f"No records in {log_path}.")
        return scores

    scores: dict[str, float] = {}
    with open(log_path) as fh:
        for line in fh:
            m = _LOG_PATTERN.search(line)
            if m:
                event_name = m.group(2)
                ap5 = float(m.group(3))
                scores[event_name] = ap5
    if not scores:
        raise ValueError(
            f"No AP@5 entries found in {log_path}. "
            "Check that the file was produced by evaluate.py."
        )
    return scores


def dominant_phase(event_json: Path) -> str:
    """Return dominant lifecycle phase (lowercase) from scored_pairs bridge labels.

    Reads the event JSON and counts phase labels among pairs with label==1
    (positive bridge pairs). Returns the most common phase, or 'unknown'.
    """
    try:
        event = json.loads(event_json.read_text())
    except (json.JSONDecodeError, OSError):
        return "unknown"

    counts: Counter = Counter()
    for pair in event.get("scored_pairs", []):
        if pair.get("label", -1) == 1:
            phase = pair.get("phase", "unknown")
            if isinstance(phase, str):
                counts[phase.lower()] += 1

    return counts.most_common(1)[0][0] if counts else "unknown"


def build_phase_map(data_dir: Path, events: list[str]) -> dict[str, str]:
    """Build {event_name: phase_str} by scanning event JSON files in data_dir."""
    phase_map: dict[str, str] = {}
    for ev in events:
        exact = data_dir / f"{ev}.json"
        if exact.exists():
            phase_map[ev] = dominant_phase(exact)
            continue
        # fall back to glob, excluding sidecar files without scored_pairs
        candidates = [c for c in (list(data_dir.glob(f"{ev}*.json"))
                                  + list(data_dir.glob(f"**/{ev}*.json")))
                      if not any(s in c.name for s in ("_s5", "_s6", "_emb"))]
        phase_map[ev] = dominant_phase(candidates[0]) if candidates else "unknown"
    return phase_map


# ── Figure builder ───────────────────────────────────────────────────────────

def build_figure(
    mbnet_scores: dict[str, float],
    simonly_scores: dict[str, float],
    phase_map: dict[str, str],
    out_stem: str,
) -> None:
    """Render and save the scatter figure as <out_stem>.pdf and <out_stem>.png."""

    common = sorted(set(mbnet_scores) & set(simonly_scores))
    if not common:
        raise ValueError("No common events between MBNet and SimOnly score dicts.")

    x = np.array([simonly_scores[e] * 100 for e in common])
    y = np.array([mbnet_scores[e]  * 100 for e in common])
    colors   = [PHASE_COLORS.get(phase_map.get(e, "unknown"), "#9E9E9E") for e in common]
    is_val   = [e in VALIDATION_EVENTS for e in common]

    n = len(common)
    mean_x, mean_y = np.mean(x), np.mean(y)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    # diagonal y = x
    lim = (0, 115)
    ax.plot(lim, lim, "k--", lw=0.8, alpha=0.4, zorder=1)

    # scatter points
    for xi, yi, c, ev, v in zip(x, y, colors, common, is_val):
        marker = "D" if v else "o"
        ms     = 7   if v else 6
        ax.scatter(
            xi, yi,
            color=c, s=ms**2, marker=marker,
            edgecolors="black" if v else "none",
            linewidths=0.8 if v else 0,
            zorder=3, alpha=0.88,
        )

    # annotations: data-driven — label the events farthest off the diagonal,
    # with a greedy vertical de-collision so labels do not overlap.
    annot = sorted(common, key=lambda e: -abs(mbnet_scores[e] - simonly_scores[e]))
    annot = [e for e in annot
             if abs(mbnet_scores[e] - simonly_scores[e]) * 100 >= ANNOTATE_MIN_DELTA
             ][:ANNOTATE_MAX_N]
    placed: list[tuple[float, float]] = []  # (tx, ty) already used

    def _declash(tx: float, ty: float, up: bool) -> float:
        step = 7.0
        for _ in range(12):
            if all(abs(tx - px) > 24 or abs(ty - py) > 6.5 for px, py in placed):
                break
            ty += step if up else -step
            ty = min(112.0, max(2.0, ty))
        return ty

    for ev in annot:
        xi = simonly_scores[ev] * 100
        yi = mbnet_scores[ev]  * 100
        delta = yi - xi
        label = f"{_short_name(ev)} ({delta:+.0f})"
        if delta > 0:                       # M-BridgeNet wins → label up-left
            tx, ty, ha = xi - 6, min(yi + 7, 112), "right"
            ty = _declash(tx, ty, up=True)
        else:                               # SimOnly wins → label down-right
            tx, ty, ha = xi + 4, max(yi - 9, 2), "left"
            ty = _declash(tx, ty, up=False)
        placed.append((tx, ty))
        ax.annotate(
            label, (xi, yi), xytext=(tx, ty),
            fontsize=6.5, color="#333333", ha=ha,
            arrowprops=dict(
                arrowstyle="-",
                color="#999999",
                lw=0.6,
                connectionstyle="arc3,rad=0.15",
            ),
        )

    # mean reference lines
    ax.axhline(mean_y, color="#E87D2B", lw=0.9, ls=":", alpha=0.7,
               label=f"M-BridgeNet mean {mean_y:.1f}%")
    ax.axvline(mean_x, color="#8B5CF6", lw=0.9, ls=":", alpha=0.7,
               label=f"SimOnly mean {mean_x:.1f}%")

    # legend: phase colours
    phase_handles = [
        mpatches.Patch(color=c, label=ph.replace("_", "-").capitalize())
        for ph, c in PHASE_COLORS.items()
        if ph != "unknown"
    ]
    val_handle = plt.Line2D(
        [0], [0], marker="D", color="w",
        markerfacecolor="gray", markeredgecolor="black",
        markersize=7, label="Validation event (†)",
    )
    legend1 = ax.legend(
        handles=phase_handles + [val_handle],
        title="Dominant phase", fontsize=7.5, title_fontsize=8,
        loc="upper left", framealpha=0.85,
    )
    ax.add_artist(legend1)
    ax.legend(fontsize=7.5, loc="lower right", framealpha=0.85)

    ax.set_xlabel("SimOnly AP@5 (%)", fontsize=10)
    ax.set_ylabel("M-BridgeNet AP@5 (%)", fontsize=10)
    ax.set_title(
        f"Per-event AP@5: M-BridgeNet vs SimOnly\n"
        f"({n}-event CPHot test set; colored by dominant lifecycle phase)",
        fontsize=9,
    )
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.grid(True, alpha=0.25, lw=0.5)

    plt.tight_layout()
    for ext in ("pdf", "png"):
        out_path = Path(f"{out_stem}.{ext}")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        print(f"Saved → {out_path}")
    plt.close(fig)

    # delta summary
    deltas = y - x
    above = (deltas > 0).sum()
    below = (deltas < 0).sum()
    equal = (deltas == 0).sum()
    print(f"\nMBN > SimOnly : {above:2d} events,  mean delta = {deltas[deltas>0].mean():.1f} pp" if above else f"\nMBN > SimOnly :  0 events")
    print(f"MBN < SimOnly : {below:2d} events,  mean delta = {deltas[deltas<0].mean():.1f} pp" if below else f"MBN < SimOnly :  0 events")
    print(f"Equal         : {equal:2d} events")
    print(f"\nOverall mean  MBN={mean_y:.1f}%  SimOnly={mean_x:.1f}%  Δ={mean_y-mean_x:+.1f} pp")


# ── CLI entry point ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot per-event AP@5 scatter: M-BridgeNet vs SimOnly."
    )
    parser.add_argument(
        "--mbnet", required=True, type=Path,
        help="Path to M-BridgeNet evaluate.py log file (e.g. logs/eval_v23_fold3.log)",
    )
    parser.add_argument(
        "--simonly", required=True, type=Path,
        help="Path to SimOnly evaluate.py log file (e.g. logs/eval_simonly_41events.log)",
    )
    parser.add_argument(
        "--data", required=False, type=Path, default=None,
        help=(
            "Directory containing event JSON files with scored_pairs. "
            "Used to derive dominant lifecycle phases. "
            "If omitted, all events are coloured grey."
        ),
    )
    parser.add_argument(
        "--out", required=True, type=str,
        help="Output path stem (without extension). Both .pdf and .png are written.",
    )
    args = parser.parse_args()

    print(f"Loading M-BridgeNet scores from: {args.mbnet}")
    mbnet_scores = parse_eval_log(args.mbnet)
    print(f"  → {len(mbnet_scores)} events")

    print(f"Loading SimOnly scores from:     {args.simonly}")
    simonly_scores = parse_eval_log(args.simonly)
    print(f"  → {len(simonly_scores)} events")

    common = sorted(set(mbnet_scores) & set(simonly_scores))
    print(f"Common events: {len(common)}")

    if args.data is not None:
        print(f"Deriving lifecycle phases from:  {args.data}")
        phase_map = build_phase_map(args.data, common)
        phase_counts = Counter(phase_map.values())
        for ph, cnt in sorted(phase_counts.items()):
            print(f"  {ph:12s}: {cnt}")
    else:
        print("No --data provided; all events will be coloured grey.")
        phase_map = {e: "unknown" for e in common}

    build_figure(mbnet_scores, simonly_scores, phase_map, args.out)


if __name__ == "__main__":
    main()
