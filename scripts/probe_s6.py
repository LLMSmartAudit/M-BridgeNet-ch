"""Cheap proof-of-concept for s6 = LLM continuous bridge-likelihood.

Question: does a single-call LLM likelihood (reasoning over temporal precedence
+ narrative direction) separate true bridges from false positives on the HARD
top-K cases — better than s5 (CrossEncoder), which is blind to them?

If s6 separates TP/FP where s5 does not, a full s6 feature + MLP retrain is
justified. If not, stop.

Generates s6 for top-K candidates of the FP-heavy events (one LLM call each),
then reports TP-vs-FP means and ROC-AUC for s5 and s6.

Usage:
  OPENAI_API_KEY=... MBRIDGENET_NO_FAISS=1 .venv/bin/python scripts/probe_s6.py \
      --data data/cphot/processed/test_topk_subset \
      --checkpoint checkpoints/mlp_v25_fold2.pt --topk 10
"""
from __future__ import annotations
import argparse, json, logging, os, time
from pathlib import Path

from openai import OpenAI
from mbridgenet.config import load_config
from mbridgenet.data.dataset import CPHotDataset
from mbridgenet.pipeline import MBridgeNetPipeline

logging.getLogger().setLevel(logging.ERROR)

SYS = (
    "You score whether post A is a CROSS-PLATFORM BRIDGE that seeded post B. "
    "A bridge requires ALL THREE: (1) temporal precedence — A precedes B; "
    "(2) narrative non-redundancy — B evolves, reframes, or reacts, not parallel "
    "copy of the same wire/announcement; (3) cross-platform audience shift. "
    "CRITICAL: high text similarity ALONE is NOT a bridge — two posts independently "
    "covering the same ongoing event (often a long, non-directional time gap with "
    "parallel framing) are coincidental, not bridges. True bridges usually transfer "
    "while the story is hot (short, directional gap). "
    'Output ONLY JSON: {"bridge_probability": <float 0..1>, "reason": "<short>"}'
)


def user_block(c) -> str:
    dt = (c.post_b.timestamp - c.post_a.timestamp).total_seconds() / 3600
    return (
        f"Post A  platform={c.post_a.platform}  time={c.post_a.timestamp.isoformat()}\n"
        f"  text: {c.post_a.text[:400]}\n"
        f"Post B  platform={c.post_b.platform}  time={c.post_b.timestamp.isoformat()}\n"
        f"  text: {c.post_b.text[:400]}\n"
        f"Signals: cosine_s1={c.s1:.3f}  crossenc_s5={getattr(c,'s5',0.0):.3f}  "
        f"time_gap_hours={dt:+.1f} (positive = A before B)\n"
        "Give the calibrated bridge_probability."
    )


def call(client, model, c, temperature) -> float:
    for delay in [0, 5, 15, 30]:
        if delay:
            time.sleep(delay)
        try:
            kw = dict(model=model,
                      messages=[{"role": "system", "content": SYS},
                                {"role": "user", "content": user_block(c)}],
                      max_completion_tokens=256)
            if temperature != 1:
                kw["temperature"] = temperature
            r = client.chat.completions.create(**kw)
            txt = r.choices[0].message.content.strip()
            if txt.startswith("```"):
                txt = txt.split("```")[1]
                if txt.startswith("json"):
                    txt = txt[4:]
            return float(json.loads(txt)["bridge_probability"])
        except Exception:
            continue
    return -1.0  # parse/api failure sentinel


def auc(scores, labels) -> float:
    """ROC-AUC via Mann-Whitney U (handles ties)."""
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return float("nan")
    wins = ties = 0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1
            elif p == n:
                ties += 1
    return (wins + 0.5 * ties) / (len(pos) * len(neg))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--low-s2-simonly", type=float, default=0.20, dest="low_s2")
    ap.add_argument("--out", default="logs/probe_s6.json")
    args = ap.parse_args()

    cfg = load_config(None)
    model = cfg.stage3.llm_model
    client = OpenAI()
    pipe = MBridgeNetPipeline(config=cfg, mlp_checkpoint=args.checkpoint, llm_client=None)
    ds = CPHotDataset(Path(args.data)); dd = Path(args.data)

    rows = []  # (event, is_tp, s1, s5, s6)
    for event in ds:
        eid = event.get("event_id", ""); gt = {p[0] for p in event["bridge_pairs"]}
        if not gt:
            continue
        s5_map = {}; p5 = dd / f"{eid}_s5.json"
        if p5.exists():
            s5_map = json.loads(p5.read_text())
        out = pipe.run(event, low_s2_simonly_threshold=args.low_s2,
                       s5_map=s5_map if s5_map else None)
        best = {}
        for c in out:
            pid = c.post_a.post_id
            if pid not in best or c.score_final > best[pid].score_final:
                best[pid] = c
        topk = sorted(best.values(), key=lambda c: c.score_final, reverse=True)[:args.topk]
        for c in topk:
            s6 = call(client, model, c, cfg.stage3.llm_temperature)
            time.sleep(0.5)
            rows.append({"event": eid,
                         "is_tp": int(c.post_a.post_id in gt),
                         "s1": round(c.s1, 4),
                         "s5": round(getattr(c, "s5", 0.0), 4),
                         "s6": round(s6, 4)})
        print(f"{eid:40s} done ({len(topk)} pairs)", flush=True)

    ok = [r for r in rows if r["s6"] >= 0]
    n_fail = len(rows) - len(ok)
    import statistics as st
    tp = [r for r in ok if r["is_tp"] == 1]
    fp = [r for r in ok if r["is_tp"] == 0]
    print("\n" + "=" * 60)
    print(f"s6 PROBE  ({len(ok)} scored, {n_fail} failed; "
          f"{len(tp)} TP, {len(fp)} FP)")
    print("=" * 60)
    for sig in ("s5", "s6"):
        mtp = st.mean([r[sig] for r in tp]) if tp else 0
        mfp = st.mean([r[sig] for r in fp]) if fp else 0
        a = auc([r[sig] for r in ok], [r["is_tp"] for r in ok])
        print(f"  {sig}:  TP_mean={mtp:.3f}  FP_mean={mfp:.3f}  "
              f"sep={mtp-mfp:+.3f}  ROC-AUC={a:.3f}")
    print("\n  AUC>0.5 = discriminates TP from FP. "
          "If s6 AUC >> s5 AUC, s6 adds signal CrossEncoder lacks.")
    Path(args.out).write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    print(f"\nSaved → {args.out}")


if __name__ == "__main__":
    main()
