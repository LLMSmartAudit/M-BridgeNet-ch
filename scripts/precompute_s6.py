"""Precompute s6 = LLM continuous bridge-likelihood for top-K Stage-1 candidates.

Single LLM call per pair → calibrated bridge_probability in [0,1], reasoning over
temporal precedence + narrative direction (the axis CrossEncoder/s5 is blind to).

Saves sidecar:  <event_dir>/<event_id>_s6.json  →  {"postA||postB": prob, ...}
(key format mirrors *_s5.json: alphabetically-sorted pair id)

Usage:
  OPENAI_API_KEY=... MBRIDGENET_NO_FAISS=1 .venv/bin/python scripts/precompute_s6.py \
      --data data/cphot/processed/test_real --checkpoint checkpoints/mlp_v25_fold2.pt \
      --top-k 20 --low-s2-simonly 0.20
"""
from __future__ import annotations
import argparse, json, logging, time
from pathlib import Path

from openai import OpenAI
from mbridgenet.config import load_config
from mbridgenet.data.dataset import CPHotDataset
from mbridgenet.pipeline import MBridgeNetPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

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
    return -1.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--low-s2-simonly", type=float, default=0.20, dest="low_s2")
    ap.add_argument("--resume", action="store_true",
                    help="skip events whose _s6.json already exists")
    args = ap.parse_args()

    cfg = load_config(None); model = cfg.stage3.llm_model
    client = OpenAI()
    pipe = MBridgeNetPipeline(config=cfg, mlp_checkpoint=args.checkpoint, llm_client=None)
    ds = CPHotDataset(Path(args.data)); dd = Path(args.data)

    for event in ds:
        eid = event.get("event_id", "")
        out_path = dd / f"{eid}_s6.json"
        if args.resume and out_path.exists():
            logger.info("skip %s (exists)", eid); continue
        s5_map = {}; p5 = dd / f"{eid}_s5.json"
        if p5.exists():
            s5_map = json.loads(p5.read_text())
        cands = pipe.run_stage1_stage2(
            event["posts"], event["hourly_volumes"],
            low_s2_simonly_threshold=args.low_s2, s5_map=s5_map if s5_map else None)
        # rank candidates by score_S; take top-K (the ranking-relevant zone)
        cands = sorted(cands, key=lambda c: c.score_S, reverse=True)[:args.top_k]
        s6_map = {}
        nfail = 0
        for c in cands:
            v = call(client, model, c, cfg.stage3.llm_temperature)
            if v < 0:
                nfail += 1; v = 0.5  # neutral on failure
            lo, hi = sorted([c.post_a.post_id, c.post_b.post_id])
            s6_map[f"{lo}||{hi}"] = round(v, 4)
            time.sleep(0.4)
        out_path.write_text(json.dumps(s6_map, ensure_ascii=False, indent=2))
        logger.info("%-40s  %d pairs scored (%d failed) → %s",
                    eid, len(s6_map), nfail, out_path.name)


if __name__ == "__main__":
    main()
