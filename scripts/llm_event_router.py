"""LLM event-router (Direction B, final attempt).

Per event, the LLM reads a sample of high-similarity candidate pairs + event
metadata and picks the scoring regime:
  A = lifecycle + migration-rarity (MLP / s1xs3)   [structured temporal spread]
  B = deep text similarity (CrossEncoder s5)        [paraphrase / unreliable temporal]

We then route by the LLM choice and compare AP@5 to the v25 heuristic, the
cheap-feature router, the oracle, and CrossEncoder.

~1 LLM call per event.
"""
from __future__ import annotations
import json, logging, time
from pathlib import Path

from openai import OpenAI
from mbridgenet.config import load_config
from mbridgenet.data.dataset import CPHotDataset
from mbridgenet.pipeline import MBridgeNetPipeline
from mbridgenet.evaluation.metrics import ap_at_k

logging.getLogger().setLevel(logging.ERROR)
LOW_S2 = 0.20

SYS = (
    "You are a routing controller for a cross-platform bridge-detection system. "
    "For the given event you must choose how to RANK candidate bridge pairs:\n"
    "  A = lifecycle+rarity scoring — best when the narrative spreads with clear "
    "temporal order and across distinct platform pairs (organic, time-structured diffusion).\n"
    "  B = deep text-similarity (cross-encoder) scoring — best when bridges are textual "
    "paraphrases/restatements, OR when temporal structure is unreliable (censored, "
    "clipped, or parallel re-coverage where timing is uninformative).\n"
    "You are shown several of the event's most similar cross-platform pairs (text A → "
    "text B, with the time gap). Decide which regime will rank true bridges better.\n"
    'Output ONLY JSON: {"regime": "A" or "B", "reason": "<short>"}'
)


def sample_block(event, cands) -> str:
    posts = event["posts"]
    plats = {}
    for p in posts:
        plats[p.platform] = plats.get(p.platform, 0) + 1
    top = sorted(cands, key=lambda c: c.s1, reverse=True)[:6]
    lines = [f"Event: {event.get('event_id','')}",
             f"posts={len(posts)}  platforms={dict(plats)}",
             "Top candidate pairs (high text similarity):"]
    for i, c in enumerate(top, 1):
        dt = (c.post_b.timestamp - c.post_a.timestamp).total_seconds() / 3600
        lines.append(f"  {i}. [{c.post_a.platform}->{c.post_b.platform}] gap={dt:+.1f}h "
                     f"s1={c.s1:.2f}\n     A: {c.post_a.text[:160]}\n     B: {c.post_b.text[:160]}")
    return "\n".join(lines)


def llm_pick(client, model, event, cands, temperature) -> int:
    for delay in [0, 5, 15]:
        if delay:
            time.sleep(delay)
        try:
            kw = dict(model=model,
                      messages=[{"role": "system", "content": SYS},
                                {"role": "user", "content": sample_block(event, cands)}],
                      max_completion_tokens=200)
            if temperature != 1:
                kw["temperature"] = temperature
            r = client.chat.completions.create(**kw)
            t = r.choices[0].message.content.strip()
            if t.startswith("```"):
                t = t.split("```")[1]
                if t.startswith("json"):
                    t = t[4:]
            reg = json.loads(t)["regime"].strip().upper()
            return 1 if reg == "B" else 0   # 1 = s5 regime
        except Exception:
            continue
    return 0   # default MLP on failure


def ap5(cands, gt, mode):
    best = {}
    for c in cands:
        sc = (getattr(c, "s5", 0.0) or c.s1) if mode == "s5" else c.score_final
        pid = c.post_a.post_id
        if pid not in best or sc > best[pid]:
            best[pid] = sc
    return ap_at_k(sorted(best.items(), key=lambda x: x[1], reverse=True), gt, 5)


def main():
    cfg = load_config(None); model = cfg.stage3.llm_model
    client = OpenAI()
    pipe = MBridgeNetPipeline(config=cfg, mlp_checkpoint="checkpoints/mlp_v25_fold2.pt",
                              llm_client=None)
    ds = CPHotDataset(Path("data/cphot/processed/test_real"))
    dd = Path("data/cphot/processed/test_real")

    import numpy as np
    am, a5, hp, lp, yo = [], [], [], [], []
    for event in ds:
        eid = event.get("event_id", ""); gt = {p[0] for p in event["bridge_pairs"]}
        if not gt:
            continue
        s5_map = {}; p5 = dd / f"{eid}_s5.json"
        if p5.exists():
            s5_map = json.loads(p5.read_text())
        cands = pipe.run(event, low_s2_simonly_threshold=0.0,
                         s5_map=s5_map if s5_map else None)
        if not cands:
            continue
        m = ap5(cands, gt, "mlp"); s = ap5(cands, gt, "s5")
        top100 = sorted(cands, key=lambda c: c.s1, reverse=True)[:100]
        med = sorted([c.s2 for c in top100])[len(top100) // 2]
        pick = llm_pick(client, model, event, cands, cfg.stage3.llm_temperature)
        time.sleep(0.3)
        am.append(m); a5.append(s); hp.append(1 if med < LOW_S2 else 0)
        lp.append(pick); yo.append(1 if s > m else 0)
        print(f"{eid:38s} llm={'B(s5)' if pick else 'A(mlp)'}  "
              f"oracle={'B' if s>m else 'A'}  mlp={m:.2f} s5={s:.2f}", flush=True)

    am = np.array(am); a5 = np.array(a5); hp = np.array(hp); lp = np.array(lp); yo = np.array(yo)
    rap = lambda p: float(np.mean(np.where(p == 1, a5, am)))
    n = len(am)
    print("\n" + "=" * 56)
    print(f"events={n}  LLM picked s5 for {lp.sum()};  oracle s5={yo.sum()}")
    print(f"  LLM router accuracy vs oracle = {(lp==yo).mean():.3f}")
    print(f"  heuristic    AP@5 = {rap(hp):.4f}")
    print(f"  LLM-router   AP@5 = {rap(lp):.4f}")
    print(f"  oracle       AP@5 = {rap(yo):.4f}")
    print(f"  CrossEncoder      = 0.8301")
    print(f"  -> LLM-router {'BEATS CrossEncoder' if rap(lp)>0.8301 else 'below CrossEncoder'}")


if __name__ == "__main__":
    main()
