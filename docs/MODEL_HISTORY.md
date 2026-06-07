# Model Development History (v1 → v25)

This is a distilled, honest changelog of how M-BridgeNet's scorer evolved. The full
machine log of every run is in [`../results/eval_results.jsonl`](../results/eval_results.jsonl).

> ⚠️ **Numbers across versions are NOT directly comparable.** The held-out test set
> grew over development (2 → 3 → 15 → 24 → 27 → 33 → 41 → **67** events). Early
> "AP@5 = 1.00" figures are on 2–3 events; the final headline (82.29%) is the
> 67-event split. Compare versions only *within* the same split.
>
> The architecture also changed: **v12–v20** ranked by an adaptive **s1×s3** head;
> the **deployed v25** ranks by a 5-signal MLP + low-s2 SimOnly fallback and does
> **not** use s1×s3 (kept only as an alternative head). See the paper.

## Architecture shift (two eras)

```
 v2–v6    3–4 signal MLP over (s1, s2, s3 [,s4])  ──softmax-weighted sum──▶  rank by score S
            └─ s4 dropped at v6 (synthetic artifact); pre_event phase added

 v12–v20  + adaptive  s1×s3  re-ranking head (PMI-like) for direct-BRIDGE pairs
            └─────────  "s1×s3 era"  → now the paper's *alternative* head  ─────────┐
                                                                                     │
 v23      4-signal MLP (s1, s2, s3, weibo_frac)  +  low-s2 SimOnly fallback          │ DROP s1×s3
            └─ degenerate per-phase weights fixed by falling back to cosine when     │
               median top-100 s2 < 0.20 (censored events)                            │
                                                                                     ▼
 v25      + s5 (CrossEncoder text score)  →  5-signal MLP + low-s2 SimOnly fallback   ★ DEPLOYED
            └─ AP@5 = 82.29%  (ties CrossEncoder at ~1/100th cost; +2.71pp vs SimOnly)
```

The deployed v25 ranks by the **MLP score**, not s1×s3. The s1×s3 head is retained
only as an alternative (it powered v12–v20 and the 27-event ablations).

## Milestones

| Version | Split | Key change | Result | Lesson |
|---|---|---|---|---|
| v2 | 18 ev (precomp.) | Initial 4-signal MLP (s1,s2,s3,s4) | F1=0.653 | baseline |
| v3 | 19 ev (precomp.) | +ai_war event | F1=0.786 | more data helps |
| **v6** | real held-out | **Removed s4** (synthetic-data artifact); added `pre_event` phase | F1=0.627 (first honest real eval) | s4 was inflating synthetic scores → exclude from scorer |
| v7–v11 | 2–3 ev | Added Weibo×Zhihu-rich training events (wangguiyuan, huxinyu, tangshan, trump) | suzhou AP@5 **0.0 → 0.20** | the censored Weibo×Zhihu case (suzhou) needs in-distribution training signal |
| **v12 + s1×s3** | 3 ev | **`score_final = s1·s3`** for BRIDGE pairs (PMI-like: high similarity × rare route) | suzhou 0.20 → **0.80**, then 3-event AP@5 = 1.00 | true bridges are *semantically close AND cross a rare route*; dominant-bucket FPs have low s3 |
| v13–v15 | 15 ev (clean) | Dataset → 32 events; 15-event clean held-out | AP@5 ≈ 0.645 | larger, harder test = lower (honest) numbers |
| v15 + **adaptive s1×s3** | 15 ev | Gate s1×s3 to multi-bucket events (bug fix: count BRIDGE-only buckets) | pangmao AP@5 **0.05 → 0.80**; mean 0.645 → **0.703** | single-bucket events (pangmao) must fall back to s1 — s3 is directional noise there |
| **v18** | 15 ev | Phase-window recalibration (12/72/120/240 h) | AP@5 0.703, AP@50 +2.4pp | short Emergence window protects fast-censored Weibo posts |
| v18 + **s1×s3 v2** | 24 ev | Two extra gates (dom-bucket s3≥0.65; minority-fraction<10%) | drone_show 0.0→0.80, xuzhou 0.6→1.0; **first significant beat over SimOnly** (0.732 vs 0.722, p=0.043) | guards against "minority bucket = FPs" rank inversion |
| v20 | 41 ev | Retrain on 38-event split | AP@5 0.666 | capacity plateau at ~16k pairs |
| **v23** | 67 ev | 4-signal MLP (s1,s2,s3,**weibo_frac**); dropped s1×s3; added **low-s2 SimOnly fallback** | AP@5 ≈ 0.80 | degenerate per-phase weights routed censored events to DISCARD → fallback to cosine when median s2<0.20 |
| **v25 (deployed)** | 67 ev | **+s5** (CrossEncoder text score) → 5-signal MLP | **AP@5 = 82.29%** | s5 adds +1.93pp; ties CrossEncoder at ~1/100th cost, +2.71pp over SimOnly |

## Recurring lessons (data & method)

- **`s4` (betweenness) is a synthetic artifact** — every synthetic bridge was the sole
  cross-platform node (s4≡1); it caused negative transfer to real events. Excluded.
- **Weibo ≥ 100 posts is a hard gate for training events.** Events with <100 Weibo
  posts (e.g. sam_altman wb=75, xuzhou wb=4) re-bias the model toward Zhihu×Bilibili
  and *hurt* the censored Weibo×Zhihu case — even when correctly annotated.
- **Homogeneous topics don't generalize.** Tech/gaming events with near-zero bridge
  rate (black_myth_wukong 0.2%, realestate_doomers 1.2%) add feature noise → excluded.
- **suzhou** (Weibo censored to s2≈0, rare Weibo×Zhihu route) was the recurring hard
  case; fixed cumulatively by (a) training-data diversity, (b) s1×s3, and finally
  (c) the low-s2 SimOnly fallback in v23/v25.
- **MABD (Stage 3 LLM debate) adds zero aggregate AP** — confirmed on the 24-, 27-,
  41-, and 67-event splits, and across GPT-5.4-nano / GPT-5.5 (model parity). The
  borderline routing zone is structurally near-empty of recoverable bridges. Its value
  is interpretability, not accuracy.
- **The LLM provides no reliable accuracy gain in any position tested** (decision-maker,
  feature, scorer, or event-router) — a thoroughly characterized negative result; see
  the paper's analysis and `scripts/router_ceiling.py`, `scripts/llm_event_router.py`.

## Reproducing intermediate versions

Checkpoints `mlp_v18_fold5.pt` (v18 + adaptive s1×s3 head) and `mlp_v23_fold3.pt`
(4-signal, no s5) are shipped alongside the deployed `mlp_v25_fold2.pt`, so the key
transitions (s5: v23→v25; s1×s3 head: v18) can be reproduced directly. See
[`RESULTS.md`](RESULTS.md) for commands.
