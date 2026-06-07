# CPHot Dataset

> ⚠️ **Public release is a SAMPLE.** Every event JSON here is truncated to **≤200
> entries** per list (posts / scored_pairs / bridge_pairs; `_s5.json` to ≤200 pairs)
> to protect the data asset. The schema is identical to the full data, but the sample
> **does not reproduce the paper's metrics**. The **full CPHot dataset is available on
> request / under a data-use agreement** — see `DATA_ACCESS.md` (contact:
> **lincrazy31@gmail.com**).

**CPHot** (Cross-Platform Hot events) covers **110 events** (43 train + 67 test)
across four Chinese platforms — **Weibo, Douyin, Bilibili, Zhihu** — with
~12,900 annotated bridge pairs and lifecycle-phase labels.

This release ships the **processed** split used for all paper results:
- `data/cphot/processed/test_real/` — 67 held-out test events
- `data/cphot/processed/train_all/` — 43 training events

## Per-event JSON schema

Each `<event_id>.json`:

```json
{
  "event_id": "beijing_flood_2023_001",
  "posts": [
    {"post_id": "dy_7261...", "text": "...", "account_id": "101663...",
     "platform": "douyin", "timestamp": "2023-07-29T03:44:26+00:00"}
  ],
  "bridge_pairs": [["bili_617170113", "wb_4934616916168703"], ...],
  "hourly_volumes": [1.0, 2.0, ...],
  "scored_pairs": [
    {"post_a_id": "bili_617...", "post_b_id": "wb_493...",
     "s1": 0.869, "s2": 0.961, "s3": 0.853, "s4": 0.088,
     "phase": "decline", "label": 1}
  ]
}
```

- **`bridge_pairs`** — ground-truth `[source_post, destination_post]` pairs
  (a source post may bridge to several destinations). AP@K ground truth = the set
  of unique source posts.
- **`scored_pairs`** — candidate pairs with precomputed signals and binary `label`
  (1 = bridge, 0 = non-bridge, −1 = unlabeled). `phase` ∈
  {`pre_event`, `emergence`, `diffusion`, `peak`, `decline`}.
- Signals: `s1` cosine similarity, `s2` lifecycle-aware temporal gap, `s3` platform
  migration rarity, `s4` betweenness (excluded from the scorer).

## CrossEncoder sidecars

`<event_id>_s5.json` maps `"postA||postB"` (post IDs sorted alphabetically) →
the fine-tuned MacBERT CrossEncoder score for that pair, precomputed for the
top-500 Stage-1 candidates per event. Pairs absent from the map get `s5 = 0`.

## Annotation

Bridge pairs were labeled with a hybrid protocol: 50% of candidate pairs were
annotated directly by human experts, and 50% were labeled with GPT-family
assistance under the same three operational criteria (temporal precedence,
narrative non-redundancy, cross-platform audience shift). Low-confidence,
malformed, criterion-inconsistent, and borderline LLM-assisted outputs were
reviewed by human experts. An independent re-annotation with a cosine-free
causal-only prompt gave Cohen's κ = 0.72 (substantial agreement); see the paper's
limitations on annotation–evaluation circularity.

## `raw/` vs `processed/`

`data/cphot/raw/<event>.json` is the pre-processing **source of truth** (94 events;
the 16 synthetic `event_0xx` training events are generated and have no raw crawl).
It differs from `processed/` in two useful ways:

1. **Annotation provenance.** Each `scored_pairs` entry additionally carries
   `llm_confidence` and `llm_reasoning` — the LLM annotator's per-pair rationale.
   This supports auditing the labels and the annotation–evaluation circularity
   discussion (re-annotation κ=0.72).
2. **Full candidate pool.** `raw` keeps every annotated candidate pair; `processed`
   ships the filtered/scored subset used for training and evaluation
   (e.g. beijing_flood: 668 raw pairs → 68 processed).

`processed/` adds the computed signals (`s1,s2,s3,s4`, lifecycle `phase`) on top of
the labels. You can regenerate `processed` from `raw` with
`scripts/prepare_training_data.py`, so the full raw→processed→eval chain is auditable.

## Notes
- `*_emb.npz` BGE embedding caches are **not** shipped (regenerate with
  `scripts/prepare_training_data.py`). They are only needed for the PairEncoder
  baseline; the main pipeline embeds posts on the fly.
- 17 crawled events were **excluded** from CPHot's 110 by the inclusion criteria
  (Weibo ≥ 100 posts; ≥ ~10 bridge pairs; non-homogeneous topic) — e.g.
  black_myth_wukong (0.2% bridge rate). They are not part of this release.
- All posts are public social-media content; provided for non-commercial research.
