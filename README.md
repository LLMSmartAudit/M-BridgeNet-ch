# M-BridgeNet: Cross-Platform Bridge-Node Detection on Chinese Social Media

Code for **M-BridgeNet**, a three-stage framework that detects *bridge posts* —
posts that carry a narrative from one social platform to another — evaluated on the
**CPHot** benchmark (110 events across Weibo, Douyin, Bilibili, Zhihu).

- 📦 **Code (this repo):** the framework, experiment scripts, and lightweight result artifacts.
- 🤗 **Data + model weights (Hugging Face):** the CPHot dataset and trained checkpoints
  live on the Hub (too large for GitHub) — see [§2](#2-get-the-data--checkpoints).

> **Headline result (v25):** AP@5 = **82.29%** on the 67-event `test_real` split —
> statistically indistinguishable from a fine-tuned CrossEncoder (83.01%, p=0.41) at
> ~1/100th the inference cost, and +2.71pp over a cosine-only baseline.

## Results at a glance (67-event `test_real`, AP@K %)

| Method | AP@5 | AP@20 | AP@50 | Inference cost |
|---|:--:|:--:|:--:|---|
| BM25 (Okapi, jieba) | 13.15 | 9.18 | 9.55 | low |
| PairEncoder (Siamese-BGE) | 54.37 | 39.79 | 34.56 | low |
| TCO (temporal decay) | 59.89 | 52.86 | 50.50 | low |
| SimOnly (cosine `s1`) | 79.58 | 69.66 | 64.53 | low |
| **M-BridgeNet (v25, ours)** | **82.29** | **71.99** | 66.79 | **low** (5-signal MLP) |
| CrossEncoder (MacBERT, cap-1000) | 83.01 | 70.17 | 68.05 | high (≥9 h uncapped) |

M-BridgeNet ties the CrossEncoder on AP@5 (Δ−0.72pp, p=0.41) while **leading at
AP@20 (+1.82pp)** at a fraction of the cost, and beats SimOnly by +2.71pp AP@5.
Stage-3 MABD (LLM debate) adds **zero aggregate AP** — its value is interpretability.
Full breakdown (ablations, τ-sweep, per-phase, routing) in [`docs/RESULTS.md`](docs/RESULTS.md).

---

## 1. Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .            # installs the `mbridgenet` package + deps
```
Python 3.10+. First run downloads BGE-large-zh (~1.3 GB) for Stage-1 embedding.
On Apple Silicon set `MBRIDGENET_NO_FAISS=1` (exact, deterministic numpy path).

## 2. Get the data & checkpoints

Checkpoints and a **public data sample** are hosted on Hugging Face. Fetch them into
the layout the scripts expect (run from the repo root):

```bash
pip install huggingface_hub
make data HF_REPO=your-org/M-BridgeNet
# equivalently:
python scripts/download_from_hf.py --repo-id your-org/M-BridgeNet
```
This populates `data/cphot/{raw,processed}/…` and `checkpoints/`.

> ⚠️ **The public HF dataset is a truncated SAMPLE** (≤200 posts/pairs per event) to
> protect the data asset. It exercises the full code path but **does not reproduce the
> paper's numbers**. The **full CPHot dataset is available on request / under a
> data-use agreement** — see the HF `DATA_ACCESS.md` (contact: **lincrazy31@gmail.com**);
> the commands in §3 reproduce the reported metrics **on the full dataset**.

> The fine-tuned MacBERT **CrossEncoder** (~391 MB) is optional: per-pair scores are
> precomputed and shipped as `*_s5.json` in the dataset, so the main eval needs no
> CrossEncoder. You only need it to run the CrossEncoder *baseline* or regenerate `s5`.

## 3. Reproduce the headline result

```bash
make reproduce            # → AP@5=0.8229  AP@10=0.7470  AP@20=0.7199  AP@50=0.6679
```
*(these numbers require the **full** dataset; on the public ≤200/event sample the
pipeline runs but the metrics will differ — see §2)*
which runs:
```bash
MBRIDGENET_NO_FAISS=1 python scripts/evaluate.py \
  --data data/cphot/processed/test_real \
  --checkpoint checkpoints/mlp_v25_fold2.pt \
  --k 5 10 20 50 --low-s2-simonly 0.20
```

Other one-command targets: `make baselines`, `make ablation`, `make tau-sweep`,
`make figure`, `make analyses`. Full commands and the numbers each produces are in
**[`docs/RESULTS.md`](docs/RESULTS.md)**. The full model-development changelog
(how performance evolved v1→v25, with the data/method lessons) is in
**[`docs/MODEL_HISTORY.md`](docs/MODEL_HISTORY.md)**.
Dataset schema is in **[`docs/DATASET.md`](docs/DATASET.md)**.

## 4. Repository layout

```
Makefile             one-command targets (make help)
src/mbridgenet/      framework (stage1 candidate gen, stage2 scorer+routing, stage3 MABD)
scripts/             evaluate.py, train.py, download_from_hf.py, analysis & plotting scripts
configs/default.yaml thresholds / model names
results/             eval_results.jsonl run log, per-event dumps, figures, τ-sweep configs
docs/                RESULTS.md, DATASET.md, MODEL_HISTORY.md (v1→v25), EXPERIMENTS.md (script→artifact map)
checkpoints/, data/  populated from Hugging Face (§2)
```

## 5. Reproducibility notes
- `MBRIDGENET_NO_FAISS=1` → exact, deterministic candidate generation (headline 82.29
  reproduces bit-for-bit). FAISS-HNSW is an optional approximate path (≲0.3pp drift).
- Low-s2 events (median top-100 `s2 < 0.20`) fall back to cosine `s1` (deployed default).
- Stage-3 MABD and the LLM router require `OPENAI_API_KEY`; not needed for non-LLM results.

<!-- ## 6. Citation
```bibtex
@article{mbridgenet, title={M-BridgeNet: ...}, author={TODO}, year={2026}, note={Preprint}}
``` -->
See `LICENSE`. Code: non-commercial research use.
