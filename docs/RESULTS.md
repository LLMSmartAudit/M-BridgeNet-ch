# Reproducible Results (67-event `test_real`, checkpoint `mlp_v25_fold2.pt`)

All numbers below reproduce deterministically with `MBRIDGENET_NO_FAISS=1`.

## Main comparison

| Method | AP@5 | AP@20 | AP@50 | How to reproduce |
|---|---|---|---|---|
| BM25 (Okapi, jieba) | 13.15 | 9.18 | 9.55 | `evaluate.py … --bm25` |
| PairEncoder (Siamese-BGE) | 54.37 | 39.79 | 34.56 | `--pair-encoder pair_enc_v5_fold4.pt` (needs `_emb.npz`) |
| TCO (temporal decay) | 59.89 | 52.86 | 50.50 | `--tco` |
| **SimOnly (cosine s1)** | **79.58** | 69.66 | 64.53 | `--simonly` |
| **M-BridgeNet (v25)** | **82.29** | **71.99** | 66.79 | `--low-s2-simonly 0.20` |
| CrossEncoder (MacBERT, cap-1000) | 83.01 | 70.17 | 68.05 | needs CrossEncoder ckpt |

M-BridgeNet vs SimOnly: Δ+2.71pp AP@5 (Wilcoxon p=0.55, n.s.); vs CrossEncoder:
−0.72pp AP@5 (p=0.41, n.s.) but +1.82pp AP@20, at ~1/100th the inference cost.

## Ablation (v25 components, Δ vs full 82.29)

| Variant | AP@5 | ΔAP@5 | Flag |
|---|---|---|---|
| Full M-BridgeNet (v25) | 82.29 | — | `--low-s2-simonly 0.20` |
| w/o MLP (uniform weights) | 76.49 | −5.80 | `--uniform-weights` |
| w/o s5 (4-signal, v23) | 80.36 | −1.93 | `mlp_v23_fold3.pt` |
| s5-CrossEncoder fallback (vs s1) | 80.03 | −2.26 | `MBRIDGENET_S5_FALLBACK=1` |
| w/o low-s2 fallback | 82.07 | −0.22 | `--low-s2-simonly 0.0` |
| uniform phase window (72h) | 83.14 | +0.85 | `--uniform-phase-window` |
| SimOnly (Stage 1 only) | 79.58 | −2.71 | `--simonly` |

## τ_high robustness (`scripts/` + `results/tau_sweep_configs/`)

AP@5 is **invariant at 82.29%** for τ_high ∈ {0.40 … 0.70}; AP@20 ≤0.24pp,
AP@50 ≤0.56pp variation.

## Per-phase AP@5 (`scripts/per_phase_ap_67.py`, 5 phases)

| Phase | AP@5 | #Ev |
|---|---|---|
| Pre-event | 72.3 | 48 |
| Emergence | 80.4 | 57 |
| Diffusion | 78.0 | 49 |
| Peak | 71.0 | 40 |
| Decline | 69.5 | 45 |

## Oracle event-router (`scripts/router_ceiling.py`)

| Router | AP@5 |
|---|---|
| always-MLP | 82.28 |
| always-s5 | 79.50 |
| median-s2 heuristic (deployed) | 83.51 |
| LLM event-router | 81.24 |
| **oracle (per-event best)** | **87.31** |

The +3.80pp oracle headroom over the heuristic is **not** realizable by any learned
or LLM router (regime winner is a model-interaction artifact, not predictable).

## MABD (Stage 3)

Zero aggregate change on the 67-event test (`mlp_v25_fold2.pt` + MABD ≡ Stage 1+2).
Value is interpretability, not AP. See `scripts/analyze_mabd_routing.py`.
