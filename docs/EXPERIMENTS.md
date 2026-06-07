# Experiment → Paper-Artifact Map

Which script produces which table/figure/number. All evaluation runs use
`MBRIDGENET_NO_FAISS=1` and the 67-event `test_real` split unless noted.
Deployed checkpoint: `checkpoints/mlp_v25_fold2.pt`.

## Data pipeline (build CPHot from raw crawls)
| Script | Produces |
|---|---|
| `collect_data.py`, `convert_mediacrawler.py`, `set_crawl.py`, `auto_crawl_all.py`, `phase0b_*.py`, `phase0c_*.sh` | Crawl → CPHot raw event JSONs |
| `llm_annotate.py`, `smart_annotate.py`, `annotate_pairs.py` | LLM/manual bridge-pair annotation |
| `reannotate_circularity_check.py` | Independent re-annotation; Cohen's κ=0.72 (annotation-circularity limitation) |
| `prepare_training_data.py` | Compute signals (s1–s4) + lifecycle phase; writes processed JSON + `_emb.npz` |
| `precompute_s5.py` | CrossEncoder `s5` scores → `*_s5.json` sidecars (needs the MacBERT ckpt) |
| `split_events.py` | train_all / test_real split |

## Training
| Script | Produces |
|---|---|
| `train.py` | LifecycleMLP (`mlp_v*.pt`); `N_SIGNALS=5` → v25, `=4` → v23 |
| `train_pair_encoder.py` | PairEncoder baseline (`pair_enc_v*.pt`) |
| `train_crossencoder.py` | MacBERT CrossEncoder (large; not shipped) |
| `train_logreg.py` | LR-NoPhase baseline |

## Main results — **Table: Main Results** + **Table: Ablation**
| Command / Script | Paper artifact |
|---|---|
| `make reproduce` (`evaluate.py … --low-s2-simonly 0.20`) | M-BridgeNet v25 row (AP@5=82.29) |
| `evaluate.py … --simonly` | SimOnly row (79.58) |
| `evaluate.py … --tco` | TCO baseline (59.89) |
| `evaluate.py --pair-encoder …` | PairEncoder row (54.37; needs `_emb.npz`) |
| `evaluate_bm25.py` | BM25 row (13.15) |
| `evaluate_crossencoder.py` | CrossEncoder row (83.01; needs MacBERT ckpt) |
| `evaluate.py … --uniform-weights / --uniform-phase-window / --low-s2-simonly 0.0` | Ablation rows (w/o MLP, uniform window, w/o fallback) |
| `evaluate.py --checkpoint mlp_v23_fold3.pt …` | "w/o s5" ablation (80.36) |
| `make ablation` | runs the above ablation set |

## Robustness — **Table: τ_high sensitivity**
| Script | Paper artifact |
|---|---|
| `make tau-sweep` (`evaluate.py --config results/tau_sweep_configs/cfg_*.yaml`) | τ_high sweep (AP@5 invariant at 82.29) |
| `tau_high_sweep.py`, `gating_threshold_sweep.py` | sweep helpers |

## Per-event & per-phase — **Tables: per-event AP, per-phase AP, phase distribution; Fig: scatter**
| Script | Paper artifact |
|---|---|
| `per_event_full_67.py` *(or* `evaluate.py` *with* `DUMP_PER_EVENT=…`*)* | Per-event AP table (M-BridgeNet vs SimOnly) |
| `bootstrap_ci.py` | Per-event bootstrap 95% CIs |
| `plot_per_event_scatter.py` | **Fig:** per-event scatter (`results/figures/`) |
| `per_phase_ap_67.py` | Per-phase AP@K table (67-event, 5 phases) |
| `phase_dist_67.py` | Lifecycle phase-distribution table |
| `per_phase_baseline.py` | 7-event per-phase F1 reference table |
| `phase_sensitivity.py`, `phase_stats.py` | Phase-label sensitivity appendix |

## LLM / routing analyses — **Tables: regime routing, per-event regime; the negative-result arc**
| Script | Paper artifact |
|---|---|
| `router_ceiling.py` | Oracle event-router ceiling (oracle 87.31, heuristic 83.51) |
| `router_feasibility.py` | Cheap-feature router CV (80.12 — can't capture headroom) |
| `llm_event_router.py` | LLM event-router (81.24, below chance vs oracle); needs `OPENAI_API_KEY` |
| `per_event_regime.py` | Per-event MLP-vs-s5 regime breakdown table |
| `precheck_highs1_separability.py` | High-s1 region separability pre-check |
| `gboost_scorer_eval.py` | GBoost scorer attempt (below v25) |
| `probe_s6.py`, `precompute_s6.py`, `eval_s6_fusion_full.py`, `analyze_s6_fusion.py` | s6 (LLM bridge-likelihood) probes — net-zero |
| `analyze_topk_fp.py` | Top-K false-positive characterization |

## MABD (Stage 3) — **Table: MABD routing-zone**
| Script | Paper artifact |
|---|---|
| `evaluate.py … --openai --mabd-limit 10` | MABD eval (zero aggregate delta); needs `OPENAI_API_KEY` |
| `analyze_mabd_routing.py` | Routing-zone occupancy (91.4% direct-BRIDGE, 2.2% MABD zone) |
| `analyze_mabd_headroom.py`, `analyze_mabd_crucial.py`, `analyze_mabd_weaktie.py`, `analyze_mabd_D_routing.py`, `diagnose_mabd.py` | MABD characterization (difficult/rare-route; "crucial" disproven) |

## Paper review helper
| Script | Purpose |
|---|---|
| `paper_judger.py`, `judge.py` | Automated reviewer that verifies numerical claims against eval logs |

> **Note on `_emb.npz` and the CrossEncoder:** scripts marked "needs `_emb.npz`" or
> "needs MacBERT ckpt" require artifacts not shipped in the dataset release. Regenerate
> embeddings with `prepare_training_data.py`; the main results (Table: Main, Ablation,
> τ-sweep, per-event/phase) need neither, because BGE embeds on the fly and the `s5`
> scores are precomputed in `*_s5.json`.
