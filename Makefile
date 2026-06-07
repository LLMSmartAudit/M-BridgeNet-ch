# M-BridgeNet reproduction Makefile.
# Usage:
#   make install
#   make data HF_REPO=your-org/M-BridgeNet
#   make reproduce          # headline AP@5 = 0.8229
#   make baselines ablation tau-sweep figure
#
# Set HF_REPO to your published Hugging Face repo id before `make data`.

PY      ?= python
DATA    ?= data/cphot/processed/test_real
CKPT    ?= checkpoints/mlp_v25_fold2.pt
HF_REPO ?= TODO/M-BridgeNet
export MBRIDGENET_NO_FAISS = 1

.PHONY: help install data reproduce baselines ablation tau-sweep figure analyses clean

help:
	@echo "Targets: install | data HF_REPO=<id> | reproduce | baselines | ablation | tau-sweep | figure | analyses | clean"

install:
	$(PY) -m pip install -e .

# Download dataset + checkpoints from the Hugging Face Hub into the expected layout.
data:
	$(PY) scripts/download_from_hf.py --repo-id $(HF_REPO) --repo-type dataset

# Headline result: M-BridgeNet v25 (AP@5 = 0.8229).
reproduce:
	$(PY) scripts/evaluate.py --data $(DATA) --checkpoint $(CKPT) \
		--k 5 10 20 50 --low-s2-simonly 0.20

# Baselines.
baselines:
	@echo ">> SimOnly (cosine s1)"; \
	$(PY) scripts/evaluate.py --data $(DATA) --checkpoint $(CKPT) --k 5 10 20 50 --simonly
	@echo ">> w/o s5 (4-signal v23)"; \
	$(PY) scripts/evaluate.py --data $(DATA) --checkpoint checkpoints/mlp_v23_fold3.pt \
		--k 5 10 20 50 --low-s2-simonly 0.20

# v25 component ablations.
ablation:
	@echo ">> w/o MLP (uniform weights)"; \
	$(PY) scripts/evaluate.py --data $(DATA) --checkpoint $(CKPT) --k 5 10 20 50 \
		--low-s2-simonly 0.20 --uniform-weights
	@echo ">> uniform phase window (72h)"; \
	$(PY) scripts/evaluate.py --data $(DATA) --checkpoint $(CKPT) --k 5 10 20 50 \
		--low-s2-simonly 0.20 --uniform-phase-window
	@echo ">> w/o low-s2 fallback"; \
	$(PY) scripts/evaluate.py --data $(DATA) --checkpoint $(CKPT) --k 5 10 20 50 \
		--low-s2-simonly 0.0

# tau_high robustness sweep (AP@5 invariant at 0.8229).
tau-sweep:
	@for th in 0.40 0.45 0.50 0.55 0.60 0.65 0.70; do \
		echo ">> tau_high=$$th"; \
		$(PY) scripts/evaluate.py --data $(DATA) --checkpoint $(CKPT) \
			--config results/tau_sweep_configs/cfg_$$th.yaml \
			--k 5 10 20 50 --low-s2-simonly 0.20; \
	done

# Per-event dump + scatter figure.
figure:
	DUMP_PER_EVENT=results/per_event/mbridgenet_v25_per_event.json \
		$(PY) scripts/evaluate.py --data $(DATA) --checkpoint $(CKPT) \
		--k 5 10 20 50 --low-s2-simonly 0.20
	$(PY) scripts/plot_per_event_scatter.py \
		--mbnet results/per_event/mbridgenet_v25_per_event.json \
		--simonly results/per_event/simonly_per_event.json \
		--data $(DATA) --out results/figures/fig_per_event_scatter

# Paper analyses (per-phase, regime breakdown, oracle router).
analyses:
	$(PY) scripts/per_phase_ap_67.py
	$(PY) scripts/phase_dist_67.py
	$(PY) scripts/router_ceiling.py

clean:
	rm -rf src/*.egg-info __pycache__ .pytest_cache
	find . -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true
