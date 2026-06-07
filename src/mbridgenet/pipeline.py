"""End-to-end M-BridgeNet pipeline.

Given a CPHot event (posts + hourly_volumes), produces ranked bridge pairs.
"""
from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm

from mbridgenet.config import MBridgeNetConfig, load_config
from mbridgenet.schemas import CandidatePair, Route
from mbridgenet.stage1.embedder import BGEEmbedder
from mbridgenet.stage1.candidate_gen import generate_candidates
from mbridgenet.stage2.lifecycle import assign_phases
from mbridgenet.stage2.signals import compute_s2, compute_s3, compute_s4_normalized
from mbridgenet.stage2.graph import build_event_graph, compute_betweenness
from mbridgenet.stage2.scorer import LifecycleMLP, route, PHASE2IDX
from mbridgenet.stage2.pair_encoder import PairEncoder, PLAT2IDX as PLAT2IDX_PE
from mbridgenet.stage3.mabd import MABDDebater

logger = logging.getLogger(__name__)


class MBridgeNetPipeline:
    def __init__(
        self,
        config: Optional[MBridgeNetConfig] = None,
        mlp_checkpoint: Optional[str] = None,
        llm_client: Optional[Any] = None,
        embedder_model: Optional[str] = None,
        drop_signal: str = "",   # ablation: "s1", "s2", or "s3" to exclude
        uniform_weights: bool = False,  # ablation: replace MLP with (s1+s2+s3)/3
        uniform_phase_window: bool = False,  # ablation: use W=72h for all phases
    ):
        self.cfg = config or load_config()
        model_name = embedder_model or self.cfg.stage1.embedding_model
        self.embedder = BGEEmbedder(model_name=model_name)
        self._drop_signal = drop_signal
        self._uniform_weights = uniform_weights
        self._uniform_phase_window = uniform_phase_window
        self._sig_keys = [k for k in ["s1", "s2", "s3"] if k != drop_signal]
        n_signals = len(self._sig_keys) + 1  # +1 for weibo_frac (event-level platform signal)
        if mlp_checkpoint:
            raw = torch.load(mlp_checkpoint, map_location="cpu", weights_only=True)
            # Detect checkpoint format:
            #   New format (v24+): {"state_dict": ..., "residual": bool, "alpha": float, "n_signals": int}
            #   Old format (v23-): plain state_dict (OrderedDict with weight keys)
            if "state_dict" in raw:
                ckpt_meta = raw
                ckpt = raw["state_dict"]
                n_signals    = ckpt_meta.get("n_signals", n_signals)
                _residual    = ckpt_meta.get("residual", False)
                _alpha       = ckpt_meta.get("alpha", 0.5)
            else:
                ckpt = raw
                _residual = False
                _alpha    = 0.5
                # Auto-detect n_signals from first weight shape
                if "mlp.0.weight" in ckpt:
                    ckpt_input_dim = ckpt["mlp.0.weight"].shape[1]
                    phase_embed_dim = 16
                    n_signals = ckpt_input_dim - phase_embed_dim
            self.mlp = LifecycleMLP(n_signals=n_signals, residual=_residual, alpha=_alpha)
            self.mlp.load_state_dict(ckpt)
            logger.info(
                "Loaded checkpoint: n_signals=%d  residual=%s  alpha=%.2f",
                n_signals, _residual, _alpha,
            )
        else:
            self.mlp = LifecycleMLP(n_signals=n_signals)
        self._mlp_n_signals = n_signals  # used in signal_tensors assembly
        self.mlp.eval()
        self.pair_encoder: Optional[PairEncoder] = None
        self.debater = (
            MABDDebater(
                client=llm_client,
                model=self.cfg.stage3.llm_model,
                temperature=self.cfg.stage3.llm_temperature,
                max_tokens=self.cfg.stage3.llm_max_tokens,
            )
            if llm_client
            else None
        )

    def load_pair_encoder(self, checkpoint: str) -> None:
        """Load a pre-trained PairEncoder as an alternative Stage-2 scorer."""
        self.pair_encoder = PairEncoder()
        self.pair_encoder.load_state_dict(
            torch.load(checkpoint, map_location="cpu", weights_only=True)
        )
        self.pair_encoder.eval()
        # tau placeholders — overwritten by calibrate_pair_encoder_thresholds()
        self._pe_tau_high: Optional[float] = None
        self._pe_tau_low:  Optional[float] = None
        logger.info("PairEncoder loaded from %s", checkpoint)

    def load_logreg(self, model_path: str) -> None:
        """Load a pre-trained sklearn LogisticRegression as an alternative Stage-2 scorer.

        The LR is trained on (s1, s2, s3) without phase conditioning (LR-NoPhase baseline).
        tau_high/tau_low are calibrated on training data via calibrate_logreg_thresholds().
        """
        import pickle
        with open(model_path, "rb") as f:
            bundle = pickle.load(f)
        self._logreg = bundle["clf"]
        self._logreg_scaler = bundle["scaler"]
        self._logreg_tau_high: Optional[float] = None
        self._logreg_tau_low: Optional[float] = None
        logger.info("LR-NoPhase loaded from %s", model_path)

    def calibrate_logreg_thresholds(self, train_data_dir: str) -> tuple[float, float]:
        """Calibrate tau_high/tau_low for LR-NoPhase using F1-max on training scored_pairs."""
        import json as _json
        from pathlib import Path as _Path

        assert hasattr(self, "_logreg"), "load_logreg() must be called first"

        bridge_scores: list[float] = []
        non_bridge_scores: list[float] = []
        for path in sorted(_Path(train_data_dir).glob("*.json")):
            if "_emb" in path.stem:
                continue
            with open(path) as f:
                event = _json.load(f)
            for p in event.get("scored_pairs", []):
                if p.get("label", -1) == -1:
                    continue
                feat = self._logreg_scaler.transform([[p["s1"], p["s2"], p["s3"]]])
                score = float(self._logreg.predict_proba(feat)[0, 1])
                if p["label"] == 1:
                    bridge_scores.append(score)
                else:
                    non_bridge_scores.append(score)

        if not bridge_scores:
            self._logreg_tau_high = 0.50
            self._logreg_tau_low = 0.35
            return 0.50, 0.35

        best_f1, best_tau = 0.0, 0.50
        all_scores = sorted(set(bridge_scores + non_bridge_scores))
        for tau in all_scores:
            tp = sum(1 for s in bridge_scores if s >= tau)
            fp = sum(1 for s in non_bridge_scores if s >= tau)
            fn = len(bridge_scores) - tp
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec  = tp / (tp + fn) if (tp + fn) else 0.0
            f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
            if f1 > best_f1:
                best_f1, best_tau = f1, tau

        tau_high = best_tau
        tau_low  = max(0.0, tau_high - 0.15)
        self._logreg_tau_high = tau_high
        self._logreg_tau_low  = tau_low
        logger.info("LR-NoPhase thresholds: tau_high=%.3f  tau_low=%.3f  (best_F1=%.3f)",
                    tau_high, tau_low, best_f1)
        return tau_high, tau_low

    def calibrate_pair_encoder_thresholds(self, train_data_dir: str) -> tuple[float, float]:
        """Auto-calibrate tau_high/tau_low from PairEncoder scores on training data.

        Loads all *_emb.npz files in train_data_dir, scores their labeled pairs,
        then sets:
          tau_high = P25 of bridge scores  (top 75% of bridges pass)
          tau_low  = P5  of bridge scores  (bottom 5% discarded)

        Returns (tau_high, tau_low).
        """
        from pathlib import Path
        from datetime import datetime, timezone

        assert self.pair_encoder is not None, "load_pair_encoder() must be called first"

        train_dir = Path(train_data_dir)
        bridge_scores: list[float] = []
        non_bridge_scores: list[float] = []

        for emb_path in sorted(train_dir.glob("*_emb.npz")):
            json_path = emb_path.with_name(emb_path.stem[:-4] + ".json")  # strip "_emb"
            if not json_path.exists():
                continue
            import json
            with open(json_path, encoding="utf-8") as f:
                ev = json.load(f)
            emb_data = np.load(emb_path)
            post_emb = {k: emb_data[k] for k in emb_data.files}

            post_meta: dict = {}
            for p in ev.get("posts", []):
                ts = datetime.fromisoformat(p["timestamp"])
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                post_meta[p["post_id"]] = {"platform": p["platform"], "timestamp": ts}

            pairs = [p for p in ev.get("scored_pairs", []) if p["label"] != -1]
            valid = [
                p for p in pairs
                if p["post_a_id"] in post_emb and p["post_b_id"] in post_emb
                and p["post_a_id"] in post_meta and p["post_b_id"] in post_meta
            ]
            if not valid:
                continue

            ea_t = torch.tensor(
                np.stack([post_emb[p["post_a_id"]] for p in valid]), dtype=torch.float32)
            eb_t = torch.tensor(
                np.stack([post_emb[p["post_b_id"]] for p in valid]), dtype=torch.float32)
            pa_t = torch.tensor(
                [PLAT2IDX_PE.get(post_meta[p["post_a_id"]]["platform"], 0) for p in valid],
                dtype=torch.long)
            pb_t = torch.tensor(
                [PLAT2IDX_PE.get(post_meta[p["post_b_id"]]["platform"], 0) for p in valid],
                dtype=torch.long)
            td_t = torch.tensor(
                [abs((post_meta[p["post_b_id"]]["timestamp"]
                      - post_meta[p["post_a_id"]]["timestamp"]).total_seconds() / 3600)
                 for p in valid],
                dtype=torch.float32)

            with torch.no_grad():
                scores = self.pair_encoder(ea_t, eb_t, pa_t, pb_t, td_t).numpy()

            for pair, score in zip(valid, scores):
                if pair["label"] == 1:
                    bridge_scores.append(float(score))
                else:
                    non_bridge_scores.append(float(score))

        if not bridge_scores:
            logger.warning("No bridge scores for calibration — using defaults 0.50 / 0.30")
            self._pe_tau_high, self._pe_tau_low = 0.50, 0.30
            return 0.50, 0.30

        # Find tau_high by maximising F1 on the combined score/label pool.
        # This is more robust than fixed percentiles when the model saturates
        # (i.e. most bridge scores cluster near 1.0 due to memorisation).
        all_scores = np.array(bridge_scores + non_bridge_scores)
        all_labels = np.array([1] * len(bridge_scores) + [0] * len(non_bridge_scores))
        best_f1, best_tau = 0.0, 0.5
        for thresh in np.linspace(0.05, 0.95, 91):
            preds = (all_scores >= thresh).astype(int)
            tp = int(((preds == 1) & (all_labels == 1)).sum())
            fp = int(((preds == 1) & (all_labels == 0)).sum())
            fn = int(((preds == 0) & (all_labels == 1)).sum())
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec  = tp / (tp + fn) if (tp + fn) else 0.0
            f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
            if f1 > best_f1:
                best_f1, best_tau = f1, float(thresh)

        tau_high = best_tau
        tau_low  = max(0.0, tau_high - 0.15)   # MABD band 0.15 wide below tau_high
        logger.info(
            "PairEncoder thresholds calibrated on %d bridges / %d non-bridges "
            "(best train F1=%.3f): tau_high=%.3f  tau_low=%.3f",
            len(bridge_scores), len(non_bridge_scores), best_f1, tau_high, tau_low,
        )
        self._pe_tau_high = tau_high
        self._pe_tau_low  = tau_low
        return tau_high, tau_low

    def run_stage1_stage2(
        self, posts: list, hourly_volumes: list,
        per_bucket_min_mabd: int = 0,
        s1_floor: float = 0.0,
        low_s2_simonly_threshold: float = 0.0,
        min_bridge_fallback: int = 0,
        s5_map: dict | None = None,
    ) -> List[CandidatePair]:
        """s5_map: optional dict {"postA||postB" (sorted): CrossEncoder_score} for
        the top-500 Stage-1 candidates of this event.  Loaded externally from a
        *_s5.json sidecar and passed in so that the 5th MLP signal (CrossEncoder
        text score) is available.  Pairs absent from the map receive s5=0.0."""
        """Run Stage 1+2 only and return all routed candidates (BRIDGE/MABD/DISCARD)."""
        cfg1, cfg2 = self.cfg.stage1, self.cfg.stage2

        logger.info("Stage 1: embedding %d posts", len(posts))
        texts = [p.text for p in posts]
        vecs = self.embedder.encode_batch(texts)
        embeddings = {p.post_id: vecs[i] for i, p in enumerate(posts)}

        if cfg1.adaptive_time_gap and len(posts) >= 4:
            ts_sorted = sorted(p.timestamp for p in posts)
            n = len(ts_sorted)
            q1 = ts_sorted[n // 4]
            q3 = ts_sorted[3 * n // 4]
            iqr_hours = (q3 - q1).total_seconds() / 3600
            effective_gap = max(48.0, min(iqr_hours * 2, 336.0))
            logger.info("Adaptive time gap: %.1f h (IQR=%.1f h)", effective_gap, iqr_hours)
        else:
            effective_gap = cfg1.max_time_gap_hours

        candidates = generate_candidates(
            posts, embeddings,
            tau_coarse=cfg1.tau_coarse,
            max_gap_hours=effective_gap,
            top_k=cfg1.top_k_candidates,
            faiss_m=cfg1.faiss_m,
            faiss_ef=cfg1.faiss_ef_search,
        )
        logger.info("Stage 1: %d raw candidates", len(candidates))

        candidates = [c for c in candidates if c.s1 >= cfg2.tau_fine]
        logger.info("Stage 1: %d after fine filter (tau_fine=%.2f)",
                    len(candidates), cfg2.tau_fine)

        if not candidates:
            return []

        timestamps = [p.timestamp for p in posts]
        phase_map = dict(zip(
            [p.post_id for p in posts],
            assign_phases(timestamps, hourly_volumes,
                          smoothing_window=cfg2.smoothing_window,
                          thresholds=cfg2.lifecycle_thresholds),
        ))

        migration_counts: Dict = {}
        for c in candidates:
            key = (c.post_a.platform, c.post_b.platform)
            migration_counts[key] = migration_counts.get(key, 0) + 1

        candidate_pair_ids = [(c.post_a.post_id, c.post_b.post_id) for c in candidates]
        G = build_event_graph(posts, candidate_pair_ids)
        raw_bc = compute_betweenness(G)
        norm_bc = compute_s4_normalized(raw_bc)

        # Event-level Weibo fraction: allows MLP to adapt signal weights for
        # Weibo-sparse events where s2/s3 carry different semantics than in
        # Weibo-dominant training events.
        n_weibo = sum(1 for p in posts if p.platform == "weibo")
        weibo_frac = n_weibo / max(len(posts), 1)

        signal_tensors, phase_tensors = [], []
        # Ablation: uniform phase window (72h for all phases) vs adaptive
        _phase_windows = ({"emergence": 72, "diffusion": 72, "peak": 72, "decline": 72}
                          if self._uniform_phase_window else cfg2.phase_windows)
        for c in candidates:
            phase = phase_map.get(c.post_a.post_id, "emergence")
            c.phase = phase
            delta_t = (c.post_b.timestamp - c.post_a.timestamp).total_seconds() / 3600
            c.s2 = compute_s2(delta_t, phase, _phase_windows)
            c.s3 = compute_s3(c.post_a.platform, c.post_b.platform, migration_counts)
            c.s4 = norm_bc.get(c.post_a.post_id, 0.0)  # kept for logging; not fed to MLP
            sigs = [getattr(c, k) for k in self._sig_keys]
            if self._mlp_n_signals > len(sigs):
                sigs = sigs + [weibo_frac]  # append weibo_frac only when model expects it
            if self._mlp_n_signals > len(sigs):
                # 5th signal: CrossEncoder text score (s5).
                # Look up in precomputed s5_map; default to 0.0 (no text signal).
                if s5_map:
                    lo, hi = sorted([c.post_a.post_id, c.post_b.post_id])
                    s5 = float(s5_map.get(f"{lo}||{hi}", 0.0))
                else:
                    s5 = 0.0
                sigs = sigs + [s5]
                c.s5 = s5  # persist for MABD context (improv. A) & disagreement routing (D)
            else:
                c.s5 = 0.0
            signal_tensors.append(sigs)
            phase_tensors.append(PHASE2IDX.get(phase, 0))

        # ── Event-level low-s2 detection → SimOnly fallback ─────────────────────
        # When the temporal signal (s2) is globally suppressed for an event
        # (e.g. Weibo-censored events where post timestamps are clustered,
        # causing lifecycle windows to assign s2≈0 to almost every pair),
        # the MLP's diffusion/peak weights (w_s2≈1.0) make ALL candidates
        # score ≈ 0 → DISCARD.  True bridges are lost.
        #
        # Detection: if the MEDIAN s2 across the TOP-100 Stage-2 candidates
        # (by cosine s1) is below `low_s2_simonly_threshold`, treat this as a
        # low-temporal-signal event and fall back to SimOnly.
        #
        # WHY top-100 instead of all candidates:
        # True bridges concentrate at the TOP of the cosine ranking (high s1).
        # Computing the median over all 10K candidates is dominated by non-bridges
        # which may have s2≈0 even when bridges have informative s2 (e.g.
        # us_election: s2_bridge=0.761, s2_nonbridge=0.141, global median≈0.14
        # → fallback incorrectly fires and discards the strong bridge-s2 signal).
        # Using only the top-100 cosine candidates gives a median that reflects
        # the bridge-side s2 distribution, preventing false fallbacks while still
        # catching censored-timeline events where ALL top candidates have s2≈0
        # (suzhou: top-100 median_s2≈0.03 → fallback correctly fires).
        if low_s2_simonly_threshold > 0.0 and candidates:
            top100 = sorted(candidates, key=lambda c: c.s1, reverse=True)[:100]
            s2_vals = [c.s2 for c in top100]
            median_s2 = float(sorted(s2_vals)[len(s2_vals) // 2])
            if median_s2 < low_s2_simonly_threshold:
                # If s5 (CrossEncoder) scores are available, prefer s5 for ranking
                # because text-level discrimination outperforms cosine for events
                # where the temporal/platform signals are uninformative.
                # Fall back to cosine (s1) for pairs outside the precomputed top-K.
                # Default fallback is SimOnly (s1) — this is the v25 paper config
                # (AP@5=82.29). An experimental s5-CrossEncoder fallback is opt-in via
                # MBRIDGENET_S5_FALLBACK=1; it empirically *underperforms* s1 on the
                # low-s2 events (AP@5 80.03) and is retained only for ablation.
                import os as _os_fb
                if s5_map and _os_fb.environ.get("MBRIDGENET_S5_FALLBACK", "0") == "1":
                    logger.info(
                        "Low-s2 event detected (median_s2=%.4f < threshold=%.2f): "
                        "falling back to s5-CrossEncoder routing (s5_map available)",
                        median_s2, low_s2_simonly_threshold,
                    )
                    for c in candidates:
                        lo, hi = sorted([c.post_a.post_id, c.post_b.post_id])
                        s5 = float(s5_map.get(f"{lo}||{hi}", 0.0))
                        # For pairs with s5 > 0, rank by s5; otherwise fall back to s1
                        c.score_S = s5 if s5 > 0.0 else c.s1
                        c.route = Route.BRIDGE
                        c.score_final = c.score_S
                else:
                    logger.info(
                        "Low-s2 event detected (median_s2=%.4f < threshold=%.2f): "
                        "falling back to SimOnly routing",
                        median_s2, low_s2_simonly_threshold,
                    )
                    for c in candidates:
                        c.score_S = c.s1
                        c.route = Route.BRIDGE
                        c.score_final = c.s1
                return candidates

        if hasattr(self, "_logreg") and self._logreg is not None:
            # LR-NoPhase path — LogisticRegression on (s1, s2, s3) without phase.
            # The logreg scaler was trained on exactly 3 features; drop weibo_frac
            # (index 3) so feature count matches regardless of MLP n_signals.
            feats = np.array(signal_tensors, dtype=np.float32)[:, :3]  # s1, s2, s3 only
            feats_scaled = self._logreg_scaler.transform(feats)
            scores = self._logreg.predict_proba(feats_scaled)[:, 1]
        elif self.pair_encoder is not None:
            # PairEncoder path — uses raw BGE embedding vectors
            emb_a_list = np.stack([embeddings[c.post_a.post_id] for c in candidates])
            emb_b_list = np.stack([embeddings[c.post_b.post_id] for c in candidates])
            ea_t = torch.tensor(emb_a_list, dtype=torch.float32)
            eb_t = torch.tensor(emb_b_list, dtype=torch.float32)
            pa_t = torch.tensor(
                [PLAT2IDX_PE.get(c.post_a.platform, 0) for c in candidates],
                dtype=torch.long,
            )
            pb_t = torch.tensor(
                [PLAT2IDX_PE.get(c.post_b.platform, 0) for c in candidates],
                dtype=torch.long,
            )
            td_t = torch.tensor(
                [abs((c.post_b.timestamp - c.post_a.timestamp).total_seconds() / 3600)
                 for c in candidates],
                dtype=torch.float32,
            )
            with torch.no_grad():
                scores = self.pair_encoder(ea_t, eb_t, pa_t, pb_t, td_t).numpy()
        else:
            # LifecycleMLP path (default)
            sig_t = torch.tensor(signal_tensors, dtype=torch.float32)
            ph_t  = torch.tensor(phase_tensors,  dtype=torch.long)
            with torch.no_grad():
                if self._uniform_weights:
                    # Ablation: bypass MLP, use mean of signals as score
                    scores = sig_t.mean(dim=-1).numpy()
                else:
                    scores = self.mlp(sig_t, ph_t).numpy()

        for c, s in zip(candidates, scores):
            c.score_S = float(s)

        # Use calibrated thresholds when available, else MLP defaults
        if hasattr(self, "_logreg") and self._logreg_tau_high is not None:
            _tau_high, _tau_low = self._logreg_tau_high, self._logreg_tau_low
        elif self.pair_encoder is not None and self._pe_tau_high is not None:
            _tau_high, _tau_low = self._pe_tau_high, self._pe_tau_low
        else:
            _tau_high, _tau_low = cfg2.tau_high, cfg2.tau_low

        for c in candidates:
            decision = route(c.score_S, _tau_high, _tau_low)
            c.route = Route(decision)

        # ── s1-floor routing: rescue high-cosine / low-temporal-signal candidates ─
        # The MLP collapses to pure-s2 scoring for diffusion/peak/pre_event phases
        # (w_s2≈1.0, w_s1≈0).  For events with very low temporal signal (s2≈0),
        # ALL candidates get score_S ≈ 0 → DISCARD, including true bridges that
        # SimOnly correctly ranks by cosine.
        #
        # Rescue condition: score_S < tau_low AND s1 ≥ s1_floor AND s2 < 0.15
        # The s2 guard is critical — it restricts rescue to the specific failure
        # mode (MLP collapse in low-s2 events) rather than broadly rescuing all
        # high-cosine discards (which includes correctly-discarded same-topic FPs
        # in emergence/decline events like harbin where MLP is already reliable).
        #
        # Rescued pairs get route=BRIDGE, score_final=s1 (cosine-ranked, SimOnly-style).
        # score_final set immediately so the later scoring block (score_final==0.0 guard)
        # skips them — they retain cosine ranking rather than reverting to score_S≈0.
        #
        # s1_floor=0.80: "unambiguously high cosine" — above tau_coarse (~0.40)
        # and tau_fine (~0.50–0.60); s2<0.15 = "negligible temporal signal".
        _S2_LOW_THRESHOLD = 0.15
        if s1_floor > 0.0:
            n_s1_promoted = 0
            for c in candidates:
                if (c.route == Route.DISCARD
                        and c.s1 >= s1_floor
                        and c.s2 < _S2_LOW_THRESHOLD):
                    c.route = Route.BRIDGE
                    c.score_final = c.s1  # rank by cosine, like SimOnly
                    n_s1_promoted += 1
            if n_s1_promoted:
                logger.info(
                    "s1-floor (%.2f, s2<%.2f): promoted %d DISCARD→BRIDGE (score_final=s1)",
                    s1_floor, _S2_LOW_THRESHOLD, n_s1_promoted,
                )

        # ── Per-bucket MABD promotion ─────────────────────────────────────────
        # For each (platform_a, platform_b) bucket, promote the top-K candidates
        # (by score_S) that landed in DISCARD up to MABD.  This ensures minority
        # platform-pair buckets (e.g. Weibo×Zhihu in suzhou) are not eliminated
        # purely because a dominant bucket (Bili×Zhihu) inflates the score scale.
        _per_bucket = per_bucket_min_mabd or cfg2.per_bucket_min_mabd
        if _per_bucket > 0:
            from collections import defaultdict as _dd
            bucket_map: dict = _dd(list)
            for c in candidates:
                bucket_map[(c.post_a.platform, c.post_b.platform)].append(c)
            n_promoted = 0
            for bucket_key, bucket in bucket_map.items():
                top_k = sorted(bucket, key=lambda c: c.score_S, reverse=True)[:_per_bucket]
                for c in top_k:
                    if c.route == Route.DISCARD:
                        c.route = Route.MABD
                        n_promoted += 1
            if n_promoted:
                logger.info(
                    "Per-bucket promotion: %d DISCARD→MABD (top-%d per bucket, %d buckets)",
                    n_promoted, _per_bucket, len(bucket_map),
                )

        # ── Sparse-BRIDGE SimOnly fallback ────────────────────────────────────
        # The MLP collapses to s2-dominant scoring for diffusion/peak phases.
        # For censored or anomalous events where s2≈0 for most pairs, the MLP
        # routes almost everything to DISCARD, leaving fewer than K pairs in the
        # BRIDGE zone.  When the BRIDGE set is too sparse to produce meaningful
        # AP@K rankings (fewer than min_bridge_fallback pairs), the MLP is
        # clearly over-discarding → fall back to SimOnly.
        #
        # Unlike median_s2 detection (which fires on ANY low-s2 event), this
        # check fires only when routing actually produced sparse BRIDGE output.
        # This avoids incorrectly forcing SimOnly for events where the MLP
        # correctly routes most non-bridges to DISCARD while keeping a small
        # but accurate BRIDGE set.
        if min_bridge_fallback > 0 and candidates:
            n_bridge = sum(1 for c in candidates if c.route == Route.BRIDGE)
            if n_bridge < min_bridge_fallback:
                logger.info(
                    "Sparse-BRIDGE fallback: only %d BRIDGE pairs (< %d threshold); "
                    "falling back to SimOnly routing",
                    n_bridge, min_bridge_fallback,
                )
                for c in candidates:
                    c.score_S = c.s1
                    c.route = Route.BRIDGE
                    c.score_final = c.s1

        return candidates

    @staticmethod
    def _s1s3_score(
        all_candidates: List["CandidatePair"],
        c: "CandidatePair",
        _DOM_S3_THRESHOLD: float = 0.45,
        _S3_GAP_THRESHOLD: float = 0.25,
    ) -> float:
        """Adaptive s1×s3 scoring.

        s1×s3 rewards pairs in rare platform-pair buckets (high s3) and
        penalises pairs in dominant buckets (low s3).  It is designed for
        the "suzhou pattern": the dominant BRIDGE-zone bucket is a VERY COMMON
        platform pair (e.g. Bili×Zhihu with s3≈0.09), while the true bridges
        occupy a RARER pair (e.g. Weibo×Zhihu with s3≈0.50).

        Three conditions must all hold before s1×s3 is applied:

        1. **Multi-bucket**: BRIDGE-zone candidates span ≥2 undirected platform
           pairs.

        2. **Dominant-bucket very common**: The largest bucket's median s3 is
           below ``_DOM_S3_THRESHOLD`` (default 0.45).  This ensures s1×s3 only
           fires for genuinely common dominant buckets (suzhou: s3≈0.09), not
           for events where all platform pairs are moderately rare (s3≈0.6+),
           where the formula introduces rank inversions instead of fixing them.
           Previous threshold was 0.65, which incorrectly fired for events like
           jiangping/zheng_qinwen/hunan_flood with dom_s3≈0.625.

        3. **Clear s3 gap**: The largest non-dominant bucket's median s3 must
           exceed the dominant bucket's median s3 by at least ``_S3_GAP_THRESHOLD``
           (default 0.25).  This ensures there is a meaningful rarity contrast —
           the minority bucket must be substantially rarer than the dominant.
           Without a gap, multiplying by s3 adds noise rather than signal.

        4. **Minority sizable**: The largest non-dominant bucket must hold ≥10%
           of BRIDGE candidates (guards against trivially-small minority buckets
           that are likely FPs).

        Parameters
        ----------
        all_candidates      : all CandidatePair for this event
        c                   : the specific pair to score
        _DOM_S3_THRESHOLD   : dominant-bucket median-s3 cutoff (default 0.45)
        _S3_GAP_THRESHOLD   : min (minority_s3_med - dom_s3_med) to apply (default 0.25)
        """
        from collections import defaultdict

        # Count distinct undirected platform-pair buckets across BRIDGE-routed candidates only.
        # MABD and DISCARD pairs may come from different platforms and would inflate the bucket
        # count even for single-bucket events (pangmao: all bridges are Bili×Zhihu but some
        # MABD/DISCARD pairs could be Weibo×Bili, making count≥2 → wrong s1×s3 applied).
        bridge_candidates = [cand for cand in all_candidates if cand.route == Route.BRIDGE]

        bucket_s3: "dict[tuple, list]" = defaultdict(list)
        for cand in bridge_candidates:
            bucket = tuple(sorted([cand.post_a.platform, cand.post_b.platform]))
            bucket_s3[bucket].append(cand.s3)

        if len(bucket_s3) < 2:
            return c.s1           # single-bucket: directional s3 noise → use s1 only

        # Find dominant bucket and its median s3
        dominant_bucket = max(bucket_s3, key=lambda b: len(bucket_s3[b]))
        dom_vals = sorted(bucket_s3[dominant_bucket])
        dom_s3_median = dom_vals[len(dom_vals) // 2]

        if dom_s3_median >= _DOM_S3_THRESHOLD:
            # Dominant bucket is not "very common" — all buckets have moderate/high s3.
            # s1×s3 would introduce rank inversions without fixing any real problem.
            # Example: jiangping/zheng_qinwen/hunan_flood (dom_s3≈0.625 ≥ 0.45),
            # drone_show (bili×weibo s3_med=0.727 ≥ 0.45).
            return c.s1

        # Guard against trivially-small minority buckets that are likely FPs.
        # Require the LARGEST non-dominant bucket to hold ≥ 10% of BRIDGE cands.
        total_bridge = len(bridge_candidates)
        # Find largest non-dominant bucket (key + size)
        minority_items = [(k, v) for k, v in bucket_s3.items() if k != dominant_bucket]
        largest_minority_key, largest_minority_vals = max(
            minority_items, key=lambda x: len(x[1])
        )
        largest_minority = len(largest_minority_vals)
        if largest_minority < 0.10 * total_bridge:
            return c.s1   # minority too small → likely FPs, not rare true bridges

        # Guard against insufficient s3 contrast.
        # Require the largest minority bucket to be substantially rarer than dominant.
        # Without a clear gap, s1×s3 adds noise rather than correcting bucket ordering.
        # Suzhou: dom_s3≈0.09, minority_s3≈0.50, gap≈0.41 > 0.25 ✓
        # Failure events with moderate dom_s3 have small gap relative to minority.
        min_vals = sorted(largest_minority_vals)
        min_s3_median = min_vals[len(min_vals) // 2]
        if min_s3_median - dom_s3_median < _S3_GAP_THRESHOLD:
            return c.s1   # s3 contrast too weak → multiplicative boost unreliable

        return c.s1 * c.s3   # multi-bucket, dominant very common, minority clearly rarer

    def run(
        self,
        event: Dict[str, Any],
        mabd_limit: Optional[int] = None,
        rerank_n: Optional[int] = None,
        rerank_final_n: Optional[int] = None,
        rescue_discard_n: Optional[int] = None,
        per_bucket_min_mabd: int = 0,
        use_s1s3_scoring: bool = False,
        always_s1s3: bool = False,
        simonly: bool = False,
        tco: bool = False,
        s1_floor: float = 0.0,
        low_s2_simonly_threshold: float = 0.0,
        score_bridge_by_s1: bool = False,
        min_bridge_fallback: int = 0,
        s5_map: dict | None = None,
    ) -> List[CandidatePair]:
        """Process one CPHot event and return scored, routed bridge pairs.

        Args:
            event:             dict from CPHotDataset.__getitem__ / load_event()
            mabd_limit:        max pairs to send through original MABD zone
            rerank_n:          if set, apply MABD to top-rerank_n candidates by s1 (cosine)
            rerank_final_n:    if set, apply MABD to top-K candidates by Stage-1+2 score_final
                               (Plan A: puts debate on the exact candidates that determine AP@K)
            rescue_discard_n:  if set, apply MABD to top-N DISCARD candidates by cosine (s1).
                               Rescues true bridges that the MLP incorrectly routed to DISCARD
                               (Plan B: fixes structural null where hard-event bridges live below
                               tau_low and are never seen by original MABD zone routing).
            use_s1s3_scoring: use adaptive s1×s3 for direct-BRIDGE scoring
            always_s1s3:     ablation — always multiply s1×s3, skip adaptive bucket check
            simonly:         baseline — rank all Stage-1 candidates by s1 only, skip MLP
            tco:             baseline — rank by s1 * exp(-|Δt_h| / 72h), no MLP
        Returns:
            List of CandidatePair with route==BRIDGE, sorted by score_final desc.
        """
        posts = event["posts"]
        hourly_volumes = event["hourly_volumes"]

        candidates = self.run_stage1_stage2(posts, hourly_volumes,
                                             per_bucket_min_mabd=per_bucket_min_mabd,
                                             s1_floor=s1_floor,
                                             low_s2_simonly_threshold=low_s2_simonly_threshold,
                                             min_bridge_fallback=min_bridge_fallback,
                                             s5_map=s5_map)
        # Note: score_bridge_by_s1 applied at end of run(), not in run_stage1_stage2
        if not candidates:
            return []

        # TCO (Temporal Co-occurrence) baseline: rank by s1 * exp(-|Δt_h| / 72h).
        # Tests whether exponential temporal proximity, combined with semantic similarity,
        # captures the ranking signal learned by the LifecycleMLP.
        if tco:
            import math
            _T_DECAY = 72.0  # exponential half-life in hours
            for c in candidates:
                c.route = Route.BRIDGE
                delta_t_h = abs(
                    (c.post_b.timestamp - c.post_a.timestamp).total_seconds() / 3600
                )
                c.score_final = c.s1 * math.exp(-delta_t_h / _T_DECAY)
            output = sorted(candidates, key=lambda c: c.score_final, reverse=True)
            logger.info("TCO: %d candidates ranked by s1*exp(-Δt/72h)", len(output))
            return output

        # SimOnly baseline: rank all Stage-1 candidates by cosine similarity (s1).
        # Bypasses Stage-2 MLP routing entirely — no lifecycle, no rarity signals.
        if simonly:
            for c in candidates:
                c.route = Route.BRIDGE
                c.score_final = c.s1
            output = sorted(candidates, key=lambda c: c.score_final, reverse=True)
            logger.info("SimOnly: %d candidates ranked by s1", len(output))
            return output

        if rerank_n is not None:
            # ── Reranker mode: MABD on top-N by cosine sim ───────────────────
            by_s1 = sorted(candidates, key=lambda c: c.s1, reverse=True)
            rerank_pairs = by_s1[:rerank_n]
            logger.info("Stage 3 (reranker): %d pairs by top-s1", len(rerank_pairs))

            if self.debater:
                n_parse_fail = 0
                for c in tqdm(rerank_pairs, desc="MABD-rerank", unit="pair", leave=False):
                    record = self.debater.debate(c)
                    if record is None:
                        n_parse_fail += 1
                        continue
                    if record.confidence >= self.cfg.stage3.min_confidence:
                        c.route = Route.BRIDGE if record.is_bridge else Route.DISCARD
                        # S_final = λ·score_S + (1-λ)·confidence  (paper eq. 10)
                        lam = self.cfg.lambda_weight
                        c.score_final = lam * c.score_S + (1 - lam) * record.confidence
                if n_parse_fail:
                    logger.warning(
                        "Stage 3 (reranker): %d/%d judge outputs failed to parse",
                        n_parse_fail, len(rerank_pairs),
                    )

            # All BRIDGE pairs without a MABD score fall back to ranking signal
            for c in candidates:
                if c.route == Route.BRIDGE and c.score_final == 0.0:
                    if use_s1s3_scoring:
                        c.score_final = (
                            c.s1 * c.s3
                            if always_s1s3
                            else self._s1s3_score(candidates, c)
                        )
                    else:
                        c.score_final = c.score_S

        elif rerank_final_n is not None:
            # ── Plan A: MABD on top-K by Stage-1+2 score_final ───────────────
            # Step 1: Assign provisional score_final using the same adaptive
            #         s1×s3 logic that Stage 1+2 would use — so we select
            #         exactly the candidates that determine AP@K in evaluation.
            for c in candidates:
                if c.score_final == 0.0:
                    if c.route == Route.BRIDGE:
                        if use_s1s3_scoring:
                            c.score_final = (
                                c.s1 * c.s3
                                if always_s1s3
                                else self._s1s3_score(candidates, c)
                            )
                        else:
                            c.score_final = c.score_S
                    elif c.route == Route.MABD:
                        # Provisional: treat borderline pairs as their MLP score
                        c.score_final = c.score_S

            # Step 2: Select top-K across BRIDGE + MABD candidates by score_final
            non_discard = [c for c in candidates if c.route != Route.DISCARD]
            by_final = sorted(non_discard, key=lambda c: c.score_final, reverse=True)
            rerank_pairs = by_final[:rerank_final_n]
            logger.info(
                "Stage 3 (rerank-final): %d pairs by top-score_final "
                "(%d BRIDGE, %d MABD)",
                len(rerank_pairs),
                sum(1 for c in rerank_pairs if c.route == Route.BRIDGE),
                sum(1 for c in rerank_pairs if c.route == Route.MABD),
            )

            if self.debater:
                n_parse_fail = 0
                n_pruned = 0
                # Conservative-prune threshold: only demote a top-K candidate when
                # the Judge is HIGHLY confident it is a non-bridge.  At the default
                # min_confidence (0.55) the verifier over-prunes (kills true bridges
                # that share the high-similarity/long-gap surface features); a higher
                # bar removes only unambiguous coincidental co-occurrences.
                import os as _os
                _prune_min = float(_os.environ.get("MABD_PRUNE_MIN_CONF",
                                                   self.cfg.stage3.min_confidence))
                for c in tqdm(rerank_pairs, desc="MABD-final", unit="pair", leave=False):
                    record = self.debater.debate(c)
                    if record is None:
                        n_parse_fail += 1
                        continue
                    if record.confidence >= _prune_min:
                        if not record.is_bridge:
                            # Filter mode: MABD removes high-ranked FPs from top-K.
                            # We do NOT re-score confirmed bridges — preserving the
                            # s1×s3 signal that determines ranking quality.  The
                            # original MABD fusion formula (λ·score_S + (1-λ)·conf)
                            # is designed for MABD-zone pairs (score_final≈score_S);
                            # for direct-BRIDGE pairs score_final is s1×s3, which
                            # already encodes migration rarity — overwriting it
                            # degrades ranking.
                            c.route = Route.DISCARD
                            n_pruned += 1
                if n_parse_fail:
                    logger.warning(
                        "Stage 3 (rerank-final): %d/%d judge outputs failed to parse",
                        n_parse_fail, len(rerank_pairs),
                    )
                logger.info(
                    "Stage 3 (rerank-final): pruned %d/%d top-K FPs via MABD",
                    n_pruned, len(rerank_pairs),
                )
            # Candidates NOT in rerank_pairs keep their provisional score_final

        else:
            # ── Original MABD zone routing ────────────────────────────────────
            mabd_pairs = [c for c in candidates if c.route == Route.MABD]
            mabd_pairs.sort(key=lambda c: c.score_S, reverse=True)
            if mabd_limit is not None:
                mabd_pairs = mabd_pairs[:mabd_limit]
            logger.info("Stage 3: %d pairs routed to MABD", len(mabd_pairs))

            if self.debater:
                n_parse_fail = 0
                for c in tqdm(mabd_pairs, desc="MABD", unit="pair", leave=False):
                    record = self.debater.debate(c)
                    if record is None:
                        n_parse_fail += 1
                        continue
                    if record.confidence >= self.cfg.stage3.min_confidence:
                        c.route = Route.BRIDGE if record.is_bridge else Route.DISCARD
                        # S_final = λ·score_S + (1-λ)·confidence  (paper eq. 10)
                        lam = self.cfg.lambda_weight
                        c.score_final = lam * c.score_S + (1 - lam) * record.confidence
                if n_parse_fail:
                    logger.warning(
                        "Stage 3: %d/%d judge outputs failed to parse",
                        n_parse_fail, len(mabd_pairs),
                    )

            # Assign final score for direct-bridge pairs
            for c in candidates:
                if c.route == Route.BRIDGE and c.score_final == 0.0:
                    if use_s1s3_scoring:
                        c.score_final = (
                            c.s1 * c.s3
                            if always_s1s3
                            else self._s1s3_score(candidates, c)
                        )
                    else:
                        c.score_final = c.score_S

        # ── Plan B: Rescue Layer ──────────────────────────────────────────────
        # Debate top-N DISCARD candidates by cosine similarity.  The MLP may
        # under-score true bridges in hard events (unusual lifecycle patterns,
        # underrepresented platform pairs) and route them to DISCARD.  MABD
        # reasons from post content and can recover these lost bridges.
        # Applied AFTER all other routing modes so it stacks correctly with
        # the original MABD zone or Plan A.
        if rescue_discard_n is not None and self.debater:
            discard_pool = [c for c in candidates if c.route == Route.DISCARD]
            top_discard = sorted(discard_pool, key=lambda c: c.s1, reverse=True)
            top_discard = top_discard[:rescue_discard_n]
            logger.info(
                "Stage 3 (rescue-discard): debating top-%d DISCARD candidates by s1",
                len(top_discard),
            )
            n_rescued = 0
            n_parse_fail = 0
            for c in tqdm(top_discard, desc="MABD-rescue", unit="pair", leave=False):
                record = self.debater.debate(c)
                if record is None:
                    n_parse_fail += 1
                    continue
                if record.confidence >= self.cfg.stage3.min_confidence and record.is_bridge:
                    c.route = Route.BRIDGE
                    # For rescued DISCARD pairs, score_S < tau_low (MLP was confidently
                    # wrong).  Blending with score_S (λ*score_S + (1-λ)*conf) gives
                    # score_final ≤ 0.575 — below all BRIDGE candidates' s1×s3 scores.
                    # Instead use MABD confidence directly so rescued bridges can rank
                    # competitively in the final output.
                    c.score_final = record.confidence
                    n_rescued += 1
            if n_parse_fail:
                logger.warning(
                    "Stage 3 (rescue-discard): %d/%d judge outputs failed to parse",
                    n_parse_fail, len(top_discard),
                )
            logger.info(
                "Stage 3 (rescue-discard): rescued %d/%d DISCARD candidates → BRIDGE",
                n_rescued, len(top_discard),
            )

        # ── Optional: rank BRIDGE zone by s1 instead of score_S ─────────────
        # The MLP's composite score_S may introduce within-BRIDGE ranking noise
        # (weights collapse to per-phase single-signal routing; score_S carries
        # limited discriminative information once the routing decision is made).
        # score_bridge_by_s1=True overrides score_final=s1 for all BRIDGE pairs,
        # turning the MLP into a pure FILTER (keeps/discards candidates) while
        # using cosine similarity for final ranking within the BRIDGE set.
        # This is equivalent to "SimOnly with strict candidate pruning".
        if score_bridge_by_s1:
            n_overridden = 0
            for c in candidates:
                if c.route == Route.BRIDGE and c.score_final != c.s1:
                    c.score_final = c.s1
                    n_overridden += 1
            if n_overridden:
                logger.info(
                    "score_bridge_by_s1: overrode score_final=s1 for %d BRIDGE pairs",
                    n_overridden,
                )

        # Return only confirmed bridge pairs, sorted by score_final desc
        output = [c for c in candidates if c.route == Route.BRIDGE]
        output.sort(key=lambda c: c.score_final, reverse=True)
        logger.info("Pipeline done: %d bridge pairs returned", len(output))
        return output
