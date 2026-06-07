from __future__ import annotations
from typing import List, Literal
import torch
import torch.nn as nn

# pre_event is included as an explicit phase (index 0) so it gets its own
# embedding rather than silently aliasing to emergence (index 0).
PHASES = ["pre_event", "emergence", "diffusion", "peak", "decline"]
PHASE2IDX = {p: i for i, p in enumerate(PHASES)}

# s4 (betweenness centrality) is excluded: it is a synthetic artifact that
# equals 1.0 for every bridge in the synthetic training/test data but carries
# no discriminative signal in real-world events.
#
# s5 (CrossEncoder text score): optional 5th signal — sigmoid output of a
# fine-tuned MacBERT CrossEncoder run on the top-500 Stage-1 candidates per
# event.  Pairs outside the top-500 receive s5=0.  The MLP learns to weight
# s5 heavily when s2/s3 are uninformative (opinion-threading events) and
# downweight it when lifecycle structural signals are discriminative.
# Set N_SIGNALS=5 to enable; 4 for v23 and earlier (no s5).
N_SIGNALS = 5   # s1, s2, s3, weibo_frac, s5 (CrossEncoder; 0 if outside top-500)


class LifecycleMLP(nn.Module):
    """Phase-conditioned MLP that scores candidate bridge pairs.

    Two architectures controlled by ``residual``:

    **Original** (``residual=False``, default for v23 and earlier):
      1. Embed phase → 16-dim
      2. Concat [s1, s2, s3, weibo_frac, phase_embed] → MLP → softmax weights
      3. Score = dot(weights, signals)

      Problem: softmax weights collapse to near-one-hot per phase (w_s2≈1 for
      diffusion/peak, w_s1≈1 for emergence/decline), causing DISCARD for
      diffusion-phase bridges in Weibo-censored events where s2≈0.

    **Residual** (``residual=True``, v24d+):
      1. Embed phase → 16-dim
      2. Concat [s2, s3, weibo_frac, phase_embed] → correction MLP → tanh → scalar
      3. gate = s2 / (s2 + 0.10)   (temporal gate: 0 when s2≈0, →1 when s2 strong)
      4. Score = clamp(s1 + alpha * correction * gate, 0, 1)

      **Temporal-gated residual**: s1 is the unconditional baseline.  The
      correction MLP sees all signals (including s2 for discrimination power),
      but its output is GATED by s2.

      Why gating matters: training data has a spurious "diffusion + low s2 →
      non-bridge" correlation (Weibo-censored events like suzhou are in test_real,
      not training).  Without gating, the correction MLP learns "low s2 in
      diffusion → strong negative correction → DISCARD", killing suzhou bridges
      that have s2≈0 due to platform censorship.  With gating:
        - s2≈0.00 → gate≈0.00 → correction≈0 → score≈s1 → BRIDGE ✓
        - s2≈0.10 → gate≈0.50 → half-strength correction
        - s2≈0.50 → gate≈0.83 → near-full correction
        - s2≈0.90 → gate≈0.90 → full correction

      This gives the model full expressive power for normal events while
      structurally guaranteeing score≈s1 when temporal signal is absent.
      alpha=0.5 bounds the max correction to ±0.5 (clamped to [0, 1]).
    """

    def __init__(
        self,
        n_signals: int = N_SIGNALS,
        n_phases: int = 5,
        hidden: int = 64,
        residual: bool = False,
        alpha: float = 0.5,
    ):
        super().__init__()
        self.residual = residual
        self.alpha = alpha
        self.phase_embed = nn.Embedding(n_phases, 16)

        if residual:
            # Correction MLP: [s2, s3, weibo_frac] + phase_embed → scalar correction
            # s2 is INCLUDED for discriminative power, but its influence is GATED
            # by s2 itself in the forward pass: correction_eff = correction * s2/(s2+0.10).
            # This structural gating prevents "low s2 → discard" without removing s2's
            # discriminative power for normal (non-censored) events.
            # n_correction = n_signals - 1 (exclude s1)
            n_correction = n_signals - 1
            self.mlp = nn.Sequential(
                nn.Linear(n_correction + 16, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden // 2),
                nn.ReLU(),
                nn.Linear(hidden // 2, 1),
                nn.Tanh(),           # output in [-1, +1]; alpha scales range
            )
        else:
            # Original: [s1, s2, s3, weibo_frac] + phase_embed → softmax weights
            self.mlp = nn.Sequential(
                nn.Linear(n_signals + 16, hidden),
                nn.ReLU(),
                nn.Linear(hidden, n_signals),
                nn.Softmax(dim=-1),
            )

    def forward(self, signals: torch.Tensor, phase_idx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            signals:   (B, 4) float tensor of [s1, s2, s3, weibo_frac]
            phase_idx: (B,)   long tensor of phase indices
        Returns:
            scores: (B,) composite scores
        """
        phase_e = self.phase_embed(phase_idx)           # (B, 16)

        if self.residual:
            s1 = signals[:, 0]                          # (B,)  cosine baseline
            s2 = signals[:, 1]                          # (B,)  temporal signal
            correction_in = signals[:, 1:]              # (B, 3) = [s2, s3, weibo_frac]
            x = torch.cat([correction_in, phase_e], dim=-1)   # (B, 3+16=19)
            correction = self.mlp(x).squeeze(-1)        # (B,) in [-1, +1]
            # Temporal gate: smoothly suppresses correction when s2≈0.
            # gate = s2 / (s2 + 0.10):  s2=0→0, s2=0.1→0.5, s2=0.5→0.83, s2=1→0.91
            # When s2≈0 (Weibo-censored events like suzhou), score→s1 (SimOnly).
            # When s2 is strong, correction applies normally.
            gate = s2 / (s2 + 0.10)
            # Clamp to [0, 1] so existing tau_high/tau_low thresholds apply.
            return torch.clamp(s1 + self.alpha * correction * gate, 0.0, 1.0)
        else:
            x = torch.cat([signals, phase_e], dim=-1)   # (B, 4+16=20)
            weights = self.mlp(x)                       # (B, 4), sums to 1
            return (weights * signals).sum(dim=-1)      # (B,)


def compute_composite_score(signals: List[float], weights: List[float]) -> float:
    """Weighted sum: S = Σ w_i * s_i."""
    return sum(w * s for w, s in zip(weights, signals))


def route(
    score_S: float,
    tau_high: float = 0.72,
    tau_low: float = 0.55,
) -> Literal["bridge", "mabd", "discard"]:
    """Three-way routing based on composite score S."""
    if score_S >= tau_high:
        return "bridge"
    if score_S >= tau_low:
        return "mabd"
    return "discard"
