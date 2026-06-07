"""Pair-level semantic encoder for bridge detection.

Takes pre-computed BGE embeddings (1024-dim, L2-normalised) for post_a and post_b,
projects them to a lower dimension, and scores the pair using element-wise
difference and product features plus platform and time signals.
"""
from __future__ import annotations
import torch
import torch.nn as nn

PLATFORMS = ["weibo", "zhihu", "bilibili", "douyin"]
PLAT2IDX  = {p: i for i, p in enumerate(PLATFORMS)}

EMB_DIM  = 1024   # BGE-large-zh-v1.5 output dimension
PROJ_DIM = 64     # bottleneck projection
PLAT_DIM = 8      # platform embedding size


class PairEncoder(nn.Module):
    """Score a post pair from their BGE embeddings.

    Input features per pair:
      - proj_a          (PROJ_DIM)
      - proj_b          (PROJ_DIM)
      - |proj_a-proj_b| (PROJ_DIM)
      - proj_a * proj_b (PROJ_DIM)
      - platform_a      (PLAT_DIM)  learned embedding
      - platform_b      (PLAT_DIM)  learned embedding
      - time_delta_norm (1)         |t_b - t_a| / 336h, clipped to [0, 1]
    Total input: 4*PROJ_DIM + 2*PLAT_DIM + 1 = 273
    """

    def __init__(
        self,
        emb_dim: int = EMB_DIM,
        proj_dim: int = PROJ_DIM,
        plat_dim: int = PLAT_DIM,
        hidden: int = 128,
        n_platforms: int = len(PLATFORMS),
    ):
        super().__init__()
        self.proj = nn.Linear(emb_dim, proj_dim, bias=False)
        self.plat_embed = nn.Embedding(n_platforms, plat_dim)
        feat_dim = 4 * proj_dim + 2 * plat_dim + 1
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        emb_a: torch.Tensor,       # (B, 1024) L2-normalised
        emb_b: torch.Tensor,       # (B, 1024) L2-normalised
        plat_a: torch.Tensor,      # (B,) long — platform index
        plat_b: torch.Tensor,      # (B,) long — platform index
        time_delta: torch.Tensor,  # (B,) float — |t_b - t_a| in hours
    ) -> torch.Tensor:
        """Return bridge probability scores in [0, 1], shape (B,)."""
        pa = self.proj(emb_a)          # (B, PROJ_DIM)
        pb = self.proj(emb_b)          # (B, PROJ_DIM)
        diff = torch.abs(pa - pb)      # (B, PROJ_DIM)
        prod = pa * pb                 # (B, PROJ_DIM)
        ea = self.plat_embed(plat_a)   # (B, PLAT_DIM)
        eb = self.plat_embed(plat_b)   # (B, PLAT_DIM)
        td = (time_delta / 336.0).clamp(0.0, 1.0).unsqueeze(-1)  # (B, 1)
        x = torch.cat([pa, pb, diff, prod, ea, eb, td], dim=-1)
        return self.mlp(x).squeeze(-1)
