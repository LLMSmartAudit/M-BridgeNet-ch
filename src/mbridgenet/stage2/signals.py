from __future__ import annotations
from typing import Dict, Tuple

# W(φ) in hours — matches configs/default.yaml
PHASE_WINDOWS: Dict[str, float] = {
    "emergence": 6.0,
    "diffusion": 24.0,
    "peak": 48.0,
    "decline": 72.0,
}


def compute_s2(delta_t_hours: float, phase: str,
               phase_windows: Dict[str, float] | None = None) -> float:
    """Lifecycle-aware temporal gap score.

    s2 = 1 - Δt / W(φ)  if Δt ≤ W(φ), else 0
    """
    windows = phase_windows or PHASE_WINDOWS
    W = windows.get(phase, 72.0)
    if delta_t_hours > W:
        return 0.0
    return 1.0 - delta_t_hours / W


def compute_s3(
    platform_a: str,
    platform_b: str,
    migration_counts: Dict[Tuple[str, str], int],
) -> float:
    """Platform migration rarity score.

    s3 = 1 - freq(A→B) / total_migrations
    Unseen migration routes return 1.0 (maximum rarity).
    """
    total = sum(migration_counts.values())
    if total == 0:
        return 1.0
    count = migration_counts.get((platform_a, platform_b), 0)
    if count == 0:
        return 1.0
    return 1.0 - count / total


def compute_s4_normalized(raw_bc: Dict[str, float]) -> Dict[str, float]:
    """Min-max normalize betweenness centrality scores to [0, 1].

    Args:
        raw_bc: {account_id: BC_value}
    Returns:
        {account_id: normalized_score}
    """
    if not raw_bc:
        return {}
    lo, hi = min(raw_bc.values()), max(raw_bc.values())
    if hi == lo:
        return {k: 0.0 for k in raw_bc}
    return {k: (v - lo) / (hi - lo) for k, v in raw_bc.items()}
