from __future__ import annotations
from datetime import datetime
from typing import List


PHASE_ORDER = ["pre_event", "emergence", "diffusion", "peak", "decline"]


def smooth_volumes(volumes: List[float], H: int = 6) -> List[float]:
    """H-hour sliding average. For t < H uses available prefix."""
    result = []
    for t in range(len(volumes)):
        window = volumes[max(0, t - H + 1): t + 1]
        result.append(sum(window) / len(window))
    return result


def detect_phase(v: float, v_star: float, is_post_peak: bool,
                 thresholds: dict | None = None) -> str:
    """Classify a single hourly volume value into a lifecycle phase.

    Args:
        v: smoothed volume at this hour
        v_star: peak smoothed volume for this event
        is_post_peak: True if this hour comes after the peak hour
        thresholds: {"emergence_start": 0.10, "diffusion_start": 0.50,
                     "peak_start": 0.80}
    """
    if thresholds is None:
        thresholds = {"emergence_start": 0.10,
                      "diffusion_start": 0.50,
                      "peak_start": 0.80}
    if v_star == 0:
        return "pre_event"

    frac = v / v_star
    if frac < thresholds["emergence_start"]:
        return "pre_event" if not is_post_peak else "decline"
    if frac >= thresholds["peak_start"]:
        return "peak"
    if is_post_peak:
        return "decline"
    if frac >= thresholds["diffusion_start"]:
        return "diffusion"
    return "emergence"


def assign_phases(
    timestamps: List[datetime],
    hourly_volumes: List[float],
    smoothing_window: int = 6,
    thresholds: dict | None = None,
) -> List[str]:
    """Assign a lifecycle phase label to each post timestamp.

    timestamps and hourly_volumes must be aligned hourly from event start.
    """
    smoothed = smooth_volumes(hourly_volumes, H=smoothing_window)
    if not smoothed:
        return []
    v_star = max(smoothed)
    peak_hour = smoothed.index(v_star)

    phases_per_hour = [
        detect_phase(smoothed[h], v_star, is_post_peak=(h > peak_hour),
                     thresholds=thresholds)
        for h in range(len(smoothed))
    ]

    if not timestamps:
        return []
    t0 = timestamps[0]
    result = []
    for ts in timestamps:
        hour_idx = int((ts - t0).total_seconds() // 3600)
        hour_idx = max(0, min(hour_idx, len(phases_per_hour) - 1))
        result.append(phases_per_hour[hour_idx])
    return result
