from __future__ import annotations
from typing import Dict, List, Set, Tuple


def f1_strict_at_k(
    predictions: List[Tuple[str, float]],
    ground_truth: Set[str],
    k: int,
) -> float:
    """F1-Strict@K: exact post-id match."""
    if not ground_truth or k == 0:
        return 0.0
    top_k = {pid for pid, _ in predictions[:k]}
    tp = len(top_k & ground_truth)
    precision = tp / k
    recall = tp / len(ground_truth)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def f1_loose_at_k(
    predictions: List[Tuple[str, float]],
    ground_truth: Set[str],
    k: int,
    related_pairs: Dict[str, str],
) -> float:
    """F1-Loose@K: a predicted id counts as TP if it or its related id is in GT.

    related_pairs: {predicted_id: gt_id} — same-account alternate posts.
    """
    if not ground_truth or k == 0:
        return 0.0
    top_k = [pid for pid, _ in predictions[:k]]
    tp = sum(
        1 for pid in top_k
        if pid in ground_truth or related_pairs.get(pid) in ground_truth
    )
    precision = tp / k
    recall = min(tp / len(ground_truth), 1.0)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def ap_at_k(
    predictions: List[Tuple[str, float]],
    ground_truth: Set[str],
    k: int,
) -> float:
    """Average Precision@K.

    Normalises by min(k, |GT|) so a perfect score of 1.0 is achievable.
    Duplicate post_ids in predictions count at most once against ground truth.
    """
    if not ground_truth or k == 0:
        return 0.0
    hits = 0
    total_precision = 0.0
    seen: Set[str] = set()
    for rank, (pid, _) in enumerate(predictions[:k], start=1):
        if pid in ground_truth and pid not in seen:
            seen.add(pid)
            hits += 1
            total_precision += hits / rank
    return total_precision / min(k, len(ground_truth))
