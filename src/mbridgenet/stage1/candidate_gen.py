from __future__ import annotations
from typing import Dict, List
import numpy as np
import faiss
from mbridgenet.schemas import Post, CandidatePair


def passes_temporal_filter(
    pa: Post, pb: Post, max_gap_hours: float = 72.0
) -> bool:
    """Return True iff (pa, pb) is a valid cross-platform ordered pair."""
    if pa.platform == pb.platform:
        return False
    delta = (pb.timestamp - pa.timestamp).total_seconds() / 3600
    return 0 < delta <= max_gap_hours


def generate_candidates(
    posts: List[Post],
    embeddings: Dict[str, np.ndarray],
    tau_coarse: float = 0.50,
    max_gap_hours: float = 72.0,
    top_k: int = 5000,
    faiss_m: int = 32,
    faiss_ef: int = 64,
) -> List[CandidatePair]:
    """Stage 1: temporal filter + FAISS-HNSW → candidate pairs.

    Algorithm:
    1. For each post pA, query HNSW index for nearest posts pB.
    2. Keep pairs where cosine similarity >= tau_coarse AND
       passes_temporal_filter(pA, pB).
    3. Return top_k pairs globally by similarity score.
    """
    n = len(posts)
    ordered = sorted(posts, key=lambda p: p.timestamp)
    ids = [p.post_id for p in ordered]
    matrix = np.stack([embeddings[pid] for pid in ids]).astype(np.float32)

    id_to_post = {p.post_id: p for p in posts}
    candidates: list[tuple[float, CandidatePair]] = []

    # For large datasets use FAISS-HNSW; for small ones use numpy brute-force.
    # MBRIDGENET_NO_FAISS=1 forces numpy path (workaround for FAISS SIMD
    # segfaults on Apple Silicon MPS environments).
    import os
    _force_numpy = os.environ.get("MBRIDGENET_NO_FAISS", "0") == "1"
    use_faiss = (not _force_numpy) and (n > 2 * faiss_m)

    if use_faiss:
        dim = matrix.shape[1]
        index = faiss.IndexHNSWFlat(dim, faiss_m, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efSearch = faiss_ef
        index.add(matrix)
        k_search = min(50, n)

    for i, pa in enumerate(ordered):
        if pa.post_id not in embeddings:
            continue

        if use_faiss:
            query = embeddings[pa.post_id].reshape(1, -1)
            sims_raw, idx_raw = index.search(query, k_search)
            neighbors = list(zip(sims_raw[0], idx_raw[0]))
        else:
            # Brute-force inner product (vectors already normalised → cosine)
            scores = matrix @ embeddings[pa.post_id]
            neighbors = sorted(enumerate(scores), key=lambda x: -x[1])
            neighbors = [(float(s), j) for j, s in neighbors]

        for sim, j in neighbors:
            j = int(j)
            if j < 0 or j == i:
                continue
            if float(sim) < tau_coarse:
                continue
            pb = id_to_post[ids[j]]
            if not passes_temporal_filter(pa, pb, max_gap_hours):
                continue
            pair = CandidatePair(post_a=pa, post_b=pb, s1=float(sim))
            candidates.append((float(sim), pair))

    # Sort by score descending, deduplicate (keep highest sim per pair)
    seen: set[tuple[str, str]] = set()
    unique: list[CandidatePair] = []
    for _, pair in sorted(candidates, key=lambda x: -x[0]):
        key = (pair.post_a.post_id, pair.post_b.post_id)
        if key not in seen:
            seen.add(key)
            unique.append(pair)
        if len(unique) >= top_k:
            break

    return unique
