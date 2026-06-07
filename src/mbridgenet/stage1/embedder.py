from __future__ import annotations
import hashlib
import logging
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class BGEEmbedder:
    """Thin wrapper around BGE-large-zh-v1.5 via sentence-transformers.

    BGE models require a query prefix for retrieval tasks:
      - query: "为这个句子生成表示以用于检索相关文章："
      - passage: no prefix needed
    For bridge pair detection we treat all posts as passages (symmetric).
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-large-zh-v1.5",
        device: str | None = None,
        batch_size: int = 64,
        cache_dir: str | Path | None = ".cache/embeddings",
    ):
        self._model = SentenceTransformer(model_name, device=device)
        self._model_name = model_name
        self.batch_size = batch_size
        self.dim = 1024
        self._cache_dir = Path(cache_dir) if cache_dir else None

    def _cache_key(self, texts: list[str]) -> str:
        h = hashlib.sha256()
        h.update(self._model_name.encode())
        for t in texts:
            h.update(t.encode())
        return h.hexdigest()[:16]

    def encode(self, text: str, normalize: bool = True) -> np.ndarray:
        """Encode a single text → float32 vector of shape (1024,)."""
        vec = self._model.encode(
            text,
            normalize_embeddings=normalize,
            convert_to_numpy=True,
        )
        return vec.astype(np.float32)

    def encode_batch(
        self, texts: list[str], normalize: bool = True
    ) -> np.ndarray:
        """Encode a list of texts → float32 matrix of shape (N, 1024).

        Results are cached to disk under cache_dir when set, keyed on a
        SHA-256 hash of the model name + all text content.
        """
        if self._cache_dir is not None:
            key = self._cache_key(texts)
            cache_path = self._cache_dir / f"{key}.npy"
            if cache_path.exists():
                logger.info("Embedding cache hit (%s) — skipping BGE inference", key)
                return np.load(cache_path)

        vecs = self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=normalize,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 200,
        )
        vecs = vecs.astype(np.float32)

        if self._cache_dir is not None:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, vecs)
            logger.info("Embedding cache saved (%s)", key)

        return vecs
