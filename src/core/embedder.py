from __future__ import annotations

import hashlib
import math
from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    model_name: str = "unknown"

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError


class HashEmbeddingProvider(EmbeddingProvider):
    """Small deterministic offline embedding fallback for tests.

    This is not the production semantic model; it exists so the retrieval test
    suite remains offline and deterministic when Sentence Transformers is absent.
    """

    model_name = "hash-embedding-test-v1"

    def __init__(self, dimensions: int = 128):
        self.dimensions = dimensions

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        tokens = [token.casefold() for token in text.split() if token.strip()]
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector))
        if norm:
            vector = [value / norm for value in vector]
        return vector


class SentenceTransformerEmbeddingProvider(EmbeddingProvider):
    """Production embedding adapter using all-MiniLM-L6-v2 by default."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "Install sentence-transformers to use SentenceTransformerEmbeddingProvider"
                ) from exc
            self._model = SentenceTransformer(self.model_name, device="cpu")
        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        model = self._ensure_model()
        vectors = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return [vector.tolist() for vector in vectors]


__all__ = [
    "EmbeddingProvider",
    "HashEmbeddingProvider",
    "SentenceTransformerEmbeddingProvider",
]
