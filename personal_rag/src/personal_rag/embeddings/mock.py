"""Deterministic Mock Embedding Provider for fast, zero-credit CPU testing."""

import hashlib
import math
from typing import List
from personal_rag.embeddings.base import BaseEmbeddingProvider


class MockEmbeddingProvider(BaseEmbeddingProvider):
    """Generates deterministic pseudo-semantic dense embeddings without downloading neural weights."""

    def __init__(self, dimension: int = 384):
        self._dim = dimension

    @property
    def dimension(self) -> int:
        return self._dim

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        return [self._embed_single(t) for t in texts]

    def embed_query(self, query: str) -> List[float]:
        return self._embed_single(query)

    def _embed_single(self, text: str) -> List[float]:
        if not text:
            return [0.0] * self._dim

        # Generate a deterministic vector from SHA-256 tokens
        vec = [0.0] * self._dim
        tokens = text.lower().split()
        for token in tokens:
            token_hash = int(hashlib.sha256(token.encode("utf-8")).hexdigest(), 16)
            for i in range(self._dim):
                # Pseudo-random component from token hash
                val = ((token_hash >> (i % 64)) & 0xFF) / 255.0 - 0.5
                vec[i] += val

        # L2 Normalize vector
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]
        else:
            vec[0] = 1.0

        return vec

