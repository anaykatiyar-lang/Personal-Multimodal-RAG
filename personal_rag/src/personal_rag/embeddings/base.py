"""Abstract Embedding Provider interface."""

from abc import ABC, abstractmethod
from typing import List


class BaseEmbeddingProvider(ABC):
    """Abstract interface for dense semantic embedding providers."""

    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Compute L2-normalized dense embeddings for a batch of strings."""
        pass

    @abstractmethod
    def embed_query(self, query: str) -> List[float]:
        """Compute L2-normalized embedding for a search query."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return vector dimension."""
        pass

