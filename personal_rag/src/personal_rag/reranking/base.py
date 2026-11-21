"""Abstract Reranker interface."""

from abc import ABC, abstractmethod
from typing import List
from personal_rag.core.types import SearchResult


class BaseReranker(ABC):
    """Abstract reranking protocol refining candidate search results."""

    @abstractmethod
    def rerank(self, query: str, candidates: List[SearchResult], top_n: int = 5) -> List[SearchResult]:
        """Score and reorder candidate chunks against the query."""
        pass

