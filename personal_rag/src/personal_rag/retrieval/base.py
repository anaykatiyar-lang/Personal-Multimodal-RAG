"""Abstract Retriever interface."""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from personal_rag.core.types import SearchResult


class BaseRetriever(ABC):
    """Abstract retrieval protocol returning ranked SearchResults."""

    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Retrieve most relevant chunks for a given query."""
        pass

