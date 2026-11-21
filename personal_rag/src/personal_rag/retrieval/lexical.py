"""Lexical keyword BM25 retriever utilizing SQLite FTS5."""

from typing import List, Optional, Dict, Any
from personal_rag.retrieval.base import BaseRetriever
from personal_rag.core.types import SearchResult
from personal_rag.storage.sqlite_store import SQLiteStore


class LexicalRetriever(BaseRetriever):
    """Retrieves chunks using BM25 ranking across SQLite FTS5 full-text indexes."""

    def __init__(self, sqlite_store: SQLiteStore):
        self.sqlite_store = sqlite_store

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        return self.sqlite_store.search_lexical(query, top_k=top_k, filters=filters)

