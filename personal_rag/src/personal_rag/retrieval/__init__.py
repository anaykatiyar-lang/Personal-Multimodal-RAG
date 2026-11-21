"""Retrieval engines: Dense, Lexical, Hybrid (RRF), MMR, and Adaptive Query Routing."""

from personal_rag.retrieval.base import BaseRetriever
from personal_rag.retrieval.dense import DenseRetriever
from personal_rag.retrieval.lexical import LexicalRetriever
from personal_rag.retrieval.hybrid import HybridRetriever
from personal_rag.retrieval.parent import ParentContextRetriever
from personal_rag.retrieval.router import AdaptiveQueryRouter, QueryIntent

__all__ = [
    "BaseRetriever",
    "DenseRetriever",
    "LexicalRetriever",
    "HybridRetriever",
    "ParentContextRetriever",
    "AdaptiveQueryRouter",
    "QueryIntent",
]

