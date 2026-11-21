"""Dense semantic vector retriever."""

from typing import List, Optional, Dict, Any
from personal_rag.retrieval.base import BaseRetriever
from personal_rag.core.types import SearchResult
from personal_rag.storage.base import BaseVectorStore
from personal_rag.embeddings.base import BaseEmbeddingProvider


class DenseRetriever(BaseRetriever):
    """Retrieves document chunks via dense embedding similarity."""

    def __init__(
        self,
        vector_store: BaseVectorStore,
        embedding_provider: BaseEmbeddingProvider
    ):
        self.vector_store = vector_store
        self.embedding_provider = embedding_provider

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        if not query.strip():
            return []
        q_vec = self.embedding_provider.embed_query(query)
        return self.vector_store.search_vector(q_vec, top_k=top_k, filters=filters)

