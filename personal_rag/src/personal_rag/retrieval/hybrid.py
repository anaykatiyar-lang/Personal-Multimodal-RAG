"""Hybrid Retriever combining dense and lexical search via Reciprocal Rank Fusion (RRF)."""

from typing import List, Optional, Dict, Any
from personal_rag.retrieval.base import BaseRetriever
from personal_rag.retrieval.dense import DenseRetriever
from personal_rag.retrieval.lexical import LexicalRetriever
from personal_rag.core.types import SearchResult, Chunk


class HybridRetriever(BaseRetriever):
    """Fuses dense semantic search and BM25 lexical search using Reciprocal Rank Fusion."""

    def __init__(
        self,
        dense_retriever: DenseRetriever,
        lexical_retriever: LexicalRetriever,
        dense_weight: float = 0.65,
        lexical_weight: float = 0.35,
        rrf_k: int = 60
    ):
        self.dense = dense_retriever
        self.lexical = lexical_retriever
        self.dense_weight = dense_weight
        self.lexical_weight = lexical_weight
        self.rrf_k = rrf_k

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        dense_weight_override: Optional[float] = None,
        lexical_weight_override: Optional[float] = None,
    ) -> List[SearchResult]:
        if not query.strip():
            return []

        w_dense = dense_weight_override if dense_weight_override is not None else self.dense_weight
        w_lex = lexical_weight_override if lexical_weight_override is not None else self.lexical_weight

        # BUG-05 FIX: Normalise weights so they always sum to 1.0
        # This prevents inflated RRF scores when caller passes e.g. 0.9 + 0.9
        total_w = w_dense + w_lex
        if total_w > 0:
            w_dense = w_dense / total_w
            w_lex = w_lex / total_w

        # Retrieve candidates from both sources (fetch 2x top_k for high-quality fusion)
        dense_results = self.dense.retrieve(query, top_k=top_k * 2, filters=filters)
        lexical_results = self.lexical.retrieve(query, top_k=top_k * 2, filters=filters)

        rrf_scores: Dict[str, float] = {}
        chunk_map: Dict[str, Chunk] = {}
        dense_score_map: Dict[str, float] = {}
        lexical_score_map: Dict[str, float] = {}

        # 1. Accumulate dense RRF
        for rank_idx, res in enumerate(dense_results, start=1):
            cid = res.chunk.chunk_id
            chunk_map[cid] = res.chunk
            dense_score_map[cid] = res.score
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (w_dense / (self.rrf_k + rank_idx))

        # 2. Accumulate lexical RRF
        for rank_idx, res in enumerate(lexical_results, start=1):
            cid = res.chunk.chunk_id
            chunk_map[cid] = res.chunk
            lexical_score_map[cid] = res.score
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (w_lex / (self.rrf_k + rank_idx))

        # 3. Sort by fused RRF score
        sorted_cids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)

        fused_results: List[SearchResult] = []
        for final_rank, cid in enumerate(sorted_cids[:top_k], start=1):
            fused_results.append(SearchResult(
                chunk=chunk_map[cid],
                score=rrf_scores[cid],
                dense_score=dense_score_map.get(cid),
                lexical_score=lexical_score_map.get(cid),
                rank=final_rank
            ))

        return fused_results

