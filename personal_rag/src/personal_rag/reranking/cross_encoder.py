"""Cross-Encoder Reranker with lazy loading and CPU-friendly fallback."""

from typing import List, Optional
from personal_rag.reranking.base import BaseReranker
from personal_rag.core.types import SearchResult
from personal_rag.core.hardware import HardwareManager


class CrossEncoderReranker(BaseReranker):
    """Refines candidate ordering using a cross-encoder model (e.g. BAAI/bge-reranker-small)."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-small",
        hardware_manager: Optional[HardwareManager] = None,
        enabled: bool = False
    ):
        self.model_name = model_name
        self.hardware_manager = hardware_manager or HardwareManager()
        self.enabled = enabled
        self._model = None

    def _lazy_load(self):
        if not self.enabled:
            return
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                device = self.hardware_manager.resolve_device()
                self._model = CrossEncoder(self.model_name, device=device)
            except Exception:
                self._model = None

    def rerank(self, query: str, candidates: List[SearchResult], top_n: int = 5) -> List[SearchResult]:
        if not candidates:
            return []

        # If reranking is disabled or model unavailable, return top_n directly
        if not self.enabled:
            return candidates[:top_n]

        self._lazy_load()
        if self._model is None:
            return candidates[:top_n]

        pairs = [[query, res.chunk.text] for res in candidates]
        scores = self._model.predict(pairs)

        reranked = []
        for idx, (res, score) in enumerate(zip(candidates, scores)):
            res.score = float(score)
            reranked.append(res)

        reranked.sort(key=lambda x: x.score, reverse=True)
        for rank_idx, item in enumerate(reranked[:top_n], start=1):
            item.rank = rank_idx

        return reranked[:top_n]

