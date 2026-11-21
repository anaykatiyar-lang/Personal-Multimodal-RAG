"""SentenceTransformers Embedding Provider supporting local CPU and GPU execution."""

from typing import List, Optional
from personal_rag.embeddings.base import BaseEmbeddingProvider
from personal_rag.core.hardware import HardwareManager


class SentenceTransformersProvider(BaseEmbeddingProvider):
    """Embeddings using sentence-transformers (e.g. BAAI/bge-small-en-v1.5)."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-en-v1.5",
        hardware_manager: Optional[HardwareManager] = None,
        batch_size: int = 32
    ):
        self.model_name = model_name
        self.hardware_manager = hardware_manager or HardwareManager()
        self.batch_size = batch_size
        self._model = None
        self._dim = 384

    def _lazy_load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            device = self.hardware_manager.resolve_device()
            self._model = SentenceTransformer(self.model_name, device=device)
            self._dim = self._model.get_sentence_embedding_dimension()

    @property
    def dimension(self) -> int:
        if self._model is None:
            return 384
        return self._dim

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        self._lazy_load()
        embeddings = self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        return embeddings.tolist()

    def embed_query(self, query: str) -> List[float]:
        self._lazy_load()
        embedding = self._model.encode(
            query,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        return embedding.tolist()

