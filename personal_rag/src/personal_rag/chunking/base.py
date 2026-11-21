"""Abstract chunker interface."""

from abc import ABC, abstractmethod
from typing import List
from personal_rag.core.types import ParsedDocument, Chunk


class BaseChunker(ABC):
    """Abstract chunker protocol converting parsed documents into indexed chunks."""

    @abstractmethod
    def chunk(self, document: ParsedDocument) -> List[Chunk]:
        """Transform a ParsedDocument into a list of Chunk objects."""
        pass

