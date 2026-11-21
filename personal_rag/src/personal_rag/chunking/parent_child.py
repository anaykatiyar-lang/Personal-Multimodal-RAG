"""Parent-Child Chunking preserving broad contextual envelopes for concise retrieved passages."""

from typing import List
from personal_rag.chunking.base import BaseChunker
from personal_rag.chunking.recursive import RecursiveChunker
from personal_rag.core.types import ParsedDocument, Chunk, ChunkType
from personal_rag.core.utils.hashing import compute_chunk_id


class ParentChildChunker(BaseChunker):
    """Generates two-tier parent-child chunk hierarchies."""

    def __init__(
        self,
        parent_chunk_size: int = 1500,
        child_chunk_size: int = 400,
        child_overlap: int = 80
    ):
        self.parent_chunk_size = parent_chunk_size
        self.child_chunk_size = child_chunk_size
        self.child_overlap = child_overlap
        self._parent_splitter = RecursiveChunker(chunk_size=parent_chunk_size, chunk_overlap=150)
        self._child_splitter = RecursiveChunker(chunk_size=child_chunk_size, chunk_overlap=child_overlap)

    def chunk(self, document: ParsedDocument) -> List[Chunk]:
        all_chunks: List[Chunk] = []
        doc_meta = document.metadata
        parent_counter = 0
        child_counter = 0

        # Step 1: Generate Parent Chunks
        raw_parents = self._parent_splitter.chunk(document)

        for p_idx, p_chunk in enumerate(raw_parents):
            parent_id = f"parent_{doc_meta.document_id[:8]}_{p_idx:04d}"
            p_chunk.chunk_id = parent_id
            p_chunk.metadata["is_parent"] = True
            all_chunks.append(p_chunk)

            # Step 2: Slice each parent chunk into children
            child_texts = self._child_splitter._split_text(
                p_chunk.text, self.child_chunk_size, self.child_overlap
            )

            for c_text in child_texts:
                c_text = c_text.strip()
                if not c_text:
                    continue

                child_id = compute_chunk_id(doc_meta.document_id, child_counter, c_text)
                child_chunk = Chunk(
                    chunk_id=child_id,
                    document_id=doc_meta.document_id,
                    source_path=doc_meta.source_path,
                    parent_id=parent_id,
                    text=c_text,
                    chunk_type=p_chunk.chunk_type,
                    page_number=p_chunk.page_number,
                    metadata={"is_parent": False, "parent_id": parent_id}
                )
                all_chunks.append(child_chunk)
                child_counter += 1

        return all_chunks

