"""Recursive structure-aware character chunker using LangChain text splitters."""

import re
from typing import List, Optional
from personal_rag.chunking.base import BaseChunker
from personal_rag.core.types import ParsedDocument, Chunk, ChunkType
from personal_rag.core.utils.hashing import compute_chunk_id

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    _HAS_LANGCHAIN = True
except ImportError:
    _HAS_LANGCHAIN = False


class RecursiveChunker(BaseChunker):
    """Splits document content hierarchically across paragraphs, lines, and sentences.
    
    Leverages LangChain's RecursiveCharacterTextSplitter when available, with
    built-in deterministic fallback.
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        separators: Optional[List[str]] = None
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", "! ", "? ", "; ", " ", ""]
        if _HAS_LANGCHAIN:
            self._lc_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=self.separators,
                keep_separator=True
            )
        else:
            self._lc_splitter = None

    def chunk(self, document: ParsedDocument) -> List[Chunk]:
        chunks: List[Chunk] = []
        doc_meta = document.metadata
        chunk_counter = 0

        for page in document.pages:
            text = page.text.strip()
            if not text:
                continue

            raw_chunks = self._split_text(text, self.chunk_size, self.chunk_overlap)

            for c_text in raw_chunks:
                c_text = c_text.strip()
                if not c_text:
                    continue

                # BUG-11 FIX: Robust Markdown table detection (avoid YAML frontmatter or HR false positives)
                lines = [l.strip() for l in c_text.splitlines() if l.strip()]
                is_table = (
                    len(lines) >= 2
                    and sum(1 for l in lines if l.startswith("|") and l.endswith("|")) >= 2
                    and any(re.match(r"^\|[-| :]+\|$", l) for l in lines)
                )
                chunk_type = ChunkType.TABLE if is_table else ChunkType.TEXT

                cid = compute_chunk_id(doc_meta.document_id, chunk_counter, c_text)
                chunks.append(Chunk(
                    chunk_id=cid,
                    document_id=doc_meta.document_id,
                    source_path=doc_meta.source_path,
                    text=c_text,
                    chunk_type=chunk_type,
                    page_number=page.page_number,
                    metadata={"doc_title": doc_meta.title}
                ))
                chunk_counter += 1

        return chunks

    def _split_text(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Split text using LangChain if available, otherwise pure Python fallback."""
        if self._lc_splitter:
            return self._lc_splitter.split_text(text)

        if len(text) <= chunk_size:
            return [text]

        # Find best separator present in text
        chosen_sep = ""
        for s in self.separators:
            if s and s in text:
                chosen_sep = s
                break

        if not chosen_sep:
            return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size - overlap)]

        splits = text.split(chosen_sep)
        result = []
        current_doc = []
        current_len = 0

        for s in splits:
            s_len = len(s) + len(chosen_sep)
            if current_len + s_len > chunk_size and current_doc:
                result.append(chosen_sep.join(current_doc))
                overlap_count = max(1, int(len(current_doc) * (overlap / chunk_size)))
                current_doc = current_doc[-overlap_count:]
                current_len = sum(len(x) + len(chosen_sep) for x in current_doc)

            current_doc.append(s)
            current_len += s_len

        if current_doc:
            result.append(chosen_sep.join(current_doc))

        return result
