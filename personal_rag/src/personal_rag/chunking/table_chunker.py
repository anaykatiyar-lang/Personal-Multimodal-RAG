"""Table-aware chunker maintaining header context across sliced tabular rows."""

from typing import List
from personal_rag.chunking.base import BaseChunker
from personal_rag.core.types import ParsedDocument, Chunk, ChunkType, ExtractedTable
from personal_rag.core.utils.hashing import compute_chunk_id


class TableChunker(BaseChunker):
    """Chunks documents while ensuring tabular records preserve their column headers."""

    def __init__(self, max_rows_per_table_chunk: int = 25):
        self.max_rows = max_rows_per_table_chunk

    def chunk(self, document: ParsedDocument) -> List[Chunk]:
        chunks: List[Chunk] = []
        doc_meta = document.metadata
        counter = 0

        # BUG-04 FIX: Track only new chunks per table, not running total.
        for table in document.tables:
            new_chunks = self._chunk_single_table(doc_meta, table, counter)
            chunks.extend(new_chunks)
            counter += len(new_chunks)  # only delta, not cumulative

        return chunks

    def _chunk_single_table(
        self,
        doc_meta,
        table: ExtractedTable,
        start_counter: int
    ) -> List[Chunk]:
        if not table.rows:
            return []

        chunks = []
        headers = table.headers
        header_line = "| " + " | ".join(headers) + " |"
        sep_line = "| " + " | ".join(["---"] * len(headers)) + " |"
        prefix = f"{header_line}\n{sep_line}\n"

        for i in range(0, len(table.rows), self.max_rows):
            slice_rows = table.rows[i:i + self.max_rows]
            row_lines = ["| " + " | ".join(r) + " |" for r in slice_rows]
            table_slice_text = prefix + "\n".join(row_lines)

            cid = compute_chunk_id(doc_meta.document_id, start_counter + len(chunks), table_slice_text)
            chunks.append(Chunk(
                chunk_id=cid,
                document_id=doc_meta.document_id,
                source_path=doc_meta.source_path,
                text=table_slice_text,
                chunk_type=ChunkType.TABLE,
                page_number=table.page_number,
                table_id=table.table_id,
                metadata={"sheet_name": table.sheet_name, "row_start": i, "row_end": i + len(slice_rows)}
            ))

        return chunks

