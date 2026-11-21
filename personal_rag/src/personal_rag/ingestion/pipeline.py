"""Incremental, fault-tolerant document ingestion pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Set

from personal_rag.parsers.factory import ParserFactory
from personal_rag.chunking.base import BaseChunker
from personal_rag.chunking.recursive import RecursiveChunker
from personal_rag.storage.sqlite_store import SQLiteStore
from personal_rag.storage.base import BaseVectorStore
from personal_rag.core.utils.hashing import compute_file_sha256
from personal_rag.core.types import Chunk


@dataclass
class IngestionReport:
    """Summary metrics of an ingestion execution."""
    total_files_scanned: int = 0
    files_indexed: int = 0
    files_skipped_unchanged: int = 0
    files_failed: int = 0
    total_chunks_created: int = 0
    errors: List[str] = field(default_factory=list)


class IngestionPipeline:
    """Orchestrates defensive parsing, chunking, deduplication, and multi-index storage."""

    def __init__(
        self,
        sqlite_store: SQLiteStore,
        vector_store: Optional[BaseVectorStore] = None,
        parser_factory: Optional[ParserFactory] = None,
        chunker: Optional[BaseChunker] = None,
    ):
        self.sqlite_store = sqlite_store
        self.vector_store = vector_store
        self.parser_factory = parser_factory or ParserFactory()
        self.chunker = chunker or RecursiveChunker()

    def ingest_directory(
        self,
        directory_path: Path,
        force_reindex: bool = False,
        prune_deleted: bool = False
    ) -> IngestionReport:
        """Scan a directory tree and ingest all supported documents incrementally."""
        root = Path(directory_path)
        if not root.is_dir():
            raise NotADirectoryError(f"Directory not found: {directory_path}")

        report = IngestionReport()
        current_seen_paths: Set[str] = set()

        for dirpath, dirnames, filenames in os.walk(root):
            for fname in sorted(filenames):
                fpath = Path(dirpath) / fname
                rel_path = str(fpath.relative_to(root)).replace("\\", "/")
                current_seen_paths.add(rel_path)
                report.total_files_scanned += 1

                try:
                    # Catch unreadable permissions (e.g. chmod 000)
                    with open(fpath, "rb") as probe:
                        _ = probe.read(8)
                except (PermissionError, OSError) as e:
                    report.files_failed += 1
                    report.errors.append(f"Unreadable file {rel_path}: {e}")
                    continue

                if not self.parser_factory.is_supported(fpath):
                    # Unknown file type — gracefully skip
                    continue

                try:
                    file_sha = compute_file_sha256(fpath)

                    # Incremental check: Skip unchanged files
                    if not force_reindex and not self.sqlite_store.is_document_modified(rel_path, file_sha):
                        report.files_skipped_unchanged += 1
                        continue

                    # Parse file
                    parsed_doc = self.parser_factory.parse_file(fpath, relative_to=root)
                    if not parsed_doc:
                        continue

                    # Clean up old chunks if re-indexing modified file
                    self.sqlite_store.delete_chunks_for_document(parsed_doc.metadata.document_id)
                    if self.vector_store:
                        self.vector_store.delete_vectors_for_document(parsed_doc.metadata.document_id)

                    # Register document
                    self.sqlite_store.upsert_document(parsed_doc.metadata)

                    # Generate chunks
                    chunks: List[Chunk] = self.chunker.chunk(parsed_doc)

                    # Index chunks into SQLite and FTS5
                    if chunks:
                        self.sqlite_store.index_chunks(chunks)
                        if self.vector_store:
                            self.vector_store.upsert_vectors(chunks)

                    report.files_indexed += 1
                    report.total_chunks_created += len(chunks)

                except Exception as e:
                    report.files_failed += 1
                    report.errors.append(f"Failed to ingest {rel_path}: {e}")

        # Optional pruning of removed documents
        if prune_deleted:
            self._prune_stale_documents(current_seen_paths)

        return report

    def ingest_single_file(self, file_path: Path, relative_to: Optional[Path] = None) -> bool:
        """Ingest or update a single document."""
        path = Path(file_path)
        if not path.is_file():
            return False

        if not self.parser_factory.is_supported(path):
            return False

        parsed_doc = self.parser_factory.parse_file(path, relative_to=relative_to)
        if not parsed_doc:
            return False

        self.sqlite_store.delete_chunks_for_document(parsed_doc.metadata.document_id)
        if self.vector_store:
            self.vector_store.delete_vectors_for_document(parsed_doc.metadata.document_id)

        self.sqlite_store.upsert_document(parsed_doc.metadata)
        chunks = self.chunker.chunk(parsed_doc)
        if chunks:
            self.sqlite_store.index_chunks(chunks)
            if self.vector_store:
                self.vector_store.upsert_vectors(chunks)
        return True

    def _prune_stale_documents(self, current_paths: Set[str]) -> None:
        """Purge documents and chunks that no longer exist on disk."""
        all_docs = self.sqlite_store.list_documents(limit=10000)
        for doc in all_docs:
            if doc.source_path not in current_paths:
                self.sqlite_store.delete_document(doc.document_id)
                if self.vector_store:
                    self.vector_store.delete_vectors_for_document(doc.document_id)

