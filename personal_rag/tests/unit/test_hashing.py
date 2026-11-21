"""Unit tests for hashing utilities and chunk ID generation."""

import tempfile
from pathlib import Path
from personal_rag.core.utils.hashing import (
    compute_content_sha256,
    compute_file_sha256,
    compute_chunk_id,
)


def test_compute_content_sha256():
    hash1 = compute_content_sha256("test document content")
    hash2 = compute_content_sha256("test document content")
    hash3 = compute_content_sha256("different content")
    assert hash1 == hash2
    assert hash1 != hash3
    assert len(hash1) == 64


def test_compute_file_sha256(temp_dir):
    test_file = temp_dir / "sample.txt"
    test_file.write_text("Hello Personal RAG!", encoding="utf-8")

    digest1 = compute_file_sha256(test_file)
    digest2 = compute_content_sha256("Hello Personal RAG!")
    assert digest1 == digest2


def test_compute_chunk_id_determinism():
    cid1 = compute_chunk_id("doc_12345678", 0, "Chunk text snippet")
    cid2 = compute_chunk_id("doc_12345678", 0, "Chunk text snippet")
    cid3 = compute_chunk_id("doc_12345678", 1, "Chunk text snippet")
    assert cid1 == cid2
    assert cid1 != cid3
    assert cid1.startswith("chk_doc_1234_0000_")

