"""Pytest configuration and deterministic test fixtures."""

import os
import shutil
import tempfile
from pathlib import Path
import pytest

from personal_rag.core.config import Settings
from personal_rag.storage.sqlite_store import SQLiteStore
from personal_rag.storage.vector_store import InMemoryVectorStore


@pytest.fixture
def temp_dir():
    """Create a temporary directory cleaned up after test completion."""
    d = tempfile.mkdtemp(prefix="personal_rag_test_")
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def test_settings(temp_dir):
    """Deterministic CPU-only test settings fixture."""
    return Settings(
        CPU_ONLY=True,
        GPU_EXECUTION_APPROVED=False,
        SQLITE_DB_PATH=temp_dir / "test_kb.db",
        QDRANT_STORAGE_PATH=temp_dir / "qdrant",
        DOCUMENTS_DIR=temp_dir / "documents",
        CACHE_DIR=temp_dir / "cache",
        LLM_PROVIDER="mock",
    )


@pytest.fixture
def sqlite_store(temp_dir):
    """Instantiate a fresh SQLite store on a temporary database."""
    db_path = temp_dir / "test.db"
    return SQLiteStore(db_path)


@pytest.fixture
def vector_store():
    """Instantiate a zero-dependency in-memory vector store."""
    return InMemoryVectorStore()

