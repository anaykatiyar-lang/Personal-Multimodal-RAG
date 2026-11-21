"""Core primitives, configuration, type contracts, and hardware management."""

from personal_rag.core.types import (
    DocumentMetadata,
    ExtractedTable,
    ParsedPage,
    ParsedDocument,
    Chunk,
    ChunkType,
    Citation,
    SearchResult,
    QueryResult,
)
from personal_rag.core.config import Settings, get_settings
from personal_rag.core.hardware import HardwareManager, DeviceInfo
from personal_rag.core.exceptions import (
    PersonalRAGException,
    HardwarePolicyViolationError,
    DocumentParsingError,
    StorageError,
    RetrievalError,
    InsufficientEvidenceError,
    CitationVerificationError,
    ModelInvocationError,
)

__all__ = [
    "DocumentMetadata",
    "ExtractedTable",
    "ParsedPage",
    "ParsedDocument",
    "Chunk",
    "ChunkType",
    "Citation",
    "SearchResult",
    "QueryResult",
    "Settings",
    "get_settings",
    "HardwareManager",
    "DeviceInfo",
    "PersonalRAGException",
    "HardwarePolicyViolationError",
    "DocumentParsingError",
    "StorageError",
    "RetrievalError",
    "InsufficientEvidenceError",
    "CitationVerificationError",
    "ModelInvocationError",
]

