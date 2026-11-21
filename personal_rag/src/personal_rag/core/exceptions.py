"""Domain exceptions for the Personal Advanced RAG Platform."""

class PersonalRAGException(Exception):
    """Base exception for all Personal RAG platform errors."""
    pass


class HardwarePolicyViolationError(PersonalRAGException):
    """Raised when an operation attempts to allocate or run GPU compute without explicit user approval."""
    pass


class DocumentParsingError(PersonalRAGException):
    """Raised when a document cannot be parsed or decoded."""
    def __init__(self, file_path: str, reason: str):
        super().__init__(f"Failed to parse document '{file_path}': {reason}")
        self.file_path = file_path
        self.reason = reason


class StorageError(PersonalRAGException):
    """Raised when storage operations (SQLite, Qdrant) fail."""
    pass


class RetrievalError(PersonalRAGException):
    """Raised when retrieval fails."""
    pass


class InsufficientEvidenceError(PersonalRAGException):
    """Raised or used to signal that the corpus does not contain sufficient verified evidence to answer."""
    pass


class CitationVerificationError(PersonalRAGException):
    """Raised when a cited document or passage fails validation."""
    pass


class ModelInvocationError(PersonalRAGException):
    """Raised when an LLM generation or model provider invocation fails."""
    pass

