"""Core domain models and data contracts for Personal RAG."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ChunkType(str, Enum):
    """Categorization of chunk content structure."""
    TEXT = "text"
    TABLE = "table"
    CODE = "code"
    IMAGE_DESCRIPTION = "image_description"
    HEADING = "heading"


class DocumentMetadata(BaseModel):
    """Metadata describing an ingested document."""
    document_id: str = Field(..., description="Stable SHA-256 fingerprint of the document bytes")
    title: str = Field(..., description="Document title or sanitized file stem")
    source_path: str = Field(..., description="Relative or canonical path of the source file")
    filename: str = Field(..., description="Original filename with extension")
    mime_type: str = Field(..., description="Standard MIME type or extension identifier")
    file_size_bytes: int = Field(0, description="Size of the raw file in bytes")
    created_at: str = Field(..., description="ISO 8601 timestamp of creation")
    modified_at: str = Field(..., description="ISO 8601 timestamp of file modification")
    parser_name: str = Field(..., description="Name of the parser component used")
    page_count: int = Field(1, description="Number of parsed pages or sheets")
    tags: List[str] = Field(default_factory=list, description="User or automated category tags")
    warnings: List[str] = Field(default_factory=list, description="Non-fatal warnings encountered")


class ExtractedTable(BaseModel):
    """Table extracted from a document with structured matrix representation."""
    table_id: str
    page_number: int = 1
    sheet_name: Optional[str] = None
    headers: List[str] = Field(default_factory=list)
    rows: List[List[str]] = Field(default_factory=list)
    markdown_repr: str = Field("", description="Standard Markdown table representation")
    caption: Optional[str] = None


class ParsedPage(BaseModel):
    """Content extracted from a single page, slide, or sheet."""
    page_number: int
    text: str
    tables: List[ExtractedTable] = Field(default_factory=list)
    image_paths: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ParsedDocument(BaseModel):
    """Unified container produced by all document parsers."""
    metadata: DocumentMetadata
    pages: List[ParsedPage] = Field(default_factory=list)
    raw_text: str = Field("", description="Consolidated raw plain text")
    tables: List[ExtractedTable] = Field(default_factory=list)


class Chunk(BaseModel):
    """A retrievable text fragment with full provenance and parent-child hierarchy."""
    chunk_id: str = Field(..., description="Unique deterministic chunk ID (hash of doc_id + offset/type)")
    document_id: str = Field(..., description="Foreign key to DocumentMetadata")
    source_path: str = Field(..., description="Path of source file for immediate attribution")
    parent_id: Optional[str] = Field(None, description="Optional ID of broader parent context chunk")
    text: str = Field(..., description="Chunk content provided to search indices and LLM")
    chunk_type: ChunkType = Field(default=ChunkType.TEXT)
    page_number: int = Field(1, description="Page, slide, or sheet number")
    section_heading: Optional[str] = Field(None, description="Enclosing heading if available")
    table_id: Optional[str] = Field(None, description="Reference to originating ExtractedTable")
    embedding: Optional[List[float]] = Field(None, description="Dense vector embedding")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Citation(BaseModel):
    """Grounding citation linking a generated statement to verified source evidence."""
    source_path: str = Field(..., description="Corpus-relative path of the cited file")
    document_id: str = Field(..., description="ID of the originating document")
    chunk_id: str = Field(..., description="Exact chunk ID supporting the claim")
    page_number: int = Field(1, description="Page number of the supporting evidence")
    excerpt: str = Field(..., description="Verbatim text excerpt containing the evidence")
    confidence: float = Field(1.0, description="Verification confidence score")


class SearchResult(BaseModel):
    """Result item returned by retrieval modules."""
    chunk: Chunk
    score: float = Field(..., description="Combined fusion score")
    dense_score: Optional[float] = None
    lexical_score: Optional[float] = None
    rank: int = 1


class QueryResult(BaseModel):
    """End-to-end response produced by the RAG platform."""
    query: str
    answer: str
    citations: List[Citation] = Field(default_factory=list)
    confidence: float = 0.0
    is_refusal: bool = False
    retrieved_chunks: List[SearchResult] = Field(default_factory=list)
    latency_breakdown_ms: Dict[str, float] = Field(default_factory=dict)
    strategy_used: str = "hybrid_rrf"

