"""Abstract document parser interface."""

import datetime
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Set

from personal_rag.core.types import ParsedDocument, DocumentMetadata
from personal_rag.core.utils.hashing import compute_file_sha256


class BaseParser(ABC):
    """Abstract parser contract guaranteeing a standard ParsedDocument output."""

    @abstractmethod
    def supported_extensions(self) -> Set[str]:
        """Return the lowercase file extensions supported by this parser (e.g. {'.pdf'})."""
        pass

    def can_parse(self, file_path: Path) -> bool:
        """Check if this parser handles the given file."""
        return file_path.suffix.lower() in self.supported_extensions()

    @abstractmethod
    def parse(self, file_path: Path, relative_to: Path = None) -> ParsedDocument:
        """Parse the target file into a unified ParsedDocument representation."""
        pass

    def _build_metadata(
        self,
        file_path: Path,
        mime_type: str,
        page_count: int = 1,
        title: str = None,
        warnings: list = None,
        relative_to: Path = None
    ) -> DocumentMetadata:
        """Build a standardized DocumentMetadata record."""
        path = Path(file_path)
        stat = path.stat()
        file_hash = compute_file_sha256(path)
        rel_path = str(path.relative_to(relative_to)).replace("\\", "/") if relative_to else str(path).replace("\\", "/")

        created_ts = datetime.datetime.fromtimestamp(stat.st_ctime, datetime.timezone.utc).isoformat()
        modified_ts = datetime.datetime.fromtimestamp(stat.st_mtime, datetime.timezone.utc).isoformat()

        return DocumentMetadata(
            document_id=file_hash,
            title=title or path.stem.replace("_", " ").title(),
            source_path=rel_path,
            filename=path.name,
            mime_type=mime_type,
            file_size_bytes=stat.st_size,
            created_at=created_ts,
            modified_at=modified_ts,
            parser_name=self.__class__.__name__,
            page_count=page_count,
            tags=[],
            warnings=warnings or []
        )

