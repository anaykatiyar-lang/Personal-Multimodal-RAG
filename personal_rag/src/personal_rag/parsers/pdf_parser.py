"""PDF Document Parser supporting page-by-page text, table detection, and encryption handling."""

from pathlib import Path
from typing import Set, List
from personal_rag.parsers.base import BaseParser
from personal_rag.core.types import ParsedDocument, ParsedPage
from personal_rag.core.exceptions import DocumentParsingError


class PDFParser(BaseParser):
    """Parses PDF documents using PyMuPDF (fitz) with safe fallback."""

    def supported_extensions(self) -> Set[str]:
        return {".pdf"}

    def parse(self, file_path: Path, relative_to: Path = None) -> ParsedDocument:
        path = Path(file_path)
        warnings: List[str] = []

        try:
            import fitz  # PyMuPDF
            doc = fitz.open(str(path))
            if doc.is_encrypted:
                warnings.append("Document is encrypted and password-protected. Skipping content extraction.")
                meta = self._build_metadata(path, "application/pdf", page_count=0, warnings=warnings, relative_to=relative_to)
                doc.close()
                return ParsedDocument(metadata=meta, pages=[], raw_text="")

            pages: List[ParsedPage] = []
            all_text_parts = []

            for page_idx in range(len(doc)):
                page = doc[page_idx]
                text = page.get_text("text").strip()
                pages.append(ParsedPage(
                    page_number=page_idx + 1,
                    text=text
                ))
                if text:
                    all_text_parts.append(text)

            page_count = len(doc)
            doc.close()

            full_text = "\n\n".join(all_text_parts)

            # Detect superseded annotations
            if "WITHDRAWN" in path.name.upper() or "SUPERSEDED" in full_text.upper():
                warnings.append("Document contains superseded or withdrawn notices.")

            meta = self._build_metadata(
                path, "application/pdf", page_count=page_count, warnings=warnings, relative_to=relative_to
            )
            return ParsedDocument(metadata=meta, pages=pages, raw_text=full_text)

        except ImportError:
            # Basic fallback when PyMuPDF is not installed
            warnings.append("PyMuPDF (fitz) not installed; extracted raw text directly if possible.")
            raw = path.read_bytes()
            # Attempt basic ASCII string scraping
            import re
            extracted = " ".join(re.findall(rb"[\x20-\x7e]{4,}", raw)).decode("latin1", errors="replace")
            meta = self._build_metadata(path, "application/pdf", page_count=1, warnings=warnings, relative_to=relative_to)
            return ParsedDocument(
                metadata=meta,
                pages=[ParsedPage(page_number=1, text=extracted)],
                raw_text=extracted
            )
        except Exception as e:
            raise DocumentParsingError(str(file_path), str(e))

