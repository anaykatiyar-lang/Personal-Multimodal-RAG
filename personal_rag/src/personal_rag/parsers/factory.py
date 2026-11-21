"""Factory for selecting and orchestrating document parsers."""

from pathlib import Path
from typing import List, Optional, Dict

from personal_rag.parsers.base import BaseParser
from personal_rag.parsers.pdf_parser import PDFParser
from personal_rag.parsers.docx_parser import DocxParser
from personal_rag.parsers.spreadsheet_parser import SpreadsheetParser
from personal_rag.parsers.code_text_parser import CodeTextParser
from personal_rag.parsers.image_parser import ImageParser
from personal_rag.core.types import ParsedDocument


class ParserFactory:
    """Registry and selector for document format parsers."""

    def __init__(self, parsers: Optional[List[BaseParser]] = None):
        self._parsers: List[BaseParser] = parsers or [
            PDFParser(),
            DocxParser(),
            SpreadsheetParser(),
            CodeTextParser(),
            ImageParser(),
        ]
        self._extension_map: Dict[str, BaseParser] = {}
        for p in self._parsers:
            for ext in p.supported_extensions():
                self._extension_map[ext.lower()] = p

    def get_parser(self, file_path: Path) -> Optional[BaseParser]:
        """Look up the appropriate parser by file extension."""
        ext = file_path.suffix.lower()
        return self._extension_map.get(ext)

    def is_supported(self, file_path: Path) -> bool:
        """Check if any registered parser supports this file format."""
        return self.get_parser(file_path) is not None

    def parse_file(self, file_path: Path, relative_to: Optional[Path] = None) -> Optional[ParsedDocument]:
        """Parse a file using the registered parser, or return None if unsupported."""
        parser = self.get_parser(file_path)
        if not parser:
            return None
        return parser.parse(file_path, relative_to=relative_to)

