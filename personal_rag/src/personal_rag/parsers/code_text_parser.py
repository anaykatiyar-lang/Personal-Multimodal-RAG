"""Parser for plain text, logs, markdown, python code, and json structures."""

import json
from pathlib import Path
from typing import Set
from personal_rag.parsers.base import BaseParser
from personal_rag.core.types import ParsedDocument, ParsedPage
from personal_rag.core.exceptions import DocumentParsingError


class CodeTextParser(BaseParser):
    """Parses text, code, logs, and markdown files with fault-tolerant utf-8 decoding."""

    def supported_extensions(self) -> Set[str]:
        return {".txt", ".log", ".py", ".md", ".json", ".html"}

    def parse(self, file_path: Path, relative_to: Path = None) -> ParsedDocument:
        path = Path(file_path)
        ext = path.suffix.lower()

        mime_map = {
            ".txt": "text/plain",
            ".log": "text/x-log",
            ".py": "text/x-python",
            ".md": "text/markdown",
            ".json": "application/json",
            ".html": "text/html"
        }
        mime = mime_map.get(ext, "text/plain")

        try:
            content = path.read_text(encoding="utf-8", errors="replace")

            # Pretty-print JSON for readable retrieval chunks
            if ext == ".json":
                try:
                    obj = json.loads(content)
                    content = json.dumps(obj, indent=2)
                except Exception:
                    pass

            meta = self._build_metadata(path, mime, page_count=1, relative_to=relative_to)
            return ParsedDocument(
                metadata=meta,
                pages=[ParsedPage(page_number=1, text=content)],
                raw_text=content
            )
        except Exception as e:
            raise DocumentParsingError(str(file_path), str(e))

