"""Image Parser for diagrams, charts, and technical photographs."""

import base64
from pathlib import Path
from typing import Set, Optional
from personal_rag.parsers.base import BaseParser
from personal_rag.core.types import ParsedDocument, ParsedPage
from personal_rag.core.exceptions import DocumentParsingError


class ImageParser(BaseParser):
    """Parses images (PNG, JPG) and prepares metadata and multimodal representations."""

    def supported_extensions(self) -> Set[str]:
        return {".png", ".jpg", ".jpeg"}

    def parse(self, file_path: Path, relative_to: Path = None) -> ParsedDocument:
        path = Path(file_path)
        ext = path.suffix.lower()
        mime = "image/png" if ext == ".png" else "image/jpeg"

        try:
            raw_bytes = path.read_bytes()
            b64_str = base64.b64encode(raw_bytes).decode("ascii")

            # Try to read dimensions with PIL if available, otherwise fallback
            width, height = 0, 0
            try:
                from PIL import Image
                with Image.open(path) as img:
                    width, height = img.size
            except Exception:
                pass

            description = f"[Image: {path.name} | Dimensions: {width}x{height} | Format: {ext.upper()}]"

            meta = self._build_metadata(path, mime, page_count=1, relative_to=relative_to)
            meta.warnings.append("OCR/Vision model extraction is pending user activation.")

            page = ParsedPage(
                page_number=1,
                text=description,
                image_paths=[str(path)],
                metadata={"width": width, "height": height, "base64_preview": b64_str[:128]}
            )
            return ParsedDocument(
                metadata=meta,
                pages=[page],
                raw_text=description
            )
        except Exception as e:
            raise DocumentParsingError(str(file_path), str(e))

