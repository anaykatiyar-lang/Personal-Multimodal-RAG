"""Word DOCX Parser extracting hierarchical headings, paragraphs, and tables."""

from pathlib import Path
from typing import Set, List
from personal_rag.parsers.base import BaseParser
from personal_rag.core.types import ParsedDocument, ParsedPage, ExtractedTable
from personal_rag.core.exceptions import DocumentParsingError


class DocxParser(BaseParser):
    """Parses Microsoft Word (.docx) documents."""

    def supported_extensions(self) -> Set[str]:
        return {".docx"}

    def parse(self, file_path: Path, relative_to: Path = None) -> ParsedDocument:
        path = Path(file_path)
        warnings: List[str] = []

        try:
            from docx import Document
            doc = Document(str(path))

            text_lines = []
            tables_extracted = []

            # 1. Paragraphs
            for p in doc.paragraphs:
                txt = p.text.strip()
                if txt:
                    text_lines.append(txt)

            # 2. Tables (Crucial for roadmaps and technical data)
            for t_idx, table in enumerate(doc.tables, start=1):
                rows_data = []
                for row in table.rows:
                    row_cells = [c.text.strip().replace("\n", " ") for c in row.cells]
                    if any(row_cells):
                        rows_data.append(row_cells)

                if rows_data:
                    headers = rows_data[0]
                    content_rows = rows_data[1:] if len(rows_data) > 1 else []

                    # Convert to standard Markdown table format
                    header_line = "| " + " | ".join(headers) + " |"
                    separator_line = "| " + " | ".join(["---"] * len(headers)) + " |"
                    row_lines = ["| " + " | ".join(r) + " |" for r in content_rows]
                    md_table = "\n".join([header_line, separator_line] + row_lines)

                    t_obj = ExtractedTable(
                        table_id=f"table_{t_idx}",
                        page_number=1,
                        headers=headers,
                        rows=content_rows,
                        markdown_repr=md_table
                    )
                    tables_extracted.append(t_obj)
                    text_lines.append(f"\n[Table {t_idx}]\n{md_table}\n")

            full_text = "\n".join(text_lines)
            page = ParsedPage(page_number=1, text=full_text, tables=tables_extracted)
            meta = self._build_metadata(
                path,
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                page_count=1,
                warnings=warnings,
                relative_to=relative_to
            )
            return ParsedDocument(
                metadata=meta,
                pages=[page],
                raw_text=full_text,
                tables=tables_extracted
            )

        except ImportError:
            # Fallback for plain unzipped XML parsing
            import zipfile
            import xml.etree.ElementTree as ET
            warnings.append("python-docx not installed; parsed word/document.xml directly.")
            try:
                with zipfile.ZipFile(str(path), "r") as z:
                    xml_content = z.read("word/document.xml")
                tree = ET.fromstring(xml_content)
                texts = [elem.text for elem in tree.iter() if elem.text]
                full_text = " ".join(texts)
                meta = self._build_metadata(path, "application/docx", page_count=1, warnings=warnings, relative_to=relative_to)
                return ParsedDocument(
                    metadata=meta,
                    pages=[ParsedPage(page_number=1, text=full_text)],
                    raw_text=full_text
                )
            except Exception as e:
                raise DocumentParsingError(str(file_path), f"XML fallback failed: {e}")
        except Exception as e:
            raise DocumentParsingError(str(file_path), str(e))

