"""Spreadsheet Parser for XLSX workbooks and CSV tabular datasets."""

import csv
from pathlib import Path
from typing import Set, List
from personal_rag.parsers.base import BaseParser
from personal_rag.core.types import ParsedDocument, ParsedPage, ExtractedTable
from personal_rag.core.exceptions import DocumentParsingError


class SpreadsheetParser(BaseParser):
    """Parses Excel (.xlsx) and Comma-Separated Values (.csv) files."""

    def supported_extensions(self) -> Set[str]:
        return {".xlsx", ".csv"}

    def parse(self, file_path: Path, relative_to: Path = None) -> ParsedDocument:
        path = Path(file_path)
        ext = path.suffix.lower()

        if ext == ".csv":
            return self._parse_csv(path, relative_to)
        else:
            return self._parse_xlsx(path, relative_to)

    def _parse_csv(self, path: Path, relative_to: Path = None) -> ParsedDocument:
        tables_extracted: List[ExtractedTable] = []
        rows_data: List[List[str]] = []

        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                reader = csv.reader(f)
                for r in reader:
                    cleaned_r = [cell.strip() for cell in r]
                    if any(cleaned_r):
                        rows_data.append(cleaned_r)

            if not rows_data:
                meta = self._build_metadata(path, "text/csv", page_count=1, relative_to=relative_to)
                return ParsedDocument(metadata=meta, pages=[ParsedPage(page_number=1, text="")], raw_text="")

            headers = rows_data[0]
            content_rows = rows_data[1:] if len(rows_data) > 1 else []

            header_line = "| " + " | ".join(headers) + " |"
            sep_line = "| " + " | ".join(["---"] * len(headers)) + " |"
            row_lines = ["| " + " | ".join(r) + " |" for r in content_rows]
            md_table = "\n".join([header_line, sep_line] + row_lines)

            table_obj = ExtractedTable(
                table_id="csv_table",
                page_number=1,
                headers=headers,
                rows=content_rows,
                markdown_repr=md_table
            )
            tables_extracted.append(table_obj)

            meta = self._build_metadata(path, "text/csv", page_count=1, relative_to=relative_to)
            return ParsedDocument(
                metadata=meta,
                pages=[ParsedPage(page_number=1, text=md_table, tables=tables_extracted)],
                raw_text=md_table,
                tables=tables_extracted
            )
        except Exception as e:
            raise DocumentParsingError(str(path), str(e))

    def _parse_xlsx(self, path: Path, relative_to: Path = None) -> ParsedDocument:
        tables_extracted: List[ExtractedTable] = []
        sheet_text_blocks: List[str] = []
        warnings: List[str] = []

        try:
            import openpyxl
            wb = openpyxl.load_workbook(str(path), data_only=True, read_only=True)

            for sheet_idx, sheet_name in enumerate(wb.sheetnames, start=1):
                ws = wb[sheet_name]
                rows = []
                for row in ws.iter_rows(values_only=True):
                    row_vals = [str(c).strip() if c is not None else "" for c in row]
                    if any(row_vals):
                        rows.append(row_vals)

                if rows:
                    headers = rows[0]
                    content_rows = rows[1:] if len(rows) > 1 else []

                    header_line = "| " + " | ".join(headers) + " |"
                    sep_line = "| " + " | ".join(["---"] * len(headers)) + " |"
                    row_lines = ["| " + " | ".join(r) + " |" for r in content_rows]
                    md_table = "\n".join([header_line, sep_line] + row_lines)

                    t_obj = ExtractedTable(
                        table_id=f"sheet_{sheet_name}",
                        page_number=sheet_idx,
                        sheet_name=sheet_name,
                        headers=headers,
                        rows=content_rows,
                        markdown_repr=md_table
                    )
                    tables_extracted.append(t_obj)
                    sheet_text_blocks.append(f"### Sheet: {sheet_name}\n\n{md_table}")

            wb.close()
            full_text = "\n\n".join(sheet_text_blocks)
            meta = self._build_metadata(
                path,
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                page_count=len(sheet_text_blocks) or 1,
                warnings=warnings,
                relative_to=relative_to
            )
            return ParsedDocument(
                metadata=meta,
                pages=[ParsedPage(page_number=1, text=full_text, tables=tables_extracted)],
                raw_text=full_text,
                tables=tables_extracted
            )
        except ImportError:
            raise DocumentParsingError(str(path), "openpyxl library is required to parse .xlsx workbooks")
        except Exception as e:
            raise DocumentParsingError(str(path), str(e))

