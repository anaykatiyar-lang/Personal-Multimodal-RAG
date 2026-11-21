"""Unit tests for document parsers and ParserFactory."""

import tempfile
from pathlib import Path
from personal_rag.parsers.factory import ParserFactory
from personal_rag.parsers.code_text_parser import CodeTextParser
from personal_rag.parsers.spreadsheet_parser import SpreadsheetParser


def test_parser_factory_resolution(temp_dir):
    factory = ParserFactory()
    assert factory.is_supported(Path("sample.pdf")) is True
    assert factory.is_supported(Path("sample.docx")) is True
    assert factory.is_supported(Path("sample.csv")) is True
    assert factory.is_supported(Path("sample.xlsx")) is True
    assert factory.is_supported(Path("sample.txt")) is True
    assert factory.is_supported(Path("sample.py")) is True
    assert factory.is_supported(Path("sample.dat")) is False  # Unknown binary skipped


def test_code_text_parser(temp_dir):
    txt_file = temp_dir / "service.py"
    txt_file.write_text("DEFAULT_BATCH_TIMEOUT_S = 180\n", encoding="utf-8")

    parser = CodeTextParser()
    doc = parser.parse(txt_file, relative_to=temp_dir)

    assert doc is not None
    assert doc.metadata.filename == "service.py"
    assert "DEFAULT_BATCH_TIMEOUT_S = 180" in doc.raw_text
    assert doc.metadata.mime_type == "text/x-python"


def test_csv_parser(temp_dir):
    csv_file = temp_dir / "bugs.csv"
    csv_file.write_text("ticket,severity,status,fixed_in\nORR-1847,S2,closed,4.3.2\n", encoding="utf-8")

    parser = SpreadsheetParser()
    doc = parser.parse(csv_file, relative_to=temp_dir)

    assert doc is not None
    assert len(doc.tables) == 1
    assert "ORR-1847" in doc.raw_text
    assert "4.3.2" in doc.raw_text
    assert doc.tables[0].headers == ["ticket", "severity", "status", "fixed_in"]

