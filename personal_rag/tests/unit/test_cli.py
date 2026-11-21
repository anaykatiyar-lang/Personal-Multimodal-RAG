"""Unit tests for personal-rag CLI commands."""

import sys
from pathlib import Path
from personal_rag.cli.main import main


def test_cli_help(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["personal-rag", "--help"])
    try:
        ret = main()
    except SystemExit as e:
        ret = e.code
    assert ret == 0
    captured = capsys.readouterr()
    assert "Personal Advanced RAG Platform" in captured.out


def test_cli_hw(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["personal-rag", "hw"])
    ret = main()
    assert ret == 0
    captured = capsys.readouterr()
    assert "Hardware Inspection" in captured.out
    assert "CPU_ONLY" in captured.out


def test_cli_docs_and_backup(monkeypatch, capsys, temp_dir):
    # Test docs listing
    monkeypatch.setattr(sys, "argv", ["personal-rag", "docs"])
    ret = main()
    assert ret == 0

    # Test backup
    backup_file = temp_dir / "test_backup.db"
    monkeypatch.setattr(sys, "argv", ["personal-rag", "backup", str(backup_file)])
    ret = main()
    assert ret == 0
    assert backup_file.exists()

    # Test restore
    monkeypatch.setattr(sys, "argv", ["personal-rag", "restore", str(backup_file)])
    ret = main()
    assert ret == 0

