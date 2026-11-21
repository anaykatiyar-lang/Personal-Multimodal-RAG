"""Command-line interface for the Personal RAG Platform."""


def main() -> int:
    from personal_rag.cli.main import main as _main
    return _main()


__all__ = ["main"]
