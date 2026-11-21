"""Deterministic cryptographic hashing and ID generation."""

import hashlib
from pathlib import Path
from typing import Union


def compute_content_sha256(data: Union[str, bytes]) -> str:
    """Compute the SHA-256 hash of a string or byte array."""
    if isinstance(data, str):
        data = data.encode("utf-8")
    hasher = hashlib.sha256()
    hasher.update(data)
    return hasher.hexdigest()


def compute_file_sha256(file_path: Union[str, Path], chunk_size: int = 65536) -> str:
    """Compute the SHA-256 hash of a file streaming in chunks."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")

    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_chunk_id(document_id: str, index: int, chunk_text: str) -> str:
    """Generate a stable, deterministic chunk identifier."""
    seed = f"{document_id}:{index}:{chunk_text[:64]}"
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    return f"chk_{document_id[:8]}_{index:04d}_{digest[:8]}"

