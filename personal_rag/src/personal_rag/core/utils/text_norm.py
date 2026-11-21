"""Text normalization, token cleaning, and entity token extraction."""

import re
from typing import Set


def normalize_token(text: str) -> str:
    """Normalize string using challenge-compatible rules:
    Uppercased, whitespace removed, and '.', '°', units normalized for exact value comparison.
    """
    if not text:
        return ""
    v = text.upper().strip()
    # Normalize degree Celsius variants like '94 C', '94°C', '94 DEGREES' to '94'
    v = re.sub(r"(\d+)\s*(?:°\s*C|°|DEG(?:REES)?|C)\b", r"\1", v)
    v = re.sub(r"[\s\.\°]", "", v)
    return v


def clean_whitespace(text: str) -> str:
    """Normalize whitespace and newlines for consistent chunking."""
    if not text:
        return ""
    # Replace non-breaking spaces and tabs
    text = text.replace("\xa0", " ").replace("\t", "    ")
    # Collapse multiple consecutive empty lines
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    # Strip trailing whitespace per line
    lines = [line.rstrip() for line in text.split("\n")]
    return "\n".join(lines).strip()


def extract_technical_entities(text: str) -> Set[str]:
    """Extract technical codes, part numbers, tickets, and error identifiers.
    Matches patterns like:
      - Ticket IDs: ORR-1847, BUG-123
      - Error codes: E7731, ERR-404
      - Part numbers: ORR-FAN-2214-B
      - Revision codes: REV-C2, V1.0.2
      - Fiscal quarters: Q3 FY27
    """
    if not text:
        return set()

    entities = set()

    # Hyphenated alphanumeric tokens with single or multi-character segments
    # e.g. ORR-FAN-2214-B, ORR-1847, REV-C2
    hyphenated = re.findall(r"\b[A-Za-z0-9]{2,6}(?:-[A-Za-z0-9]{1,8})+\b", text)
    entities.update(hyphenated)

    # Error codes like E7731, E1002
    errors = re.findall(r"\bE\d{4,5}\b", text)
    entities.update(errors)

    # Fiscal quarters: Q[1-4] FY\d{2,4}
    quarters = re.findall(r"\bQ[1-4]\s*FY\d{2,4}\b", text, re.IGNORECASE)
    entities.update([re.sub(r"\s+", " ", q).upper() for q in quarters])

    # Semantic version numbers: 4.3.2, 1.0.0-rc1
    versions = re.findall(r"\b\d+\.\d+\.\d+(?:-[a-zA-Z0-9\.]+)?\b", text)
    entities.update(versions)

    return entities
