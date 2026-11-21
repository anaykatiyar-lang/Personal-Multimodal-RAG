"""Unit tests for token normalization and entity extraction."""

from personal_rag.core.utils.text_norm import (
    normalize_token,
    clean_whitespace,
    extract_technical_entities,
)


def test_normalize_token_amd_rules():
    assert normalize_token("94 C") == "94"
    assert normalize_token("94°C") == "94"
    assert normalize_token("  94   ") == "94"
    assert normalize_token("Q3 FY27") == "Q3FY27"
    assert normalize_token("ORR-FAN-2214-B") == "ORR-FAN-2214-B"
    assert normalize_token("REV-C2") == "REV-C2"


def test_clean_whitespace():
    raw = "Line 1   \n\n\n\nLine 2\t\xa0with spaces   "
    cleaned = clean_whitespace(raw)
    assert "Line 1\n\nLine 2" in cleaned
    assert not cleaned.endswith(" ")


def test_extract_technical_entities():
    text = (
        "Incident logged against ORR-1847 due to thermal throttle. "
        "Error E7731 was triggered. Fixed in firmware 4.3.2. "
        "Hardware part ORR-FAN-2214-B installed for Q3 FY27 revision REV-C2."
    )
    entities = extract_technical_entities(text)
    assert "ORR-1847" in entities
    assert "E7731" in entities
    assert "4.3.2" in entities
    assert "ORR-FAN-2214-B" in entities
    assert "REV-C2" in entities
    assert any("Q3" in e and "FY27" in e for e in entities)

