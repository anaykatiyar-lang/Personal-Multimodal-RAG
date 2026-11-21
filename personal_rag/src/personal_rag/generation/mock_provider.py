"""Deterministic Mock Model Provider for test suites."""

import re
from typing import Optional
from personal_rag.generation.base import BaseModelProvider


class MockProvider(BaseModelProvider):
    """Deterministic model provider extracting verifiable factual answers from context.

    Matching is done against the QUESTION line only (not the full prompt) so
    that conversation-history prefixes from previous turns do not interfere.
    """

    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
        **kwargs
    ) -> str:
        # Extract only the QUESTION line so previous conversation history cannot
        # accidentally trigger the wrong case.
        question = ""
        for line in prompt.splitlines():
            if line.strip().upper().startswith("QUESTION:"):
                question = line.split(":", 1)[1].strip().lower()
                break
        if not question:
            question = prompt.lower()

        # Helper to find which file in prompt contains target token
        def find_source_file_for_token(tok: str) -> Optional[str]:
            blocks = prompt.split("--- FILE: ")
            for b in blocks[1:]:
                header = b.splitlines()[0]
                source_p = header.split(" (Page")[0].strip()
                if tok.lower() in b.lower():
                    return source_p
            return None

        # Case 1: TQ-40 junction temp
        if "maximum junction temperature" in question:
            if "94" in prompt:
                src = find_source_file_for_token("94") or "specs/tq40_datasheet_r2.pdf"
                return f"ANSWER: 94\nCITATIONS: {src}"

        # Case 2: TQ-60 customer sampling
        if "customer sampling" in question:
            if "q3 fy27" in prompt.lower():
                src = find_source_file_for_token("Q3 FY27") or "planning/roadmap_fy27.docx"
                return f"ANSWER: Q3 FY27\nCITATIONS: {src}"

        # Case 3: Fan part number
        if "fan assembly" in question:
            if "orr-fan-2214-b" in prompt.lower():
                src = find_source_file_for_token("ORR-FAN-2214-B") or "support/rma_parts.xlsx"
                return f"ANSWER: ORR-FAN-2214-B\nCITATIONS: {src}"

        # Case 4: Firmware fixed ticket
        if "fixed ticket orr-1847" in question or "fixed ticket" in question:
            if "4.3.2" in prompt:
                src = find_source_file_for_token("4.3.2") or "support/bug_database.csv"
                return f"ANSWER: 4.3.2\nCITATIONS: {src}"

        # Case 5: Thermal throttle error code
        if ("thermal throttle engages" in question or "thermal throttle error" in question
                or "error code" in question):
            if "e7731" in prompt.lower():
                src = find_source_file_for_token("E7731") or "logs/prod_inference_2026-09-02.log"
                return f"ANSWER: E7731\nCITATIONS: {src}"

        # Case 6: Ingest service default timeout
        if "default batch timeout" in question or "batch timeout" in question:
            if "180" in prompt:
                src = find_source_file_for_token("180") or "engineering/ingest_service.py"
                return f"ANSWER: 180\nCITATIONS: {src}"

        # Case 7: Backplane pin
        if "pin carries therm_alert" in question or "backplane pin" in question:
            if "b14" in prompt.lower():
                src = find_source_file_for_token("B14") or "specs/backplane_pinout.png"
                return f"ANSWER: B14\nCITATIONS: {src}"

        # Case 8: Asset label board revision
        if "board revision" in question and "label" in question:
            if "rev-c2" in prompt.lower():
                src = find_source_file_for_token("REV-C2") or "support/asset_label.jpg"
                return f"ANSWER: REV-C2\nCITATIONS: {src}"

        # Case 9: Multi-hop log + bug database
        if ("thermal throttle incident" in question
                and "fixed the underlying defect" in question):
            if "4.3.2" in prompt and "orr-1847" in prompt.lower():
                src1 = find_source_file_for_token("ORR-1847") or "logs/prod_inference_2026-09-02.log"
                src2 = find_source_file_for_token("4.3.2") or "support/bug_database.csv"
                return f"ANSWER: 4.3.2\nCITATIONS: {src1}, {src2}"

        # Default refusal if no answer found
        return "ANSWER:\nCITATIONS:"
