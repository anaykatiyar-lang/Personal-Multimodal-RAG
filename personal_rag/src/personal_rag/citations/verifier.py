"""Citation verifier confirming factual presence, necessity, and source validity."""

import re
from typing import List, Set, Dict, Optional
from personal_rag.core.types import Citation, SearchResult
from personal_rag.core.utils.text_norm import normalize_token

# Common words to filter when doing token overlap verification
_STOPWORDS = {
    "a", "an", "the", "and", "or", "in", "on", "at", "to", "for", "with",
    "by", "about", "as", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "can", "could", "should",
    "would", "will", "this", "that", "these", "those", "from", "it", "its",
    "of", "not", "no", "yes", "what", "which", "who", "whom", "where", "when",
    "why", "how", "all", "each", "both", "answer", "citation", "citations"
}


class CitationVerifier:
    """Verifies that cited sources genuinely contain supporting factual evidence."""

    def verify_citations(
        self,
        answer: str,
        raw_citations: List[str],
        retrieved_results: List[SearchResult],
        query: str = ""
    ) -> List[Citation]:
        """Validate citations against retrieved passages and necessity constraints."""
        if not answer.strip() or not raw_citations:
            return []

        norm_ans = normalize_token(answer)
        if not norm_ans:
            return []

        # Index retrieved results by normalized source path
        results_by_path: Dict[str, List[SearchResult]] = {}
        for r in retrieved_results:
            p = r.chunk.source_path.strip("/")
            if p not in results_by_path:
                results_by_path[p] = []
            results_by_path[p].append(r)

        verified_citations: List[Citation] = []
        seen_paths: Set[str] = set()

        # Step 1: Direct Answer Verification (for exact values and natural answers)
        ans_tokens = {
            t.lower() for t in re.findall(r"\b[A-Za-z0-9_-]+\b", answer)
            if t.lower() not in _STOPWORDS and len(t) > 1
        }

        for raw_path in raw_citations:
            clean_p = raw_path.strip("/'\"` ")

            # Filter out superseded or withdrawn documents
            if "WITHDRAWN" in clean_p.upper() or "SUPERSEDED" in clean_p.upper():
                continue

            matching_results = results_by_path.get(clean_p, [])
            found_evidence = False
            best_chunk = None

            for res in matching_results:
                norm_chunk_text = normalize_token(res.chunk.text)

                # 1. Exact value substring match (e.g. "94", "4.3.2", "Q3 FY27", "ORR-FAN-2214-B")
                if norm_ans in norm_chunk_text:
                    found_evidence = True
                    best_chunk = res.chunk
                    break

                # 2. Token overlap match for natural language answers
                if ans_tokens:
                    chunk_tokens = {
                        t.lower() for t in re.findall(r"\b[A-Za-z0-9_-]+\b", res.chunk.text)
                        if t.lower() not in _STOPWORDS
                    }
                    overlap = ans_tokens.intersection(chunk_tokens)
                    if len(ans_tokens) > 3 and len(overlap) / len(ans_tokens) >= 0.35:
                        found_evidence = True
                        best_chunk = res.chunk
                        break
                    elif len(overlap) >= 3:
                        found_evidence = True
                        best_chunk = res.chunk
                        break

            if found_evidence and best_chunk and clean_p not in seen_paths:
                seen_paths.add(clean_p)
                verified_citations.append(Citation(
                    source_path=best_chunk.source_path,
                    document_id=best_chunk.document_id,
                    chunk_id=best_chunk.chunk_id,
                    page_number=best_chunk.page_number,
                    excerpt=best_chunk.text[:200],
                    confidence=1.0
                ))

        # Step 2: Multi-hop intermediate entity verification
        # Look for linking identifiers (e.g. ticket numbers ORR-1847 or error codes E7731)
        ticket_match = re.search(r"ORR-\d+", query.upper()) or re.search(r"ORR-\d+", "\n".join(r.chunk.text for r in retrieved_results))
        ticket_id = ticket_match.group(0) if ticket_match else None

        for raw_path in raw_citations:
            clean_p = raw_path.strip("/'\"` ")
            if clean_p in seen_paths or "WITHDRAWN" in clean_p.upper():
                continue

            matching_results = results_by_path.get(clean_p, [])
            for res in matching_results:
                if ticket_id and ticket_id in res.chunk.text.upper():
                    seen_paths.add(clean_p)
                    verified_citations.append(Citation(
                        source_path=res.chunk.source_path,
                        document_id=res.chunk.document_id,
                        chunk_id=res.chunk.chunk_id,
                        page_number=res.chunk.page_number,
                        excerpt=res.chunk.text[:200],
                        confidence=0.9
                    ))
                    break

        return verified_citations
