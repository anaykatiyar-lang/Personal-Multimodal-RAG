"""Parent Context Retriever expanding child chunks into cohesive parent passages."""

from typing import List, Dict
from personal_rag.core.types import SearchResult, Chunk
from personal_rag.storage.sqlite_store import SQLiteStore


class ParentContextRetriever:
    """Expands retrieved child chunks with their broader parent context envelopes."""

    def __init__(self, sqlite_store: SQLiteStore):
        self.sqlite_store = sqlite_store

    def expand_context(self, search_results: List[SearchResult]) -> List[SearchResult]:
        """Replace child chunks with parent chunks where available, avoiding duplicate parents."""
        expanded: List[SearchResult] = []
        seen_parent_ids: Dict[str, SearchResult] = {}

        for res in search_results:
            child = res.chunk
            if child.parent_id:
                if child.parent_id in seen_parent_ids:
                    # BUG-06 FIX: Keep the highest-scoring sibling's score
                    existing = seen_parent_ids[child.parent_id]
                    if res.score > existing.score:
                        existing.score = res.score
                    continue

                parent = self.sqlite_store.get_chunk(child.parent_id)
                if parent:
                    # Parent adopts child's search score and rank
                    parent_res = SearchResult(
                        chunk=parent,
                        score=res.score,
                        dense_score=res.dense_score,
                        lexical_score=res.lexical_score,
                        rank=res.rank
                    )
                    seen_parent_ids[child.parent_id] = parent_res
                    expanded.append(parent_res)
                    continue

            # If no parent or parent lookup failed, keep original chunk
            expanded.append(res)

        return expanded

