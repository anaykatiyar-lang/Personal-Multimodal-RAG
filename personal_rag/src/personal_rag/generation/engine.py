"""Generation Engine building grounded prompts and managing model interactions."""

from typing import List, Tuple, Optional
from personal_rag.generation.base import BaseModelProvider
from personal_rag.core.types import SearchResult, QueryResult, Citation


class GenerationEngine:
    """Orchestrates prompt formatting, context injection, and LLM generation."""

    def __init__(self, provider: BaseModelProvider, strict_grounding: bool = True):
        self.provider = provider
        self.strict_grounding = strict_grounding

    def generate_answer(
        self,
        query: str,
        retrieved_results: List[SearchResult],
        conversation_context: Optional[str] = None
    ) -> Tuple[str, List[str]]:
        """Construct prompt and generate grounded answer and raw citations."""
        if not retrieved_results:
            return "", []

        # 1. Format document context inside security tags
        doc_blocks = []
        for r in retrieved_results:
            p = r.chunk.source_path
            doc_blocks.append(f"--- FILE: {p} (Page {r.chunk.page_number}) ---\n{r.chunk.text}")

        context_str = "\n\n".join(doc_blocks)

        # BUG-14 FIX: Inject conversation history into prompt so multi-turn memory works
        conv_section = ""
        if conversation_context and conversation_context.strip():
            conv_section = f"\n\n{conversation_context.strip()}\n"

        prompt = f"""You are a personal research assistant. You answer questions strictly from the provided documents.
Rules:
1. Provide the complete factual value or direct answer. No filler phrases.
2. If the documents do not provide the exact answer, leave the answer blank.
3. List only the file paths that are strictly necessary to prove the answer.
4. Do not invent citations. Do not cite superseded files marked as WITHDRAWN.
{conv_section}
<untrusted_document_context>
{context_str}
</untrusted_document_context>

QUESTION: {query}

OUTPUT FORMAT:
ANSWER: <value or empty>
CITATIONS: <file1, file2 or empty>"""

        response = self.provider.generate(prompt=prompt, temperature=0.0)

        answer_val = ""
        citations: List[str] = []

        for line in response.splitlines():
            line = line.strip()
            if line.upper().startswith("ANSWER:"):
                answer_val = line[7:].strip()
            elif line.upper().startswith("CITATIONS:"):
                c_str = line[10:].strip()
                if c_str:
                    citations = [c.strip().strip("'\"`") for c in c_str.split(",") if c.strip()]

        # BUG-09 FIX: Lowercase comparison to avoid case-sensitivity edge cases
        refusals = {"", "none", "n/a", "unknown", "not found", "unanswerable", "i don't know"}
        if answer_val.strip().lower() in refusals:
            return "", []

        return answer_val, citations

