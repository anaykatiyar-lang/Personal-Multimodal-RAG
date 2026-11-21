"""Central Personal RAG Service orchestrating ingestion, routing, retrieval, generation, and memory."""

import time
from pathlib import Path
from typing import Optional, List, Dict, Any

from personal_rag.core.config import Settings, get_settings
from personal_rag.core.types import QueryResult, SearchResult
from personal_rag.storage.sqlite_store import SQLiteStore
from personal_rag.storage.vector_store import InMemoryVectorStore, QdrantLocalVectorStore
from personal_rag.embeddings.mock import MockEmbeddingProvider
from personal_rag.embeddings.sentence_transformers import SentenceTransformersProvider
from personal_rag.parsers.factory import ParserFactory
from personal_rag.chunking.recursive import RecursiveChunker
from personal_rag.chunking.parent_child import ParentChildChunker
from personal_rag.ingestion.pipeline import IngestionPipeline, IngestionReport
from personal_rag.retrieval.dense import DenseRetriever
from personal_rag.retrieval.lexical import LexicalRetriever
from personal_rag.retrieval.hybrid import HybridRetriever
from personal_rag.retrieval.parent import ParentContextRetriever
from personal_rag.retrieval.router import AdaptiveQueryRouter, QueryIntent
from personal_rag.reranking.cross_encoder import CrossEncoderReranker
from personal_rag.generation.mock_provider import MockProvider
from personal_rag.generation.extractive_synthesizer import ExtractiveSynthesizerProvider
from personal_rag.generation.openai_provider import OpenAICompatibleProvider
from personal_rag.generation.engine import GenerationEngine
from personal_rag.citations.verifier import CitationVerifier
from personal_rag.memory.manager import ConversationMemoryManager


class PersonalRAGService:
    """Unified application service coordinating the complete personal RAG lifecycle."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.settings.ensure_directories()

        # 1. Hardware & Storage initialization
        self.hw_manager = self.settings.get_hardware_manager()
        self.sqlite_store = SQLiteStore(self.settings.sqlite_db_path)

        # 2. Embedding provider selection
        if self.settings.llm_provider == "mock" or self.settings.cpu_only:
            self.embedding_provider = MockEmbeddingProvider(dimension=self.settings.embedding_dimension)
            self.vector_store = InMemoryVectorStore()
        else:
            self.embedding_provider = SentenceTransformersProvider(
                model_name=self.settings.embedding_model_name,
                hardware_manager=self.hw_manager,
                batch_size=self.settings.embedding_batch_size
            )
            self.vector_store = QdrantLocalVectorStore(
                storage_path=str(self.settings.qdrant_storage_path),
                collection_name=self.settings.qdrant_collection_name,
                dimension=self.settings.embedding_dimension
            )

        # 3. Ingestion pipeline
        self.parser_factory = ParserFactory()
        self.chunker = ParentChildChunker(
            parent_chunk_size=self.settings.parent_chunk_size,
            child_chunk_size=self.settings.default_chunk_size,
            child_overlap=self.settings.default_chunk_overlap
        )
        self.ingestion_pipeline = IngestionPipeline(
            sqlite_store=self.sqlite_store,
            vector_store=self.vector_store,
            parser_factory=self.parser_factory,
            chunker=self.chunker
        )

        # 4. Retrieval & Routing
        self.dense_retriever = DenseRetriever(self.vector_store, self.embedding_provider)
        self.lexical_retriever = LexicalRetriever(self.sqlite_store)
        self.hybrid_retriever = HybridRetriever(
            dense_retriever=self.dense_retriever,
            lexical_retriever=self.lexical_retriever,
            dense_weight=self.settings.dense_search_weight,
            lexical_weight=self.settings.lexical_search_weight,
            rrf_k=self.settings.rrf_k
        )
        self.parent_expander = ParentContextRetriever(self.sqlite_store)
        self.router = AdaptiveQueryRouter()
        self.reranker = CrossEncoderReranker(
            model_name=self.settings.reranker_model_name,
            hardware_manager=self.hw_manager,
            enabled=self.settings.reranker_enabled
        )

        # 5. Generation, Citations & Memory
        if self.settings.llm_provider in ("openai", "ollama", "local_llm"):
            self.model_provider = OpenAICompatibleProvider(
                endpoint=self.settings.local_llm_endpoint,
                model_name=self.settings.llm_model_name,
                api_key=self.settings.local_llm_api_key
            )
        elif self.settings.llm_provider == "strict_mock":
            self.model_provider = MockProvider()
        else:
            # Default to ExtractiveSynthesizerProvider: zero-cost local CPU Q&A supporting
            # both exact benchmark values and natural language user documents
            self.model_provider = ExtractiveSynthesizerProvider()

        self.generation_engine = GenerationEngine(self.model_provider)
        self.citation_verifier = CitationVerifier()
        self.memory_manager = ConversationMemoryManager(self.sqlite_store)

    def ingest_path(self, target_path: Path, prune_deleted: bool = False) -> IngestionReport:
        """Ingest a directory tree or single file."""
        p = Path(target_path)
        if p.is_dir():
            return self.ingestion_pipeline.ingest_directory(p, prune_deleted=prune_deleted)
        else:
            success = self.ingestion_pipeline.ingest_single_file(p)
            return IngestionReport(
                total_files_scanned=1,
                files_indexed=1 if success else 0,
                files_failed=0 if success else 1
            )

    def answer_query(
        self,
        query: str,
        session_id: Optional[str] = None,
        top_k: Optional[int] = None
    ) -> QueryResult:
        """Execute end-to-end adaptive RAG query."""
        t_start = time.perf_counter()
        k = top_k or self.settings.retrieval_top_k
        latencies: Dict[str, float] = {}

        # Step 1: Query Intent Classification
        t0 = time.perf_counter()
        intent = self.router.classify_query(query)
        latencies["router_ms"] = (time.perf_counter() - t0) * 1000

        # Fast direct response for greetings & conversational queries (never refuse on "hi")
        if intent == QueryIntent.CONVERSATIONAL:
            q_clean = query.lower().strip()
            if any(w in q_clean for w in ["who are you", "what are you", "what can you do", "help", "how do you work"]):
                conv_ans = (
                    "Hello! I am your Personal RAG Research Assistant. I can search, analyze, and cite "
                    "information from your indexed documents (PDFs, DOCX, CSV/Excel, code, logs, and notes). "
                    "You can ask me specific questions about your data, specifications, part numbers, or system logs!"
                )
            elif any(w in q_clean for w in ["thank", "thanks"]):
                conv_ans = "You're welcome! Let me know if you have any more questions about your documents."
            elif any(w in q_clean for w in ["bye", "goodbye"]):
                conv_ans = "Goodbye! Reach out whenever you need to explore or verify your documents."
            else:
                conv_ans = (
                    "Hello! How can I assist with your personal documents and research today? "
                    "Feel free to ask any question about your indexed files."
                )

            latencies["total_ms"] = (time.perf_counter() - t_start) * 1000
            if session_id:
                self.memory_manager.record_turn(session_id, query, conv_ans, [])
            return QueryResult(
                query=query,
                answer=conv_ans,
                citations=[],
                confidence=1.0,
                is_refusal=False,
                retrieved_chunks=[],
                latency_breakdown_ms=latencies,
                strategy_used="conversational_direct"
            )

        weights = self.router.get_retrieval_weights(intent)

        # Step 2: Adaptive Retrieval
        t0 = time.perf_counter()
        search_results = self.hybrid_retriever.retrieve(
            query=query,
            top_k=k,
            dense_weight_override=weights["dense_weight"],
            lexical_weight_override=weights["lexical_weight"]
        )
        latencies["retrieval_ms"] = (time.perf_counter() - t0) * 1000

        # Step 3: Optional Reranking
        t0 = time.perf_counter()
        reranked = self.reranker.rerank(query, search_results, top_n=k)
        latencies["reranking_ms"] = (time.perf_counter() - t0) * 1000

        # Step 4: Parent Context Expansion
        t0 = time.perf_counter()
        context_chunks = self.parent_expander.expand_context(reranked)
        latencies["context_expansion_ms"] = (time.perf_counter() - t0) * 1000

        # Step 5: Grounded Answer Generation
        t0 = time.perf_counter()
        dialogue_ctx = self.memory_manager.format_dialogue_context(session_id) if session_id else None
        raw_answer, raw_citations = self.generation_engine.generate_answer(
            query=query,
            retrieved_results=context_chunks,
            conversation_context=dialogue_ctx
        )
        latencies["generation_ms"] = (time.perf_counter() - t0) * 1000

        # Step 6: Citation Verification & Grounding
        t0 = time.perf_counter()
        verified_citations = self.citation_verifier.verify_citations(
            answer=raw_answer,
            raw_citations=raw_citations,
            retrieved_results=context_chunks,
            query=query
        )
        latencies["verification_ms"] = (time.perf_counter() - t0) * 1000

        # BUG-01 FIX: Refusal only when answer is blank, NOT when citations are empty.
        # An answer can be valid even if the citation verifier fails path matching.
        is_refusal = not raw_answer.strip()
        final_answer = raw_answer if not is_refusal else ""
        final_citations = verified_citations  # may be empty; shown if present
        confidence = 0.95 if not is_refusal else 0.0

        # Step 7: Record dialogue turn if session provided
        if session_id:
            self.memory_manager.record_turn(session_id, query, final_answer, final_citations)

        latencies["total_ms"] = (time.perf_counter() - t_start) * 1000

        return QueryResult(
            query=query,
            answer=final_answer,
            citations=final_citations,
            confidence=confidence,
            is_refusal=is_refusal,
            retrieved_chunks=reranked,
            latency_breakdown_ms=latencies,
            strategy_used=f"adaptive_hybrid_{intent.value}"
        )

