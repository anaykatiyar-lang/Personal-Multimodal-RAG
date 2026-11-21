"""Benchmarking suite evaluating retrieval strategies on CPU fixtures."""

import time
from typing import List, Dict, Any, Set
from personal_rag.service import PersonalRAGService
from personal_rag.evaluation.metrics import (
    compute_recall_at_k,
    compute_precision_at_k,
    compute_mrr,
    compute_citation_f1
)
from personal_rag.core.utils.text_norm import normalize_token


class RAGBenchmarker:
    """Executes comparative evaluation across retrieval strategies."""

    def __init__(self, service: PersonalRAGService):
        self.service = service

    def evaluate_suite(self, benchmark_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate questions comparing Dense-only, Lexical-only, and Adaptive Hybrid strategies."""
        strategies = ["dense_only", "lexical_only", "adaptive_hybrid"]
        metrics_by_strategy: Dict[str, Dict[str, float]] = {s: {"recall@5": 0.0, "mrr": 0.0, "latency_ms": 0.0, "accuracy": 0.0, "citation_f1": 0.0} for s in strategies}

        total_queries = len(benchmark_items)
        if total_queries == 0:
            return {}

        for item in benchmark_items:
            q = item["query"]
            expected_ans = normalize_token(item.get("expected_answer", ""))
            expected_cits = item.get("expected_citations", [])
            expected_sources = set(c.strip("/") for c in expected_cits)

            # Strategy 1: Dense Only
            t0 = time.perf_counter()
            dense_res = self.service.dense_retriever.retrieve(q, top_k=5)
            dense_lat = (time.perf_counter() - t0) * 1000
            dense_sources = [r.chunk.source_path.strip("/") for r in dense_res]
            metrics_by_strategy["dense_only"]["recall@5"] += compute_recall_at_k(dense_sources, expected_sources, 5)
            metrics_by_strategy["dense_only"]["mrr"] += compute_mrr(dense_sources, expected_sources)
            metrics_by_strategy["dense_only"]["latency_ms"] += dense_lat

            # Strategy 2: Lexical Only
            t0 = time.perf_counter()
            lex_res = self.service.lexical_retriever.retrieve(q, top_k=5)
            lex_lat = (time.perf_counter() - t0) * 1000
            lex_sources = [r.chunk.source_path.strip("/") for r in lex_res]
            metrics_by_strategy["lexical_only"]["recall@5"] += compute_recall_at_k(lex_sources, expected_sources, 5)
            metrics_by_strategy["lexical_only"]["mrr"] += compute_mrr(lex_sources, expected_sources)
            metrics_by_strategy["lexical_only"]["latency_ms"] += lex_lat

            # Strategy 3: End-to-End Adaptive Hybrid
            t0 = time.perf_counter()
            hybrid_qr = self.service.answer_query(q, top_k=5)
            hybrid_lat = (time.perf_counter() - t0) * 1000
            hybrid_sources = [c.source_path.strip("/") for c in hybrid_qr.citations]

            metrics_by_strategy["adaptive_hybrid"]["recall@5"] += compute_recall_at_k(hybrid_sources, expected_sources, 5)
            metrics_by_strategy["adaptive_hybrid"]["mrr"] += compute_mrr(hybrid_sources, expected_sources)
            metrics_by_strategy["adaptive_hybrid"]["latency_ms"] += hybrid_lat
            metrics_by_strategy["adaptive_hybrid"]["citation_f1"] += compute_citation_f1(hybrid_sources, expected_cits)

            if normalize_token(hybrid_qr.answer) == expected_ans:
                metrics_by_strategy["adaptive_hybrid"]["accuracy"] += 1.0

        # Normalize averages
        for s in strategies:
            for m in metrics_by_strategy[s]:
                metrics_by_strategy[s][m] /= total_queries

        return {
            "total_queries": total_queries,
            "metrics": metrics_by_strategy,
            "summary_markdown": self._format_report(metrics_by_strategy, total_queries)
        }

    def _format_report(self, metrics: Dict[str, Dict[str, float]], n: int) -> str:
        lines = [
            "# Retrieval & RAG Benchmark Report",
            f"**Evaluated Queries:** {n} | **Execution Mode:** CPU Only (Deterministic)",
            "",
            "| Strategy | Recall@5 | MRR | Avg Latency (ms) | Citation F1 | Accuracy |",
            "| :--- | :---: | :---: | :---: | :---: | :---: |"
        ]
        for s, m in metrics.items():
            name = s.replace("_", " ").title()
            lines.append(
                f"| **{name}** | {m['recall@5']:.3f} | {m['mrr']:.3f} | {m['latency_ms']:.1f} ms | {m.get('citation_f1', 0):.3f} | {m.get('accuracy', 0):.1%} |"
            )
        return "\n".join(lines)

