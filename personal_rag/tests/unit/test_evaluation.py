"""Unit tests for the evaluation metrics and benchmarker."""

from personal_rag.evaluation.metrics import (
    compute_recall_at_k,
    compute_precision_at_k,
    compute_mrr,
    compute_ndcg,
    compute_citation_f1
)


def test_metrics_calculation():
    ground_truth = {"specs/doc_a.pdf", "specs/doc_b.pdf"}
    retrieved = ["specs/doc_a.pdf", "other/irrelevant.txt", "specs/doc_b.pdf"]

    recall = compute_recall_at_k(retrieved, ground_truth, k=3)
    assert recall == 1.0

    precision = compute_precision_at_k(retrieved, ground_truth, k=3)
    assert abs(precision - 2/3) < 1e-4

    mrr = compute_mrr(retrieved, ground_truth)
    assert mrr == 1.0  # first item is in ground truth

    ndcg = compute_ndcg(retrieved, ground_truth, k=3)
    assert ndcg > 0.8


def test_citation_f1():
    expected = ["specs/doc_a.pdf", "specs/doc_b.pdf"]
    actual = ["specs/doc_a.pdf", "specs/doc_b.pdf"]
    assert compute_citation_f1(actual, expected) == 1.0

    actual_partial = ["specs/doc_a.pdf"]
    f1 = compute_citation_f1(actual_partial, expected)
    assert 0.6 < f1 < 0.7

    assert compute_citation_f1([], []) == 1.0
    assert compute_citation_f1(["doc.pdf"], []) == 0.0

