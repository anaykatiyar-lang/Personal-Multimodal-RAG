"""Information Retrieval and RAG evaluation metrics."""

import math
from typing import List, Set


def compute_recall_at_k(retrieved_items: List[str], ground_truth: Set[str], k: int = 5) -> float:
    """Proportion of relevant documents retrieved in the top K."""
    if not ground_truth:
        return 1.0 if not retrieved_items else 0.0
    top_k_set = set(retrieved_items[:k])
    hits = len(top_k_set.intersection(ground_truth))
    return float(hits / len(ground_truth))


def compute_precision_at_k(retrieved_items: List[str], ground_truth: Set[str], k: int = 5) -> float:
    """Proportion of top K retrieved items that are relevant."""
    if k <= 0:
        return 0.0
    top_k_items = retrieved_items[:k]
    if not top_k_items:
        return 0.0
    hits = len(set(top_k_items).intersection(ground_truth))
    return float(hits / len(top_k_items))


def compute_mrr(retrieved_items: List[str], ground_truth: Set[str]) -> float:
    """Mean Reciprocal Rank: reciprocal of the rank of the first relevant document."""
    if not ground_truth:
        return 1.0 if not retrieved_items else 0.0
    for idx, item in enumerate(retrieved_items, start=1):
        if item in ground_truth:
            return 1.0 / idx
    return 0.0


def compute_ndcg(retrieved_items: List[str], ground_truth: Set[str], k: int = 5) -> float:
    """Normalized Discounted Cumulative Gain at K."""
    if not ground_truth:
        return 1.0 if not retrieved_items else 0.0

    dcg = 0.0
    for idx, item in enumerate(retrieved_items[:k], start=1):
        rel = 1.0 if item in ground_truth else 0.0
        dcg += rel / math.log2(idx + 1)

    # Ideal DCG
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(ground_truth), k) + 1))
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def compute_citation_f1(actual_citations: List[str], expected_citations: List[str]) -> float:
    """Compute exact citation set F1 score."""
    act_set = set(c.strip("/") for c in actual_citations)
    exp_set = set(c.strip("/") for c in expected_citations)

    if not act_set and not exp_set:
        return 1.0
    if not act_set or not exp_set:
        return 0.0

    tp = len(act_set.intersection(exp_set))
    precision = tp / len(act_set)
    recall = tp / len(exp_set)
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

