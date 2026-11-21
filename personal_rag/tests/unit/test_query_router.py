"""Unit tests for the Adaptive Query Router."""

from personal_rag.retrieval.router import AdaptiveQueryRouter, QueryIntent


def test_query_router_classification():
    router = AdaptiveQueryRouter()

    # Multimodal queries
    assert router.classify_query("Which backplane pin carries THERM_ALERT#?") == QueryIntent.MULTIMODAL
    assert router.classify_query("What board revision is printed on the asset label?") == QueryIntent.MULTIMODAL

    # Exact lookup queries
    assert router.classify_query("Which firmware version fixed ticket ORR-1847?") == QueryIntent.EXACT_LOOKUP
    assert router.classify_query("What error code is logged when E7731 occurs?") == QueryIntent.EXACT_LOOKUP

    # Table / numerical queries
    assert router.classify_query("What is the unit price of the TQ-40 at 10,000 unit volume?") == QueryIntent.TABLE_NUMERICAL

    # Multi-hop queries
    assert router.classify_query("The production log shows a thermal incident. Which firmware release fixed the underlying defect?") == QueryIntent.MULTI_HOP

    # Semantic discovery
    assert router.classify_query("Explain how the ingest service handles backlogged queues") == QueryIntent.SEMANTIC_DISCOVERY


def test_query_router_weight_adaptation():
    router = AdaptiveQueryRouter()

    exact_weights = router.get_retrieval_weights(QueryIntent.EXACT_LOOKUP)
    assert exact_weights["lexical_weight"] > exact_weights["dense_weight"]

    semantic_weights = router.get_retrieval_weights(QueryIntent.SEMANTIC_DISCOVERY)
    assert semantic_weights["dense_weight"] > semantic_weights["lexical_weight"]

