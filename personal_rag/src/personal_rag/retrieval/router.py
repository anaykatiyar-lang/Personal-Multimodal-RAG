"""Adaptive Query Router classifying question complexity and configuring retrieval strategies."""

import re
from enum import Enum
from typing import Dict, Any, List
from personal_rag.core.utils.text_norm import extract_technical_entities


class QueryIntent(str, Enum):
    """Categorization of query informational intent."""
    CONVERSATIONAL = "conversational"         # Greetings, pleasantries, who-are-you, help
    EXACT_LOOKUP = "exact_lookup"             # Part numbers, error codes, ticket IDs, versions
    SEMANTIC_DISCOVERY = "semantic_discovery" # Conceptual explanations, summaries, themes
    MULTI_HOP = "multi_hop"                   # Bridging two entities (e.g. log incident to fixed version)
    TABLE_NUMERICAL = "table_numerical"       # Pricing, metrics, sheet data
    MULTIMODAL = "multimodal"                 # Diagrams, pinouts, asset labels, photos


class AdaptiveQueryRouter:
    """Classifies user queries and tunes retrieval weights and workflows."""

    def classify_query(self, query: str) -> QueryIntent:
        q_lower = query.lower()

        # 0. Conversational & greetings intent (e.g., "hi", "hii", "hello", "who are you")
        stripped = re.sub(r"[^\w\s]", "", q_lower).strip()
        conversational_exact = {
            "hi", "hii", "hiii", "hello", "helloo", "hey", "heyy", "hola",
            "greetings", "good morning", "good afternoon", "good evening",
            "howdy", "sup", "yo", "who are you", "what are you", "what can you do",
            "help", "how does this work", "thanks", "thank you", "bye", "goodbye"
        }
        if stripped in conversational_exact:
            return QueryIntent.CONVERSATIONAL
        if any(stripped.startswith(g + " ") for g in ["hi", "hii", "hello", "hey"]):
            tech_terms = ["temperature", "part", "ticket", "version", "spec", "error", "pin", "rev", "cost", "price"]
            if not any(term in q_lower for term in tech_terms):
                return QueryIntent.CONVERSATIONAL

        # 1. Multimodal intent
        multimodal_terms = {"pin", "pinout", "label", "diagram", "schematic", "photo", "drawing", "asset label"}
        if any(term in q_lower for term in multimodal_terms):
            return QueryIntent.MULTIMODAL

        # 2. Multi-hop indicators (Explicit bridging between two stages: e.g. production log AND fix/defect)
        if ("log" in q_lower and ("fixed" in q_lower or "defect" in q_lower)) or \
           ("underlying defect" in q_lower) or \
           ("caused by" in q_lower and "resolved" in q_lower):
            return QueryIntent.MULTI_HOP

        # 3. Table and numerical queries (checked BEFORE entity extraction because
        #    numbers like "10,000" can trigger the entity pattern)
        table_terms = {"price", "cost", "quantity", "unit volume", "unit price", "lead time", "sheet", "row"}
        if any(term in q_lower for term in table_terms):
            return QueryIntent.TABLE_NUMERICAL

        # 4. Exact lookup intent (Presence of explicit part numbers, ticket IDs, error codes, or version queries)
        entities = extract_technical_entities(query)
        if entities or "part number" in q_lower or "error code" in q_lower:
            return QueryIntent.EXACT_LOOKUP

        # 5. Default to semantic discovery
        return QueryIntent.SEMANTIC_DISCOVERY

    def get_retrieval_weights(self, intent: QueryIntent) -> Dict[str, float]:
        """Tuning RRF weights based on classified intent."""
        if intent == QueryIntent.CONVERSATIONAL:
            return {"dense_weight": 0.50, "lexical_weight": 0.50}
        elif intent == QueryIntent.EXACT_LOOKUP:
            return {"dense_weight": 0.15, "lexical_weight": 0.85}
        elif intent == QueryIntent.TABLE_NUMERICAL:
            return {"dense_weight": 0.30, "lexical_weight": 0.70}
        elif intent == QueryIntent.SEMANTIC_DISCOVERY:
            return {"dense_weight": 0.80, "lexical_weight": 0.20}
        elif intent == QueryIntent.MULTI_HOP:
            return {"dense_weight": 0.50, "lexical_weight": 0.50}
        elif intent == QueryIntent.MULTIMODAL:
            return {"dense_weight": 0.40, "lexical_weight": 0.60}
        return {"dense_weight": 0.65, "lexical_weight": 0.35}

    def decompose_query(self, query: str, intent: QueryIntent) -> List[str]:
        """Generate focused sub-queries for multi-hop synthesis."""
        if intent != QueryIntent.MULTI_HOP:
            return [query]

        sub_queries = [query]
        entities = extract_technical_entities(query)
        for ent in entities:
            sub_queries.append(ent)
            sub_queries.append(f"{ent} fixed in release")

        return sub_queries
