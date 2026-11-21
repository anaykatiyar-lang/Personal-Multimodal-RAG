"""Model provider abstractions, local inference, and grounded generation engines."""

from personal_rag.generation.base import BaseModelProvider
from personal_rag.generation.mock_provider import MockProvider
from personal_rag.generation.extractive_synthesizer import ExtractiveSynthesizerProvider
from personal_rag.generation.openai_provider import OpenAICompatibleProvider
from personal_rag.generation.custom_llm_provider import CustomScratchLLMProvider
from personal_rag.generation.engine import GenerationEngine

__all__ = [
    "BaseModelProvider",
    "MockProvider",
    "ExtractiveSynthesizerProvider",
    "OpenAICompatibleProvider",
    "CustomScratchLLMProvider",
    "GenerationEngine",
]
