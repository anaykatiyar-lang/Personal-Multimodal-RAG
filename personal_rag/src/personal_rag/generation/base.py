"""Abstract Model Provider interface."""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List


class BaseModelProvider(ABC):
    """Abstract interface for LLM text generation backends."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
        **kwargs
    ) -> str:
        """Generate response text given prompt and parameters."""
        pass

