"""OpenAI-compatible LLM Provider for local Ollama, vLLM, LM Studio, or OpenAI API."""

import json
import urllib.request
import urllib.error
from typing import Optional, Dict, Any
from personal_rag.generation.base import BaseModelProvider
from personal_rag.core.exceptions import ModelInvocationError


class OpenAICompatibleProvider(BaseModelProvider):
    """Integrates any OpenAI-compatible completions endpoint without external dependencies."""

    def __init__(
        self,
        endpoint: str = "http://127.0.0.1:11434/v1",
        model_name: str = "qwen2.5-vl:7b",
        api_key: Optional[str] = None,
        timeout_seconds: int = 45
    ):
        self.endpoint = endpoint.rstrip("/")
        self.model_name = model_name
        self.api_key = api_key or "none"
        self.timeout = timeout_seconds

    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
        **kwargs
    ) -> str:
        url = f"{self.endpoint}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                res_json = json.loads(resp.read().decode("utf-8"))
                choice = res_json.get("choices", [{}])[0]
                return choice.get("message", {}).get("content", "").strip()
        except urllib.error.URLError as e:
            raise ModelInvocationError(
                f"Failed to connect to LLM at {url}: {e}. Ensure the local server is running."
            )
        except Exception as e:
            raise ModelInvocationError(f"Error invoking LLM endpoint {url}: {e}")

