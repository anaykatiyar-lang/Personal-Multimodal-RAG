"""Custom LLM Provider for integrating custom-trained or scratch LLMs (e.g., 500M parameter models)."""

import os
from pathlib import Path
from typing import Optional, Any
from personal_rag.generation.base import BaseModelProvider
from personal_rag.core.exceptions import ModelInvocationError


class CustomScratchLLMProvider(BaseModelProvider):
    """Integrates custom from-scratch trained models (e.g. 500M parameter PyTorch/HF models).
    
    Compatible with:
    - Custom PyTorch nn.Module checkpoints (.pt, .pth, .bin)
    - HuggingFace AutoModelForCausalLM local directories
    - ONNX / GGUF (via llama-cpp or onnxruntime)
    
    Zero GPU cost by default (CPU inference).
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cpu",
        max_context_length: int = 2048,
        custom_generate_fn: Optional[Any] = None
    ):
        self.model_path = model_path
        self.device = device
        self.max_context_length = max_context_length
        self.custom_generate_fn = custom_generate_fn
        self.model = None
        self.tokenizer = None
        self._is_loaded = False

    def load_model(self) -> None:
        """Lazy loader for custom 500M weights."""
        if self._is_loaded:
            return

        # If a custom generation function or pipeline is passed directly:
        if self.custom_generate_fn is not None:
            self._is_loaded = True
            return

        if not self.model_path or not os.path.exists(self.model_path):
            raise ModelInvocationError(
                f"Custom LLM checkpoint path not found: {self.model_path}. "
                "Provide a valid path to your weights directory or .pt file."
            )

        try:
            # 1. Attempt HuggingFace local directory load
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype="auto",
                low_cpu_mem_usage=True
            ).to(self.device)
            self.model.eval()
            self._is_loaded = True
        except Exception as e_hf:
            try:
                # 2. Attempt raw PyTorch torch.load
                import torch
                checkpoint = torch.load(self.model_path, map_location=self.device)
                self.model = checkpoint
                self._is_loaded = True
            except Exception as e_torch:
                raise ModelInvocationError(
                    f"Could not load custom model from {self.model_path}. "
                    f"HF error: {e_hf}. Torch error: {e_torch}"
                )

    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
        **kwargs
    ) -> str:
        """Generate response using the custom 500M parameter model."""
        if not self._is_loaded:
            self.load_model()

        # If custom callable is supplied:
        if self.custom_generate_fn:
            return self.custom_generate_fn(prompt, max_tokens=max_tokens, temperature=temperature)

        if self.tokenizer and self.model:
            import torch
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.max_context_length).to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=(temperature > 0.0),
                    temperature=temperature if temperature > 0.0 else 1.0,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            generated_text = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            return generated_text.strip()

        raise ModelInvocationError("Custom model is loaded but no inference routine was able to execute.")
