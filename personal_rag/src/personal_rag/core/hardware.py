"""Hardware inspection, device resolution, and strict credit protection policy."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Dict, Any
from personal_rag.core.exceptions import HardwarePolicyViolationError


@dataclass
class DeviceInfo:
    """Snapshot of system hardware state."""
    device_type: str  # "cpu", "cuda", "rocm", "mps"
    device_name: str
    is_gpu_available: bool
    is_rocm: bool
    total_memory_bytes: int = 0
    available_memory_bytes: int = 0
    cpu_count: int = os.cpu_count() or 1


class HardwareManager:
    """Enforces zero-cost, CPU-default execution and manages device resolution."""

    def __init__(
        self,
        cpu_only: bool = True,
        gpu_execution_approved: bool = False,
        preferred_device: str = "cpu"
    ):
        self.cpu_only = cpu_only
        self.gpu_execution_approved = gpu_execution_approved
        self.preferred_device = preferred_device.lower()

    def get_device_info(self) -> DeviceInfo:
        """Inspect hardware capabilities without allocating GPU memory."""
        # Check environment overrides
        if self.cpu_only or not self.gpu_execution_approved:
            return DeviceInfo(
                device_type="cpu",
                device_name="CPU Execution Mode (GPU Protected)",
                is_gpu_available=False,
                is_rocm=False,
                cpu_count=os.cpu_count() or 1
            )

        # Lazy check for torch without triggering model loading
        try:
            import torch
            if torch.cuda.is_available():
                name = torch.cuda.get_device_name(0)
                is_rocm = "rocm" in torch.__version__.lower() or "gfx" in name.lower()
                props = torch.cuda.get_device_properties(0)
                return DeviceInfo(
                    device_type="rocm" if is_rocm else "cuda",
                    device_name=name,
                    is_gpu_available=True,
                    is_rocm=is_rocm,
                    total_memory_bytes=props.total_memory,
                    available_memory_bytes=props.total_memory,
                    cpu_count=os.cpu_count() or 1
                )
        except ImportError:
            pass

        return DeviceInfo(
            device_type="cpu",
            device_name="Host CPU",
            is_gpu_available=False,
            is_rocm=False,
            cpu_count=os.cpu_count() or 1
        )

    def resolve_device(self, requested_device: Optional[str] = None) -> str:
        """Resolve the active compute device string according to strict security policy.

        Returns 'cpu' unless GPU_EXECUTION_APPROVED is explicitly True and CPU_ONLY is False.
        """
        target = (requested_device or self.preferred_device).lower()

        if target in {"cuda", "rocm", "gpu"}:
            if self.cpu_only:
                raise HardwarePolicyViolationError(
                    "GPU execution requested, but CPU_ONLY policy is enforced. "
                    "Set CPU_ONLY=false and GPU_EXECUTION_APPROVED=true to proceed."
                )
            if not self.gpu_execution_approved:
                raise HardwarePolicyViolationError(
                    "GPU execution requested without explicit user authorization. "
                    "GPU credits are protected."
                )

        # Default safe fallback
        if self.cpu_only or not self.gpu_execution_approved:
            return "cpu"

        return target

    @staticmethod
    def generate_gpu_proposal(
        experiment_name: str,
        model_name: str,
        dataset_name: str,
        estimated_vram_gb: float,
        expected_runtime_minutes: float,
        target_metric: str,
        justification: str,
        low_cost_alternative: str
    ) -> Dict[str, Any]:
        """Construct the mandatory 7-point impact report required before any GPU execution."""
        estimated_cost_usd = (expected_runtime_minutes / 60.0) * 1.50  # Estimated $1.50/hr on MI300X cloud
        return {
            "1_exact_experiment": experiment_name,
            "2_model_and_dataset": f"Model: {model_name} | Dataset: {dataset_name}",
            "3_estimated_vram_gb": f"{estimated_vram_gb:.1f} GiB",
            "4_expected_runtime": f"{expected_runtime_minutes:.1f} minutes (Estimated cost: ~${estimated_cost_usd:.2f})",
            "5_measurable_improvement": target_metric,
            "6_gpu_necessity_justification": justification,
            "7_low_cost_alternative": low_cost_alternative,
        }

