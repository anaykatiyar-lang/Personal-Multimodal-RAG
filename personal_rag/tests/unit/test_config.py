"""Unit tests for configuration validation and hardware policy enforcement."""

import pytest
from personal_rag.core.config import Settings
from personal_rag.core.exceptions import HardwarePolicyViolationError


def test_default_settings_enforce_cpu_only():
    """Verify that default settings strictly enforce CPU_ONLY and GPU_EXECUTION_APPROVED=False."""
    settings = Settings()
    assert settings.cpu_only is True
    assert settings.gpu_execution_approved is False
    assert settings.device.lower() == "cpu"


def test_hardware_manager_blocks_gpu_when_unapproved():
    """Verify that HardwareManager raises HardwarePolicyViolationError when GPU is requested without approval."""
    settings = Settings(CPU_ONLY=True, GPU_EXECUTION_APPROVED=False)
    hw = settings.get_hardware_manager()

    with pytest.raises(HardwarePolicyViolationError):
        hw.resolve_device("cuda")

    with pytest.raises(HardwarePolicyViolationError):
        hw.resolve_device("rocm")


def test_hardware_proposal_generation():
    """Verify that the 7-point impact report is formatted correctly."""
    hw = Settings().get_hardware_manager()
    report = hw.generate_gpu_proposal(
        experiment_name="Benchmark Qwen2.5-VL-7B",
        model_name="Qwen2.5-VL-7B-Instruct",
        dataset_name="MC3 Corpus",
        estimated_vram_gb=15.5,
        expected_runtime_minutes=10.0,
        target_metric="Recall@5 > 0.95",
        justification="Multimodal table and pinout image QA evaluation",
        low_cost_alternative="Pre-extracted OCR text on CPU"
    )
    assert "1_exact_experiment" in report
    assert "15.5 GiB" in report["3_estimated_vram_gb"]
    assert "7_low_cost_alternative" in report

