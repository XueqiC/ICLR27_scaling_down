"""Explicit CPU harness for frozen training regressions.

Load with PYTHONPATH=tests:. python3 -m pytest -p v87_cpu_guard ... .
Frozen V12's state helper and PyTorch's optimizer health check otherwise query
CUDA even with CPU parameters. Substitute CPU-only equivalents in this harness.
"""
import pytest
import torch

from analysis import eval_records, v12_distill


@pytest.fixture(autouse=True)
def cpu_only_frozen_regressions(monkeypatch):
    monkeypatch.setattr(v12_distill, "preserve_training_state", eval_records.preserve_cpu_training_state)

    monkeypatch.setattr(torch.optim.Optimizer, "_cuda_graph_capture_health_check", eval_records.cpu_optimizer_health_check)

    def forbidden(*args, **kwargs):
        pytest.fail("CPU regression tried an accelerator query/action")

    for name in ("is_available", "is_initialized", "manual_seed_all", "get_rng_state_all",
                 "set_rng_state_all", "synchronize", "empty_cache", "is_current_stream_capturing"):
        monkeypatch.setattr(torch.cuda, name, forbidden)
