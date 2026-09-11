"""V70's throughput stop must preserve the full scheduler horizon. No model training."""
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from analysis import v12_distill as v12


class Length:
    def __init__(self, n):
        self.n = n

    def numel(self):
        return self.n


class Loss:
    def __truediv__(self, _):
        return self

    def backward(self):
        pass

    def detach(self):
        return 1.0


class Model:
    def parameters(self):
        return [SimpleNamespace(requires_grad=True)]

    def to(self, _):
        return self

    def train(self):
        return self


class Optimizer:
    def __init__(self, parameters, lr):
        self.lr = lr

    def zero_grad(self, **kwargs):
        pass

    def step(self):
        pass


class Scheduler:
    def __init__(self, optimizer, num_warmup_steps, num_training_steps):
        self.step_number = 0
        self.total = num_training_steps
        self.warmup = num_warmup_steps

    def step(self):
        self.step_number += 1

    def get_last_lr(self):
        # A deterministic horizon-sensitive stand-in; no tensors or weights.
        return [self.step_number / self.total + self.warmup / 10000]


def test_throughput_is_an_exact_prefix_without_optimizer_or_model_updates(monkeypatch):
    import transformers

    monkeypatch.setattr(v12.torch.optim, "AdamW", Optimizer)
    monkeypatch.setattr(transformers, "get_cosine_schedule_with_warmup", Scheduler)
    monkeypatch.setattr(v12, "masked_causal_loss", lambda *args: (Loss(), 3))
    monkeypatch.setattr(v12, "preserve_training_state", lambda model: nullcontext())
    monkeypatch.setattr(v12, "EFFECTIVE_BATCH_SIZE", 2)
    examples = [{"example_id": str(i), "input_ids": Length(10 + i)} for i in range(4)]
    callbacks = []
    full = v12._train(Model(), examples, "cpu", 99, 1e-4, schedule_updates=100,
                      trajectory_tokens=[50], trajectory_callback=lambda p: None)
    prefix = v12._train(Model(), examples, "cpu", 99, 1e-4, schedule_updates=100,
                        trajectory_tokens=[50], trajectory_callback=lambda p: callbacks.append(dict(p)),
                        stop_after_trajectory=True)
    assert prefix["updates"] == callbacks[0]["updates"] < full["updates"]
    assert prefix["loss_curve"] == full["loss_curve"][:prefix["updates"]]
    assert prefix["total_updates_planned"] == full["total_updates_planned"] == 100
    assert prefix["warmup_steps"] == full["warmup_steps"] == 3
    assert prefix["stopped_after_trajectory"] and not full["stopped_after_trajectory"]
    assert prefix["trajectory_tokens_unreached"] == []


def test_early_stop_needs_a_fixed_horizon_before_any_training():
    with pytest.raises(ValueError, match="fixed schedule"):
        v12._train(None, [], "cpu", 99, 1e-4, trajectory_tokens=[10],
                    trajectory_callback=lambda p: None, stop_after_trajectory=True)


def test_cli_dry_run_preserves_full_schedule_and_does_not_load(monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        pytest.fail("No model, data, CUDA or seed operations allowed")

    for name in ("seed_everything", "load_sft_records", "build_probes", "load_text_causal_lm"):
        monkeypatch.setattr(v12, name, forbidden)
    monkeypatch.setattr(v12, "OUT_BASE", tmp_path / "unused")
    p = v12.run_distillation("gemma3-1b", "gpt-5.6-luna", "full", ["math", "qa", "code"],
        200, 99, "cuda:0", 1e-4, output_suffix="_p2v3conf_throughput", seed=0, data_seed=31,
        schedule_tokens=1000000, save_trajectory=True, trajectory_tokens=[170000],
        stop_after_trajectory=True, dry_run=True)
    assert p["schedule_tokens"] == 1000000 and p["stop_after_trajectory"]
    assert p["trajectory_tokens"] == [170000]
    assert not (tmp_path / "unused").exists()
