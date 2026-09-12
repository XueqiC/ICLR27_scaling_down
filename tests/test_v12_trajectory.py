import json
import random

import numpy as np
import pytest
import torch

from analysis import v12_distill as distill


def examples():
    # Duplicate rows count as repeated exposure, not additional unique data.
    return [{"example_id": key, "input_ids": torch.arange(n),
             "attention_mask": torch.ones(n, dtype=torch.long),
             "labels": torch.arange(n), "n_completion_tokens": n - 1}
            for key, n in (("a", 3), ("b", 5), ("a", 3))]


def test_unique_vs_seen_accounts_for_duplicates_repetition_and_padding():
    pool = examples()
    pool[1]["input_ids"] = torch.arange(7)
    pool[1]["attention_mask"] = torch.tensor([1, 1, 1, 1, 1, 0, 0])
    accounting = distill.TokenAccounting(pool)
    assert accounting.snapshot()["unique_data_pool_tokens"] == 8
    assert accounting.snapshot()["unique_data_tokens"] == 0
    for index in (0, 2, 1, 0, 1):
        accounting.observe(index)
    result = accounting.snapshot()
    assert result["seen_tokens"] == result["processed_tokens"] == 19
    assert result["unique_data_tokens"] == result["unique_data_pool_tokens"] == 8
    assert result["unique_examples_seen"] == result["unique_data_pool_examples"] == 2


def test_milestones_cross_at_updates_and_do_not_change_training(monkeypatch):
    monkeypatch.setattr(distill, "EFFECTIVE_BATCH_SIZE", 2)

    def loss(model, example, device):
        assert device == "cpu"
        assert model.training
        # Detect evaluation disturbing any of these RNGs or module modes.
        noise = random.random() + np.random.random() + float(torch.rand(()))
        return (model.weight * noise).square().sum(), example["n_completion_tokens"]

    monkeypatch.setattr(distill, "masked_causal_loss", loss)
    logs, weights = [], []
    snapshots = []
    for use_trajectory in (False, True):
        random.seed(8)
        np.random.seed(8)
        torch.random.default_generator.manual_seed(8)
        model = torch.nn.Linear(1, 1)

        def evaluate(progress):
            snapshots.append(dict(progress))
            model.eval()
            random.random()
            np.random.random()
            torch.rand(50)

        log = distill._train(model, examples(), "cpu", 3, 1e-3, seed=8,
                             trajectory_tokens=[1, 2, 12, 22, 33, 100] if use_trajectory else (),
                             trajectory_callback=evaluate if use_trajectory else None)
        logs.append(log)
        weights.append(model.weight.detach().clone())
    assert logs[0]["loss_curve"] == logs[1]["loss_curve"]
    torch.testing.assert_close(*weights, rtol=0, atol=0)
    assert snapshots[0]["requested_token_milestones"] == [1, 2]
    assert snapshots[0]["updates"] == 1
    assert snapshots[-1]["processed_tokens"] == 33
    assert snapshots[-1]["unique_data_tokens"] == 8
    assert logs[1]["trajectory_tokens_unreached"] == [100]
    assert logs[1]["optimizer_steps"] == logs[0]["optimizer_steps"] == 6
    assert logs[1]["processed_tokens"] == logs[1]["tokens_seen"] == 33
    for snapshot in snapshots:
        update = snapshot["updates"]
        previous = logs[1]["loss_curve"][update - 2]["processed_tokens"] if update > 1 else 0
        assert all(previous < milestone <= snapshot["processed_tokens"]
                   for milestone in snapshot["requested_token_milestones"])


@pytest.mark.parametrize("milestones", [[0], [-1], [2, 1], [1, 1], [2.5]])
def test_invalid_milestones_fail(milestones):
    with pytest.raises(ValueError):
        distill.validate_trajectory_tokens(milestones)


def test_dry_run_skips_all_loads_seed_cuda_and_writes(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("dry run must not load or seed")

    for name in ("seed_everything", "load_sft_records", "build_probes", "load_text_causal_lm"):
        monkeypatch.setattr(distill, name, forbidden)
    monkeypatch.setattr(distill, "OUT_BASE", tmp_path / "outputs")
    payload = distill.run_distillation("gemma3-270m", "gpt-5.6-luna", "full",
                                       ["math"], 10, 2, "cuda:0", None, seed=7,
                                       save_trajectory=True, trajectory_tokens=[3, 12], dry_run=True)
    assert payload["seed"] == 7
    assert payload["trajectory_tokens"] == [3, 12]
    assert not (tmp_path / "outputs").exists()


def test_eval_files_record_token_counts_seed_and_shared_trajectory(tmp_path, monkeypatch):
    from test_v12 import TinyTokenizer

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.adapter = torch.nn.Module()
            self.adapter.lora_A = torch.nn.Parameter(torch.ones(1))

        @property
        def weight(self):
            return self.adapter.lora_A

        def save_pretrained(self, path):
            torch.save(self.state_dict(), path / "weights.pt")

    monkeypatch.setattr(distill, "OUT_BASE", tmp_path)
    monkeypatch.setattr(distill, "seed_everything", lambda seed: None)
    monkeypatch.setattr(distill, "load_sft_records", lambda **kw: (
        [{"prompt": "ab", "completion": "XY", "domain": "math"}] * 2, {}))
    monkeypatch.setattr(distill, "build_probes", lambda *a, **kw: {
        c: [{"prompt": "a", "completion": "B"}] * 4 for c in distill.CAPABILITIES})
    monkeypatch.setattr(distill, "load_text_causal_lm", lambda *a: (Model(), TinyTokenizer()))
    monkeypatch.setattr(distill, "resolve_training_mode", lambda mode: "lora")
    monkeypatch.setattr(distill, "configure_training", lambda m, **kw: (m, "lora", ["adapter.lora_A"]))
    monkeypatch.setattr(distill, "masked_causal_loss", lambda model, example, device: (model.weight.square().sum(), 2))

    def measure(model, *args):
        model.eval()
        return ({c: float(model.weight.detach()) for c in distill.CAPABILITIES},
                {c: 4 for c in distill.CAPABILITIES})

    monkeypatch.setattr(distill, "measure_capability_losses", measure)
    result = distill.run_distillation("gemma3-270m", "gpt-5.6-luna", "full", ["math"],
                                      2, 2, "cpu", None, seed=7, save_trajectory=True,
                                      trajectory_tokens=[1, 3, 11, 100])
    root = tmp_path / "gemma3-270m" / result["run_name"]
    snapshots = sorted((root / "trajectory").glob("*/eval.json"))
    assert len(snapshots) == 3  # baseline + 2 update states, not 4 fake replicates
    for path in [*snapshots, root / "eval.json"]:
        payload = json.loads(path.read_text())
        assert payload["seed"] == payload["training_seed"] == payload["data_sampling_seed"] == 7
        assert payload["probe_seed"] == 0
        assert payload["trajectory_run_id"] == result["trajectory_run_id"]
        assert payload["unique_data_pool_tokens"] == 5  # distinct example counted once
        assert payload["seen_tokens"] == payload["processed_tokens"]
        assert payload["capability_losses"].keys() == set(distill.CAPABILITIES)
    assert result["processed_tokens"] == 20
    assert result["unique_data_tokens"] == 5
    assert result["trajectory_tokens_unreached"] == [100]
    first = json.loads(snapshots[1].read_text())
    assert first["requested_token_milestones"] == [1, 3]
    assert first["is_independent_seed"] is False
    assert (snapshots[1].parent / "adapter" / "weights.pt").is_file()


def test_full_delta_snapshot_does_not_mutate_cpu_parameters(tmp_path):
    model = torch.nn.Linear(1, 1, bias=False)
    initial = tmp_path / "initial"
    manifest = distill._snapshot_full_parameters(model, ["weight"], initial)
    with torch.no_grad():
        model.weight.add_(3)
    expected = model.weight.detach().clone()
    distill._save_full_delta(model, initial, manifest, tmp_path / "adapter", "base")
    torch.testing.assert_close(model.weight, expected, rtol=0, atol=0)
    torch.testing.assert_close(torch.load(tmp_path / "adapter" / "delta-00000.pt", weights_only=True), torch.full_like(expected, 3))


def test_trajectory_requires_fresh_directory_before_seeding_or_training(tmp_path, monkeypatch):
    monkeypatch.setattr(distill, "OUT_BASE", tmp_path)
    monkeypatch.setattr(distill, "seed_everything", lambda *a: pytest.fail("must refuse before seeding"))
    monkeypatch.setattr(distill, "resolve_training_mode", lambda mode: "lora")
    run = tmp_path / "gemma3-270m" / "gpt-5.6-luna_full_1_lora"
    run.mkdir(parents=True)
    marker = run / "eval.json"
    marker.write_text('{"existing": true}')
    with pytest.raises(FileExistsError, match="fresh run directory"):
        distill.run_distillation("gemma3-270m", "gpt-5.6-luna", "full", ["math"],
                                 1, 1, "cpu", None, save_trajectory=True)
    assert marker.read_text() == '{"existing": true}'
