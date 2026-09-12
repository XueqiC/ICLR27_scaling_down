"""CPU-only protocol checks: synthetic adapters, no model downloads or training."""
import builtins
import json
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from analysis import v12_distill as distill


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = torch.nn.Linear(2, 2)
        self.norm = torch.nn.LayerNorm(2)

    def save_pretrained(self, path):
        (path / "adapter_config.json").write_text("{}")


@pytest.fixture
def missing_peft(monkeypatch):
    original_import = builtins.__import__

    def import_without_peft(name, *args, **kwargs):
        if name == "peft" or name.startswith("peft."):
            raise ImportError("synthetic missing peft")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_peft)


@pytest.fixture
def fake_peft(monkeypatch):
    peft = ModuleType("peft")
    peft.LoraConfig = SimpleNamespace

    def get_peft_model(model, config):
        model.requires_grad_(False)
        model.q_proj.lora_A = torch.nn.Linear(2, 1, bias=False)
        model.q_proj.lora_B = torch.nn.Linear(1, 2, bias=False)
        model.peft_config = {"default": config}
        return model

    peft.get_peft_model = get_peft_model
    monkeypatch.setitem(sys.modules, "peft", peft)


@pytest.mark.parametrize("kwargs", [{}, {"training_mode": "lora"}])
def test_lora_without_peft_raises_before_changing_parameters(missing_peft, kwargs):
    model = TinyModel()
    before = [parameter.requires_grad for parameter in model.parameters()]
    with pytest.raises(RuntimeError, match="--training-mode lora requires peft") as error:
        distill.configure_training(model, **kwargs)
    assert isinstance(error.value.__cause__, ImportError)
    assert [parameter.requires_grad for parameter in model.parameters()] == before


def test_lora_run_without_peft_aborts_before_loading_or_writing(tmp_path, monkeypatch, missing_peft):
    monkeypatch.setattr(distill, "OUT_BASE", tmp_path)

    def forbidden(*args, **kwargs):
        pytest.fail("Missing PEFT must abort before data/model loads or training")

    for name in ("seed_everything", "load_sft_records", "load_text_causal_lm", "_train"):
        monkeypatch.setattr(distill, name, forbidden)
    with pytest.raises(RuntimeError, match="no full fine-tuning fallback"):
        distill.run_distillation("gemma3-270m", "gpt-5.6-luna", "full", ["math"],
                                 1, 1, "cpu", None)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("requested", ["full", "auto"])
def test_full_without_peft_preserves_matrix_only_protocol(missing_peft, capsys, requested):
    model, actual, names = distill.configure_training(TinyModel(), training_mode=requested)
    manifest = distill.build_training_manifest(model, requested, actual)
    assert actual == "full"
    assert names == ["q_proj.weight"]
    assert manifest == {
        "version": "v12-training-protocol-v1",
        "requested_training_mode": requested,
        "training_mode": "full",
        "trainable_parameters": 4,
        "total_parameters": 10,
        "target_modules": [],
    }
    if requested == "auto":
        assert "!!! TRAINING MODE AUTO -> FULL" in capsys.readouterr().out


@pytest.mark.parametrize("requested", ["lora", "auto"])
def test_lora_manifest_counts_actual_adapters(fake_peft, capsys, requested):
    model, actual, _ = distill.configure_training(TinyModel(), training_mode=requested)
    manifest = distill.build_training_manifest(model, requested, actual)
    assert manifest["training_mode"] == "lora"
    assert manifest["requested_training_mode"] == requested
    assert manifest["trainable_parameters"] == 4
    assert manifest["total_parameters"] == 14
    assert manifest["target_modules"] == ["q_proj"]
    assert not model.q_proj.weight.requires_grad
    if requested == "auto":
        assert "!!! TRAINING MODE AUTO -> LORA" in capsys.readouterr().out


def test_full_with_peft_installed_does_not_inject_adapters(fake_peft):
    model, actual, _ = distill.configure_training(TinyModel(), training_mode="full")
    assert actual == "full"
    assert not hasattr(model.q_proj, "lora_A")


@pytest.mark.parametrize("requested,configured", [("lora", "full"), ("lora", "lora"), ("auto", "lora")])
def test_manifest_rejects_mismatch_and_mislabeled_full_model(requested, configured):
    with pytest.raises(RuntimeError, match="Training protocol mismatch"):
        distill.build_training_manifest(TinyModel(), requested, configured)


def test_manifest_rejects_full_request_with_actual_lora(fake_peft):
    model, actual, _ = distill.configure_training(TinyModel())
    with pytest.raises(RuntimeError, match="Training protocol mismatch"):
        distill.build_training_manifest(model, "full", actual)


def test_manifest_rejects_unfrozen_lora_base_and_zero_trainable_parameters(fake_peft):
    model, actual, _ = distill.configure_training(TinyModel())
    model.q_proj.weight.requires_grad_(True)
    with pytest.raises(RuntimeError, match="frozen base parameters"):
        distill.build_training_manifest(model, "lora", actual)
    model.requires_grad_(False)
    with pytest.raises(RuntimeError, match="positive trainable-parameter count"):
        distill.build_training_manifest(model, "lora", actual)


@pytest.mark.parametrize("suffix", ["", "control", "_lora", "_full"])
@pytest.mark.parametrize("seed", [0, 7])
def test_output_paths_separate_modes_and_keep_seed_and_suffix(suffix, seed):
    names = {mode: distill.make_run_name("teacher", "full", 1, suffix, seed,
                                        training_mode=mode) for mode in ("lora", "full")}
    assert names["lora"] != names["full"]
    for mode, name in names.items():
        assert name.endswith(f"_{mode}_seed{seed}" if seed else f"_{mode}")
    assert distill.make_run_name("teacher", "full", 1, "_lora") == "teacher_full_1_lora"
    with pytest.raises(ValueError, match="actual training mode"):
        distill.make_run_name("teacher", "full", 1, training_mode="auto")


@pytest.mark.parametrize("requested,available", [("lora", True), ("full", True), ("auto", True), ("auto", False)])
def test_run_persists_manifest_before_training_and_in_all_evals(
    tmp_path, monkeypatch, request, requested, available,
):
    from test_v12 import TinyTokenizer

    request.getfixturevalue("fake_peft" if available else "missing_peft")
    actual = "lora" if available and requested != "full" else "full"
    monkeypatch.setattr(distill, "OUT_BASE", tmp_path)
    monkeypatch.setattr(distill, "seed_everything", lambda seed: None)
    monkeypatch.setattr(distill, "load_sft_records", lambda **kw: (
        [{"prompt": "ab", "completion": "XY", "domain": "math"}], {}))
    monkeypatch.setattr(distill, "build_probes", lambda *a, **kw: {
        c: [{"prompt": "a", "completion": "B"}] * 4 for c in distill.CAPABILITIES})
    monkeypatch.setattr(distill, "load_text_causal_lm", lambda *a: (TinyModel(), TinyTokenizer()))
    monkeypatch.setattr(distill, "measure_capability_losses", lambda *a: (
        {c: 2.0 for c in distill.CAPABILITIES}, {c: 4 for c in distill.CAPABILITIES}))
    before_training = []

    def no_training(**kwargs):
        paths = list(tmp_path.glob("*/*/train_log.json"))
        assert len(paths) == 1
        log = json.loads(paths[0].read_text())
        assert log["status"] == "configured"
        assert not (paths[0].parent / "eval.json").exists()
        before_training.append(log["training_manifest"])
        accounting = distill.TokenAccounting(kwargs["examples"])
        accounting.observe(0)
        progress = {**accounting.snapshot(), "updates": 1,
                    "completion_tokens_seen": 2, "requested_token_milestones": [1]}
        kwargs["trajectory_callback"](progress)
        return {**progress, "trajectory_tokens_unreached": []}

    monkeypatch.setattr(distill, "_train", no_training)
    result = distill.run_distillation("gemma3-270m", "gpt-5.6-luna", "full", ["math"],
                                      1, 1, "cpu", None, training_mode=requested,
                                      save_trajectory=True, trajectory_tokens=[1])
    manifest = before_training[0]
    assert manifest["requested_training_mode"] == requested
    assert manifest["training_mode"] == actual
    assert manifest["trainable_parameters"] == 4
    assert manifest["total_parameters"] == (14 if actual == "lora" else 10)
    assert manifest["target_modules"] == (["q_proj"] if actual == "lora" else [])
    root = tmp_path / "gemma3-270m" / result["run_name"]
    assert root.name.endswith(f"_{actual}")
    log = json.loads((root / "train_log.json").read_text())
    assert log["training_manifest"] == manifest == result["training_manifest"]
    assert log["status"] == "trained"
    assert log["trainable_parameters"] == manifest["trainable_parameters"]
    assert log["total_parameters"] == manifest["total_parameters"]
    evals = list(root.rglob("eval.json"))
    assert len(evals) == 3  # Baseline, simulated update, final.
    assert all(json.loads(path.read_text())["training_manifest"] == manifest for path in evals)


@pytest.mark.parametrize("requested", ["lora", "full", "auto"])
def test_dry_run_does_not_import_peft_and_labels_possible_outputs(tmp_path, monkeypatch, missing_peft, requested):
    monkeypatch.setattr(distill, "OUT_BASE", tmp_path)
    plan = distill.run_distillation("gemma3-270m", "gpt-5.6-luna", "full", ["math"],
                                    1, 1, "cpu", None, training_mode=requested, dry_run=True)
    assert plan["requested_training_mode"] == requested
    assert plan["output_candidates"]["lora"].endswith("_lora")
    assert plan["output_candidates"]["full"].endswith("_full")
    assert plan["output"] == (None if requested == "auto" else plan["output_candidates"][requested])
    assert list(tmp_path.iterdir()) == []


def test_cli_training_mode_default_and_choices(monkeypatch):
    calls = []
    monkeypatch.setattr(distill, "run_distillation", lambda **kw: calls.append(kw))
    argv = ["v12_distill.py", "--teacher", "gpt-5.6-luna", "--recipe", "full"]
    for value in (None, "lora", "full", "auto"):
        monkeypatch.setattr(sys, "argv", argv + (["--training-mode", value] if value else []))
        distill.main()
    assert [call["training_mode"] for call in calls] == ["lora", "lora", "full", "auto"]
    monkeypatch.setattr(sys, "argv", argv + ["--training-mode", "typo"])
    with pytest.raises(SystemExit) as error:
        distill.main()
    assert error.value.code == 2
