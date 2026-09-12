"""Checkpoint selection regressions: CPU only, with no Hub/data downloads."""
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from analysis import v6_capability_geometry as geometry
from analysis import v10_quantization as quantization
from analysis import v12_distill as distill
from analysis.model_registry import (
    MODEL_REGISTRY,
    is_prc_model,
    require_compliant,
    resolve_model,
    resolve_model_and_revision,
)


@pytest.mark.parametrize("size", ["410m", "1.4b", "2.8b"])
@pytest.mark.parametrize("revision", [None, "step16000", "step143000"])
def test_pythia_registry_and_revision(size, revision, monkeypatch):
    monkeypatch.delenv("SDL_ALLOW_RESTRICTED", raising=False)
    base = f"pythia-{size}"
    requested = f"{base}@{revision}" if revision else base
    expected = f"EleutherAI/{base}"
    assert MODEL_REGISTRY[base]["family"] == "pythia"
    assert resolve_model_and_revision(requested) == (expected, revision)
    assert resolve_model(requested) == expected
    assert require_compliant(requested) == expected
    assert not is_prc_model(requested)


@pytest.mark.parametrize("requested,resolved,revision,tag", [
    ("pythia-1.4b@step16000", "EleutherAI/pythia-1.4b", "step16000", "pythia-1.4b--step16000"),
    ("pythia-1.4b@step143000", "EleutherAI/pythia-1.4b", "step143000", "pythia-1.4b--step143000"),
    ("pythia-1.4b", "EleutherAI/pythia-1.4b", None, "pythia-1.4b"),
    ("gemma3-4b", "google/gemma-3-4b-pt", None, "gemma3-4b"),
    ("olmo3-7b", "allenai/Olmo-3-1025-7B", None, "olmo3-7b"),
    ("Qwen3-4B", "Qwen/Qwen3-4B", None, "Qwen3-4B"),
    ("org/custom", "org/custom", None, "org--custom"),
    ("EleutherAI/pythia-1.4b@step16000", "EleutherAI/pythia-1.4b", "step16000", "EleutherAI--pythia-1.4b--step16000"),
    ("org/custom@refs/pr/1", "org/custom", "refs/pr/1", "org--custom--refs--pr--1"),
])
def test_resolution_and_output_tags(requested, resolved, revision, tag):
    assert resolve_model_and_revision(requested) == (resolved, revision)
    assert resolve_model(requested) == resolved
    assert geometry.model_output_tag(requested, resolved) == tag
    assert Path(tag).name == tag


@pytest.mark.parametrize("requested", ["pythia-1.4b@", "@step16000"])
def test_empty_checkpoint_parts_are_rejected(requested):
    with pytest.raises(ValueError, match="@revision"):
        resolve_model_and_revision(requested)


def test_compliance_checks_base_model_only(monkeypatch):
    monkeypatch.delenv("SDL_ALLOW_RESTRICTED", raising=False)
    assert require_compliant("pythia-1.4b@qwen-comparison") == "EleutherAI/pythia-1.4b"
    with pytest.raises(RuntimeError, match="the shared cluster policy"):
        require_compliant("Qwen3-4B@step16000")
    monkeypatch.setenv("SDL_ALLOW_RESTRICTED", "1")
    assert require_compliant("Qwen3-4B@step16000") == "Qwen/Qwen3-4B"


@pytest.mark.parametrize("revision", [None, "step16000", "step143000"])
@pytest.mark.parametrize("fallback", [False, True])
def test_revision_reaches_config_weights_and_tokenizer(monkeypatch, revision, fallback):
    config = SimpleNamespace(tie_word_embeddings=False)
    model = torch.nn.Linear(2, 2)
    tokenizer = object()
    fake = ModuleType("transformers")
    fake.AutoConfig = SimpleNamespace(from_pretrained=Mock(return_value=config))
    primary = Mock(return_value=(model, {}))
    if fallback:
        primary.side_effect = ValueError("Use the fallback loader")
    secondary = Mock(return_value=(model, {}))
    fake.AutoModelForCausalLM = SimpleNamespace(from_pretrained=primary)
    fake.AutoModelForImageTextToText = SimpleNamespace(from_pretrained=secondary)
    fake.AutoTokenizer = SimpleNamespace(from_pretrained=Mock(return_value=tokenizer))
    monkeypatch.setitem(sys.modules, "transformers", fake)
    monkeypatch.setattr(geometry, "_checkpoint_sanity_forward", lambda *args: 0.0)

    actual = geometry.load_text_causal_lm("EleutherAI/pythia-1.4b", torch.float32, revision)

    assert actual == (model, tokenizer)
    loaders = [fake.AutoConfig.from_pretrained, primary, fake.AutoTokenizer.from_pretrained]
    if fallback:
        loaders.append(secondary)
    else:
        secondary.assert_not_called()
    for loader in loaders:
        assert loader.call_args.args == ("EleutherAI/pythia-1.4b",)
        assert loader.call_args.kwargs.get("revision") == revision
        if revision is None:
            assert "revision" not in loader.call_args.kwargs


@pytest.mark.parametrize("requested,hf_id,revision,tag", [
    ("pythia-1.4b@step16000", "EleutherAI/pythia-1.4b", "step16000", "pythia-1.4b--step16000"),
    ("pythia-1.4b@step143000", "EleutherAI/pythia-1.4b", "step143000", "pythia-1.4b--step143000"),
    ("gemma3-4b", "google/gemma-3-4b-pt", None, "gemma3-4b"),
    ("olmo3-7b", "allenai/Olmo-3-1025-7B", None, "olmo3-7b"),
    ("Qwen3-4B", "Qwen/Qwen3-4B", None, "Qwen3-4B"),
])
@pytest.mark.parametrize("pipeline", ["prune", "fisher", "quant", "distill"])
def test_cli_preserves_checkpoint_through_stage_to_loader(
    tmp_path, monkeypatch, requested, hf_id, revision, tag, pipeline,
):
    class ReachedLoader(Exception):
        pass

    def load(model_name, dtype, revision=None):
        assert (model_name, revision) == (hf_id, expected_revision)
        raise ReachedLoader

    expected_revision = revision
    monkeypatch.setenv("SDL_ALLOW_RESTRICTED", "1")  # Qwen compatibility test, no cluster use.
    module = {"prune": geometry, "fisher": geometry,
              "quant": quantization, "distill": distill}[pipeline]
    monkeypatch.setattr(module, "OUT_BASE", tmp_path)
    monkeypatch.setattr(module, "load_text_causal_lm", load)
    argv = ["pipeline", "--device", "cpu"]
    if pipeline == "distill":
        monkeypatch.setattr(distill, "seed_everything", lambda *args: None)
        monkeypatch.setattr(distill, "load_sft_records", lambda **kwargs: ([], {}))
        monkeypatch.setattr(distill, "build_probes", lambda *args, **kwargs: {
            capability: [{}, {}] for capability in distill.CAPABILITIES
        })
        argv += ["--student", requested, "--teacher", "gpt-5.6-luna", "--recipe", "full",
                 "--training-mode", "full"]
    else:
        argv += ["--model", requested]
        if pipeline in ("prune", "fisher"):
            argv += ["--stage", pipeline]
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(ReachedLoader):
        module.main()

    assert [path.name for path in tmp_path.iterdir()] == [tag]


@pytest.mark.parametrize("revision", ["step16000", "step143000"])
def test_distillation_explicit_revision_matches_suffix(tmp_path, monkeypatch, revision):
    monkeypatch.setattr(distill, "OUT_BASE", tmp_path)
    kwargs = dict(teacher="gpt-5.6-luna", recipe="full", domains=["math"],
                  n_per_domain=1, epochs=1, device="cpu", learning_rate=None, dry_run=True)
    explicit = distill.run_distillation("pythia-1.4b", revision=revision, **kwargs)
    suffixed = distill.run_distillation(f"pythia-1.4b@{revision}", **kwargs)
    assert explicit == suffixed
    assert explicit["revision"] == revision
    assert Path(explicit["output"]).parent.name == f"pythia-1.4b--{revision}"
    assert list(tmp_path.iterdir()) == []
    with pytest.raises(ValueError, match="conflicts"):
        distill.run_distillation("pythia-1.4b@other", revision=revision, **kwargs)


def test_pythia_lora_targets_work_on_tiny_cpu_model():
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("peft")
    config = transformers.GPTNeoXConfig(
        vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
        num_attention_heads=2, max_position_embeddings=32,
    )
    model = transformers.GPTNeoXForCausalLM(config)
    model, mode, names = distill.configure_training(model, allow_full_fallback=False)
    assert mode == "lora"
    manifest = distill.build_training_manifest(model, "lora", mode)
    assert 0 < manifest["trainable_parameters"] < manifest["total_parameters"]
    assert len(manifest["target_modules"]) == 4
    assert next(model.parameters()).device.type == "cpu"
    for target in ("query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"):
        assert any(f".{target}.lora_" in name for name in names)
    for family in ("gemma3", "olmo3", "qwen3"):
        other = SimpleNamespace(config=SimpleNamespace(model_type=family))
        assert distill.lora_target_modules(other) == distill.LORA_TARGET_MODULES
