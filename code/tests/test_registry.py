import pytest

from analysis.model_registry import (MODEL_REGISTRY, is_prc_model,
                                     require_compliant, resolve_model)


def test_registry_tags_resolve():
    assert resolve_model("gemma3-4b") == "google/gemma-3-4b-pt"
    assert resolve_model("olmo3-32b") == "allenai/Olmo-3-1125-32B"
    assert MODEL_REGISTRY["muse-30b"]["family"] == "muse_glimmer"
    assert resolve_model("some-org/custom-model") == "some-org/custom-model"


@pytest.mark.parametrize("model", [
    "Qwen/Qwen3-4B",
    "deepseek-ai/DeepSeek-R1",
])
def test_prc_detection_catches_known_families(model):
    assert is_prc_model(model)


@pytest.mark.parametrize("model", [
    "gemma3-27b",
    "allenai/Olmo-3-1025-7B",
    "meta-models/Muse-Glimmer-30B",
])
def test_prc_detection_passes_compliant_families(model):
    assert not is_prc_model(model)


def test_require_compliant_raises_without_escape_hatch(monkeypatch):
    monkeypatch.delenv("SDL_ALLOW_PRC", raising=False)
    with pytest.raises(RuntimeError, match="HiPerGator policy"):
        require_compliant("Qwen/Qwen3-4B")

