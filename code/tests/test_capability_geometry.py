from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

from analysis import v6_capability_geometry as geometry


class _TinyTokenizer:
    def __call__(self, text, return_tensors="pt"):
        assert text == geometry.SANITY_PROMPT
        assert return_tensors == "pt"
        return SimpleNamespace(input_ids=torch.tensor([[0, 1, 2]]))


class _TinyModel(torch.nn.Module):
    def __init__(self, good=True):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.good = good

    def forward(self, input_ids, **kwargs):
        logits = torch.zeros((*input_ids.shape, 3), device=input_ids.device)
        score = 20.0 if self.good else -20.0
        logits[0, 0, 1] = score
        logits[0, 1, 2] = score
        return SimpleNamespace(logits=logits)


def test_loading_info_rejects_missing_and_mismatched_keys():
    info = {
        "missing_keys": ["model.layers.0.weight", "model.layers.1.weight"],
        "mismatched_keys": [("model.embed_tokens.weight", (2, 3), (4, 3))],
    }

    with pytest.raises(RuntimeError) as exc_info:
        geometry._raise_for_loading_info(
            "org/broken", info, SimpleNamespace(tie_word_embeddings=False))

    message = str(exc_info.value)
    assert "Checkpoint-integrity failure" in message
    assert "missing_keys=2" in message
    assert "model.layers.0.weight" in message
    assert "mismatched_keys=1" in message
    assert "model.embed_tokens.weight" in message


def test_loading_info_allows_only_tied_lm_head_missing_key():
    tied = SimpleNamespace(tie_word_embeddings=True)
    geometry._raise_for_loading_info(
        "org/tied", {"missing_keys": ["lm_head.weight"]}, tied)

    with pytest.raises(RuntimeError, match="missing_keys=1"):
        geometry._raise_for_loading_info(
            "org/not-tied", {"missing_keys": ["lm_head.weight"]},
            SimpleNamespace(tie_word_embeddings=False),
        )


def test_sanity_forward_rejects_random_level_loss():
    assert geometry._checkpoint_sanity_forward(
        _TinyModel(good=True), _TinyTokenizer(), "org/good") < 0.01

    with pytest.raises(RuntimeError, match="Checkpoint-integrity failure"):
        geometry._checkpoint_sanity_forward(
            _TinyModel(good=False), _TinyTokenizer(), "org/broken")


def test_abs_weight_sample_is_deterministic_and_bounded():
    params = [
        ("a", torch.arange(1000, dtype=torch.float32)),
        ("b", -torch.arange(1000, dtype=torch.float32)),
        ("c", torch.full((1000,), 7.0)),
    ]

    first = geometry._sample_abs_weights(params, sample_size=300, seed=17)
    second = geometry._sample_abs_weights(params, sample_size=300, seed=17)

    assert first.shape == (300,)
    np.testing.assert_array_equal(first, second)
    assert np.all(first >= 0)


def test_primary_large_unexpected_set_uses_fallback(monkeypatch):
    calls = []
    config = SimpleNamespace(
        text_config=SimpleNamespace(tie_word_embeddings=False),
        architectures=None,
        tie_word_embeddings=False,
    )

    class AutoConfig:
        @classmethod
        def from_pretrained(cls, model_name):
            return config

    class Primary:
        @classmethod
        def from_pretrained(cls, model_name, **kwargs):
            calls.append(("primary", kwargs))
            return _TinyModel(), {
                "unexpected_keys": [f"vision.{i}" for i in range(11)],
            }

    class Fallback:
        @classmethod
        def from_pretrained(cls, model_name, **kwargs):
            calls.append(("fallback", kwargs))
            return _TinyModel(), {
                "missing_keys": [],
                "mismatched_keys": [],
                "unexpected_keys": ["vision.checkpoint_only"],
            }

    class AutoTokenizer:
        @classmethod
        def from_pretrained(cls, model_name):
            return _TinyTokenizer()

    fake_transformers = ModuleType("transformers")
    fake_transformers.AutoConfig = AutoConfig
    fake_transformers.AutoModelForCausalLM = Primary
    fake_transformers.AutoModelForImageTextToText = Fallback
    fake_transformers.AutoTokenizer = AutoTokenizer
    monkeypatch.setitem(__import__("sys").modules, "transformers",
                        fake_transformers)

    model, _ = geometry.load_text_causal_lm("org/model", torch.float32)

    assert isinstance(model, _TinyModel)
    assert [kind for kind, _ in calls] == ["primary", "fallback"]
    assert all(kwargs["output_loading_info"] is True for _, kwargs in calls)
