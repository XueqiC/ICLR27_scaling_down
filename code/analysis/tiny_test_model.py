"""Offline CPU test doubles: a tiny tokenizer, a tiny causal model, and probe sets.

Used by the offline test suite and by the descriptor self-check. Kept separate from
any experiment module so that tests do not depend on research code under
development.
"""
from types import SimpleNamespace

import torch


class TinyTokenizer:
    name_or_path = "v87-tiny"
    bos_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return ([0] if add_special_tokens else []) + [1 + ord(c) % 16 for c in text]

    def __call__(self, text, return_tensors=None, truncation=False, max_length=None,
                 add_special_tokens=False, return_offsets_mapping=False):
        ids = self.encode(text, add_special_tokens)
        if truncation:
            ids = ids[:max_length]
        if return_offsets_mapping:
            return {"input_ids": ids, "offset_mapping": [(i, i + 1) for i in range(len(text))]}
        if return_tensors == "pt":
            return SimpleNamespace(input_ids=torch.tensor([ids], dtype=torch.long))
        return {"input_ids": ids}


class TinyModel(torch.nn.Module):
    """Random causal transition model, with an actual final softcap."""
    def __init__(self, dtype=torch.float32):
        super().__init__()
        generator = torch.Generator(device="cpu").manual_seed(87)
        self.weight = torch.nn.Parameter(torch.randn(17, 17, generator=generator).to(dtype))
        self.config = SimpleNamespace(_commit_hash="random-seed-87")

    def forward(self, input_ids, use_cache=False):
        assert input_ids.device.type == "cpu" and not use_cache
        return SimpleNamespace(logits=1.7 * torch.tanh(self.weight[input_ids] / 1.7))


PROBES = {"math": [{"sample_id": "short", "prompt": "ab", "completion": "c"},
                    {"sample_id": "long", "prompt": "cd", "completion": "efghijk"}],
          "code": [{"sample_id": "unicode", "prompt": "lm", "completion": "én"}],
          "qa": [{"sample_id": "qa", "prompt": "op", "completion": "qrs"}]}
