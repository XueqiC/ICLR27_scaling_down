#!/usr/bin/env python3
"""Model aliases and the HiPerGator PRC-model compliance guard."""
from __future__ import annotations

import os
import re


MODEL_REGISTRY: dict[str, dict[str, str]] = {
    "gemma3-270m": {
        "hf_id": "google/gemma-3-270m",
        "family": "gemma3",
        "notes": "Smallest checkpoint in the Gemma 3 capability ladder.",
    },
    "gemma3-1b": {
        "hf_id": "google/gemma-3-1b-pt",
        "family": "gemma3",
        "notes": "Base/pretrained Gemma 3 capability-ladder checkpoint.",
    },
    "gemma3-4b": {
        "hf_id": "google/gemma-3-4b-pt",
        "family": "gemma3",
        "notes": "Base/pretrained multimodal checkpoint; analyze text LM only.",
    },
    "gemma3-12b": {
        "hf_id": "google/gemma-3-12b-pt",
        "family": "gemma3",
        "notes": "Base/pretrained multimodal checkpoint; analyze text LM only.",
    },
    "gemma3-27b": {
        "hf_id": "google/gemma-3-27b-pt",
        "family": "gemma3",
        "notes": "Largest Gemma 3 ladder checkpoint; analyze text LM only.",
    },
    "gemma4-31b": {
        "hf_id": "google/gemma-4-31B",
        "family": "gemma4",
        "notes": "Cross-family 30B panel; analyze the text decoder only.",
    },
    "muse-30b": {
        "hf_id": "meta-models/Muse-Glimmer-30B",
        "family": "muse_glimmer",
        "notes": "Cross-family 30B panel; exclude the perception encoder.",
    },
    "olmo3-7b": {
        "hf_id": "allenai/Olmo-3-1025-7B",
        "family": "olmo3",
        "notes": "OLMo 3 mechanism-line checkpoint.",
    },
    "olmo3-32b": {
        "hf_id": "allenai/Olmo-3-1125-32B",
        "family": "olmo3",
        "notes": "Cross-family 30B panel checkpoint.",
    },
    "pythia-160m": {
        "hf_id": "EleutherAI/pythia-160m",
        "family": "pythia",
        "notes": "Controlled training-step checkpoints; non-PRC, HiPerGator-eligible; 3rd (smallest) panel size.",
    },
    "pythia-410m": {
        "hf_id": "EleutherAI/pythia-410m",
        "family": "pythia",
        "notes": "Controlled training-step checkpoints; non-PRC, HiPerGator-eligible.",
    },
    "pythia-1.4b": {
        "hf_id": "EleutherAI/pythia-1.4b",
        "family": "pythia",
        "notes": "Controlled training-step checkpoints; non-PRC, HiPerGator-eligible.",
    },
    "pythia-2.8b": {
        "hf_id": "EleutherAI/pythia-2.8b",
        "family": "pythia",
        "notes": "Controlled training-step checkpoints; non-PRC, HiPerGator-eligible.",
    },
    "Qwen3-0.6B": {
        "hf_id": "Qwen/Qwen3-0.6B",
        "family": "qwen3",
        "notes": "PRC-developed; RAI-only (SDL_ALLOW_PRC=1), never on HiPerGator.",
    },
    "Qwen3-1.7B": {
        "hf_id": "Qwen/Qwen3-1.7B",
        "family": "qwen3",
        "notes": "PRC-developed; RAI-only (SDL_ALLOW_PRC=1), never on HiPerGator.",
    },
    "Qwen3-4B": {
        "hf_id": "Qwen/Qwen3-4B",
        "family": "qwen3",
        "notes": "PRC-developed; RAI-only (SDL_ALLOW_PRC=1), never on HiPerGator.",
    },
    "Qwen3-8B": {
        "hf_id": "Qwen/Qwen3-8B",
        "family": "qwen3",
        "notes": "PRC-developed Base; RAI-only (SDL_ALLOW_PRC=1); NEW-SOURCE prospective test.",
    },
    "Qwen3-14B": {
        "hf_id": "Qwen/Qwen3-14B",
        "family": "qwen3",
        "notes": "PRC-developed; RAI-only (SDL_ALLOW_PRC=1); NEW-SOURCE prospective (pruning+quant).",
    },
}


_PRC_PATTERN = re.compile(
    r"qwen|deepseek|chatglm|glm|yi-|baichuan|internlm|minimax|kimi|moonshot",
    re.IGNORECASE,
)


def resolve_model_and_revision(model: str) -> tuple[str, str | None]:
    """Resolve a registry tag or raw HF id with an optional ``@revision``."""
    base, separator, revision = model.partition("@")
    if separator and (not base or not revision):
        raise ValueError("Expected a model tag/id followed by a non-empty @revision")
    entry = MODEL_REGISTRY.get(base)
    return (entry["hf_id"] if entry is not None else base,
            revision if separator else None)


def resolve_model(model: str) -> str:
    """Resolve a registry tag or raw HF id, stripping any ``@revision``."""
    return resolve_model_and_revision(model)[0]


def is_prc_model(hf_id_or_tag: str) -> bool:
    """Return whether a tag/id belongs to a known PRC-developed family."""
    base = hf_id_or_tag.partition("@")[0]
    candidate = f"{base} {resolve_model(base)}"
    return _PRC_PATTERN.search(candidate) is not None


def require_compliant(model: str) -> str:
    """Enforce cluster policy and return the resolved Hugging Face id.

    ``SDL_ALLOW_PRC=1`` exists only for approved, non-cluster RAI work.  The
    HiPerGator job wrapper has an additional unconditional guard.
    """
    resolved = resolve_model(model)
    if is_prc_model(model) and os.environ.get("SDL_ALLOW_PRC") != "1":
        raise RuntimeError(
            "UF HiPerGator policy prohibits running PRC-developed models "
            f"({resolved!r} was refused). SDL_ALLOW_PRC=1 is an RAI-only "
            "escape hatch and must never be set on HiPerGator."
        )
    return resolved

