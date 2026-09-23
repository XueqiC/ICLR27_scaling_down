#!/usr/bin/env python3
"""V12: distill teacher traces into a student and measure capability loss.

The training examples come from ``results/traces-pilot``.  Three recipes
control how much of each teacher completion is retained, providing the
coverage-deletion axis used by the scaling-down-law experiments.  Evaluation
uses the odd-indexed (measurement) half of the clean V6 probes.

Example:
  python3 analysis/v12_distill.py --student gemma3-270m \
      --teacher gpt-5.6-luna --recipe full --device cuda:0

Artifacts are written under
``results/v12-distill/<student_tag>/<teacher>_<recipe>_<n>_<lora|full>[_seedN]/``.
``--training-mode`` defaults to strict LoRA (requires PEFT); ``full`` opts into
the original full matrix fine-tuning protocol, and ``auto`` permits fallback.
``--seed`` controls training and data shuffling; measurement probes stay fixed.
``--save-trajectory --trajectory-tokens 250000 500000 1000000`` saves a dense
baseline plus eval/adapter snapshots at completed updates within this same run.
Processed tokens include repeated prompt+target exposure; unique encountered
tokens and fixed-pool tokens are recorded separately in every eval.json.
Snapshots share one seed and schedule and are not independent replicates.
``--dry-run`` prints the design without loading data/models or writing outputs.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import random
import re
import tempfile
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch.nn import functional as F

try:
    from .v6_capability_geometry import (
        build_probes,
        completion_loss,
        language_weight_parameters,
        load_text_causal_lm,
        model_output_tag,
        require_compliant,
        resolve_model_and_revision,
    )
except ImportError:  # direct execution: python analysis/v12_distill.py
    from v6_capability_geometry import (
        build_probes,
        completion_loss,
        language_weight_parameters,
        load_text_causal_lm,
        model_output_tag,
        require_compliant,
        resolve_model_and_revision,
    )


ROOT = Path(__file__).resolve().parents[1]
TRACE_BASE = ROOT / "results/traces-pilot"
OUT_BASE = ROOT / "results/v12-distill"

TEACHERS = ("gpt-5.6-luna", "claude-sonnet-4-6")
DOMAINS = ("math", "qa", "code")
CAPABILITIES = ("math", "code", "qa")
RECIPES = ("full", "answer_only", "no_code_fence")
TRAINING_MODES = ("lora", "full", "auto")
TRAINING_BENCHMARKS = {
    "math": "GSM8K",
    "code": "CodeAlpaca",
    "qa": "HotpotQA",
}
MEASUREMENT_BENCHMARKS = {
    "math": "MATH-500",
    "code": "MBPP",
    "qa": "2WikiMultihopQA",
}
LORA_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

SEED = 0
MAX_LEN = 1024
EFFECTIVE_BATCH_SIZE = 16
DEFAULT_N_PROBE = 128
LORA_LR = 1e-4
FULL_LR = 1e-5
WARMUP_RATIO = 0.03
DEFAULT_TRAJECTORY_TOKENS = (250000, 500000, 1000000, 2000000, 4000000)
TOKEN_ACCOUNTING = {
    "version": "v12-token-accounting-v1",
    "processed_tokens": "seen_tokens: cumulative non-padding input tokens, prompt + target + BOS, after truncation; includes repetitions",
    "unique_data_tokens": "sum of input lengths of distinct training examples encountered so far, counted once",
    "unique_data_pool_tokens": "sum of input lengths of distinct examples in the fixed selected/tokenized pool",
    "identity": "SHA256 of source prompt/completion; fallback to tokenized input_ids and labels for direct _train callers",
    "completion_tokens_seen": "cumulative supervised target tokens, separately from processed input tokens",
}

_FENCED_CODE_RE = re.compile(
    r"```[^\r\n]*\r?\n(?P<body>.*?)```", re.DOTALL
)


def extract_full(response: str, domain: str) -> str:
    """Retain the complete teacher response."""
    del domain
    return response


def _last_line(text: str) -> str:
    lines = text.splitlines()
    return lines[-1].strip() if lines else text.strip()


def extract_answer_only(response: str, domain: str) -> str:
    """Retain only the domain-specific final-answer segment."""
    if domain == "math":
        if "####" in response:
            return response.rsplit("####", 1)[1].strip()
        return _last_line(response)
    if domain == "qa":
        return next(
            (line.strip() for line in reversed(response.splitlines()) if line.strip()),
            "",
        )
    if domain == "code":
        matches = list(_FENCED_CODE_RE.finditer(response))
        if matches:
            return matches[-1].group("body").strip()
        return _last_line(response)
    raise ValueError(f"Unknown domain {domain!r}; expected one of {DOMAINS}")


def extract_no_code_fence(response: str, domain: str) -> str:
    """Retain the full response except for complete fenced code blocks."""
    del domain
    return _FENCED_CODE_RE.sub("", response)


RECIPE_EXTRACTORS = {
    "full": extract_full,
    "answer_only": extract_answer_only,
    "no_code_fence": extract_no_code_fence,
}


def apply_recipe(response: str, domain: str, recipe: str) -> str:
    """Apply one named coverage recipe to a teacher response."""
    try:
        extractor = RECIPE_EXTRACTORS[recipe]
    except KeyError as exc:
        raise ValueError(f"Unknown recipe {recipe!r}; expected one of {RECIPES}") from exc
    if domain not in DOMAINS:
        raise ValueError(f"Unknown domain {domain!r}; expected one of {DOMAINS}")
    return extractor(response, domain)


def parse_domains(value: str | Sequence[str]) -> tuple[str, ...]:
    """Parse and validate a deterministic, duplicate-free domain sequence."""
    parts = value.split(",") if isinstance(value, str) else list(value)
    domains = tuple(str(part).strip() for part in parts if str(part).strip())
    if not domains:
        raise argparse.ArgumentTypeError("--domains must contain at least one domain")
    unknown = [domain for domain in domains if domain not in DOMAINS]
    if unknown:
        raise argparse.ArgumentTypeError(
            "unknown domain(s): " + ", ".join(unknown)
        )
    if len(set(domains)) != len(domains):
        raise argparse.ArgumentTypeError("--domains must not contain duplicates")
    return domains


def make_run_name(
    teacher: str,
    recipe: str,
    n_per_domain: int,
    output_suffix: str = "",
    seed: int = SEED,
    training_mode: str = "lora",
    data_seed: int | None = None,
) -> str:
    """Label the actual mode; preserve existing explicit ``_lora`` suffixes."""
    if training_mode not in ("lora", "full"):
        raise ValueError("Run names require an actual training mode: lora or full")
    base = f"{teacher}_{recipe}_{n_per_domain}"
    if output_suffix:
        safe_suffix = re.sub(r"[^A-Za-z0-9._-]+", "--", output_suffix).strip(".-_")
        if not safe_suffix:
            raise ValueError("--output-suffix must contain a path-safe character")
        base = f"{base}_{safe_suffix}"
    if not base.endswith(f"_{training_mode}"):
        base = f"{base}_{training_mode}"
    if data_seed is not None:  # keep pool-seed runs in distinct dirs (default None preserves old paths)
        base = f"{base}_dseed{data_seed}"
    return f"{base}_seed{seed}" if seed != 0 else base


def load_sft_records(
    teacher: str,
    domains: Sequence[str],
    n_per_domain: int,
    recipe: str,
    trace_base: Path = TRACE_BASE,
    seed: int = SEED,
    data_seed: int | None = None,
) -> tuple[list[dict[str, str]], dict[str, dict[str, int]]]:
    """Load exactly ``n_per_domain`` source rows and apply the SFT recipe.

    Pool selection: when ``data_seed`` is None the pool is the FIRST ``n_per_domain`` non-empty rows
    (unchanged, reproducible default). When ``data_seed`` is set, the pool is a RANDOM sample of
    ``n_per_domain`` rows per domain drawn with ``random.Random(f"{data_seed}-{domain}")`` — this is the
    pool-SAMPLING seed, independent of the training ``seed``, so pool-selection variance can be estimated.

    Empty completions created by coverage deletion are intentionally omitted:
    they contain no target token on which a causal-LM objective can train.
    The returned counts make that deletion explicit in ``train_log.json``.
    """
    if teacher not in TEACHERS:
        raise ValueError(f"Unknown teacher {teacher!r}; expected one of {TEACHERS}")
    if n_per_domain <= 0:
        raise ValueError("n_per_domain must be positive")
    if recipe not in RECIPES:
        raise ValueError(f"Unknown recipe {recipe!r}; expected one of {RECIPES}")
    parsed_domains = parse_domains(domains)

    records: list[dict[str, str]] = []
    counts: dict[str, dict[str, int]] = {}
    for domain in parsed_domains:
        path = trace_base / f"{teacher}_{domain}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"Teacher trace file does not exist: {path}")
        eligible: list[Mapping] = []
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if data_seed is None and len(eligible) == n_per_domain:
                    break  # first-U default: stop early
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {path}:{line_number}: {exc}") from exc
                eligible.append(row)
        if len(eligible) < n_per_domain:
            raise ValueError(
                f"{path} contains only {len(eligible)} non-empty rows; "
                f"requested {n_per_domain}"
            )
        if data_seed is None:
            selected = eligible[:n_per_domain]
        else:
            selected = random.Random(f"{data_seed}-{domain}").sample(eligible, n_per_domain)

        retained = 0
        for row_index, row in enumerate(selected, start=1):
            prompt = row.get("prompt")
            response = row.get("response")
            if not isinstance(prompt, str) or not isinstance(response, str):
                raise ValueError(
                    f"{path} selected row {row_index} needs string prompt/response fields"
                )
            row_teacher = row.get("teacher")
            if row_teacher is not None and row_teacher != teacher:
                raise ValueError(
                    f"{path} selected row {row_index} names teacher {row_teacher!r}, "
                    f"expected {teacher!r}"
                )
            completion = apply_recipe(response, domain, recipe)
            if not completion.strip():
                continue
            records.append(
                {"prompt": prompt, "completion": completion, "domain": domain}
            )
            retained += 1
        counts[domain] = {
            "source_rows": len(selected),
            "training_rows": retained,
            "coverage_deleted_rows": len(selected) - retained,
        }

    if not records:
        raise RuntimeError("The selected traces and recipe produced no training examples")

    # One seeded shuffle avoids capability blocks while remaining invariant
    # across machines and Python invocations.
    random.Random(seed).shuffle(records)
    return records, counts


def load_mixed_trace_records(
    trace_base: Path = TRACE_BASE,
    seed: int = SEED,
) -> tuple[list[dict[str, str]], dict[str, dict[str, int]]]:
    """Load and deterministically mix all teacher/domain JSONL traces.

    Unlike :func:`load_sft_records`, this helper intentionally scans every
    top-level ``*.jsonl`` file rather than selecting one teacher or a fixed
    number of rows.  It is used by generic continued-pretraining experiments
    where the mixed trace pool itself is the recovery-data contrast.
    """
    trace_base = Path(trace_base)
    paths = sorted(trace_base.glob("*.jsonl"))
    if not paths:
        raise FileNotFoundError(f"No teacher trace JSONL files found in {trace_base}")

    records: list[dict[str, str]] = []
    mix: dict[str, dict[str, int]] = {}
    for path in paths:
        stem = path.stem
        domain = next(
            (candidate for candidate in DOMAINS if stem.endswith(f"_{candidate}")),
            None,
        )
        if domain is None:
            raise ValueError(
                f"Cannot infer a domain from {path.name}; expected a filename "
                f"ending in one of {tuple(f'_{item}.jsonl' for item in DOMAINS)}"
            )
        filename_teacher = stem[: -(len(domain) + 1)]
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {path}:{line_number}: {exc}") from exc
                prompt = row.get("prompt")
                response = row.get("response")
                if not isinstance(prompt, str) or not isinstance(response, str):
                    raise ValueError(
                        f"{path}:{line_number} needs string prompt/response fields"
                    )
                teacher = row.get("teacher", filename_teacher)
                if not isinstance(teacher, str) or not teacher:
                    raise ValueError(
                        f"{path}:{line_number} needs a non-empty string teacher"
                    )
                if teacher != filename_teacher:
                    raise ValueError(
                        f"{path}:{line_number} names teacher {teacher!r}, "
                        f"expected {filename_teacher!r} from the filename"
                    )
                completion = apply_recipe(response, domain, "full")
                if not completion.strip():
                    continue
                records.append(
                    {
                        "prompt": prompt,
                        "completion": completion,
                        "domain": domain,
                        "teacher": teacher,
                    }
                )
                teacher_mix = mix.setdefault(teacher, {})
                teacher_mix[domain] = teacher_mix.get(domain, 0) + 1

    if not records:
        raise RuntimeError(f"Teacher traces in {trace_base} contain no usable rows")
    random.Random(seed).shuffle(records)
    return records, mix


def _encoding_input_ids(encoding) -> torch.Tensor:
    ids = encoding["input_ids"] if isinstance(encoding, Mapping) else encoding.input_ids
    if not isinstance(ids, torch.Tensor):
        ids = torch.tensor(ids, dtype=torch.long)
    if ids.ndim == 1:
        ids = ids.unsqueeze(0)
    if ids.ndim != 2 or ids.shape[0] != 1:
        raise ValueError("Tokenizer must return one rank-2 input_ids tensor")
    return ids


def tokenize_sft_example(
    tokenizer,
    prompt: str,
    completion: str,
    max_len: int = MAX_LEN,
    append_eos: bool = False,
) -> dict[str, torch.Tensor | int]:
    """Tokenize one pair and mask every prompt label with ``-100``.

    With ``append_eos`` the tokenizer's end-of-sequence id follows the completion tokens and is
    supervised like them (the P3 end-marker control); the default protocol appends nothing.

    This mirrors :func:`v6_capability_geometry.completion_loss`: direct
    tokenization, at most half the sequence for each segment, prompt special
    tokens only, and the Gemma single-BOS guard.  With causal shifting, the
    first unmasked label is the first completion token.
    """
    if max_len < 2:
        raise ValueError("max_len must be at least 2")
    half_len = max_len // 2
    prompt_ids = _encoding_input_ids(
        tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=half_len,
            add_special_tokens=True,
        )
    )
    tokenizer_name = (
        f"{getattr(tokenizer, 'name_or_path', '')} {type(tokenizer).__name__}"
    ).lower()
    bos_id = getattr(tokenizer, "bos_token_id", None)
    if (
        "gemma" in tokenizer_name
        and bos_id is not None
        and (prompt_ids.shape[1] == 0 or int(prompt_ids[0, 0]) != bos_id)
    ):
        bos = torch.tensor([[bos_id]], dtype=prompt_ids.dtype)
        prompt_ids = torch.cat((bos, prompt_ids), dim=1)[:, :half_len]

    completion_ids = _encoding_input_ids(
        tokenizer(
            completion,
            return_tensors="pt",
            truncation=True,
            max_length=half_len - (1 if append_eos else 0),
            add_special_tokens=False,
        )
    )
    if append_eos:
        eos_id = getattr(tokenizer, "eos_token_id", None)
        if eos_id is None:
            raise ValueError("--append-eos needs a tokenizer with an end-of-sequence id")
        eos = torch.tensor([[eos_id]], dtype=completion_ids.dtype)
        completion_ids = torch.cat((completion_ids, eos), dim=1)
    input_ids = torch.cat((prompt_ids, completion_ids), dim=1)[:, -max_len:]
    n_prompt = (
        min(prompt_ids.shape[1], input_ids.shape[1] - completion_ids.shape[1])
        if input_ids.shape[1] > completion_ids.shape[1]
        else 0
    )
    labels = input_ids.clone()
    labels[:, :n_prompt] = -100
    attention_mask = torch.ones_like(input_ids)
    n_completion_tokens = int((labels[:, 1:] != -100).sum())
    return {
        "input_ids": input_ids[0],
        "attention_mask": attention_mask[0],
        "labels": labels[0],
        "n_completion_tokens": n_completion_tokens,
    }


def tokenize_text_example(
    tokenizer,
    text: str,
    max_len: int = MAX_LEN,
) -> dict[str, torch.Tensor | int]:
    """Tokenize one plain document for full causal-LM supervision."""
    if max_len < 2:
        raise ValueError("max_len must be at least 2")
    input_ids = _encoding_input_ids(
        tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_len,
            add_special_tokens=True,
        )
    )[:, :max_len]
    labels = input_ids.clone()
    attention_mask = torch.ones_like(input_ids)
    n_document_tokens = max(int(input_ids.shape[1]) - 1, 0)
    return {
        "input_ids": input_ids[0],
        "attention_mask": attention_mask[0],
        "labels": labels[0],
        "n_completion_tokens": n_document_tokens,
        "n_document_tokens": n_document_tokens,
    }


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def measure_capability_losses(
    model,
    tokenizer,
    probes: Mapping[str, Sequence[Mapping[str, str]]],
    device: str,
    max_len: int = MAX_LEN,
    loss_function=completion_loss,
) -> tuple[dict[str, float], dict[str, int]]:
    losses: dict[str, float] = {}
    token_counts: dict[str, int] = {}
    model.eval()
    with torch.no_grad():
        for capability in CAPABILITIES:
            total_loss = 0.0
            total_tokens = 0
            for sample in probes[capability]:
                loss, n_tokens = loss_function(
                    model,
                    tokenizer,
                    sample["prompt"],
                    sample["completion"],
                    device,
                    max_len=max_len,
                )
                total_loss += float(loss)
                total_tokens += n_tokens
            if total_tokens == 0:
                raise RuntimeError(
                    f"{capability} measurement half produced no completion tokens"
                )
            losses[capability] = total_loss / total_tokens
            token_counts[capability] = total_tokens
    return losses, token_counts


def lora_target_modules(model) -> tuple[str, ...]:
    """Select GPT-NeoX projections for Pythia; retain other families' targets."""
    if getattr(getattr(model, "config", None), "model_type", None) == "gpt_neox":
        return ("query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h")
    return LORA_TARGET_MODULES


def resolve_training_mode(training_mode: str) -> str:
    """Check PEFT before loading data/models; only explicit auto may fall back."""
    if training_mode not in TRAINING_MODES:
        raise ValueError(f"Unknown training mode {training_mode!r}; expected {TRAINING_MODES}")
    if training_mode == "full":
        return "full"
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:
        if training_mode == "lora":
            raise RuntimeError(
                "--training-mode lora requires peft, but its import failed. "
                "Install a working peft environment or explicitly request "
                "--training-mode full (intentional full-FT) or auto (allow fallback). "
                "Aborting; no full fine-tuning fallback is allowed."
            ) from exc
        print(f"!!! TRAINING MODE AUTO -> FULL: peft import failed ({exc}); "
              "FULL MATRIX FINE-TUNING SELECTED !!!", flush=True)
        return "full"
    if training_mode == "auto":
        print("!!! TRAINING MODE AUTO -> LORA: peft available; LoRA SELECTED !!!", flush=True)
    return "lora"


def configure_training(
    model,
    training_mode: str = "lora",
    *,
    allow_full_fallback: bool = False,
) -> tuple[torch.nn.Module, str, list[str]]:
    """Configure the requested protocol; retain legacy strict callers' keyword."""
    if allow_full_fallback:
        raise ValueError("Use training_mode='auto' to explicitly allow full-FT fallback")
    training_mode = resolve_training_mode(training_mode)
    if training_mode == "full":
        model.requires_grad_(False)
        parameters = language_weight_parameters(model)
        if not parameters:
            raise ValueError("No language weight matrices found for fine-tuning")
        for _, parameter in parameters:
            parameter.requires_grad_(True)
        return model, "full", [name for name, _ in parameters]

    from peft import LoraConfig, get_peft_model

    config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.0,
        bias="none",
        target_modules=list(lora_target_modules(model)),
    )
    model = get_peft_model(model, config)
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError(
            "PEFT found no LoRA targets; expected attention/MLP projections "
            f"named {', '.join(lora_target_modules(model))}"
        )
    return model, "lora", trainable


def build_training_manifest(model, requested_mode: str, configured_mode: str) -> dict:
    """Assert the protocol from actual parameters, before any optimizer update."""
    if requested_mode not in TRAINING_MODES:
        raise ValueError(f"Unknown training mode {requested_mode!r}")
    parameters = dict(model.named_parameters())
    trainable = {name for name, parameter in parameters.items() if parameter.requires_grad}
    adapters = {name for name in parameters if ".lora_" in name}
    actual_mode = "lora" if adapters else "full"
    if configured_mode != actual_mode or requested_mode not in ("auto", actual_mode):
        raise RuntimeError(
            f"Training protocol mismatch: requested={requested_mode}, "
            f"configured={configured_mode}, actual={actual_mode}"
        )
    trainable_parameters = sum(parameters[name].numel() for name in trainable)
    total_parameters = sum(parameter.numel() for parameter in parameters.values())
    if not 0 < trainable_parameters <= total_parameters:
        raise RuntimeError("Training protocol requires a positive trainable-parameter count")
    if actual_mode == "lora" and (not trainable & adapters or trainable - adapters):
        raise RuntimeError("LoRA protocol requires trainable adapters and frozen base parameters")
    return {
        "version": "v12-training-protocol-v1",
        "requested_training_mode": requested_mode,
        "training_mode": actual_mode,
        "trainable_parameters": trainable_parameters,
        "total_parameters": total_parameters,
        # Fully qualified module names reflect injected adapters, not a target wish list.
        "target_modules": sorted({name.split(".lora_", 1)[0] for name in adapters}),
    }


def _snapshot_full_parameters(
    model: torch.nn.Module,
    names: Sequence[str],
    destination: Path,
) -> list[dict[str, str]]:
    """Save initial full-FT matrices one at a time to bound host memory."""
    destination.mkdir(parents=True, exist_ok=True)
    wanted = set(names)
    manifest: list[dict[str, str]] = []
    for index, (name, parameter) in enumerate(model.named_parameters()):
        if name not in wanted:
            continue
        filename = f"{index:05d}.pt"
        torch.save(parameter.detach().cpu(), destination / filename)
        manifest.append({"name": name, "file": filename})
    if len(manifest) != len(wanted):
        saved = {entry["name"] for entry in manifest}
        raise RuntimeError(f"Could not snapshot parameters: {sorted(wanted - saved)}")
    return manifest


def _save_full_delta(
    model: torch.nn.Module,
    snapshot_dir: Path,
    snapshot_manifest: Sequence[Mapping[str, str]],
    adapter_dir: Path,
    base_model: str,
    revision: str | None = None,
) -> None:
    """Write fp32 delta shards and a name-to-file manifest."""
    adapter_dir.mkdir(parents=True, exist_ok=True)
    current = dict(model.named_parameters())
    delta_manifest = []
    for index, entry in enumerate(snapshot_manifest):
        name = entry["name"]
        initial = torch.load(
            snapshot_dir / entry["file"], map_location="cpu", weights_only=True
        )
        delta = current[name].detach().to(device="cpu", dtype=torch.float32, copy=True)
        delta.sub_(initial.to(dtype=torch.float32))
        filename = f"delta-{index:05d}.pt"
        torch.save(delta, adapter_dir / filename)
        delta_manifest.append(
            {"name": name, "file": filename, "shape": list(delta.shape), "dtype": "float32"}
        )
        del initial, delta
    metadata = {"format": "v12-full-finetune-delta-v1", "base_model": base_model,
                "parameters": delta_manifest}
    if revision is not None:
        metadata["revision"] = revision
    write_json_atomic(
        adapter_dir / "delta_manifest.json",
        metadata,
    )


def autocast_context(device: str):
    device_type = torch.device(device).type
    if device_type in {"cpu", "cuda"}:
        return torch.autocast(device_type=device_type, dtype=torch.bfloat16)
    return nullcontext()


def masked_causal_loss(
    model,
    example: Mapping[str, torch.Tensor | int],
    device: str,
):
    input_ids = example["input_ids"].unsqueeze(0).to(device)
    attention_mask = example["attention_mask"].unsqueeze(0).to(device)
    labels = example["labels"].unsqueeze(0).to(device)
    with autocast_context(device):
        logits = model(
            input_ids=input_ids, attention_mask=attention_mask, use_cache=False
        ).logits
        shifted_labels = labels[:, 1:]
        loss = F.cross_entropy(
            logits[:, :-1].float().transpose(1, 2),
            shifted_labels,
            ignore_index=-100,
            reduction="sum",
        )
        n_tokens = int((shifted_labels != -100).sum())
        if n_tokens:
            loss = loss / n_tokens
    return loss, n_tokens


def validate_trajectory_tokens(tokens: Sequence[int]) -> tuple[int, ...]:
    milestones = tuple(tokens)
    if any(type(value) is not int or value <= 0 for value in milestones):
        raise ValueError("--trajectory-tokens must be positive integers")
    if tuple(sorted(set(milestones))) != milestones:
        raise ValueError("--trajectory-tokens must be strictly increasing and unique")
    return milestones


class TokenAccounting:
    """Track the fixed pool, distinct encountered examples, and repeated exposure."""

    def __init__(self, examples: Sequence[Mapping]):
        self.keys = []
        self.lengths = []
        pool = {}
        for example in examples:
            identity = example.get("example_id")
            if identity is None:
                identity = hashlib.sha256(json.dumps({
                    key: example[key].tolist() for key in ("input_ids", "labels")
                    if key in example
                }, sort_keys=True).encode()).hexdigest()
            length = int(example["attention_mask"].sum()) if "attention_mask" in example else int(example["input_ids"].numel())
            if identity in pool and pool[identity] != length:
                raise ValueError("One example identity has inconsistent token lengths")
            pool[identity] = length
            self.keys.append(identity)
            self.lengths.append(length)
        self.pool_tokens = sum(pool.values())
        self.pool_examples = len(pool)
        self.encountered: set[str] = set()
        self.processed = self.unique = 0

    def observe(self, index: int) -> None:
        self.processed += self.lengths[index]
        if self.keys[index] not in self.encountered:
            self.unique += self.lengths[index]
            self.encountered.add(self.keys[index])

    def snapshot(self) -> dict:
        return {"processed_tokens": self.processed, "seen_tokens": self.processed,
                "unique_data_tokens": self.unique,
                "unique_data_pool_tokens": self.pool_tokens,
                "unique_examples_seen": len(self.encountered),
                "unique_data_pool_examples": self.pool_examples}


@contextmanager
def preserve_training_state(model):
    """Evaluation must not change the training RNG stream or module modes."""
    modes = [(module, module.training) for module in model.modules()]
    python_state, numpy_state = random.getstate(), np.random.get_state()
    cpu_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
        for module, training in modes:
            module.training = training


def _train(
    model: torch.nn.Module,
    examples: Sequence[Mapping[str, torch.Tensor | int]],
    device: str,
    epochs: int,
    learning_rate: float,
    seed: int = SEED,
    data_sampling_seed: int | None = None,
    trajectory_tokens: Sequence[int] = (),
    trajectory_callback: Callable[[Mapping], None] | None = None,
    schedule_updates: int | None = None,
    stop_after_trajectory: bool = False,
) -> dict:
    milestones = validate_trajectory_tokens(trajectory_tokens)
    if milestones and trajectory_callback is None:
        raise ValueError("Trajectory milestones require an evaluation callback")
    if stop_after_trajectory and (not milestones or schedule_updates is None):
        raise ValueError("stop_after_trajectory requires milestones and a fixed schedule")
    if data_sampling_seed is None:
        data_sampling_seed = seed
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning rate must be finite and positive")
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("No trainable parameters")
    if not examples:
        raise ValueError("No tokenized examples with completion tokens")

    from transformers import get_cosine_schedule_with_warmup

    updates_per_epoch = math.ceil(len(examples) / EFFECTIVE_BATCH_SIZE)
    if schedule_updates is not None:
        # absolute-exposure protocol: the cosine horizon and the stop point are a fixed update budget
        # (derived from a fixed processed-token budget), identical for every pool size U.
        if schedule_updates <= 0:
            raise ValueError("schedule_updates must be positive")
        total_updates = int(schedule_updates)
        epochs = math.ceil(total_updates / updates_per_epoch)
    else:
        total_updates = epochs * updates_per_epoch
    warmup_steps = int(WARMUP_RATIO * total_updates)
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_updates,
    )
    model.to(device)
    model.train()
    start_time = time.monotonic()
    tokens_seen = 0
    completion_tokens_seen = 0
    optimizer_step = 0
    loss_curve = []
    accounting = TokenAccounting(examples)
    next_milestone = 0
    trajectory = []
    trajectory_stop = False

    for epoch in range(epochs):
        order = list(range(len(examples)))
        random.Random(data_sampling_seed + epoch).shuffle(order)
        for group_start in range(0, len(order), EFFECTIVE_BATCH_SIZE):
            group = order[group_start : group_start + EFFECTIVE_BATCH_SIZE]
            optimizer.zero_grad(set_to_none=True)
            group_losses = []
            group_completion_tokens = 0
            group_tokens = 0
            for example_index in group:
                example = examples[example_index]
                loss, n_tokens = masked_causal_loss(
                    model, example, device
                )
                if n_tokens == 0:
                    continue
                (loss / len(group)).backward()
                group_losses.append(float(loss.detach()))
                group_completion_tokens += n_tokens
                group_tokens += accounting.lengths[example_index]
                accounting.observe(example_index)
                del loss
            if not group_losses:
                continue
            optimizer.step()
            scheduler.step()
            optimizer_step += 1
            budget_exhausted = schedule_updates is not None and optimizer_step >= total_updates
            tokens_seen += group_tokens
            completion_tokens_seen += group_completion_tokens
            loss_curve.append(
                {
                    "step": optimizer_step,
                    "epoch": epoch + 1,
                    "loss": float(np.mean(group_losses)),
                    "tokens": group_tokens,
                    "completion_tokens": group_completion_tokens,
                    "tokens_seen": tokens_seen,
                    "completion_tokens_seen": completion_tokens_seen,
                    "lr": float(scheduler.get_last_lr()[0]),
                    **accounting.snapshot(),
                }
            )
            crossed = []
            while next_milestone < len(milestones) and accounting.processed >= milestones[next_milestone]:
                crossed.append(milestones[next_milestone])
                next_milestone += 1
            if crossed:
                snapshot = {**accounting.snapshot(), "updates": optimizer_step,
                            "epoch": epoch + 1, "requested_token_milestones": crossed,
                            "milestone_overshoot_tokens": [accounting.processed - t for t in crossed],
                            "completion_tokens_seen": completion_tokens_seen,
                            "lr": float(scheduler.get_last_lr()[0])}
                with preserve_training_state(model), torch.no_grad():
                    trajectory_callback(snapshot)
                trajectory.append(snapshot)
                trajectory_stop = stop_after_trajectory and next_milestone == len(milestones)
            if budget_exhausted or trajectory_stop:
                break
        if trajectory_stop or (schedule_updates is not None and optimizer_step >= total_updates):
            break

    wall_time = time.monotonic() - start_time
    return {
        "loss_curve": loss_curve,
        "tokens_seen": tokens_seen,
        **accounting.snapshot(),
        "token_accounting": TOKEN_ACCOUNTING,
        "updates": optimizer_step,
        "trajectory": trajectory,
        "trajectory_tokens_requested": list(milestones),
        "trajectory_tokens_unreached": list(milestones[next_milestone:]),
        "completion_tokens_seen": completion_tokens_seen,
        "wall_time_seconds": wall_time,
        "optimizer_steps": optimizer_step,
        "total_updates_planned": total_updates,
        "stopped_after_trajectory": trajectory_stop,
        "warmup_steps": warmup_steps,
        "effective_batch_size_sequences": EFFECTIVE_BATCH_SIZE,
        "max_len": MAX_LEN,
        "seed": seed,
        "training_seed": seed,
        "data_sampling_seed": data_sampling_seed,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "optimizer": "AdamW",
        "scheduler": "cosine",
        "warmup_ratio": WARMUP_RATIO,
        "dtype": "bfloat16",
    }


def write_json_atomic(path: Path, payload: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


# Preserve the pre-refactor internal names for existing notebook callers.
_seed_everything = seed_everything
_measure_capability_losses = measure_capability_losses
_configure_training = configure_training
_autocast_context = autocast_context
_masked_causal_loss = masked_causal_loss
_write_json_atomic = write_json_atomic


def _print_summary(eval_payload: Mapping, out: Path) -> None:
    print("\ncapability   dense      post       delta")
    print("----------  ---------  ---------  ---------")
    for capability in CAPABILITIES:
        dense = eval_payload["dense"][capability]
        post = eval_payload["post_training"][capability]
        delta = eval_payload["delta"][capability]
        print(f"{capability:<10}  {dense:9.5f}  {post:9.5f}  {delta:+9.5f}")
    print(f"artifacts: {out}")


def run_distillation(
    student: str,
    teacher: str,
    recipe: str,
    domains: Sequence[str],
    n_per_domain: int,
    epochs: int,
    device: str,
    learning_rate: float | None,
    output_suffix: str = "",
    seed: int = SEED,
    save_trajectory: bool = False,
    trajectory_tokens: Sequence[int] | None = None,
    dry_run: bool = False,
    revision: str | None = None,
    training_mode: str = "lora",
    data_seed: int | None = None,
    schedule_tokens: int | None = None,
    stop_after_trajectory: bool = False,
    append_eos: bool = False,
) -> dict:
    """Run dense evaluation, SFT, post-training evaluation, and persistence."""
    if not 0 <= seed < 2**32:
        raise ValueError("--seed must be in [0, 2**32)")
    if data_seed is not None and not 0 <= data_seed < 2**32:
        raise ValueError("--data-seed must be in [0, 2**32)")
    # Pool-sampling seed governs WHICH examples are drawn and their shuffle; independent of training seed.
    data_sampling_seed = data_seed if data_seed is not None else seed
    if epochs <= 0 or n_per_domain <= 0:
        raise ValueError("--epochs and --n-per-domain must be positive")
    if learning_rate is not None and (not math.isfinite(learning_rate) or learning_rate <= 0):
        raise ValueError("--lr must be finite and positive")
    if training_mode not in TRAINING_MODES:
        raise ValueError(f"Unknown training mode {training_mode!r}; expected {TRAINING_MODES}")
    if trajectory_tokens is not None and not save_trajectory:
        raise ValueError("--trajectory-tokens requires --save-trajectory")
    milestones = validate_trajectory_tokens(
        (DEFAULT_TRAJECTORY_TOKENS if trajectory_tokens is None else trajectory_tokens)
        if save_trajectory else ()
    )
    if stop_after_trajectory and (not milestones or schedule_tokens is None):
        raise ValueError("--stop-after-trajectory requires --save-trajectory and --schedule-tokens")
    parsed_domains = parse_domains(domains)
    model_name = require_compliant(student)
    _, requested_revision = resolve_model_and_revision(student)
    if revision is not None and requested_revision not in (None, revision):
        raise ValueError("revision conflicts with the student's @revision")
    revision = revision if revision is not None else requested_revision
    student_tag = model_output_tag(student, model_name, revision)
    output_paths = {
        mode: OUT_BASE / student_tag / make_run_name(
            teacher, recipe, n_per_domain, output_suffix, seed, training_mode=mode, data_seed=data_seed
        ) for mode in ("lora", "full")
    }
    if dry_run:
        plan = {"dry_run": True, "student": model_name, "teacher": teacher,
                "recipe": recipe, "domains": list(parsed_domains),
                "n_per_domain": n_per_domain, "epochs": epochs, "seed": seed, "append_eos": append_eos,
                "requested_training_mode": training_mode,
                "output": str(output_paths[training_mode]) if training_mode != "auto" else None,
                "output_candidates": {mode: str(path) for mode, path in output_paths.items()},
                "save_trajectory": save_trajectory,
                "trajectory_tokens": list(milestones),
                "schedule_tokens": schedule_tokens,
                "stop_after_trajectory": stop_after_trajectory,
                "trajectory_policy": "baseline plus first completed optimizer update crossing each milestone; group crossings share one snapshot; no schedule restart or extension",
                "token_accounting": TOKEN_ACCOUNTING,
                "independent_replicates": "one continuous run/seed; snapshots are dependent repeated measurements",
                "probe_seed": SEED, "probe_half": "measurement v[1::2]",
                "status": "plan only; no model, tokenizer, data, CUDA, or output writes"}
        if revision is not None:
            plan["revision"] = revision
        print(json.dumps(plan, indent=2))
        return plan
    requested_training_mode = training_mode
    selected_mode = resolve_training_mode(requested_training_mode)
    out = output_paths[selected_mode]
    run_name = out.name
    adapter_dir = out / "adapter"
    if save_trajectory and ((out / "eval.json").exists() or (out / "trajectory").exists()):
        raise FileExistsError(
            f"Trajectory requires a fresh run directory: {out}; choose a new --output-suffix"
        )
    seed_everything(seed)
    out.mkdir(parents=True, exist_ok=True)

    records, trace_counts = load_sft_records(
        teacher=teacher,
        domains=parsed_domains,
        n_per_domain=n_per_domain,
        recipe=recipe,
        seed=data_sampling_seed,
        data_seed=data_seed,
    )
    all_probes = build_probes(DEFAULT_N_PROBE, seed=SEED)
    probes = {
        capability: all_probes[capability][1::2] for capability in CAPABILITIES
    }
    if any(not probes[capability] for capability in CAPABILITIES):
        raise RuntimeError("V6 measurement-half probes must be non-empty")

    model, tokenizer = load_text_causal_lm(model_name, torch.bfloat16, revision)
    model.to(device).eval()
    dense_losses, measurement_tokens = measure_capability_losses(
        model, tokenizer, probes, device
    )
    print(f"[eval dense] {dense_losses}", flush=True)

    model, training_mode, trainable_names = configure_training(model, training_mode=selected_mode)
    training_manifest = build_training_manifest(model, requested_training_mode, training_mode)
    if training_mode != selected_mode:
        raise RuntimeError(f"Training mode changed after output path selection: {selected_mode} -> {training_mode}")
    print("[TRAINING PROTOCOL] " + json.dumps(training_manifest, sort_keys=True), flush=True)
    if revision is not None and training_mode == "lora":
        for config in model.peft_config.values():
            config.revision = revision
    selected_lr = (
        learning_rate
        if learning_rate is not None
        else (LORA_LR if training_mode == "lora" else FULL_LR)
    )
    if selected_lr <= 0:
        raise ValueError("--lr must be positive")

    tokenized = []
    tokenization_deleted = {domain: 0 for domain in parsed_domains}
    for record in records:
        example = tokenize_sft_example(
            tokenizer, record["prompt"], record["completion"], max_len=MAX_LEN, append_eos=append_eos
        )
        if int(example["n_completion_tokens"]) == 0:
            tokenization_deleted[record["domain"]] += 1
            continue
        example["example_id"] = hashlib.sha256(json.dumps(
            [record["prompt"], record["completion"]], ensure_ascii=False
        ).encode("utf-8")).hexdigest()
        tokenized.append(example)
    if not tokenized:
        raise RuntimeError("Tokenization produced no completion tokens")

    initial_accounting = TokenAccounting(tokenized).snapshot()
    trajectory_run_id = str(out.relative_to(OUT_BASE))
    eval_metadata = {
        "version": 12, "student": student, "resolved_student": model_name,
        "student_tag": student_tag, "teacher": teacher, "recipe": recipe,
        "domains": list(parsed_domains), "n_per_domain": n_per_domain,
        "output_suffix": output_suffix, "run_name": run_name,
        "seed": seed, "training_seed": seed, "data_sampling_seed": data_sampling_seed,
        "data_seed": data_seed, "schedule_tokens": schedule_tokens, "append_eos": append_eos,
        "stop_after_trajectory": stop_after_trajectory,
        "training_mode": training_mode, "probe_seed": SEED,
        "training_manifest": training_manifest,
        "data_selection": ("first n_per_domain rows; seeded initial and epoch shuffles"
                           if data_seed is None else
                           f"random sample of n_per_domain rows per domain via data_seed={data_seed}; seeded shuffles"),
        "training_benchmarks": {c: TRAINING_BENCHMARKS[c] for c in parsed_domains},
        "data_pool_sha256": hashlib.sha256(json.dumps(sorted(
            (example["example_id"], int(example["input_ids"].numel()))
            for example in tokenized
        )).encode()).hexdigest(),
        "epochs": epochs, "learning_rate": selected_lr,
        "optimizer": "AdamW", "scheduler": "cosine", "warmup_ratio": WARMUP_RATIO,
        "total_updates_planned": epochs * math.ceil(len(tokenized) / EFFECTIVE_BATCH_SIZE),
        "probe_source": "analysis.v6_capability_geometry.build_probes",
        "n_probe_requested": DEFAULT_N_PROBE,
        "probe_half": "measurement (odd indices, v[1::2])",
        "measurement_benchmarks": MEASUREMENT_BENCHMARKS,
        "measurement_samples": {c: len(probes[c]) for c in CAPABILITIES},
        "measurement_tokens": measurement_tokens, "dense": dense_losses,
        "token_accounting": TOKEN_ACCOUNTING,
        "trajectory_run_id": trajectory_run_id,
        "trajectory_tokens_requested": list(milestones),
        "independent_seed_unit": trajectory_run_id,
        "trajectory_note": "Snapshots share one run, fixed data pool, optimizer and cosine schedule; never independent seeds. Adapter snapshots are for evaluation, not optimizer resume.",
        "loss_definition": "sum_target_CE / sum_target_tokens (nats/token, legacy V12)",
    }
    if revision is not None:
        eval_metadata["revision"] = revision

    train_metadata = {
        "student": student,
        "resolved_student": model_name,
        "student_tag": student_tag,
        "teacher": teacher,
        "recipe": recipe,
        "domains": list(parsed_domains),
        "n_per_domain": n_per_domain,
        "run_name": run_name,
        "trace_counts": trace_counts,
        "tokenization_deleted_rows": tokenization_deleted,
        "training_examples": len(tokenized),
        "training_mode": training_mode,
        "requested_training_mode": requested_training_mode,
        "trainable_parameters": training_manifest["trainable_parameters"],
        "total_parameters": training_manifest["total_parameters"],
        "training_manifest": training_manifest,
        "lora": (
            {"r": 16, "alpha": 32, "dropout": 0.0,
             "target_modules": list(lora_target_modules(model))}
            if training_mode == "lora" else None
        ),
    }
    if revision is not None:
        train_metadata["revision"] = revision
    # Persist the asserted protocol even if training crashes. eval.json remains
    # the completion marker and receives this same manifest only after success.
    write_json_atomic(out / "train_log.json", {
        **train_metadata, **initial_accounting, "status": "configured",
        "seed": seed, "training_seed": seed, "data_sampling_seed": data_sampling_seed,
        "data_seed": data_seed,
    })

    def save_trajectory_snapshot(progress):
        snapshot_losses, snapshot_tokens = measure_capability_losses(model, tokenizer, probes, device)
        if snapshot_tokens != measurement_tokens:
            raise RuntimeError("Trajectory evaluation changed measurement tokens")
        destination = out / "trajectory" / f"update-{progress['updates']:08d}"
        if training_mode == "lora":
            (destination / "adapter").mkdir(parents=True, exist_ok=True)
            model.save_pretrained(destination / "adapter")
        else:
            _save_full_delta(model, snapshot_dir, snapshot_manifest,
                             destination / "adapter", model_name, revision)
        write_json_atomic(destination / "eval.json", {
            **eval_metadata, **progress, "snapshot_kind": "trajectory",
            "is_independent_seed": False, "post_training": snapshot_losses,
            "capability_losses": snapshot_losses,
            "delta": {c: snapshot_losses[c] - dense_losses[c] for c in CAPABILITIES},
        })
        print(f"[trajectory] updates={progress['updates']} processed_tokens={progress['processed_tokens']} losses={snapshot_losses}", flush=True)

    snapshot_manifest: list[dict[str, str]] = []
    with tempfile.TemporaryDirectory(prefix=".full-snapshot-", dir=out) as temporary:
        snapshot_dir = Path(temporary)
        if training_mode == "full":
            snapshot_manifest = _snapshot_full_parameters(
                model, trainable_names, snapshot_dir
            )
        if save_trajectory:
            write_json_atomic(out / "trajectory" / "update-00000000" / "eval.json", {
                **eval_metadata, **initial_accounting,
                "updates": 0, "completion_tokens_seen": 0,
                "requested_token_milestones": [], "snapshot_kind": "trajectory_baseline",
                "is_independent_seed": False, "post_training": dense_losses,
                "capability_losses": dense_losses,
                "delta": {c: 0.0 for c in CAPABILITIES},
                "checkpoint": "unmodified resolved_student (no adapter)",
            })
        schedule_updates = None
        if schedule_tokens is not None:
            if schedule_tokens <= 0:
                raise ValueError("--schedule-tokens must be positive")
            _pool_processed = sum(int(e["input_ids"].numel()) for e in tokenized)
            _upe = math.ceil(len(tokenized) / EFFECTIVE_BATCH_SIZE)
            schedule_updates = math.ceil(schedule_tokens / (_pool_processed / _upe))
        train_log = _train(
            model=model,
            examples=tokenized,
            device=device,
            epochs=epochs,
            learning_rate=selected_lr,
            seed=seed,
            data_sampling_seed=data_sampling_seed,
            schedule_updates=schedule_updates,
            trajectory_tokens=milestones,
            trajectory_callback=save_trajectory_snapshot if save_trajectory else None,
            stop_after_trajectory=stop_after_trajectory,
        )
        if training_mode == "lora":
            adapter_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(adapter_dir)
        else:
            _save_full_delta(
                model,
                snapshot_dir=snapshot_dir,
                snapshot_manifest=snapshot_manifest,
                adapter_dir=adapter_dir,
                base_model=model_name,
                revision=revision,
            )

    train_log.update(train_metadata)
    train_log["status"] = "trained"
    write_json_atomic(out / "train_log.json", train_log)

    model.eval()
    post_losses, post_measurement_tokens = measure_capability_losses(
        model, tokenizer, probes, device
    )
    if post_measurement_tokens != measurement_tokens:
        raise RuntimeError("Dense and post-training evaluations used different tokens")
    delta = {
        capability: post_losses[capability] - dense_losses[capability]
        for capability in CAPABILITIES
    }
    eval_payload = {
        **eval_metadata,
        "post_training": post_losses,
        "delta": delta,
        "capability_losses": post_losses,
        "snapshot_kind": "final",
        **{key: train_log[key] for key in initial_accounting},
        "updates": train_log["updates"],
        "completion_tokens_seen": train_log["completion_tokens_seen"],
        "save_trajectory": save_trajectory,
        "trajectory_tokens_unreached": train_log["trajectory_tokens_unreached"],
    }
    # eval.json is written last and atomically; its existence is the sweep's
    # completion marker.
    write_json_atomic(out / "eval.json", eval_payload)
    _print_summary(eval_payload, out)

    del model, tokenizer, tokenized
    gc.collect()
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return eval_payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--student",
        default="gemma3-270m",
        help="model-registry tag or raw Hugging Face model id, optionally @revision",
    )
    parser.add_argument("--teacher", required=True, choices=TEACHERS)
    parser.add_argument("--recipe", required=True, choices=RECIPES)
    parser.add_argument("--training-mode", choices=TRAINING_MODES, default="lora",
                        help="lora requires peft (default); full intentionally trains language weight matrices; auto allows a loudly logged fallback")
    parser.add_argument(
        "--domains",
        type=parse_domains,
        default=parse_domains("math,qa,code"),
        help="comma-separated training domains (default: math,qa,code)",
    )
    parser.add_argument("--n-per-domain", type=int, default=600)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--schedule-tokens", type=int, default=None,
                        help="absolute-exposure protocol: cosine horizon AND stop point = this many processed tokens "
                             "(converted to an update budget from the pool); --epochs is then only an upper bound")
    parser.add_argument("--data-seed", type=int, default=None,
                        help="pool-SAMPLING seed (which examples are drawn), independent of --seed; "
                             "default None = deterministic first-n_per_domain pool")
    parser.add_argument("--seed", type=int, default=SEED,
                        help="training and data-shuffle seed (default: 0; probes fixed)")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="override learning rate (defaults: LoRA 1e-4, full 1e-5)",
    )
    parser.add_argument("--output-suffix", default="")
    parser.add_argument("--save-trajectory", action="store_true",
                        help="save evals and adapters during one continuous training run")
    parser.add_argument("--trajectory-tokens", type=int, nargs="+", default=None,
                        help="strictly increasing processed-input-token milestones (not unique tokens)")
    parser.add_argument("--stop-after-trajectory", action="store_true",
                        help="stop after the last requested checkpoint, retaining the full --schedule-tokens cosine/warmup horizon")
    parser.add_argument("--append-eos", action="store_true",
                        help="append the tokenizer's end-of-sequence id to every training target (P3 control)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the protocol without data/model loads or writes")
    args = parser.parse_args()

    run_distillation(
        student=args.student,
        teacher=args.teacher,
        recipe=args.recipe,
        training_mode=args.training_mode,
        domains=args.domains,
        n_per_domain=args.n_per_domain,
        epochs=args.epochs,
        device=args.device,
        learning_rate=args.lr,
        output_suffix=args.output_suffix,
        seed=args.seed,
        data_seed=args.data_seed,
        schedule_tokens=args.schedule_tokens,
        save_trajectory=args.save_trajectory,
        trajectory_tokens=args.trajectory_tokens,
        stop_after_trajectory=args.stop_after_trajectory,
        append_eos=args.append_eos,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
