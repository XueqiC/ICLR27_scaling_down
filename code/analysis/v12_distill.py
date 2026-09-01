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
``results/v12-distill/<student_tag>/<teacher>_<recipe>_<n>/``.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import random
import re
import tempfile
import time
import warnings
from contextlib import nullcontext
from pathlib import Path
from typing import Iterable, Mapping, Sequence

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
    )
except ImportError:  # direct execution: python analysis/v12_distill.py
    from v6_capability_geometry import (
        build_probes,
        completion_loss,
        language_weight_parameters,
        load_text_causal_lm,
        model_output_tag,
        require_compliant,
    )


ROOT = Path(__file__).resolve().parents[1]
TRACE_BASE = ROOT / "results/traces-pilot"
OUT_BASE = ROOT / "results/v12-distill"

TEACHERS = ("gpt-5.6-luna", "claude-sonnet-4-6")
DOMAINS = ("math", "qa", "code")
CAPABILITIES = ("math", "code", "qa")
RECIPES = ("full", "answer_only", "no_code_fence")
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
) -> str:
    """Return the stable run directory name shared with the sweep driver."""
    base = f"{teacher}_{recipe}_{n_per_domain}"
    if not output_suffix:
        return base
    safe_suffix = re.sub(r"[^A-Za-z0-9._-]+", "--", output_suffix).strip(".-_")
    if not safe_suffix:
        raise ValueError("--output-suffix must contain a path-safe character")
    return f"{base}_{safe_suffix}"


def load_sft_records(
    teacher: str,
    domains: Sequence[str],
    n_per_domain: int,
    recipe: str,
    trace_base: Path = TRACE_BASE,
) -> tuple[list[dict[str, str]], dict[str, dict[str, int]]]:
    """Load exactly ``n_per_domain`` source rows and apply the SFT recipe.

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
        selected: list[Mapping] = []
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if len(selected) == n_per_domain:
                    break
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {path}:{line_number}: {exc}") from exc
                selected.append(row)
        if len(selected) < n_per_domain:
            raise ValueError(
                f"{path} contains only {len(selected)} non-empty rows; "
                f"requested {n_per_domain}"
            )

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
    random.Random(SEED).shuffle(records)
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
) -> dict[str, torch.Tensor | int]:
    """Tokenize one pair and mask every prompt label with ``-100``.

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
            max_length=half_len,
            add_special_tokens=False,
        )
    )
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


def configure_training(
    model,
    *,
    allow_full_fallback: bool = True,
) -> tuple[torch.nn.Module, str, list[str]]:
    """Configure LoRA when PEFT imports, otherwise full matrix fine-tuning."""
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:
        if not allow_full_fallback:
            raise RuntimeError("peft is required for LoRA training") from exc
        warnings.warn(
            "peft is unavailable; falling back to full fine-tuning of all "
            "language weight matrices",
            RuntimeWarning,
            stacklevel=2,
        )
        model.requires_grad_(False)
        parameters = language_weight_parameters(model)
        if not parameters:
            raise ValueError("No language weight matrices found for fine-tuning")
        for _, parameter in parameters:
            parameter.requires_grad_(True)
        return model, "full", [name for name, _ in parameters]

    config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.0,
        bias="none",
        target_modules=list(LORA_TARGET_MODULES),
    )
    model = get_peft_model(model, config)
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError(
            "PEFT found no LoRA targets; expected attention/MLP projections "
            f"named {', '.join(LORA_TARGET_MODULES)}"
        )
    return model, "lora", trainable


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
        delta = current[name].detach().to(device="cpu", dtype=torch.float32)
        delta.sub_(initial.to(dtype=torch.float32))
        filename = f"delta-{index:05d}.pt"
        torch.save(delta, adapter_dir / filename)
        delta_manifest.append(
            {"name": name, "file": filename, "shape": list(delta.shape), "dtype": "float32"}
        )
        del initial, delta
    write_json_atomic(
        adapter_dir / "delta_manifest.json",
        {"format": "v12-full-finetune-delta-v1", "base_model": base_model, "parameters": delta_manifest},
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


def _train(
    model: torch.nn.Module,
    examples: Sequence[Mapping[str, torch.Tensor | int]],
    device: str,
    epochs: int,
    learning_rate: float,
) -> dict:
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if learning_rate <= 0:
        raise ValueError("learning rate must be positive")
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("No trainable parameters")
    if not examples:
        raise ValueError("No tokenized examples with completion tokens")

    from transformers import get_cosine_schedule_with_warmup

    updates_per_epoch = math.ceil(len(examples) / EFFECTIVE_BATCH_SIZE)
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

    for epoch in range(epochs):
        order = list(range(len(examples)))
        random.Random(SEED + epoch).shuffle(order)
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
                group_tokens += int(example["input_ids"].numel())
                del loss
            if not group_losses:
                continue
            optimizer.step()
            scheduler.step()
            optimizer_step += 1
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
                }
            )

    wall_time = time.monotonic() - start_time
    return {
        "loss_curve": loss_curve,
        "tokens_seen": tokens_seen,
        "completion_tokens_seen": completion_tokens_seen,
        "wall_time_seconds": wall_time,
        "optimizer_steps": optimizer_step,
        "total_updates_planned": total_updates,
        "warmup_steps": warmup_steps,
        "effective_batch_size_sequences": EFFECTIVE_BATCH_SIZE,
        "max_len": MAX_LEN,
        "seed": SEED,
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
) -> dict:
    """Run dense evaluation, SFT, post-training evaluation, and persistence."""
    seed_everything()
    parsed_domains = parse_domains(domains)
    model_name = require_compliant(student)
    student_tag = model_output_tag(student, model_name)
    run_name = make_run_name(teacher, recipe, n_per_domain, output_suffix)
    out = OUT_BASE / student_tag / run_name
    adapter_dir = out / "adapter"
    out.mkdir(parents=True, exist_ok=True)

    records, trace_counts = load_sft_records(
        teacher=teacher,
        domains=parsed_domains,
        n_per_domain=n_per_domain,
        recipe=recipe,
    )
    all_probes = build_probes(DEFAULT_N_PROBE, seed=SEED)
    probes = {
        capability: all_probes[capability][1::2] for capability in CAPABILITIES
    }
    if any(not probes[capability] for capability in CAPABILITIES):
        raise RuntimeError("V6 measurement-half probes must be non-empty")

    model, tokenizer = load_text_causal_lm(model_name, torch.bfloat16)
    model.to(device).eval()
    dense_losses, measurement_tokens = measure_capability_losses(
        model, tokenizer, probes, device
    )
    print(f"[eval dense] {dense_losses}", flush=True)

    model, training_mode, trainable_names = configure_training(model)
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
            tokenizer, record["prompt"], record["completion"], max_len=MAX_LEN
        )
        if int(example["n_completion_tokens"]) == 0:
            tokenization_deleted[record["domain"]] += 1
            continue
        tokenized.append(example)
    if not tokenized:
        raise RuntimeError("Tokenization produced no completion tokens")

    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    snapshot_manifest: list[dict[str, str]] = []
    with tempfile.TemporaryDirectory(prefix=".full-snapshot-", dir=out) as temporary:
        snapshot_dir = Path(temporary)
        if training_mode == "full":
            snapshot_manifest = _snapshot_full_parameters(
                model, trainable_names, snapshot_dir
            )
        train_log = _train(
            model=model,
            examples=tokenized,
            device=device,
            epochs=epochs,
            learning_rate=selected_lr,
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
            )

    train_log.update(
        {
            "student": student,
            "resolved_student": model_name,
            "student_tag": student_tag,
            "teacher": teacher,
            "recipe": recipe,
            "domains": list(parsed_domains),
            "n_per_domain": n_per_domain,
            "trace_counts": trace_counts,
            "tokenization_deleted_rows": tokenization_deleted,
            "training_examples": len(tokenized),
            "training_mode": training_mode,
            "trainable_parameters": trainable_parameters,
            "total_parameters": total_parameters,
            "lora": (
                {
                    "r": 16,
                    "alpha": 32,
                    "dropout": 0.0,
                    "target_modules": list(LORA_TARGET_MODULES),
                }
                if training_mode == "lora"
                else None
            ),
        }
    )
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
        "version": 12,
        "student": student,
        "resolved_student": model_name,
        "student_tag": student_tag,
        "teacher": teacher,
        "recipe": recipe,
        "domains": list(parsed_domains),
        "n_per_domain": n_per_domain,
        "output_suffix": output_suffix,
        "run_name": run_name,
        "training_mode": training_mode,
        "training_benchmarks": {
            capability: TRAINING_BENCHMARKS[capability]
            for capability in parsed_domains
        },
        "measurement_benchmarks": MEASUREMENT_BENCHMARKS,
        "probe_source": "analysis.v6_capability_geometry.build_probes",
        "probe_seed": SEED,
        "n_probe_requested": DEFAULT_N_PROBE,
        "probe_half": "measurement (odd indices, v[1::2])",
        "measurement_samples": {
            capability: len(probes[capability]) for capability in CAPABILITIES
        },
        "measurement_tokens": measurement_tokens,
        "dense": dense_losses,
        "post_training": post_losses,
        "delta": delta,
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
        help="model-registry tag or raw Hugging Face model id",
    )
    parser.add_argument("--teacher", required=True, choices=TEACHERS)
    parser.add_argument("--recipe", required=True, choices=RECIPES)
    parser.add_argument(
        "--domains",
        type=parse_domains,
        default=parse_domains("math,qa,code"),
        help="comma-separated training domains (default: math,qa,code)",
    )
    parser.add_argument("--n-per-domain", type=int, default=600)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="override learning rate (defaults: LoRA 1e-4, full 1e-5)",
    )
    parser.add_argument("--output-suffix", default="")
    args = parser.parse_args()

    run_distillation(
        student=args.student,
        teacher=args.teacher,
        recipe=args.recipe,
        domains=args.domains,
        n_per_domain=args.n_per_domain,
        epochs=args.epochs,
        device=args.device,
        learning_rate=args.lr,
        output_suffix=args.output_suffix,
    )


if __name__ == "__main__":
    main()
