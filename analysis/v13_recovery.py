#!/usr/bin/env python3
"""V13: fit per-capability recovery laws after model compression.

The model is evaluated dense, damaged once by pruning or fake quantization,
then trained continuously with LoRA on generic C4 documents or a fixed mix of
teacher traces.  The odd-indexed V6 measurement probes are evaluated whenever
the cumulative recovery-token count crosses a requested budget.

Example:
  python3 analysis/v13_recovery.py --model gemma3-270m \
      --compression prune --density 0.5 --recovery-source c4 \
      --budgets 0.5M,2M,8M,32M --device cuda:0
"""
from __future__ import annotations

import argparse
import gc
import math
import re
import time
import warnings
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

import numpy as np
import torch

try:
    from .v6_capability_geometry import (
        _sample_abs_weights,
        build_probes,
        completion_loss,
        language_weight_parameters,
        load_text_causal_lm,
        model_output_tag,
        require_compliant,
    )
    from .v10_quantization import fake_quantize_per_output_channel
    from .v12_distill import (
        CAPABILITIES,
        EFFECTIVE_BATCH_SIZE,
        LORA_LR,
        LORA_TARGET_MODULES,
        TRACE_BASE,
        WARMUP_RATIO,
        configure_training,
        load_mixed_trace_records,
        masked_causal_loss,
        measure_capability_losses,
        seed_everything,
        tokenize_sft_example,
        tokenize_text_example,
        write_json_atomic,
    )
except ImportError:  # direct execution: python analysis/v13_recovery.py
    from v6_capability_geometry import (
        _sample_abs_weights,
        build_probes,
        completion_loss,
        language_weight_parameters,
        load_text_causal_lm,
        model_output_tag,
        require_compliant,
    )
    from v10_quantization import fake_quantize_per_output_channel
    from v12_distill import (
        CAPABILITIES,
        EFFECTIVE_BATCH_SIZE,
        LORA_LR,
        LORA_TARGET_MODULES,
        TRACE_BASE,
        WARMUP_RATIO,
        configure_training,
        load_mixed_trace_records,
        masked_causal_loss,
        measure_capability_losses,
        seed_everything,
        tokenize_sft_example,
        tokenize_text_example,
        write_json_atomic,
    )


ROOT = Path(__file__).resolve().parents[1]
OUT_BASE = ROOT / "results/v13-recovery"
DEFAULT_BUDGETS = "0.5M,2M,8M,32M"
DEFAULT_MAX_LEN = 1024
C4_DATASET = "allenai/c4"
C4_CONFIG = "en"
C4_SPLIT = "train"
C4_SHUFFLE_BUFFER = 10_000

_BUDGET_RE = re.compile(
    r"^(?P<number>(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
    r"(?P<suffix>[kKmMbB]?)$"
)
_BUDGET_MULTIPLIERS = {
    "": Decimal(1),
    "k": Decimal(1_000),
    "m": Decimal(1_000_000),
    "b": Decimal(1_000_000_000),
}


def parse_budgets(value: str | Sequence[int]) -> list[int]:
    """Parse a strictly increasing recovery-token schedule.

    String values are comma-separated and accept case-insensitive decimal
    ``K``, ``M``, and ``B`` suffixes, for example ``0.5M,2M``.
    """
    if isinstance(value, str):
        raw_parts = [part.strip() for part in value.split(",")]
        if not raw_parts or any(not part for part in raw_parts):
            raise argparse.ArgumentTypeError(
                "--budgets must be a comma-separated list of token counts"
            )
        budgets: list[int] = []
        for part in raw_parts:
            match = _BUDGET_RE.fullmatch(part)
            if match is None:
                raise argparse.ArgumentTypeError(
                    f"invalid recovery budget {part!r}; use values such as 500K or 2M"
                )
            try:
                scaled = Decimal(match.group("number")) * _BUDGET_MULTIPLIERS[
                    match.group("suffix").lower()
                ]
            except InvalidOperation as exc:
                raise argparse.ArgumentTypeError(
                    f"invalid recovery budget {part!r}"
                ) from exc
            integral = scaled.to_integral_value()
            if scaled != integral:
                raise argparse.ArgumentTypeError(
                    f"recovery budget {part!r} is not an integer token count"
                )
            budgets.append(int(integral))
    else:
        try:
            budgets = [int(item) for item in value]
        except (TypeError, ValueError) as exc:
            raise argparse.ArgumentTypeError(
                "--budgets must contain integer token counts"
            ) from exc

    if not budgets:
        raise argparse.ArgumentTypeError("--budgets must contain at least one value")
    if any(budget <= 0 for budget in budgets):
        raise argparse.ArgumentTypeError("all --budgets values must be positive")
    if any(right <= left for left, right in zip(budgets, budgets[1:])):
        raise argparse.ArgumentTypeError(
            "--budgets must be unique and strictly increasing"
        )
    return budgets


# Descriptive alias for callers that prefer the full helper name.
parse_budget_schedule = parse_budgets


def _empty_fit(status: str, n_points: int) -> dict[str, float | int | str | None]:
    return {
        "status": status,
        "n_points": n_points,
        "L_inf": None,
        "B": None,
        "beta": None,
        "r2": None,
        "residual_vs_dense": None,
    }


def fit_recovery_power_law(
    tokens: Sequence[int | float],
    losses: Sequence[int | float],
    dense_loss: float | None = None,
) -> dict[str, float | int | str | None]:
    """Fit ``L(D) = L_inf + B * D**(-beta)`` with bounded restarts.

    The optimization uses ``D / min(D)`` internally to avoid the extreme
    coefficient scaling caused by million-token x values.  ``B`` is converted
    back to the original token units before it is returned.
    """
    x = np.asarray(tokens, dtype=np.float64)
    y = np.asarray(losses, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape:
        raise ValueError("tokens and losses must be one-dimensional and equal length")

    valid = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y >= 0)
    x = x[valid]
    y = y[valid]
    if x.size:
        order = np.argsort(x)
        x, y = x[order], y[order]
        unique_x, inverse = np.unique(x, return_inverse=True)
        if unique_x.size != x.size:
            sums = np.zeros(unique_x.size, dtype=np.float64)
            counts = np.zeros(unique_x.size, dtype=np.int64)
            np.add.at(sums, inverse, y)
            np.add.at(counts, inverse, 1)
            x, y = unique_x, sums / counts

    n_points = int(x.size)
    if n_points < 3:
        return _empty_fit("too_few_points", n_points)

    from scipy.optimize import OptimizeWarning, curve_fit

    x_scale = float(x[0])
    normalized_x = x / x_scale

    def model(normalized_tokens, asymptote, amplitude, exponent):
        return asymptote + amplitude * np.power(normalized_tokens, -exponent)

    y_min = float(np.min(y))
    y_max = float(np.max(y))
    y_span = max(y_max - y_min, abs(y_max) * 1e-3, 1e-6)
    lower = np.array([0.0, 0.0, 1e-4], dtype=np.float64)
    upper = np.array(
        [max(y_max + 10.0 * y_span, 1.0), max(100.0 * y_span, 1.0), 5.0],
        dtype=np.float64,
    )
    asymptote_guesses = {
        0.0,
        max(0.0, y_min - y_span),
        max(0.0, y_min - 0.25 * y_span),
        max(0.0, 0.95 * y_min),
    }
    exponent_guesses = (0.1, 0.25, 0.5, 0.8, 1.2, 2.0)
    best: tuple[float, np.ndarray] | None = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", OptimizeWarning)
        for asymptote_guess in asymptote_guesses:
            amplitude_guess = max(float(y[0]) - asymptote_guess, y_span * 0.1)
            initial_asymptote = min(asymptote_guess, upper[0] * (1.0 - 1e-9))
            initial_amplitude = min(amplitude_guess, upper[1] * (1.0 - 1e-9))
            for exponent_guess in exponent_guesses:
                try:
                    parameters, _ = curve_fit(
                        model,
                        normalized_x,
                        y,
                        p0=(initial_asymptote, initial_amplitude, exponent_guess),
                        bounds=(lower, upper),
                        maxfev=50_000,
                    )
                except (RuntimeError, ValueError, FloatingPointError):
                    continue
                predictions = model(normalized_x, *parameters)
                residual_sum = float(np.sum(np.square(y - predictions)))
                if best is None or residual_sum < best[0]:
                    best = residual_sum, parameters

    if best is None:
        return _empty_fit("fit_failed", n_points)

    residual_sum, parameters = best
    asymptote, normalized_amplitude, exponent = map(float, parameters)
    try:
        coefficient = normalized_amplitude * math.pow(x_scale, exponent)
    except OverflowError:
        coefficient = math.inf
    total_sum = float(np.sum(np.square(y - np.mean(y))))
    if total_sum <= np.finfo(np.float64).eps:
        r2 = 1.0 if residual_sum <= np.finfo(np.float64).eps else 0.0
    else:
        r2 = 1.0 - residual_sum / total_sum
    residual_vs_dense = (
        asymptote - float(dense_loss)
        if dense_loss is not None and np.isfinite(dense_loss)
        else None
    )
    return {
        "status": "ok",
        "n_points": n_points,
        "L_inf": asymptote,
        "B": float(coefficient),
        "beta": exponent,
        "r2": float(r2),
        "residual_vs_dense": residual_vs_dense,
    }


# Short alias matching the terminology used in analysis notebooks.
fit_power_law = fit_recovery_power_law


def select_recovery_source(
    recovery_source: str,
    *,
    trace_base: Path = TRACE_BASE,
    seed: int = 0,
) -> tuple[Iterable[Mapping], dict]:
    """Return recovery rows and JSON-serializable source metadata."""
    if recovery_source == "traces":
        records, mix = load_mixed_trace_records(trace_base=trace_base, seed=seed)
        domains = sorted(
            {domain for teacher_counts in mix.values() for domain in teacher_counts}
        )
        metadata = {
            "name": "traces",
            "trace_base": str(Path(trace_base)),
            "streaming": False,
            "shuffled": True,
            "seed": seed,
            "n_records": len(records),
            "mix": mix,
            "mix_shape": {
                "teachers": len(mix),
                "domains": len(domains),
                "teacher_domain_cells": sum(len(counts) for counts in mix.values()),
            },
        }
        return records, metadata
    if recovery_source == "c4":
        from datasets import load_dataset

        dataset = load_dataset(
            C4_DATASET,
            C4_CONFIG,
            split=C4_SPLIT,
            streaming=True,
        )
        dataset = dataset.shuffle(seed=seed, buffer_size=C4_SHUFFLE_BUFFER)
        metadata = {
            "name": "c4",
            "dataset": C4_DATASET,
            "config": C4_CONFIG,
            "split": C4_SPLIT,
            "streaming": True,
            "shuffled": True,
            "shuffle_buffer": C4_SHUFFLE_BUFFER,
            "seed": seed,
        }
        return dataset, metadata
    raise ValueError("recovery_source must be 'c4' or 'traces'")


def _tokenized_recovery_examples(
    rows: Iterable[Mapping],
    recovery_source: str,
    tokenizer,
    max_len: int,
) -> Iterator[Mapping[str, torch.Tensor | int]]:
    if recovery_source == "traces":
        records = list(rows)
        if not records:
            raise RuntimeError("Trace recovery source contains no records")
        while True:
            for row in records:
                example = tokenize_sft_example(
                    tokenizer,
                    str(row["prompt"]),
                    str(row["completion"]),
                    max_len=max_len,
                )
                if int(example["n_completion_tokens"]) > 0:
                    yield example
    else:
        for row in rows:
            text = row.get("text")
            if not isinstance(text, str) or not text.strip():
                continue
            example = tokenize_text_example(tokenizer, text, max_len=max_len)
            if int(example["n_document_tokens"]) > 0:
                yield example


def _compression_run_name(
    compression: str,
    density: float,
    bits: int,
    recovery_source: str,
) -> str:
    if compression == "prune":
        strength = f"{density:.8g}"
    elif compression == "quant":
        strength = str(bits)
    else:
        strength = "control"
    return f"{compression}_{strength}_{recovery_source}"


def _apply_permanent_compression(
    model: torch.nn.Module,
    compression: str,
    density: float,
    bits: int,
    seed: int,
) -> dict:
    parameters = language_weight_parameters(model)
    if not parameters:
        raise ValueError("No language weight matrices found for compression")
    total_parameters = sum(parameter.numel() for _, parameter in parameters)
    metadata: dict = {
        "type": compression,
        "scope": "all language weight matrices",
        "parameters_in_scope": total_parameters,
        "permanent_base_weights": True,
    }
    if compression == "none":
        metadata["strength"] = "control"
        return metadata
    if compression == "prune":
        if not 0.0 < density <= 1.0:
            raise ValueError("--density must be in (0, 1]")
        sampled_weights = _sample_abs_weights(parameters, seed=seed)
        threshold = float(np.quantile(sampled_weights, 1.0 - density))
        del sampled_weights
        retained = 0
        with torch.no_grad():
            for _, parameter in parameters:
                mask = parameter.abs() > threshold
                retained += int(mask.sum())
                parameter.mul_(mask)
        metadata.update(
            {
                "density_requested": density,
                "density_actual": retained / total_parameters,
                "magnitude_threshold": threshold,
                "retained_parameters": retained,
                "method": "global magnitude pruning",
            }
        )
        return metadata
    if compression == "quant":
        if bits < 2:
            raise ValueError("--bits must be at least 2")
        with torch.no_grad():
            for _, parameter in parameters:
                parameter.copy_(fake_quantize_per_output_channel(parameter, bits))
        metadata.update(
            {
                "bits": bits,
                "method": "symmetric per-output-channel RTN fake quantization",
            }
        )
        return metadata
    raise ValueError("compression must be 'prune', 'quant', or 'none'")


def _cosine_warmup_factor(tokens_seen: int, total_tokens: int) -> float:
    progress = min(max(tokens_seen / total_tokens, 0.0), 1.0)
    if WARMUP_RATIO > 0 and progress < WARMUP_RATIO:
        return progress / WARMUP_RATIO
    decay_progress = (progress - WARMUP_RATIO) / max(1.0 - WARMUP_RATIO, 1e-12)
    return 0.5 * (1.0 + math.cos(math.pi * decay_progress))


def _set_learning_rate(
    optimizer: torch.optim.Optimizer,
    base_learning_rate: float,
    tokens_seen: int,
    total_tokens: int,
) -> float:
    learning_rate = base_learning_rate * _cosine_warmup_factor(
        tokens_seen, total_tokens
    )
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
    return learning_rate


def _train_recovery_ladder(
    model: torch.nn.Module,
    examples: Iterator[Mapping[str, torch.Tensor | int]],
    tokenizer,
    probes: Mapping[str, Sequence[Mapping[str, str]]],
    device: str,
    budgets: Sequence[int],
    max_len: int,
    measurement_tokens: Mapping[str, int],
) -> tuple[list[dict], list[dict]]:
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError("No trainable LoRA parameters")
    optimizer = torch.optim.AdamW(trainable, lr=LORA_LR)
    maximum_budget = int(budgets[-1])
    tokens_seen = 0
    optimizer_step = 0
    training_curve: list[dict] = []
    checkpoints: list[dict] = []
    start_time = time.monotonic()
    model.to(device).train()

    for requested_budget in budgets:
        while tokens_seen < requested_budget:
            group = []
            try:
                for _ in range(EFFECTIVE_BATCH_SIZE):
                    group.append(next(examples))
            except StopIteration as exc:
                raise RuntimeError(
                    "Recovery dataset was exhausted before the largest token budget"
                ) from exc

            learning_rate = _set_learning_rate(
                optimizer, LORA_LR, tokens_seen, maximum_budget
            )
            optimizer.zero_grad(set_to_none=True)
            group_losses: list[float] = []
            group_tokens = 0
            for example in group:
                loss, n_tokens = masked_causal_loss(model, example, device)
                if n_tokens <= 0:
                    continue
                (loss / EFFECTIVE_BATCH_SIZE).backward()
                group_losses.append(float(loss.detach()))
                group_tokens += n_tokens
                del loss
            if len(group_losses) != EFFECTIVE_BATCH_SIZE or group_tokens <= 0:
                raise RuntimeError(
                    "Recovery example stream yielded an incomplete effective batch"
                )
            optimizer.step()
            optimizer_step += 1
            tokens_seen += group_tokens
            training_curve.append(
                {
                    "step": optimizer_step,
                    "loss": float(np.mean(group_losses)),
                    "tokens": group_tokens,
                    "tokens_seen": tokens_seen,
                    "lr": learning_rate,
                }
            )

        losses, checkpoint_measurement_tokens = measure_capability_losses(
            model,
            tokenizer,
            probes,
            device,
            max_len=max_len,
            loss_function=completion_loss,
        )
        if dict(checkpoint_measurement_tokens) != dict(measurement_tokens):
            raise RuntimeError("Recovery evaluations used different measurement tokens")
        checkpoint = {
            "budget_requested": int(requested_budget),
            "tokens_seen": tokens_seen,
            "optimizer_step": optimizer_step,
            "losses": losses,
            "wall_time_seconds": time.monotonic() - start_time,
        }
        checkpoints.append(checkpoint)
        print(
            f"[recovery] requested={requested_budget:,} seen={tokens_seen:,}: {losses}",
            flush=True,
        )
        model.train()

    return checkpoints, training_curve


def _curves_from_checkpoints(
    checkpoints: Sequence[Mapping],
    dense: Mapping[str, float],
    damaged: Mapping[str, float],
) -> dict[str, list[dict]]:
    curves: dict[str, list[dict]] = {capability: [] for capability in CAPABILITIES}
    for checkpoint in checkpoints:
        for capability in CAPABILITIES:
            loss = float(checkpoint["losses"][capability])
            curves[capability].append(
                {
                    "budget_requested": int(checkpoint["budget_requested"]),
                    "tokens_seen": int(checkpoint["tokens_seen"]),
                    "loss": loss,
                    "delta_vs_damaged": loss - float(damaged[capability]),
                    "delta_vs_dense": loss - float(dense[capability]),
                }
            )
    return curves


def _format_optional(value, digits: int = 6) -> str:
    if value is None:
        return "n/a"
    try:
        if not np.isfinite(value):
            return "n/a"
    except TypeError:
        return str(value)
    return f"{value:.{digits}g}"


def write_report(out: Path, payload: Mapping) -> None:
    dense = payload["anchors"]["dense"]
    damaged = payload["anchors"]["damaged"]
    lines = [
        f"# V13 recovery report — {payload['resolved_model']}",
        "",
        f"Compression: `{payload['run_name']}`. Recovery checkpoints continue "
        "from one LoRA run; reported token counts are completion tokens for "
        "traces and causal document-target tokens for C4.",
        "",
        "## Anchors",
        "",
        "| capability | dense loss | damaged loss | initial damage |",
        "|---|---:|---:|---:|",
    ]
    for capability in CAPABILITIES:
        initial_damage = damaged[capability] - dense[capability]
        lines.append(
            f"| {capability} | {dense[capability]:.6f} | "
            f"{damaged[capability]:.6f} | {initial_damage:+.6f} |"
        )

    lines.extend(
        [
            "",
            "## Recovery ladder",
            "",
            "| requested tokens | tokens seen | capability | loss | "
            "delta vs damaged | delta vs dense |",
            "|---:|---:|---|---:|---:|---:|",
        ]
    )
    for checkpoint in payload["ladder"]:
        for capability in CAPABILITIES:
            loss = checkpoint["losses"][capability]
            lines.append(
                f"| {checkpoint['budget_requested']:,} | "
                f"{checkpoint['tokens_seen']:,} | {capability} | {loss:.6f} | "
                f"{loss - damaged[capability]:+.6f} | "
                f"{loss - dense[capability]:+.6f} |"
            )

    lines.extend(
        [
            "",
            "## Recovery-law fits",
            "",
            "Fit: `L_c(D_R) = L_c^inf + B'_c D_R^(-beta'_c)`.",
            "",
            "| capability | points | status | L_inf | B | beta | R^2 | "
            "L_inf - L_dense |",
            "|---|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for capability in CAPABILITIES:
        fit = payload["fits"][capability]
        lines.append(
            f"| {capability} | {fit['n_points']} | {fit['status']} | "
            f"{_format_optional(fit['L_inf'])} | {_format_optional(fit['B'])} | "
            f"{_format_optional(fit['beta'])} | {_format_optional(fit['r2'])} | "
            f"{_format_optional(fit['residual_vs_dense'])} |"
        )
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _print_summary(payload: Mapping, out: Path) -> None:
    dense = payload["anchors"]["dense"]
    damaged = payload["anchors"]["damaged"]
    last = payload["ladder"][-1]["losses"]
    print("\ncapability   dense    damaged   final    beta   residual")
    print("----------  -------  --------  -------  ------  --------")
    for capability in CAPABILITIES:
        fit = payload["fits"][capability]
        print(
            f"{capability:<10}  {dense[capability]:7.4f}  "
            f"{damaged[capability]:8.4f}  {last[capability]:7.4f}  "
            f"{_format_optional(fit['beta'], 4):>6}  "
            f"{_format_optional(fit['residual_vs_dense'], 4):>8}"
        )
    print(f"artifacts: {out}")


def run_recovery(
    model: str,
    compression: str,
    density: float,
    bits: int,
    recovery_source: str,
    budgets: Sequence[int],
    n_probe: int,
    device: str,
    model_dtype: str,
    seed: int,
    max_len: int,
    *,
    trace_base: Path = TRACE_BASE,
) -> dict:
    """Run dense/damaged anchors, one continuous recovery ladder, and fits."""
    budgets = parse_budgets(budgets)
    if compression not in {"prune", "quant", "none"}:
        raise ValueError("compression must be 'prune', 'quant', or 'none'")
    if recovery_source not in {"c4", "traces"}:
        raise ValueError("recovery_source must be 'c4' or 'traces'")
    if n_probe < 2:
        raise ValueError("--n-probe must be at least 2 for a measurement half")
    if max_len < 2:
        raise ValueError("--max-len must be at least 2")
    if model_dtype not in {"fp32", "bf16"}:
        raise ValueError("model_dtype must be 'fp32' or 'bf16'")
    if compression == "prune" and not 0.0 < density <= 1.0:
        raise ValueError("--density must be in (0, 1]")
    if compression == "quant" and bits < 2:
        raise ValueError("--bits must be at least 2")

    seed_everything(seed)
    resolved_model = require_compliant(model)
    model_tag = model_output_tag(model, resolved_model)
    run_name = _compression_run_name(compression, density, bits, recovery_source)
    out = OUT_BASE / model_tag / run_name
    out.mkdir(parents=True, exist_ok=True)

    all_probes = build_probes(n_probe, seed=seed)
    probes = {
        capability: all_probes[capability][1::2] for capability in CAPABILITIES
    }
    if any(not probes[capability] for capability in CAPABILITIES):
        raise RuntimeError("V6 measurement-half probes must be non-empty")

    dtype = torch.float32 if model_dtype == "fp32" else torch.bfloat16
    student, tokenizer = load_text_causal_lm(resolved_model, dtype)
    student.to(device).eval()
    dense_losses, measurement_tokens = measure_capability_losses(
        student,
        tokenizer,
        probes,
        device,
        max_len=max_len,
        loss_function=completion_loss,
    )
    print(f"[eval dense] {dense_losses}", flush=True)

    compression_metadata = _apply_permanent_compression(
        student, compression, density, bits, seed
    )
    student.eval()
    damaged_losses, damaged_measurement_tokens = measure_capability_losses(
        student,
        tokenizer,
        probes,
        device,
        max_len=max_len,
        loss_function=completion_loss,
    )
    if damaged_measurement_tokens != measurement_tokens:
        raise RuntimeError("Dense and damaged evaluations used different tokens")
    print(f"[eval damaged] {damaged_losses}", flush=True)

    student, training_mode, trainable_names = configure_training(
        student, allow_full_fallback=False
    )
    if training_mode != "lora":
        raise RuntimeError("V13 recovery requires LoRA training")
    recovery_rows, source_metadata = select_recovery_source(
        recovery_source, trace_base=trace_base, seed=seed
    )
    examples = _tokenized_recovery_examples(
        recovery_rows, recovery_source, tokenizer, max_len
    )
    ladder, training_curve = _train_recovery_ladder(
        student,
        examples,
        tokenizer,
        probes,
        device,
        budgets,
        max_len,
        measurement_tokens,
    )
    curves = _curves_from_checkpoints(ladder, dense_losses, damaged_losses)
    fits = {
        capability: fit_recovery_power_law(
            [point["tokens_seen"] for point in curves[capability]],
            [point["loss"] for point in curves[capability]],
            dense_loss=dense_losses[capability],
        )
        for capability in CAPABILITIES
    }

    payload = {
        "version": 13,
        "model": model,
        "resolved_model": resolved_model,
        "model_tag": model_tag,
        "run_name": run_name,
        "compression": compression_metadata,
        "recovery_source": source_metadata,
        "budgets_requested": list(budgets),
        "anchors": {"dense": dense_losses, "damaged": damaged_losses},
        "ladder": ladder,
        "curves": curves,
        "fits": fits,
        "training": {
            "mode": training_mode,
            "optimizer": "AdamW",
            "scheduler": "cosine",
            "warmup_ratio": WARMUP_RATIO,
            "learning_rate": LORA_LR,
            "dtype": "bfloat16 autocast",
            "effective_batch_size_sequences": EFFECTIVE_BATCH_SIZE,
            "max_len": max_len,
            "seed": seed,
            "optimizer_steps": ladder[-1]["optimizer_step"],
            "tokens_seen": ladder[-1]["tokens_seen"],
            "trainable_parameters": sum(
                parameter.numel()
                for parameter in student.parameters()
                if parameter.requires_grad
            ),
            "trainable_parameter_names": trainable_names,
            "lora": {
                "r": 16,
                "alpha": 32,
                "dropout": 0.0,
                "target_modules": list(LORA_TARGET_MODULES),
            },
            "loss_curve": training_curve,
        },
        "measurement": {
            "probe_source": "analysis.v6_capability_geometry.build_probes",
            "probe_seed": seed,
            "n_probe_requested": n_probe,
            "probe_half": "measurement (odd indices, v[1::2])",
            "samples": {
                capability: len(probes[capability]) for capability in CAPABILITIES
            },
            "tokens": measurement_tokens,
        },
    }
    write_json_atomic(out / "recovery.json", payload)
    write_report(out, payload)
    _print_summary(payload, out)

    del student, tokenizer
    gc.collect()
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="gemma3-270m",
        help="model-registry tag or raw Hugging Face model id",
    )
    parser.add_argument(
        "--compression", choices=("prune", "quant", "none"), default="prune"
    )
    parser.add_argument("--density", type=float, default=0.5)
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument(
        "--recovery-source", choices=("c4", "traces"), default="c4"
    )
    parser.add_argument(
        "--budgets",
        type=parse_budgets,
        default=parse_budgets(DEFAULT_BUDGETS),
        help=f"comma-separated recovery-token budgets (default: {DEFAULT_BUDGETS})",
    )
    parser.add_argument("--n-probe", type=int, default=128)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--model-dtype", choices=("fp32", "bf16"), default="bf16"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-len", type=int, default=DEFAULT_MAX_LEN)
    args = parser.parse_args()

    run_recovery(
        model=args.model,
        compression=args.compression,
        density=args.density,
        bits=args.bits,
        recovery_source=args.recovery_source,
        budgets=args.budgets,
        n_probe=args.n_probe,
        device=args.device,
        model_dtype=args.model_dtype,
        seed=args.seed,
        max_len=args.max_len,
    )


if __name__ == "__main__":
    main()
