#!/usr/bin/env python3
"""V11: compare geometric and behavioral capability damage under pruning.

For the three V6 capabilities (math, code, and QA), this analysis estimates
diagonal empirical-Fisher signatures on the even-indexed probe half and
completion loss on the held-out odd-indexed half.  At each pruning density it
re-estimates the signatures on the pruned model and compares them with the
dense signatures.

Large Fisher vectors are saved as flattened CPU fp32 tensors and loaded with
``mmap=True``.  Cross-capability operations are chunked, so no analysis step
needs all six dense/pruned vectors resident in RAM at once.

Example:
  python3 analysis/v11_geometry_damage.py --model gemma3-1b --device cuda:0

Artifacts are written under
``results/v11-geometry-damage/<model_tag>/``.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import tempfile
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

try:
    from .v6_capability_geometry import (
        PRUNE_THRESHOLD_SAMPLE_SIZE,
        _sample_abs_weights,
        build_probes,
        completion_loss,
        language_weight_parameters,
        load_text_causal_lm,
        model_output_tag,
        require_compliant,
    )
except ImportError:  # direct execution: python analysis/v11_geometry_damage.py
    from v6_capability_geometry import (
        PRUNE_THRESHOLD_SAMPLE_SIZE,
        _sample_abs_weights,
        build_probes,
        completion_loss,
        language_weight_parameters,
        load_text_causal_lm,
        model_output_tag,
        require_compliant,
    )


ROOT = Path(__file__).resolve().parents[1]
OUT_BASE = ROOT / "results/v11-geometry-damage"
CAPABILITIES = ("math", "code", "qa")
DENSITIES = (0.9, 0.8, 0.7, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3)
LOG_EPS = 1e-12
DENSE_RESIDUAL_TOP_FRACTION = 0.001
PRUNED_FISHER_TOP_FRACTION = 0.01
DEFAULT_CHUNK_SIZE = 5_000_000
CLIFF_THRESHOLD = 1.0
GEOMETRIC_METRICS = (
    "residual_cosine",
    "log_fisher_cosine",
    "mass_retention",
)


def _check_vectors(vectors: Sequence[torch.Tensor]) -> int:
    if not vectors:
        raise ValueError("At least one Fisher vector is required")
    size = vectors[0].numel()
    if size == 0:
        raise ValueError("Fisher vectors must be nonempty")
    for index, vector in enumerate(vectors):
        if vector.ndim != 1:
            raise ValueError(f"Fisher vector {index} is not one-dimensional")
        if vector.numel() != size:
            raise ValueError("All Fisher vectors must have equal length")
        if not vector.is_floating_point():
            raise TypeError("Fisher vectors must be floating-point tensors")
    return size


def _logged_vectors(
    vectors: Sequence[torch.Tensor], eps: float
) -> list[torch.Tensor]:
    if eps <= 0:
        raise ValueError("eps must be positive")
    _check_vectors(vectors)
    logged = []
    for index, vector in enumerate(vectors):
        if not bool(torch.isfinite(vector).all()):
            raise ValueError(f"Fisher vector {index} contains non-finite values")
        if bool((vector < 0).any()):
            raise ValueError(f"Fisher vector {index} contains negative values")
        item = vector.detach().to(device="cpu", dtype=torch.float32, copy=True)
        item.add_(eps).log_()
        logged.append(item)
    return logged


def residual_log_vectors(
    fishers: Sequence[torch.Tensor], eps: float = LOG_EPS
) -> list[torch.Tensor]:
    """Return capability residuals of log-Fisher vectors.

    For each capability ``c``, the residual is
    ``log(F_c + eps) - mean_c' log(F_c' + eps)``.  The helper is also used on
    bounded chunks by the full pipeline.
    """
    logged = _logged_vectors(fishers, eps)
    shared = torch.zeros_like(logged[0])
    for vector in logged:
        shared.add_(vector)
    shared.div_(len(logged))
    for vector in logged:
        vector.sub_(shared)
    return logged


def _product_sum(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(torch.sum(left * right, dtype=torch.float64))


def _cosine(dot: float, left_norm: float, right_norm: float) -> float:
    denominator = math.sqrt(max(left_norm, 0.0) * max(right_norm, 0.0))
    if denominator == 0:
        return 0.0
    return float(np.clip(dot / denominator, -1.0, 1.0))


def _merge_topk(
    values: torch.Tensor,
    indices: torch.Tensor,
    new_values: torch.Tensor,
    new_indices: torch.Tensor,
    keep: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    values = torch.cat((values, new_values.to(device="cpu")))
    indices = torch.cat(
        (indices, new_indices.to(device="cpu", dtype=torch.int64))
    )
    if values.numel() > keep:
        values, selected = torch.topk(values, keep, largest=True, sorted=False)
        indices = indices[selected]
    return values, indices


def top_residual_indices(
    fishers: Sequence[torch.Tensor],
    capability_index: int,
    fraction: float = DENSE_RESIDUAL_TOP_FRACTION,
    eps: float = LOG_EPS,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> torch.Tensor:
    """Find exact top positive-residual coordinates with bounded memory."""
    size = _check_vectors(fishers)
    if not 0 <= capability_index < len(fishers):
        raise IndexError("capability_index is outside the Fisher list")
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    keep = max(1, int(math.floor(size * fraction)))
    candidate_values = torch.empty(0, dtype=torch.float32)
    candidate_indices = torch.empty(0, dtype=torch.int64)
    for start in range(0, size, chunk_size):
        end = min(start + chunk_size, size)
        residual = residual_log_vectors(
            [vector[start:end] for vector in fishers], eps
        )[capability_index]
        local_keep = min(keep, residual.numel())
        values, indices = torch.topk(
            residual, local_keep, largest=True, sorted=False
        )
        indices.add_(start)
        candidate_values, candidate_indices = _merge_topk(
            candidate_values,
            candidate_indices,
            values,
            indices,
            keep,
        )
        del residual, values, indices
    return torch.sort(candidate_indices).values


def _top_fraction_boundary(
    vector: torch.Tensor,
    fraction: float,
    chunk_size: int,
) -> float:
    """Return the largest excluded value below an exact top-fraction set.

    With distinct values, exactly ``floor(n * fraction)`` coordinates are
    strictly above this boundary.  Ties at the boundary are conservatively
    excluded, matching the requested "above threshold" definition.
    """
    size = _check_vectors([vector])
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if fraction == 1 or size == 1:
        return -math.inf
    top_count = max(1, int(math.floor(size * fraction)))
    keep = min(size, top_count + 1)
    candidates = torch.empty(0, dtype=vector.dtype, device="cpu")
    for start in range(0, size, chunk_size):
        chunk = vector[start : min(start + chunk_size, size)]
        if not bool(torch.isfinite(chunk).all()):
            raise ValueError("Fisher vector contains non-finite values")
        local_keep = min(keep, chunk.numel())
        values = torch.topk(
            chunk, local_keep, largest=True, sorted=False
        ).values.to(device="cpu")
        candidates = torch.cat((candidates, values))
        if candidates.numel() > keep:
            candidates = torch.topk(
                candidates, keep, largest=True, sorted=False
            ).values
    return float(candidates.min())


def mass_retention(
    dense_residual_top_indices: torch.Tensor,
    pruned_fisher: torch.Tensor,
    top_fraction: float = PRUNED_FISHER_TOP_FRACTION,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> float:
    """Fraction of dense region coordinates above the pruned top threshold."""
    _check_vectors([pruned_fisher])
    indices = dense_residual_top_indices.to(
        device="cpu", dtype=torch.int64
    ).reshape(-1)
    if indices.numel() == 0:
        raise ValueError("dense_residual_top_indices must be nonempty")
    if int(indices.min()) < 0 or int(indices.max()) >= pruned_fisher.numel():
        raise IndexError("dense residual coordinate is outside pruned Fisher")
    threshold = _top_fraction_boundary(
        pruned_fisher, top_fraction, chunk_size
    )
    retained = pruned_fisher[indices] > threshold
    return float(retained.to(dtype=torch.float64).mean())


def geometric_metrics(
    dense_fishers: Sequence[torch.Tensor],
    pruned_fishers: Sequence[torch.Tensor],
    dense_top_indices: Sequence[torch.Tensor],
    capabilities: Sequence[str] = CAPABILITIES,
    eps: float = LOG_EPS,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> dict[str, dict[str, float]]:
    """Compute all three dense-vs-pruned metrics in bounded chunks."""
    size = _check_vectors([*dense_fishers, *pruned_fishers])
    count = len(capabilities)
    if len(dense_fishers) != count or len(pruned_fishers) != count:
        raise ValueError("Capabilities and Fisher lists must have equal length")
    if len(dense_top_indices) != count:
        raise ValueError("One dense top-coordinate set is required per capability")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    raw_dot = np.zeros(count, dtype=np.float64)
    raw_dense_norm = np.zeros(count, dtype=np.float64)
    raw_pruned_norm = np.zeros(count, dtype=np.float64)
    residual_dot = np.zeros(count, dtype=np.float64)
    residual_dense_norm = np.zeros(count, dtype=np.float64)
    residual_pruned_norm = np.zeros(count, dtype=np.float64)

    for start in range(0, size, chunk_size):
        end = min(start + chunk_size, size)
        dense_logs = _logged_vectors(
            [vector[start:end] for vector in dense_fishers], eps
        )
        pruned_logs = _logged_vectors(
            [vector[start:end] for vector in pruned_fishers], eps
        )
        for index in range(count):
            raw_dot[index] += _product_sum(
                dense_logs[index], pruned_logs[index]
            )
            raw_dense_norm[index] += _product_sum(
                dense_logs[index], dense_logs[index]
            )
            raw_pruned_norm[index] += _product_sum(
                pruned_logs[index], pruned_logs[index]
            )

        dense_shared = torch.zeros_like(dense_logs[0])
        pruned_shared = torch.zeros_like(pruned_logs[0])
        for dense_log, pruned_log in zip(dense_logs, pruned_logs):
            dense_shared.add_(dense_log)
            pruned_shared.add_(pruned_log)
        dense_shared.div_(count)
        pruned_shared.div_(count)
        for index in range(count):
            dense_logs[index].sub_(dense_shared)
            pruned_logs[index].sub_(pruned_shared)
            residual_dot[index] += _product_sum(
                dense_logs[index], pruned_logs[index]
            )
            residual_dense_norm[index] += _product_sum(
                dense_logs[index], dense_logs[index]
            )
            residual_pruned_norm[index] += _product_sum(
                pruned_logs[index], pruned_logs[index]
            )
        del dense_logs, pruned_logs, dense_shared, pruned_shared

    result = {}
    for index, capability in enumerate(capabilities):
        result[capability] = {
            "residual_cosine": _cosine(
                residual_dot[index],
                residual_dense_norm[index],
                residual_pruned_norm[index],
            ),
            "log_fisher_cosine": _cosine(
                raw_dot[index],
                raw_dense_norm[index],
                raw_pruned_norm[index],
            ),
            "mass_retention": mass_retention(
                dense_top_indices[index],
                pruned_fishers[index],
                top_fraction=PRUNED_FISHER_TOP_FRACTION,
                chunk_size=chunk_size,
            ),
        }
    return result


def _load_fisher_mmap(path: Path) -> torch.Tensor:
    try:
        vector = torch.load(
            path, map_location="cpu", weights_only=True, mmap=True
        )
    except TypeError as exc:
        raise RuntimeError(
            "V11 requires torch.load(..., mmap=True) to keep Fisher analysis "
            "memory bounded"
        ) from exc
    if not isinstance(vector, torch.Tensor):
        raise TypeError(f"{path} does not contain a tensor")
    if vector.ndim != 1 or vector.dtype != torch.float32:
        raise ValueError(
            f"{path} must contain a flattened fp32 tensor; got "
            f"shape={tuple(vector.shape)}, dtype={vector.dtype}"
        )
    return vector


def estimate_fisher(
    model,
    tokenizer,
    parameters: Sequence[tuple[str, torch.Tensor]],
    samples: Sequence[Mapping[str, str]],
    device: str,
    destination: Path,
    label: str,
) -> int:
    """Accumulate a flattened fp32 empirical Fisher on the model device."""
    n_parameters = sum(parameter.numel() for _, parameter in parameters)
    if n_parameters == 0:
        raise ValueError("No language weight matrices found for Fisher scope")
    accumulator = torch.zeros(
        n_parameters, dtype=torch.float32, device=device
    )
    used = 0
    for sample_index, sample in enumerate(samples, start=1):
        model.zero_grad(set_to_none=True)
        loss, n_tokens = completion_loss(
            model,
            tokenizer,
            sample["prompt"],
            sample["completion"],
            device,
        )
        if n_tokens <= 0:
            del loss
            continue
        (loss / n_tokens).backward()
        del loss
        offset = 0
        with torch.no_grad():
            for _, parameter in parameters:
                end = offset + parameter.numel()
                if parameter.grad is not None:
                    gradient_squared = parameter.grad.detach().float().square()
                    accumulator[offset:end].add_(gradient_squared.reshape(-1))
                    del gradient_squared
                offset = end
        used += 1
        print(
            f"[fisher] {label}: {sample_index}/{len(samples)}",
            end="\r",
            flush=True,
        )

    model.zero_grad(set_to_none=True)
    print(" " * 80, end="\r", flush=True)
    if used == 0:
        del accumulator
        raise RuntimeError(f"{label} produced no completion-token gradients")
    accumulator.div_(used)
    cpu_fisher = accumulator.to(device="cpu")
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cpu_fisher, destination)
    print(f"[fisher] {label}: saved {used} samples to {destination}", flush=True)
    del cpu_fisher, accumulator
    gc.collect()
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return used


def measure_capability_losses(
    model,
    tokenizer,
    probes: Mapping[str, Sequence[Mapping[str, str]]],
    device: str,
) -> tuple[dict[str, float], dict[str, int]]:
    """Return token-weighted completion CE and token counts by capability."""
    losses: dict[str, float] = {}
    token_counts: dict[str, int] = {}
    with torch.no_grad():
        for capability in CAPABILITIES:
            total_loss = 0.0
            total_tokens = 0
            for sample in probes[capability]:
                loss, n_tokens = completion_loss(
                    model,
                    tokenizer,
                    sample["prompt"],
                    sample["completion"],
                    device,
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


def _restore_dense_weights(
    parameters: Sequence[tuple[str, torch.Tensor]],
    dense_weights: Sequence[torch.Tensor],
) -> None:
    with torch.no_grad():
        for (_, parameter), dense in zip(parameters, dense_weights):
            parameter.copy_(dense)


def _apply_magnitude_pruning(
    parameters: Sequence[tuple[str, torch.Tensor]],
    dense_weights: Sequence[torch.Tensor],
    threshold: float,
) -> float:
    nonzero = 0
    total = 0
    with torch.no_grad():
        for (_, parameter), dense in zip(parameters, dense_weights):
            parameter.copy_(dense)
            parameter.masked_fill_(dense.abs() <= threshold, 0)
            nonzero += int(torch.count_nonzero(parameter))
            total += parameter.numel()
    return nonzero / total if total else 0.0


def _shape_before_cliff(
    curve: Sequence[Mapping[str, float]],
    metric: str,
    cliff_index: int | None,
    rebound_tolerance: float = 0.03,
) -> dict[str, float | str | bool | int]:
    """Describe whether pre-cliff geometry declines gradually or by a step."""
    end = len(curve) if cliff_index is None else cliff_index
    values = [1.0] + [float(row[metric]) for row in curve[:end]]
    if len(values) < 3:
        return {
            "classification": "insufficient pre-cliff points",
            "smooth_decline": False,
            "n_pruned_points": len(values) - 1,
            "total_decline": max(0.0, 1.0 - values[-1]),
            "largest_step_share": 0.0,
        }
    changes = [current - previous for previous, current in zip(values, values[1:])]
    drops = [max(-change, 0.0) for change in changes]
    total_drop = sum(drops)
    total_decline = max(0.0, 1.0 - values[-1])
    largest_step_share = max(drops, default=0.0) / total_drop if total_drop else 0.0
    has_rebound = any(change > rebound_tolerance for change in changes)
    if total_decline <= rebound_tolerance:
        classification = "flat / no clear decline"
        smooth = False
    elif has_rebound:
        classification = "non-monotonic"
        smooth = False
    elif largest_step_share > 0.75:
        classification = "step-like decline"
        smooth = False
    else:
        classification = "smooth decline"
        smooth = True
    return {
        "classification": classification,
        "smooth_decline": smooth,
        "n_pruned_points": len(values) - 1,
        "total_decline": total_decline,
        "largest_step_share": largest_step_share,
    }


def summarize_cliff_crossing(
    curve: Sequence[Mapping[str, float]],
    threshold: float = CLIFF_THRESHOLD,
) -> dict:
    """Summarize the first ordered density whose loss damage exceeds a limit."""
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")
    cliff_index = next(
        (
            index
            for index, row in enumerate(curve)
            if float(row["delta_loss"]) > threshold
        ),
        None,
    )
    cliff_row = None if cliff_index is None else curve[cliff_index]
    geometry_at_cliff = (
        None
        if cliff_row is None
        else {metric: float(cliff_row[metric]) for metric in GEOMETRIC_METRICS}
    )
    return {
        "threshold": float(threshold),
        "crossed": cliff_row is not None,
        "density": None if cliff_row is None else float(cliff_row["density"]),
        "delta_loss": (
            None if cliff_row is None else float(cliff_row["delta_loss"])
        ),
        "geometric_at_cliff": geometry_at_cliff,
        "geometric_shape_before_cliff": {
            metric: _shape_before_cliff(curve, metric, cliff_index)
            for metric in GEOMETRIC_METRICS
        },
    }


def _format_float(value: float | None, digits: int = 4) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value:.{digits}f}"


def write_report(out: Path, payload: Mapping) -> None:
    """Write one curve table per capability and the behavioral-cliff summary."""
    lines = [
        f"# V11 geometry-damage report — {payload['model']}",
        "",
        "Even-indexed V6 probes estimate diagonal empirical Fishers; the "
        "disjoint odd-indexed halves measure completion loss. `Delta L_c` is "
        "relative to the dense capability loss.",
        "",
    ]
    for capability in CAPABILITIES:
        record = payload["capabilities"][capability]
        lines.extend(
            [
                f"## {capability}",
                "",
                f"Dense measurement loss: {record['dense_loss']:.6f}",
                "",
                "| d | Delta L_c | residual cosine | log cosine | mass retention |",
                "|---:|---:|---:|---:|---:|",
            ]
        )
        for row in record["curve"]:
            lines.append(
                f"| {row['density']:.2f} | {row['delta_loss']:+.6f} | "
                f"{row['residual_cosine']:.4f} | "
                f"{row['log_fisher_cosine']:.4f} | "
                f"{row['mass_retention']:.4f} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Behavioral cliffs and geometric shape",
            "",
            "The behavioral cliff is the first listed density, moving from "
            "dense toward sparse, where `Delta L_c > 1.0`. Candidate critical "
            "values are the three geometric metrics at that density.",
            "",
            "A pre-cliff curve is labeled smooth when it has a clear decline, "
            "no rebound larger than 0.03, and no single density step accounts "
            "for more than 75% of its cumulative decline. The dense baseline "
            "for each geometric metric is 1.0.",
            "",
            "| capability | behavioral cliff d | residual before cliff | "
            "log-Fisher before cliff | mass before cliff | candidate values "
            "at cliff (residual / log / mass) |",
            "|---|---:|---|---|---|---|",
        ]
    )
    for capability in CAPABILITIES:
        summary = payload["capabilities"][capability]["summary"]
        shapes = summary["geometric_shape_before_cliff"]
        at_cliff = summary["geometric_at_cliff"]
        if at_cliff is None:
            candidate = "n/a"
        else:
            candidate = " / ".join(
                _format_float(at_cliff[metric]) for metric in GEOMETRIC_METRICS
            )
        lines.append(
            f"| {capability} | {_format_float(summary['density'], 2)} | "
            f"{shapes['residual_cosine']['classification']} | "
            f"{shapes['log_fisher_cosine']['classification']} | "
            f"{shapes['mass_retention']['classification']} | {candidate} |"
        )
    lines.append("")
    (out / "report.md").write_text("\n".join(lines))


def _write_results(out: Path, payload: Mapping) -> None:
    (out / "geometry_damage.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )
    write_report(out, payload)


def run_geometry_damage(
    model_name: str,
    model_tag: str,
    device: str,
    model_dtype: str,
    n_probe: int,
    out: Path,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    threshold_sample_size: int = PRUNE_THRESHOLD_SAMPLE_SIZE,
) -> None:
    """Run dense and all pruning-density measurements for one model."""
    if n_probe < 2:
        raise ValueError("n_probe must be at least 2 to form disjoint halves")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if threshold_sample_size <= 0:
        raise ValueError("threshold_sample_size must be positive")
    all_probes = build_probes(n_probe)
    if tuple(all_probes) != CAPABILITIES:
        missing = [cap for cap in CAPABILITIES if cap not in all_probes]
        if missing:
            raise RuntimeError("V6 probes are missing: " + ", ".join(missing))
    estimation = {
        capability: all_probes[capability][0::2]
        for capability in CAPABILITIES
    }
    measurement = {
        capability: all_probes[capability][1::2]
        for capability in CAPABILITIES
    }
    for capability in CAPABILITIES:
        if not estimation[capability] or not measurement[capability]:
            raise RuntimeError(
                f"{capability} needs nonempty estimation and measurement halves"
            )

    dtype = torch.float32 if model_dtype == "fp32" else torch.bfloat16
    model, tokenizer = load_text_causal_lm(model_name, dtype)
    model.to(device).eval()
    model.requires_grad_(False)
    parameters = language_weight_parameters(model)
    for _, parameter in parameters:
        parameter.requires_grad_(True)
    n_parameters = sum(parameter.numel() for _, parameter in parameters)
    if n_parameters == 0:
        raise ValueError("No language weight matrices found for pruning")

    out.mkdir(parents=True, exist_ok=True)
    dense_losses, measurement_tokens = measure_capability_losses(
        model, tokenizer, measurement, device
    )
    print(f"[dense loss] {dense_losses}", flush=True)
    dense_paths = {
        capability: out / f"fisher_dense_{capability}.pt"
        for capability in CAPABILITIES
    }
    fisher_samples = {}
    for capability in CAPABILITIES:
        fisher_samples[capability] = estimate_fisher(
            model,
            tokenizer,
            parameters,
            estimation[capability],
            device,
            dense_paths[capability],
            f"dense/{capability}",
        )

    abs_sample = _sample_abs_weights(
        parameters, sample_size=threshold_sample_size, seed=0
    )
    thresholds = {
        density: float(np.quantile(abs_sample, 1.0 - density))
        for density in DENSITIES
    }
    del abs_sample

    dense_fishers = [
        _load_fisher_mmap(dense_paths[capability])
        for capability in CAPABILITIES
    ]
    dense_top_indices = []
    for index, capability in enumerate(CAPABILITIES):
        indices = top_residual_indices(
            dense_fishers,
            index,
            fraction=DENSE_RESIDUAL_TOP_FRACTION,
            chunk_size=chunk_size,
        )
        dense_top_indices.append(indices)
        torch.save(indices, out / f"dense_top_residual_{capability}.pt")
        print(
            f"[dense region] {capability}: {indices.numel():,} coordinates",
            flush=True,
        )

    dense_weights = [
        parameter.detach().clone() for _, parameter in parameters
    ]
    payload = {
        "version": 11,
        "model": model_name,
        "model_tag": model_tag,
        "settings": {
            "model_dtype": model_dtype,
            "n_probe_requested": n_probe,
            "probe_seed": 0,
            "densities": list(DENSITIES),
            "n_parameters": n_parameters,
            "parameter_names": [name for name, _ in parameters],
            "threshold_sample_size": threshold_sample_size,
            "log_eps": LOG_EPS,
            "dense_residual_top_fraction": DENSE_RESIDUAL_TOP_FRACTION,
            "pruned_fisher_top_fraction": PRUNED_FISHER_TOP_FRACTION,
            "chunk_size": chunk_size,
            "behavioral_cliff_threshold": CLIFF_THRESHOLD,
        },
        "pruning_thresholds": {
            str(density): thresholds[density] for density in DENSITIES
        },
        "capabilities": {
            capability: {
                "dense_loss": dense_losses[capability],
                "dense_fisher_file": dense_paths[capability].name,
                "dense_top_residual_file": (
                    f"dense_top_residual_{capability}.pt"
                ),
                "estimation_samples": fisher_samples[capability],
                "measurement_samples": len(measurement[capability]),
                "measurement_tokens": measurement_tokens[capability],
                "curve": [],
            }
            for capability in CAPABILITIES
        },
    }

    try:
        with tempfile.TemporaryDirectory(
            prefix=".pruned-fishers-", dir=out
        ) as temporary_directory:
            temporary_path = Path(temporary_directory)
            for density in DENSITIES:
                realized_density = _apply_magnitude_pruning(
                    parameters, dense_weights, thresholds[density]
                )
                pruned_losses, _ = measure_capability_losses(
                    model, tokenizer, measurement, device
                )
                pruned_paths = []
                for capability in CAPABILITIES:
                    path = temporary_path / f"fisher_{capability}.pt"
                    estimate_fisher(
                        model,
                        tokenizer,
                        parameters,
                        estimation[capability],
                        device,
                        path,
                        f"d={density}/{capability}",
                    )
                    pruned_paths.append(path)
                pruned_fishers = [
                    _load_fisher_mmap(path) for path in pruned_paths
                ]
                metrics = geometric_metrics(
                    dense_fishers,
                    pruned_fishers,
                    dense_top_indices,
                    chunk_size=chunk_size,
                )
                for capability in CAPABILITIES:
                    row = {
                        "density": density,
                        "realized_density": realized_density,
                        "loss": pruned_losses[capability],
                        "delta_loss": (
                            pruned_losses[capability] - dense_losses[capability]
                        ),
                        **metrics[capability],
                    }
                    payload["capabilities"][capability]["curve"].append(row)
                print(
                    f"[density {density}] realized={realized_density:.5f}; "
                    f"losses={pruned_losses}",
                    flush=True,
                )
                del pruned_fishers, metrics
                gc.collect()
                _restore_dense_weights(parameters, dense_weights)
                # Persist completed densities so a long run leaves useful data.
                for capability in CAPABILITIES:
                    curve = payload["capabilities"][capability]["curve"]
                    payload["capabilities"][capability]["summary"] = (
                        summarize_cliff_crossing(curve)
                    )
                _write_results(out, payload)
    finally:
        _restore_dense_weights(parameters, dense_weights)
        model.zero_grad(set_to_none=True)

    for capability in CAPABILITIES:
        curve = payload["capabilities"][capability]["curve"]
        payload["capabilities"][capability]["summary"] = (
            summarize_cliff_crossing(curve)
        )
    _write_results(out, payload)

    del dense_weights, dense_top_indices, dense_fishers
    del parameters, model, tokenizer
    gc.collect()
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="gemma3-1b",
        help="model-registry tag or raw Hugging Face model id",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--model-dtype", choices=("fp32", "bf16"), default="bf16"
    )
    parser.add_argument("--n-probe", type=int, default=128)
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="CPU coordinates processed per geometric-analysis chunk",
    )
    parser.add_argument(
        "--threshold-sample-size",
        type=int,
        default=PRUNE_THRESHOLD_SAMPLE_SIZE,
    )
    args = parser.parse_args()

    model_name = require_compliant(args.model)
    tag = model_output_tag(args.model, model_name)
    out = OUT_BASE / tag
    run_geometry_damage(
        model_name=model_name,
        model_tag=tag,
        device=args.device,
        model_dtype=args.model_dtype,
        n_probe=args.n_probe,
        out=out,
        chunk_size=args.chunk_size,
        threshold_sample_size=args.threshold_sample_size,
    )


if __name__ == "__main__":
    main()
