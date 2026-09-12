#!/usr/bin/env python3
"""V9c: causal deletion tests for capability and control coordinates.

For each of math, code, and QA, this script computes the capability-average
residual log-Fisher from V9's saved per-benchmark Fisher artifacts.  A
capability region is the exact top fraction of that residual excluding every
coordinate selected by either other capability.  The selected weights are
temporarily zeroed, completion loss is measured on held-out probe halves, and
the original weights are restored before testing the next capability.

Experiment B1 adds random, magnitude-matched, global-Fisher, and
non-residualized controls.  Every requested control/fraction pair is measured
from one loaded dense snapshot and is restored before the next deletion.

Example:
  python3 analysis/v9c_ablation.py --mode capability --mode random \
    --mode magnitude_matched --mode global_fisher --mode no_residual \
    --top-fracs 0.0001,0.0005,0.002

Artifacts are written under results/v9c-ablation/<model_tag>/.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import random
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

import torch

try:
    from .v6_capability_geometry import (
        completion_loss,
        language_weight_parameters,
        load_text_causal_lm,
        model_output_tag,
        require_compliant,
    )
    from .v9_capability_regions import (
        BENCHMARK_CAPABILITY,
        CORE_CAPABILITIES,
        LOG_EPS,
        OUT_BASE as V9_OUT_BASE,
        PROBE_SEED,
        _analysis_inventory,
        _load_tensor_mmap,
        build_probe_registry,
    )
except ImportError:  # direct execution: python analysis/v9c_ablation.py
    from v6_capability_geometry import (
        completion_loss,
        language_weight_parameters,
        load_text_causal_lm,
        model_output_tag,
        require_compliant,
    )
    from v9_capability_regions import (
        BENCHMARK_CAPABILITY,
        CORE_CAPABILITIES,
        LOG_EPS,
        OUT_BASE as V9_OUT_BASE,
        PROBE_SEED,
        _analysis_inventory,
        _load_tensor_mmap,
        build_probe_registry,
    )


ROOT = Path(__file__).resolve().parents[1]
OUT_BASE = ROOT / "results/v9c-ablation"
DEFAULT_TOP_FRACTION = 0.0005
SELECTION_CHUNK_SIZE = 1_000_000
MAGNITUDE_BINS = 20
CONTROL_SEED = 0
ABLATION_MODES = (
    "capability",
    "random",
    "magnitude_matched",
    "global_fisher",
    "no_residual",
)


def parse_top_fractions(value: str | Sequence[float]) -> list[float]:
    """Parse and validate the comma-separated ``--top-fracs`` value."""
    if isinstance(value, str):
        try:
            fractions = [float(part.strip()) for part in value.split(",")]
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                "--top-fracs must be a comma-separated list of numbers"
            ) from exc
    else:
        fractions = [float(fraction) for fraction in value]
    if not fractions:
        raise argparse.ArgumentTypeError("--top-fracs must not be empty")
    if any(
        not math.isfinite(fraction) or not 0 < fraction <= 1 for fraction in fractions
    ):
        raise argparse.ArgumentTypeError(
            "all --top-fracs values must be finite and in (0, 1]"
        )
    if len(set(fractions)) != len(fractions):
        raise argparse.ArgumentTypeError("--top-fracs values must be unique")
    return fractions


def _fraction_key(fraction: float) -> str:
    return f"{fraction:.12g}"


def _stable_control_seed(seed: int, *parts: object) -> int:
    payload = "\0".join((str(seed), *(str(part) for part in parts))).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (
        2**63 - 1
    )


def _merge_topk(
    candidate_values: torch.Tensor,
    candidate_indices: torch.Tensor,
    values: torch.Tensor,
    indices: torch.Tensor,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    values = values.to(device="cpu", dtype=torch.float32)
    indices = indices.to(device="cpu", dtype=torch.int64)
    if candidate_values.numel() == 0:
        combined_values = values
        combined_indices = indices
    else:
        combined_values = torch.cat((candidate_values, values))
        combined_indices = torch.cat((candidate_indices, indices))
    if combined_values.numel() > k:
        combined_values, keep = torch.topk(
            combined_values, k, largest=True, sorted=False
        )
        combined_indices = combined_indices[keep]
    return combined_values, combined_indices


def _sorted_difference(selected: torch.Tensor, excluded: torch.Tensor) -> torch.Tensor:
    selected = torch.sort(selected.to(device="cpu", dtype=torch.int64)).values
    excluded = torch.sort(excluded.to(device="cpu", dtype=torch.int64)).values
    if selected.numel() == 0 or excluded.numel() == 0:
        return selected
    positions = torch.searchsorted(excluded, selected)
    present = positions < excluded.numel()
    matches = torch.zeros_like(present)
    matches[present] = excluded[positions[present]] == selected[present]
    return selected[~matches]


def _take_candidate_topk(
    values: torch.Tensor, indices: torch.Tensor, k: int
) -> torch.Tensor:
    if k <= 0 or values.numel() < k or indices.numel() != values.numel():
        raise ValueError("candidate top-k buffers are inconsistent")
    _, keep = torch.topk(values, k, largest=True, sorted=False)
    return torch.sort(indices[keep]).values


def _select_fisher_coordinate_sets(
    fisher_by_benchmark: Mapping[str, torch.Tensor],
    benchmark_capability: Mapping[str, str],
    top_fractions: Sequence[float],
    modes: Iterable[str],
    eps: float = LOG_EPS,
    chunk_size: int = SELECTION_CHUNK_SIZE,
) -> tuple[
    dict[str, dict[float, dict[str, torch.Tensor]]],
    dict[float, dict[str, int]],
    dict[float, dict[str, torch.Tensor]],
]:
    """Select Fisher-based coordinate sets in one streaming pass.

    Random and magnitude-matched controls depend on the capability-exclusive
    cardinalities, so requesting either also computes the capability top sets.
    The largest requested top-k buffer is retained and exact smaller top-k sets
    are derived from it.
    """
    if not fisher_by_benchmark:
        raise ValueError("At least one Fisher vector is required")
    if eps <= 0 or chunk_size <= 0:
        raise ValueError("eps and chunk_size must be positive")
    fractions = [float(fraction) for fraction in top_fractions]
    if not fractions or any(
        not math.isfinite(fraction) or not 0 < fraction <= 1 for fraction in fractions
    ):
        raise ValueError("top_fractions must contain values in (0, 1]")
    requested_modes = tuple(dict.fromkeys(modes))
    invalid_modes = set(requested_modes) - set(ABLATION_MODES)
    if invalid_modes:
        raise ValueError(f"Unknown ablation modes: {sorted(invalid_modes)}")
    need_capability = bool(
        {"capability", "random", "magnitude_matched"} & set(requested_modes)
    )
    need_no_residual = "no_residual" in requested_modes
    need_global = "global_fisher" in requested_modes
    if not (need_capability or need_no_residual or need_global):
        raise ValueError("At least one Fisher-based selection is required")

    names = list(fisher_by_benchmark)
    vectors = [fisher_by_benchmark[name].reshape(-1) for name in names]
    n_coordinates = vectors[0].numel()
    if n_coordinates == 0 or any(vector.numel() != n_coordinates for vector in vectors):
        raise ValueError("Fisher vectors must be nonempty and equally sized")
    capability_counts = {
        capability: sum(benchmark_capability.get(name) == capability for name in names)
        for capability in CORE_CAPABILITIES
    }
    missing = [
        capability for capability, count in capability_counts.items() if count == 0
    ]
    if missing:
        raise ValueError(f"No Fisher benchmarks for capabilities: {', '.join(missing)}")

    max_k = max(
        max(1, int(math.floor(n_coordinates * fraction))) for fraction in fractions
    )
    candidate_values: dict[tuple[str, str], torch.Tensor] = {}
    candidate_indices: dict[tuple[str, str], torch.Tensor] = {}
    if need_capability:
        for capability in CORE_CAPABILITIES:
            key = ("capability", capability)
            candidate_values[key] = torch.empty(0, dtype=torch.float32)
            candidate_indices[key] = torch.empty(0, dtype=torch.int64)
    if need_no_residual:
        for capability in CORE_CAPABILITIES:
            key = ("no_residual", capability)
            candidate_values[key] = torch.empty(0, dtype=torch.float32)
            candidate_indices[key] = torch.empty(0, dtype=torch.int64)
    if need_global:
        key = ("global_fisher", "all")
        candidate_values[key] = torch.empty(0, dtype=torch.float32)
        candidate_indices[key] = torch.empty(0, dtype=torch.int64)

    for start in range(0, n_coordinates, chunk_size):
        end = min(start + chunk_size, n_coordinates)
        shared_log_sum = (
            torch.zeros(end - start, dtype=torch.float32) if need_capability else None
        )
        capability_log_sums = (
            {
                capability: torch.zeros(end - start, dtype=torch.float32)
                for capability in CORE_CAPABILITIES
            }
            if need_capability or need_no_residual
            else {}
        )
        global_fisher_sum = (
            torch.zeros(end - start, dtype=torch.float32) if need_global else None
        )
        for name, vector in zip(names, vectors):
            chunk = vector[start:end]
            if not bool(torch.isfinite(chunk).all()):
                raise ValueError(f"Fisher vector {name!r} contains non-finite values")
            if bool((chunk < 0).any()):
                raise ValueError(f"Fisher vector {name!r} contains negative values")
            if global_fisher_sum is not None:
                global_fisher_sum.add_(chunk)
            capability = benchmark_capability.get(name)
            if shared_log_sum is not None or capability in capability_log_sums:
                logged = chunk.to(device="cpu", dtype=torch.float32, copy=True)
                logged.add_(eps).log_()
                if shared_log_sum is not None:
                    shared_log_sum.add_(logged)
                if capability in capability_log_sums:
                    capability_log_sums[capability].add_(logged)
                del logged

        score_chunks: dict[tuple[str, str], torch.Tensor] = {}
        if shared_log_sum is not None:
            shared_log_sum.div_(len(vectors))
        for capability in CORE_CAPABILITIES:
            if capability in capability_log_sums:
                capability_log_sums[capability].div_(capability_counts[capability])
            if need_no_residual:
                score_chunks[("no_residual", capability)] = capability_log_sums[
                    capability
                ]
            if need_capability:
                residual = capability_log_sums[capability].clone()
                residual.sub_(shared_log_sum)
                score_chunks[("capability", capability)] = residual
        if global_fisher_sum is not None:
            global_fisher_sum.div_(len(vectors))
            score_chunks[("global_fisher", "all")] = global_fisher_sum

        for key, scores in score_chunks.items():
            local_k = min(max_k, scores.numel())
            values, indices = torch.topk(scores, local_k, largest=True, sorted=False)
            indices.add_(start)
            candidate_values[key], candidate_indices[key] = _merge_topk(
                candidate_values[key],
                candidate_indices[key],
                values,
                indices,
                max_k,
            )
        del score_chunks, shared_log_sum, capability_log_sums, global_fisher_sum

    selections: dict[str, dict[float, dict[str, torch.Tensor]]] = {}
    top_counts: dict[float, dict[str, int]] = {}
    capability_top_sets: dict[float, dict[str, torch.Tensor]] = {}
    if need_capability:
        selections["capability"] = {}
    if need_no_residual:
        selections["no_residual"] = {}
    if need_global:
        selections["global_fisher"] = {}
    for fraction in fractions:
        k = max(1, int(math.floor(n_coordinates * fraction)))
        if need_capability:
            top_indices = {
                capability: _take_candidate_topk(
                    candidate_values[("capability", capability)],
                    candidate_indices[("capability", capability)],
                    k,
                )
                for capability in CORE_CAPABILITIES
            }
            exclusive = {}
            for capability in CORE_CAPABILITIES:
                excluded = torch.unique(
                    torch.cat(
                        [
                            top_indices[other]
                            for other in CORE_CAPABILITIES
                            if other != capability
                        ]
                    ),
                    sorted=True,
                )
                exclusive[capability] = _sorted_difference(
                    top_indices[capability], excluded
                )
            selections["capability"][fraction] = exclusive
            capability_top_sets[fraction] = top_indices
            top_counts[fraction] = {
                capability: int(top_indices[capability].numel())
                for capability in CORE_CAPABILITIES
            }
        if need_no_residual:
            selections["no_residual"][fraction] = {
                capability: _take_candidate_topk(
                    candidate_values[("no_residual", capability)],
                    candidate_indices[("no_residual", capability)],
                    k,
                )
                for capability in CORE_CAPABILITIES
            }
        if need_global:
            global_coordinates = _take_candidate_topk(
                candidate_values[("global_fisher", "all")],
                candidate_indices[("global_fisher", "all")],
                k,
            )
            selections["global_fisher"][fraction] = {
                capability: global_coordinates for capability in CORE_CAPABILITIES
            }
    return selections, top_counts, capability_top_sets


def _select_residual_top_and_exclusive(
    fisher_by_benchmark: Mapping[str, torch.Tensor],
    benchmark_capability: Mapping[str, str],
    top_fraction: float = DEFAULT_TOP_FRACTION,
    eps: float = LOG_EPS,
    chunk_size: int = SELECTION_CHUNK_SIZE,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Select exact residual-log-Fisher top sets and exclusive subsets."""
    selections, _, capability_top_sets = _select_fisher_coordinate_sets(
        fisher_by_benchmark,
        benchmark_capability,
        [top_fraction],
        ["capability"],
        eps=eps,
        chunk_size=chunk_size,
    )
    exclusive = selections["capability"][top_fraction]
    return capability_top_sets[top_fraction], exclusive


def select_exclusive_residual_top_coordinates(
    fisher_by_benchmark: Mapping[str, torch.Tensor],
    benchmark_capability: Mapping[str, str],
    top_fraction: float = DEFAULT_TOP_FRACTION,
    eps: float = LOG_EPS,
    chunk_size: int = SELECTION_CHUNK_SIZE,
) -> dict[str, torch.Tensor]:
    """Return capability top sets after excluding either other top set."""
    _, exclusive = _select_residual_top_and_exclusive(
        fisher_by_benchmark,
        benchmark_capability,
        top_fraction=top_fraction,
        eps=eps,
        chunk_size=chunk_size,
    )
    return exclusive


# Short public alias for callers that do not need the implementation wording.
select_capability_coordinates = select_exclusive_residual_top_coordinates


def sample_uniform_coordinates(
    n_coordinates: int, count: int, seed: int = CONTROL_SEED
) -> torch.Tensor:
    """Uniformly sample global coordinates without a model-sized permutation."""
    if n_coordinates < 0 or count < 0 or count > n_coordinates:
        raise ValueError("count must be between zero and n_coordinates")
    sampled = random.Random(seed).sample(range(n_coordinates), count)
    return torch.sort(torch.tensor(sampled, dtype=torch.int64)).values


def _parameter_magnitudes_at_coordinates(
    params: Sequence[tuple[str, torch.Tensor]], coordinates: torch.Tensor
) -> torch.Tensor:
    selected = torch.sort(
        torch.unique(coordinates.to(device="cpu", dtype=torch.int64))
    ).values
    total = sum(parameter.numel() for _, parameter in params)
    if selected.numel() and (int(selected[0]) < 0 or int(selected[-1]) >= total):
        raise IndexError("A magnitude-matching coordinate is outside the scope")
    pieces = []
    offset = 0
    for _, parameter in params:
        end = offset + parameter.numel()
        left = int(torch.searchsorted(selected, offset).item())
        right = int(torch.searchsorted(selected, end).item())
        if right > left:
            local = (selected[left:right] - offset).to(parameter.device)
            values = parameter.detach().reshape(-1).index_select(0, local)
            pieces.append(values.float().abs().to(device="cpu"))
        offset = end
    if not pieces:
        return torch.empty(0, dtype=torch.float32)
    result = torch.cat(pieces)
    if not bool(torch.isfinite(result).all()):
        raise ValueError("Selected weights contain non-finite magnitudes")
    return result


def _magnitude_quantile_edges(
    reference_magnitudes: torch.Tensor, n_bins: int
) -> torch.Tensor:
    reference = (
        reference_magnitudes.detach().reshape(-1).to(device="cpu", dtype=torch.float64)
    )
    if reference.numel() == 0:
        raise ValueError("reference_magnitudes must be nonempty")
    if n_bins <= 0 or not bool(torch.isfinite(reference).all()):
        raise ValueError("n_bins must be positive and magnitudes finite")
    quantiles = torch.linspace(0.0, 1.0, n_bins + 1, dtype=torch.float64)
    return torch.quantile(reference, quantiles)


def magnitude_quantile_bin_indices(
    magnitudes: torch.Tensor,
    reference_magnitudes: torch.Tensor,
    n_bins: int = MAGNITUDE_BINS,
) -> torch.Tensor:
    """Assign magnitudes to quantile bins defined by a reference sample."""
    values = magnitudes.detach().reshape(-1).to(device="cpu", dtype=torch.float64)
    if not bool(torch.isfinite(values).all()):
        raise ValueError("magnitudes must be finite")
    edges = _magnitude_quantile_edges(reference_magnitudes, n_bins)
    return torch.bucketize(values, edges[1:-1], right=False)


def _sample_magnitude_matched_from_parameters(
    params: Sequence[tuple[str, torch.Tensor]],
    capability_coordinates: torch.Tensor,
    n_bins: int,
    seed: int,
    chunk_size: int,
) -> torch.Tensor:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    selected = torch.sort(
        torch.unique(capability_coordinates.to(device="cpu", dtype=torch.int64))
    ).values
    if selected.numel() == 0:
        return selected
    total = sum(parameter.numel() for _, parameter in params)
    if int(selected[0]) < 0 or int(selected[-1]) >= total:
        raise IndexError("A magnitude-matching coordinate is outside the scope")

    selected_magnitudes = _parameter_magnitudes_at_coordinates(params, selected)
    edges = _magnitude_quantile_edges(selected_magnitudes, n_bins)
    selected_bins = torch.bucketize(
        selected_magnitudes.to(dtype=torch.float64), edges[1:-1], right=False
    )
    quotas = torch.bincount(selected_bins, minlength=n_bins).to(dtype=torch.int64)
    candidate_keys = [torch.empty(0, dtype=torch.float32) for _ in range(n_bins)]
    candidate_indices = [torch.empty(0, dtype=torch.int64) for _ in range(n_bins)]
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    offset = 0
    for _, parameter in params:
        flat = parameter.detach().reshape(-1)
        for local_start in range(0, flat.numel(), chunk_size):
            local_end = min(local_start + chunk_size, flat.numel())
            start = offset + local_start
            end = offset + local_end
            magnitudes = flat[local_start:local_end].float().abs().to(device="cpu")
            if not bool(torch.isfinite(magnitudes).all()):
                raise ValueError("Model weights contain non-finite magnitudes")
            bins = torch.bucketize(
                magnitudes.to(dtype=torch.float64), edges[1:-1], right=False
            )
            eligible = torch.ones(end - start, dtype=torch.bool)
            left = int(torch.searchsorted(selected, start).item())
            right = int(torch.searchsorted(selected, end).item())
            if right > left:
                eligible[selected[left:right] - start] = False
            keys = torch.rand(end - start, generator=generator)
            for bin_index in range(n_bins):
                quota = int(quotas[bin_index].item())
                if quota == 0:
                    continue
                mask = eligible & (bins == bin_index)
                if not bool(mask.any()):
                    continue
                local_indices = torch.nonzero(mask, as_tuple=False).reshape(-1)
                global_indices = local_indices.to(dtype=torch.int64).add(start)
                (
                    candidate_keys[bin_index],
                    candidate_indices[bin_index],
                ) = _merge_topk(
                    candidate_keys[bin_index],
                    candidate_indices[bin_index],
                    keys[local_indices],
                    global_indices,
                    quota,
                )
            del magnitudes, bins, eligible, keys
        offset += flat.numel()

    sampled = []
    for bin_index, quota_tensor in enumerate(quotas):
        quota = int(quota_tensor.item())
        if candidate_indices[bin_index].numel() != quota:
            raise RuntimeError(
                "Not enough non-capability coordinates in magnitude bin "
                f"{bin_index}: needed {quota}, found "
                f"{candidate_indices[bin_index].numel()}"
            )
        sampled.append(candidate_indices[bin_index])
    result = torch.sort(torch.cat(sampled)).values
    if result.numel() != selected.numel():
        raise AssertionError("Magnitude-matched sampler changed the set size")
    return result


def sample_magnitude_matched_coordinates(
    weights: torch.Tensor,
    capability_coordinates: torch.Tensor,
    n_bins: int = MAGNITUDE_BINS,
    seed: int = CONTROL_SEED,
    chunk_size: int = SELECTION_CHUNK_SIZE,
) -> torch.Tensor:
    """Sample a same-size complement matching capability-set |w| bin counts."""
    return _sample_magnitude_matched_from_parameters(
        [("weights", weights)],
        capability_coordinates,
        n_bins=n_bins,
        seed=seed,
        chunk_size=chunk_size,
    )


def _measurement_probes(n_probe: int) -> dict[str, list[dict[str, str]]]:
    """Combine odd-indexed (measurement-half) benchmarks by capability."""
    benchmark_probes = build_probe_registry(n_probe, seed=PROBE_SEED)
    combined = {capability: [] for capability in CORE_CAPABILITIES}
    for benchmark, samples in benchmark_probes.items():
        capability = BENCHMARK_CAPABILITY[benchmark]
        if capability in combined:
            combined[capability].extend(samples[1::2])
    empty = [capability for capability, samples in combined.items() if not samples]
    if empty:
        raise RuntimeError(
            "Measurement halves are empty for: "
            f"{', '.join(empty)}; increase --n-probe."
        )
    return combined


def _measure_losses(
    model,
    tokenizer,
    probes: Mapping[str, Sequence[dict[str, str]]],
    device: str,
) -> dict[str, float]:
    losses = {}
    with torch.no_grad():
        for capability in CORE_CAPABILITIES:
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
                    f"No completion tokens while measuring {capability!r}"
                )
            losses[capability] = total_loss / total_tokens
    return losses


def _validate_parameter_scope(
    params: Sequence[tuple[str, torch.Tensor]], metadata: Mapping
) -> int:
    names = [name for name, _ in params]
    numels = [parameter.numel() for _, parameter in params]
    expected_names = metadata.get("parameter_names")
    expected_numels = metadata.get("parameter_numels")
    if expected_names is not None and names != expected_names:
        raise RuntimeError(
            "Loaded model's language-weight parameter order does not match V9 "
            "Fisher metadata; refusing to ablate misaligned coordinates."
        )
    if expected_numels is not None and numels != expected_numels:
        raise RuntimeError(
            "Loaded model's parameter sizes do not match V9 Fisher metadata."
        )
    n_parameters = sum(numels)
    expected_total = metadata.get("n_parameters")
    if expected_total is not None and n_parameters != expected_total:
        raise RuntimeError(
            f"Loaded parameter count {n_parameters} does not match V9 "
            f"metadata {expected_total}."
        )
    return n_parameters


@contextmanager
def _zero_global_coordinates(
    params: Sequence[tuple[str, torch.Tensor]], coordinates: torch.Tensor
) -> Iterator[None]:
    """Temporarily zero sorted global coordinates and restore them exactly."""
    selected = torch.sort(
        torch.unique(coordinates.to(device="cpu", dtype=torch.int64))
    ).values
    total = sum(parameter.numel() for _, parameter in params)
    if selected.numel() and (int(selected[0]) < 0 or int(selected[-1]) >= total):
        raise IndexError("Ablation coordinate is outside the parameter scope")
    saved: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    offset = 0
    try:
        with torch.no_grad():
            for _, parameter in params:
                end = offset + parameter.numel()
                left = int(torch.searchsorted(selected, offset).item())
                right = int(torch.searchsorted(selected, end).item())
                if right > left:
                    if not parameter.is_contiguous():
                        raise RuntimeError(
                            "Cannot apply coordinate ablation to a non-contiguous "
                            "weight tensor."
                        )
                    local = (selected[left:right] - offset).to(parameter.device)
                    flat = parameter.view(-1)
                    original = flat.index_select(0, local).clone()
                    flat.index_fill_(0, local, 0)
                    saved.append((flat, local, original))
                offset = end
        yield
    finally:
        with torch.no_grad():
            for flat, local, original in saved:
                flat.index_copy_(0, local, original)


def _markdown_loss_matrix(
    title: str, rows: Mapping[str, Mapping[str, float]], signed: bool = False
) -> list[str]:
    lines = [
        f"## {title}",
        "",
        "| ablated capability | math | code | qa |",
        "|---|---:|---:|---:|",
    ]
    for ablated in CORE_CAPABILITIES:
        formatter = (
            (lambda value: f"{value:+.6f}")
            if signed
            else (lambda value: f"{value:.6f}")
        )
        values = " | ".join(
            formatter(rows[ablated][measured]) for measured in CORE_CAPABILITIES
        )
        lines.append(f"| {ablated} | {values} |")
    return lines


def _write_report(out: Path, payload: Mapping) -> None:
    lines = [
        f"# V9c capability-coordinate ablation — {payload['model']}",
        "",
        f"Top fraction per capability before exclusivity: "
        f"{payload['top_fraction']:.4%}. Coordinates are selected from the "
        "capability-average residual log-Fisher and removed if either other "
        "capability also selects them.",
        "",
        "## Selected coordinates",
        "",
        "| capability | top set | exclusive set | exclusive share of scope |",
        "|---|---:|---:|---:|",
    ]
    for capability in CORE_CAPABILITIES:
        lines.append(
            f"| {capability} | {payload['top_coordinate_counts'][capability]:,} | "
            f"{payload['exclusive_coordinate_counts'][capability]:,} | "
            f"{payload['exclusive_coordinate_fractions'][capability]:.6%} |"
        )
    dense_values = " | ".join(
        f"{payload['dense_losses'][capability]:.6f}" for capability in CORE_CAPABILITIES
    )
    lines.extend(
        [
            "",
            "## Dense completion loss",
            "",
            "| math | code | qa |",
            "|---:|---:|---:|",
            f"| {dense_values} |",
            "",
        ]
    )
    lines.extend(
        _markdown_loss_matrix("Ablated completion loss", payload["ablated_losses"])
    )
    lines.extend([""])
    lines.extend(
        _markdown_loss_matrix(
            "Damage relative to dense", payload["damage"], signed=True
        )
    )
    diagonal_wins = payload["diagonal_dominance"]
    lines.extend(
        [
            "",
            "## Prediction check",
            "",
            "The gold-standard prediction is row-wise diagonal dominance: "
            "deleting a capability-exclusive region should damage that "
            "capability more than either of the others.",
            "",
        ]
    )
    for capability in CORE_CAPABILITIES:
        lines.append(f"- {capability}: {'yes' if diagonal_wins[capability] else 'no'}")
    (out / "report.md").write_text("\n".join(lines) + "\n")


def _selectivity(damage: Mapping[str, Mapping[str, float]]) -> dict[str, dict]:
    result = {}
    for capability in CORE_CAPABILITIES:
        target = float(damage[capability][capability])
        off_target = max(
            float(damage[capability][measured])
            for measured in CORE_CAPABILITIES
            if measured != capability
        )
        result[capability] = {
            "target_damage": target,
            "max_off_target_damage": off_target,
            "target_over_max_off_target": (
                target / off_target if off_target != 0 else None
            ),
        }
    return result


def _normalize_modes(modes: Sequence[str]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    requested = tuple(dict.fromkeys(modes or ("capability",)))
    invalid = set(requested) - set(ABLATION_MODES)
    if invalid:
        raise ValueError(f"Unknown ablation modes: {sorted(invalid)}")
    evaluated = requested
    if (
        any(mode != "capability" for mode in requested)
        and "capability" not in requested
    ):
        evaluated = ("capability", *requested)
    return requested, evaluated


def _write_controls_report(out: Path, payload: Mapping) -> None:
    fraction_labels = ", ".join(
        f"{fraction:.4%}" for fraction in payload["top_fractions"]
    )
    lines = [
        f"# V9c Experiment B1 controls — {payload['model']}",
        "",
        "Every deletion below starts from the same loaded dense snapshot. "
        "The context manager restores selected weights exactly before the next "
        "mode/fraction pair.",
        "",
        f"Modes evaluated: {', '.join(payload['evaluated_modes'])}. Top "
        f"fractions: {fraction_labels}.",
        "",
        "## Dense completion loss",
        "",
        "| math | code | qa |",
        "|---:|---:|---:|",
        "| "
        + " | ".join(
            f"{payload['dense_losses'][capability]:.6f}"
            for capability in CORE_CAPABILITIES
        )
        + " |",
    ]
    for mode in payload["evaluated_modes"]:
        for fraction_key, result in payload["results"][mode].items():
            lines.extend(
                [
                    "",
                    f"## {mode} — top fraction {result['top_fraction_label']}",
                    "",
                    "| ablated capability | selected coordinates | share of scope |",
                    "|---|---:|---:|",
                ]
            )
            for capability in CORE_CAPABILITIES:
                lines.append(
                    f"| {capability} | "
                    f"{result['selected_coordinate_counts'][capability]:,} | "
                    f"{result['selected_coordinate_fractions'][capability]:.6%} |"
                )
            lines.extend([""])
            lines.extend(
                _markdown_loss_matrix(
                    "Ablated completion loss", result["ablated_losses"]
                )
            )
            lines.extend([""])
            lines.extend(
                _markdown_loss_matrix(
                    "Damage relative to dense", result["damage"], signed=True
                )
            )
            lines.extend(
                [
                    "",
                    "| target capability | target damage | max off-target damage | selectivity |",
                    "|---|---:|---:|---:|",
                ]
            )
            for capability in CORE_CAPABILITIES:
                item = result["selectivity"][capability]
                ratio = item["target_over_max_off_target"]
                ratio_text = "n/a" if ratio is None else f"{ratio:.4f}"
                lines.append(
                    f"| {capability} | {item['target_damage']:+.6f} | "
                    f"{item['max_off_target_damage']:+.6f} | {ratio_text} |"
                )

    lines.extend(
        [
            "",
            "## Matched-fraction selectivity comparison",
            "",
            "Selectivity is target damage divided by the maximum off-target "
            "damage in the same ablation row.",
            "",
        ]
    )
    comparisons = payload["selectivity_comparisons"]
    if comparisons:
        lines.extend(
            [
                "| top fraction | capability | control | capability selectivity | control selectivity | difference |",
                "|---:|---|---|---:|---:|---:|",
            ]
        )
        for item in comparisons:
            baseline = item["capability_selectivity"]
            control = item["control_selectivity"]
            difference = item["selectivity_difference"]
            format_optional = lambda value: "n/a" if value is None else f"{value:.4f}"
            lines.append(
                f"| {item['top_fraction_label']} | {item['capability']} | "
                f"{item['control_mode']} | {format_optional(baseline)} | "
                f"{format_optional(control)} | {format_optional(difference)} |"
            )
    else:
        lines.append("No control mode was requested alongside capability mode.")
    (out / "report_controls.md").write_text("\n".join(lines) + "\n")


def _write_legacy_capability_outputs(
    out: Path,
    payload: Mapping,
    top_counts: Mapping[str, int],
    fraction_key: str,
) -> None:
    result = payload["results"]["capability"][fraction_key]
    legacy = {
        "version": "9c",
        "model": payload["model"],
        "model_dtype": payload["model_dtype"],
        "n_probe_requested": payload["n_probe_requested"],
        "measurement_half": payload["measurement_half"],
        "top_fraction": result["top_fraction"],
        "log_eps": payload["log_eps"],
        "n_parameters": payload["n_parameters"],
        "fisher_artifact_dir": payload["fisher_artifact_dir"],
        "top_coordinate_counts": dict(top_counts),
        "exclusive_coordinate_counts": result["selected_coordinate_counts"],
        "exclusive_coordinate_fractions": result["selected_coordinate_fractions"],
        "measurement_samples": payload["measurement_samples"],
        "dense_losses": payload["dense_losses"],
        "ablated_losses": result["ablated_losses"],
        "damage": result["damage"],
        "diagonal_dominance": result["diagonal_dominance"],
    }
    (out / "ablation.json").write_text(json.dumps(legacy, indent=2) + "\n")
    _write_report(out, legacy)


def run_control_ablations(
    model_name: str,
    device: str,
    n_probe: int,
    top_fractions: Sequence[float],
    modes: Sequence[str],
    fisher_out: Path,
    out: Path,
    model_dtype: str = "fp32",
    seed: int = CONTROL_SEED,
) -> dict:
    fractions = parse_top_fractions(top_fractions)
    requested_modes, evaluated_modes = _normalize_modes(modes)
    inventory, fisher_metadata = _analysis_inventory(fisher_out)
    fisher_by_benchmark = {
        item["name"]: _load_tensor_mmap(item["path"]) for item in inventory
    }
    capability_by_benchmark = {item["name"]: item["capability"] for item in inventory}
    selections, top_counts, _ = _select_fisher_coordinate_sets(
        fisher_by_benchmark,
        capability_by_benchmark,
        fractions,
        evaluated_modes,
    )
    n_parameters = next(iter(fisher_by_benchmark.values())).numel()
    del fisher_by_benchmark
    gc.collect()

    probes = _measurement_probes(n_probe)
    dtype = torch.float32 if model_dtype == "fp32" else torch.bfloat16
    model, tokenizer = load_text_causal_lm(model_name, dtype)
    model.to(device).eval()
    params = language_weight_parameters(model)
    loaded_parameters = _validate_parameter_scope(params, fisher_metadata)
    if loaded_parameters != n_parameters:
        raise RuntimeError(
            f"Fisher coordinate count {n_parameters} does not match loaded "
            f"parameter count {loaded_parameters}."
        )

    if "random" in evaluated_modes:
        selections["random"] = {}
        for fraction in fractions:
            selections["random"][fraction] = {}
            for capability in CORE_CAPABILITIES:
                count = selections["capability"][fraction][capability].numel()
                control_seed = _stable_control_seed(
                    seed, "random", _fraction_key(fraction), capability
                )
                selections["random"][fraction][capability] = sample_uniform_coordinates(
                    n_parameters, count, control_seed
                )
    if "magnitude_matched" in evaluated_modes:
        selections["magnitude_matched"] = {}
        for fraction in fractions:
            selections["magnitude_matched"][fraction] = {}
            for capability in CORE_CAPABILITIES:
                control_seed = _stable_control_seed(
                    seed,
                    "magnitude_matched",
                    _fraction_key(fraction),
                    capability,
                )
                selections["magnitude_matched"][fraction][
                    capability
                ] = _sample_magnitude_matched_from_parameters(
                    params,
                    selections["capability"][fraction][capability],
                    n_bins=MAGNITUDE_BINS,
                    seed=control_seed,
                    chunk_size=SELECTION_CHUNK_SIZE,
                )

    dense_losses = _measure_losses(model, tokenizer, probes, device)
    print(f"[ablation] dense: {dense_losses}", flush=True)
    results = {}
    measurement_cache: dict[tuple[int, int], dict[str, float]] = {}
    for mode in evaluated_modes:
        results[mode] = {}
        for fraction in fractions:
            fraction_key = _fraction_key(fraction)
            selected = selections[mode][fraction]
            ablated_losses = {}
            for ablated in CORE_CAPABILITIES:
                coordinates = selected[ablated]
                cache_key = (coordinates.data_ptr(), coordinates.numel())
                if cache_key not in measurement_cache:
                    with _zero_global_coordinates(params, coordinates):
                        measurement_cache[cache_key] = _measure_losses(
                            model, tokenizer, probes, device
                        )
                ablated_losses[ablated] = dict(measurement_cache[cache_key])
                print(
                    f"[ablation] {mode} {fraction_key} zero {ablated} "
                    f"({coordinates.numel():,} coordinates): "
                    f"{ablated_losses[ablated]}",
                    flush=True,
                )
            damage = {
                ablated: {
                    measured: (
                        ablated_losses[ablated][measured] - dense_losses[measured]
                    )
                    for measured in CORE_CAPABILITIES
                }
                for ablated in CORE_CAPABILITIES
            }
            diagonal_dominance = {
                ablated: damage[ablated][ablated]
                > max(
                    damage[ablated][measured]
                    for measured in CORE_CAPABILITIES
                    if measured != ablated
                )
                for ablated in CORE_CAPABILITIES
            }
            results[mode][fraction_key] = {
                "top_fraction": fraction,
                "top_fraction_label": f"{fraction:.4%}",
                "selected_coordinate_counts": {
                    capability: int(selected[capability].numel())
                    for capability in CORE_CAPABILITIES
                },
                "selected_coordinate_fractions": {
                    capability: selected[capability].numel() / n_parameters
                    for capability in CORE_CAPABILITIES
                },
                "ablated_losses": ablated_losses,
                "damage": damage,
                "diagonal_dominance": diagonal_dominance,
                "selectivity": _selectivity(damage),
            }

    comparisons = []
    if "capability" in results:
        for fraction in fractions:
            fraction_key = _fraction_key(fraction)
            baseline = results["capability"][fraction_key]["selectivity"]
            for mode in evaluated_modes:
                if mode == "capability":
                    continue
                control = results[mode][fraction_key]["selectivity"]
                for capability in CORE_CAPABILITIES:
                    baseline_ratio = baseline[capability]["target_over_max_off_target"]
                    control_ratio = control[capability]["target_over_max_off_target"]
                    comparisons.append(
                        {
                            "top_fraction": fraction,
                            "top_fraction_label": f"{fraction:.4%}",
                            "capability": capability,
                            "control_mode": mode,
                            "capability_selectivity": baseline_ratio,
                            "control_selectivity": control_ratio,
                            "selectivity_difference": (
                                baseline_ratio - control_ratio
                                if baseline_ratio is not None
                                and control_ratio is not None
                                else None
                            ),
                        }
                    )

    payload = {
        "version": "9c-b1",
        "experiment": "B1",
        "model": model_name,
        "model_dtype": model_dtype,
        "n_probe_requested": n_probe,
        "measurement_half": "odd-indexed samples",
        "requested_modes": list(requested_modes),
        "evaluated_modes": list(evaluated_modes),
        "top_fractions": fractions,
        "control_seed": seed,
        "magnitude_quantile_bins": MAGNITUDE_BINS,
        "log_eps": LOG_EPS,
        "n_parameters": n_parameters,
        "fisher_artifact_dir": str(fisher_out),
        "measurement_samples": {
            capability: len(probes[capability]) for capability in CORE_CAPABILITIES
        },
        "dense_losses": dense_losses,
        "results": results,
        "selectivity_comparisons": comparisons,
    }
    (out / "controls.json").write_text(json.dumps(payload, indent=2) + "\n")
    _write_controls_report(out, payload)
    if "capability" in results:
        first_fraction = fractions[0]
        _write_legacy_capability_outputs(
            out,
            payload,
            top_counts[first_fraction],
            _fraction_key(first_fraction),
        )
    print(
        f"[ablation] wrote {out / 'controls.json'} and "
        f"{out / 'report_controls.md'}",
        flush=True,
    )

    del params, model, tokenizer
    gc.collect()
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return payload


def run_ablation(
    model_name: str,
    device: str,
    n_probe: int,
    top_fraction: float,
    fisher_out: Path,
    out: Path,
    model_dtype: str = "fp32",
) -> None:
    """Backward-compatible capability-only entry point."""
    run_control_ablations(
        model_name,
        device,
        n_probe,
        [top_fraction],
        ["capability"],
        fisher_out,
        out,
        model_dtype=model_dtype,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="gemma3-1b",
        help="model-registry tag or raw Hugging Face model id",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model-dtype", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--n-probe", type=int, default=64)
    parser.add_argument(
        "--mode",
        dest="modes",
        action="append",
        choices=ABLATION_MODES,
        help=(
            "ablation mode; repeat to evaluate multiple controls from the same "
            "dense snapshot (default: capability)"
        ),
    )
    fractions = parser.add_mutually_exclusive_group()
    fractions.add_argument(
        "--top-fracs",
        type=parse_top_fractions,
        help="comma-separated top fractions (default: 0.0005)",
    )
    fractions.add_argument(
        "--top-frac",
        type=float,
        help="legacy alias for one top fraction",
    )
    parser.add_argument("--seed", type=int, default=CONTROL_SEED)
    args = parser.parse_args()

    model_name = require_compliant(args.model)
    tag = model_output_tag(args.model, model_name)
    fisher_out = V9_OUT_BASE / tag
    out = OUT_BASE / tag
    out.mkdir(parents=True, exist_ok=True)
    top_fractions = (
        args.top_fracs
        if args.top_fracs is not None
        else [args.top_frac if args.top_frac is not None else DEFAULT_TOP_FRACTION]
    )
    run_control_ablations(
        model_name,
        args.device,
        args.n_probe,
        top_fractions,
        args.modes or ["capability"],
        fisher_out,
        out,
        model_dtype=args.model_dtype,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
