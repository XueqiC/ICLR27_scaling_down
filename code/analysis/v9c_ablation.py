#!/usr/bin/env python3
"""V9c: causal deletion test for capability-exclusive Fisher coordinates.

For each of math, code, and QA, this script computes the capability-average
residual log-Fisher from V9's saved per-benchmark Fisher artifacts.  A
capability region is the exact top fraction of that residual excluding every
coordinate selected by either other capability.  The selected weights are
temporarily zeroed, completion loss is measured on held-out probe halves, and
the original weights are restored before testing the next capability.

Artifacts are written under results/v9c-ablation/<model_tag>/.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Mapping, Sequence

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


def _sorted_difference(
    selected: torch.Tensor, excluded: torch.Tensor
) -> torch.Tensor:
    selected = torch.sort(selected.to(device="cpu", dtype=torch.int64)).values
    excluded = torch.sort(excluded.to(device="cpu", dtype=torch.int64)).values
    if selected.numel() == 0 or excluded.numel() == 0:
        return selected
    positions = torch.searchsorted(excluded, selected)
    present = positions < excluded.numel()
    matches = torch.zeros_like(present)
    matches[present] = excluded[positions[present]] == selected[present]
    return selected[~matches]


def _select_residual_top_and_exclusive(
    fisher_by_benchmark: Mapping[str, torch.Tensor],
    benchmark_capability: Mapping[str, str],
    top_fraction: float = DEFAULT_TOP_FRACTION,
    eps: float = LOG_EPS,
    chunk_size: int = SELECTION_CHUNK_SIZE,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Select exact residual-log-Fisher top sets and exclusive subsets."""
    if not fisher_by_benchmark:
        raise ValueError("At least one Fisher vector is required")
    if not 0 < top_fraction <= 1:
        raise ValueError("top_fraction must be in (0, 1]")
    if eps <= 0 or chunk_size <= 0:
        raise ValueError("eps and chunk_size must be positive")
    names = list(fisher_by_benchmark)
    vectors = [fisher_by_benchmark[name].reshape(-1) for name in names]
    n_coordinates = vectors[0].numel()
    if n_coordinates == 0 or any(
        vector.numel() != n_coordinates for vector in vectors
    ):
        raise ValueError("Fisher vectors must be nonempty and equally sized")
    capability_counts = {
        capability: sum(
            benchmark_capability.get(name) == capability for name in names
        )
        for capability in CORE_CAPABILITIES
    }
    missing = [
        capability
        for capability, count in capability_counts.items()
        if count == 0
    ]
    if missing:
        raise ValueError(f"No Fisher benchmarks for capabilities: {', '.join(missing)}")

    k = max(1, int(math.floor(n_coordinates * top_fraction)))
    candidate_values = {
        capability: torch.empty(0, dtype=torch.float32)
        for capability in CORE_CAPABILITIES
    }
    candidate_indices = {
        capability: torch.empty(0, dtype=torch.int64)
        for capability in CORE_CAPABILITIES
    }

    for start in range(0, n_coordinates, chunk_size):
        end = min(start + chunk_size, n_coordinates)
        shared_sum = torch.zeros(end - start, dtype=torch.float32)
        capability_sums = {
            capability: torch.zeros(end - start, dtype=torch.float32)
            for capability in CORE_CAPABILITIES
        }
        for name, vector in zip(names, vectors):
            chunk = vector[start:end]
            if not bool(torch.isfinite(chunk).all()):
                raise ValueError(f"Fisher vector {name!r} contains non-finite values")
            if bool((chunk < 0).any()):
                raise ValueError(f"Fisher vector {name!r} contains negative values")
            logged = chunk.to(device="cpu", dtype=torch.float32, copy=True)
            logged.add_(eps).log_()
            shared_sum.add_(logged)
            capability = benchmark_capability.get(name)
            if capability in capability_sums:
                capability_sums[capability].add_(logged)
            del logged
        shared_sum.div_(len(vectors))
        for capability in CORE_CAPABILITIES:
            residual = capability_sums[capability]
            residual.div_(capability_counts[capability]).sub_(shared_sum)
            local_k = min(k, residual.numel())
            values, indices = torch.topk(
                residual, local_k, largest=True, sorted=False
            )
            indices.add_(start)
            (
                candidate_values[capability],
                candidate_indices[capability],
            ) = _merge_topk(
                candidate_values[capability],
                candidate_indices[capability],
                values,
                indices,
                k,
            )
        del shared_sum, capability_sums

    top_indices = {
        capability: torch.sort(candidate_indices[capability]).values
        for capability in CORE_CAPABILITIES
    }
    exclusive = {}
    for capability in CORE_CAPABILITIES:
        other_sets = [
            top_indices[other]
            for other in CORE_CAPABILITIES
            if other != capability
        ]
        excluded = torch.unique(torch.cat(other_sets), sorted=True)
        exclusive[capability] = _sorted_difference(
            top_indices[capability], excluded
        )
    return top_indices, exclusive


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
    if selected.numel() and (
        int(selected[0]) < 0 or int(selected[-1]) >= total
    ):
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
        formatter = (lambda value: f"{value:+.6f}") if signed else (
            lambda value: f"{value:.6f}"
        )
        values = " | ".join(
            formatter(rows[ablated][measured])
            for measured in CORE_CAPABILITIES
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
        f"{payload['dense_losses'][capability]:.6f}"
        for capability in CORE_CAPABILITIES
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
        _markdown_loss_matrix(
            "Ablated completion loss", payload["ablated_losses"]
        )
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
        lines.append(
            f"- {capability}: {'yes' if diagonal_wins[capability] else 'no'}"
        )
    (out / "report.md").write_text("\n".join(lines) + "\n")


def run_ablation(
    model_name: str,
    device: str,
    n_probe: int,
    top_fraction: float,
    fisher_out: Path,
    out: Path,
    model_dtype: str = "fp32",
) -> None:
    inventory, fisher_metadata = _analysis_inventory(fisher_out)
    fisher_by_benchmark = {
        item["name"]: _load_tensor_mmap(item["path"]) for item in inventory
    }
    capability_by_benchmark = {
        item["name"]: item["capability"] for item in inventory
    }
    top_indices, exclusive = _select_residual_top_and_exclusive(
        fisher_by_benchmark,
        capability_by_benchmark,
        top_fraction=top_fraction,
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

    dense_losses = _measure_losses(model, tokenizer, probes, device)
    print(f"[ablation] dense: {dense_losses}", flush=True)
    ablated_losses = {}
    for ablated in CORE_CAPABILITIES:
        with _zero_global_coordinates(params, exclusive[ablated]):
            ablated_losses[ablated] = _measure_losses(
                model, tokenizer, probes, device
            )
        print(
            f"[ablation] zero {ablated} ({exclusive[ablated].numel():,} "
            f"coordinates): {ablated_losses[ablated]}",
            flush=True,
        )
    damage = {
        ablated: {
            measured: ablated_losses[ablated][measured] - dense_losses[measured]
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
    payload = {
        "version": "9c",
        "model": model_name,
        "model_dtype": model_dtype,
        "n_probe_requested": n_probe,
        "measurement_half": "odd-indexed samples",
        "top_fraction": top_fraction,
        "log_eps": LOG_EPS,
        "n_parameters": n_parameters,
        "fisher_artifact_dir": str(fisher_out),
        "top_coordinate_counts": {
            capability: int(top_indices[capability].numel())
            for capability in CORE_CAPABILITIES
        },
        "exclusive_coordinate_counts": {
            capability: int(exclusive[capability].numel())
            for capability in CORE_CAPABILITIES
        },
        "exclusive_coordinate_fractions": {
            capability: exclusive[capability].numel() / n_parameters
            for capability in CORE_CAPABILITIES
        },
        "measurement_samples": {
            capability: len(probes[capability]) for capability in CORE_CAPABILITIES
        },
        "dense_losses": dense_losses,
        "ablated_losses": ablated_losses,
        "damage": damage,
        "diagonal_dominance": diagonal_dominance,
    }
    (out / "ablation.json").write_text(json.dumps(payload, indent=2) + "\n")
    _write_report(out, payload)
    print(f"[ablation] wrote {out / 'ablation.json'} and {out / 'report.md'}")

    del params, model, tokenizer
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
    parser.add_argument("--model-dtype", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--n-probe", type=int, default=64)
    parser.add_argument("--top-frac", type=float, default=DEFAULT_TOP_FRACTION)
    args = parser.parse_args()

    model_name = require_compliant(args.model)
    tag = model_output_tag(args.model, model_name)
    fisher_out = V9_OUT_BASE / tag
    out = OUT_BASE / tag
    out.mkdir(parents=True, exist_ok=True)
    run_ablation(
        model_name,
        args.device,
        args.n_probe,
        args.top_frac,
        fisher_out,
        out,
        model_dtype=args.model_dtype,
    )


if __name__ == "__main__":
    main()
