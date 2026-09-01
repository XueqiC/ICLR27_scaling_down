#!/usr/bin/env python3
"""V9: test whether capabilities occupy distinct regions of gradient space.

The Fisher stage estimates one diagonal empirical-Fisher vector per benchmark
from reference-completion losses.  The analyze stage compares the saved
vectors with raw, log-space, shared-component-removed, and top-coordinate
similarities and asks whether same-capability benchmarks form blocks.

Examples:
  python3 analysis/v9_capability_regions.py --model gemma3-1b --stage fisher
  python3 analysis/v9_capability_regions.py --model gemma3-1b --stage analyze

Artifacts are written under results/v9-capability-regions/<model_tag>/.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import torch

try:
    from .v6_capability_geometry import (
        completion_loss,
        language_weight_parameters,
        load_text_causal_lm,
        model_output_tag,
        require_compliant,
    )
except ImportError:  # direct execution: python analysis/v9_capability_regions.py
    from v6_capability_geometry import (
        completion_loss,
        language_weight_parameters,
        load_text_causal_lm,
        model_output_tag,
        require_compliant,
    )


ROOT = Path(__file__).resolve().parents[1]
OUT_BASE = ROOT / "results/v9-capability-regions"
PROBE_SEED = 0
LOG_EPS = 1e-12
TOP_FRACTION = 0.001
DEFAULT_CHUNK_SIZE = 50_000_000
CORE_CAPABILITIES = ("math", "code", "qa")


@dataclass(frozen=True)
class ProbeSpec:
    name: str
    capability: str
    builder: Callable[[int, int], list[dict[str, str]]]


def _sample_indices(length: int, n: int, seed: int) -> np.ndarray:
    if length <= 0:
        return np.empty(0, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return rng.choice(length, size=min(n, length), replace=False)


def _context_text(context: object, max_chars: int = 4000) -> str:
    """Render 2Wiki/Hotpot-style title + sentence context structures."""
    parts: list[str] = []
    if isinstance(context, dict):
        titles = context.get("title", [])
        sentence_groups = context.get(
            "sentences", context.get("content", context.get("sentence", []))
        )
        for title, sentences in zip(titles, sentence_groups):
            if isinstance(sentences, (list, tuple)):
                body = " ".join(str(sentence) for sentence in sentences)
            else:
                body = str(sentences)
            parts.append(f"{title}: {body}")
    elif isinstance(context, (list, tuple)):
        for item in context:
            if isinstance(item, dict):
                title = item.get("title", "")
                sentences = item.get(
                    "sentences", item.get("content", item.get("sentence", ""))
                )
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                title, sentences = item[0], item[1]
            else:
                parts.append(str(item))
                continue
            if isinstance(sentences, (list, tuple)):
                body = " ".join(str(sentence) for sentence in sentences)
            else:
                body = str(sentences)
            parts.append(f"{title}: {body}")
    else:
        parts.append(str(context))
    return "\n".join(parts)[:max_chars]


def build_gsm8k_probes(n: int, seed: int = PROBE_SEED) -> list[dict[str, str]]:
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="test")
    return [
        {
            "prompt": f"Problem: {ds[int(i)]['question']}\nSolution:",
            "completion": " " + str(ds[int(i)]["answer"]),
        }
        for i in _sample_indices(len(ds), n, seed)
    ]


def build_math500_probes(n: int, seed: int = PROBE_SEED) -> list[dict[str, str]]:
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    return [
        {
            "prompt": f"Problem: {ds[int(i)]['problem']}\nSolution:",
            "completion": " " + str(ds[int(i)]["solution"]),
        }
        for i in _sample_indices(len(ds), n, seed)
    ]


def build_svamp_probes(n: int, seed: int = PROBE_SEED) -> list[dict[str, str]]:
    from datasets import load_dataset

    ds = load_dataset("ChilleD/SVAMP", split="test")
    probes = []
    for i in _sample_indices(len(ds), n, seed):
        row = dict(ds[int(i)])
        question = row.get("question_concat")
        if not question:
            body = row.get("Body", row.get("body"))
            tail = row.get("Question", row.get("question"))
            if body is None or tail is None:
                raise KeyError(
                    "SVAMP needs question_concat or both Body and Question; "
                    f"available fields are {sorted(row)}"
                )
            question = f"{body} {tail}".strip()
        answer = row.get("Answer", row.get("answer"))
        if answer is None:
            raise KeyError(
                f"SVAMP Answer field missing; available fields are {sorted(row)}"
            )
        probes.append(
            {
                "prompt": f"Problem: {question}\nSolution:",
                "completion": str(answer),
            }
        )
    return probes


def build_humaneval_probes(n: int, seed: int = PROBE_SEED) -> list[dict[str, str]]:
    from datasets import load_dataset

    ds = load_dataset("openai/openai_humaneval", split="test")
    return [
        {
            "prompt": str(ds[int(i)]["prompt"]),
            "completion": str(ds[int(i)]["canonical_solution"]),
        }
        for i in _sample_indices(len(ds), n, seed)
    ]


def build_mbpp_probes(n: int, seed: int = PROBE_SEED) -> list[dict[str, str]]:
    from datasets import load_dataset

    ds = load_dataset("google-research-datasets/mbpp", "full", split="test")
    return [
        {
            "prompt": (
                f"# Task: {ds[int(i)]['text']}\n# Write a Python function.\n"
            ),
            "completion": str(ds[int(i)]["code"]),
        }
        for i in _sample_indices(len(ds), n, seed)
    ]


def build_2wiki_probes(n: int, seed: int = PROBE_SEED) -> list[dict[str, str]]:
    from datasets import load_dataset

    ds = load_dataset("framolfese/2WikiMultihopQA", split="validation")
    probes = []
    for i in _sample_indices(len(ds), n, seed):
        row = dict(ds[int(i)])
        context = _context_text(row["context"])
        probes.append(
            {
                "prompt": (
                    f"Context:\n{context}\n\nQuestion: {row['question']}\nAnswer:"
                ),
                "completion": " " + str(row["answer"]),
            }
        )
    return probes


def build_hotpotqa_probes(n: int, seed: int = PROBE_SEED) -> list[dict[str, str]]:
    from datasets import load_dataset

    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    probes = []
    for i in _sample_indices(len(ds), n, seed):
        row = dict(ds[int(i)])
        context = _context_text(row["context"])
        probes.append(
            {
                "prompt": (
                    f"Context:\n{context}\n\nQuestion: {row['question']}\nAnswer:"
                ),
                "completion": " " + str(row["answer"]),
            }
        )
    return probes


def build_triviaqa_probes(n: int, seed: int = PROBE_SEED) -> list[dict[str, str]]:
    from datasets import load_dataset

    ds = load_dataset(
        "mandarjoshi/trivia_qa", "rc.nocontext", split="validation"
    )
    probes = []
    for i in _sample_indices(len(ds), n, seed):
        row = dict(ds[int(i)])
        answer = row["answer"]
        if not isinstance(answer, dict) or "value" not in answer:
            raise KeyError(
                "TriviaQA answer must be a mapping with a 'value' field; "
                f"got {type(answer).__name__}"
            )
        probes.append(
            {
                "prompt": f"Question: {row['question']}\nAnswer:",
                "completion": " " + str(answer["value"]),
            }
        )
    return probes


def build_c4_probes(n: int, seed: int = PROBE_SEED) -> list[dict[str, str]]:
    del seed  # C4 streaming order is deterministic; take the first n documents.
    from datasets import load_dataset

    ds = load_dataset("allenai/c4", "en", split="validation", streaming=True)
    probes = []
    for row in ds.take(n):
        text = str(row["text"])
        if len(text) <= 200:
            continue
        probes.append({"prompt": text[:200], "completion": text[200:600]})
    return probes


PROBE_REGISTRY: tuple[ProbeSpec, ...] = (
    ProbeSpec("gsm8k", "math", build_gsm8k_probes),
    ProbeSpec("math500", "math", build_math500_probes),
    ProbeSpec("svamp", "math", build_svamp_probes),
    ProbeSpec("humaneval", "code", build_humaneval_probes),
    ProbeSpec("mbpp", "code", build_mbpp_probes),
    ProbeSpec("2wiki", "qa", build_2wiki_probes),
    ProbeSpec("hotpotqa", "qa", build_hotpotqa_probes),
    ProbeSpec("triviaqa", "qa", build_triviaqa_probes),
    ProbeSpec("c4", "control", build_c4_probes),
)
BENCHMARK_CAPABILITY = {spec.name: spec.capability for spec in PROBE_REGISTRY}


def _validate_probes(name: str, probes: object) -> list[dict[str, str]]:
    if not isinstance(probes, list) or not probes:
        raise ValueError(f"{name} returned no usable probe samples")
    for index, sample in enumerate(probes):
        if not isinstance(sample, dict):
            raise TypeError(f"sample {index} is not a dictionary")
        for field in ("prompt", "completion"):
            if not isinstance(sample.get(field), str) or not sample[field]:
                raise ValueError(
                    f"sample {index} has an empty/non-string {field!r} field"
                )
    return probes


def _require_probe_coverage(capabilities: Iterable[str]) -> None:
    counts: dict[str, int] = {}
    for capability in capabilities:
        counts[capability] = counts.get(capability, 0) + 1
    insufficient = [
        capability
        for capability in CORE_CAPABILITIES
        if counts.get(capability, 0) < 2
    ]
    if insufficient:
        detail = ", ".join(
            f"{capability}={counts.get(capability, 0)}" for capability in insufficient
        )
        raise RuntimeError(
            "At least two surviving benchmarks are required for each of math, "
            f"code, and qa; insufficient coverage: {detail}."
        )
    if counts.get("control", 0) < 1:
        raise RuntimeError(
            "The C4 control benchmark is required for shared-component analysis."
        )


def build_probe_registry(
    n: int, seed: int = PROBE_SEED
) -> dict[str, list[dict[str, str]]]:
    """Build all available benchmarks, warning and skipping failed datasets."""
    if n <= 0:
        raise ValueError("n must be positive")
    probes: dict[str, list[dict[str, str]]] = {}
    for spec in PROBE_REGISTRY:
        try:
            samples = _validate_probes(spec.name, spec.builder(n, seed))
        except Exception as exc:  # dataset/schema failures must not kill registry build
            print(
                f"WARNING: skipping {spec.name} ({spec.capability}): "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
                flush=True,
            )
            continue
        probes[spec.name] = samples
        if len(samples) < n:
            print(
                f"WARNING: {spec.name} produced {len(samples)} of {n} requested "
                "probe samples.",
                file=sys.stderr,
                flush=True,
            )
    _require_probe_coverage(BENCHMARK_CAPABILITY[name] for name in probes)
    return probes


def _write_fisher_metadata(out: Path, metadata: dict) -> None:
    (out / "fisher_meta.json").write_text(json.dumps(metadata, indent=2) + "\n")


def stage_fisher(
    model_name: str,
    device: str,
    n_probe: int,
    out: Path,
    model_dtype: str = "fp32",
) -> None:
    """Estimate and save one flattened CPU fp32 Fisher per benchmark."""
    probes = build_probe_registry(n_probe, seed=PROBE_SEED)
    dtype = torch.float32 if model_dtype == "fp32" else torch.bfloat16
    model, tokenizer = load_text_causal_lm(model_name, dtype)
    model.to(device).eval()
    model.requires_grad_(False)
    params = language_weight_parameters(model)
    for _, param in params:
        param.requires_grad_(True)

    n_parameters = sum(param.numel() for _, param in params)
    if n_parameters == 0:
        raise RuntimeError("No language weight matrices were found for Fisher scope.")
    metadata = {
        "version": 9,
        "model": model_name,
        "model_dtype": model_dtype,
        "n_probe_requested": n_probe,
        "probe_seed": PROBE_SEED,
        "n_parameters": n_parameters,
        "parameter_names": [name for name, _ in params],
        "parameter_numels": [param.numel() for _, param in params],
        "benchmarks": [],
    }
    # resume support: keep benchmarks whose artifact already exists (e.g.
    # after a walltime kill) instead of recomputing hours of gradients
    metadata_path = out / "fisher_meta.json"
    if metadata_path.exists():
        try:
            prior = json.loads(metadata_path.read_text())
            if prior.get("n_parameters") == n_parameters:
                metadata["benchmarks"] = [
                    b for b in prior.get("benchmarks", [])
                    if (out / b.get("file", "")).exists()
                ]
        except Exception:
            pass
    done_benchmarks = {b["name"] for b in metadata["benchmarks"]}
    _write_fisher_metadata(out, metadata)

    for benchmark, samples in probes.items():
        if benchmark in done_benchmarks:
            print(f"[fisher] {benchmark}: artifact exists, skipping",
                  flush=True)
            continue
        capability = BENCHMARK_CAPABILITY[benchmark]
        fisher = torch.zeros(n_parameters, dtype=torch.float32, device="cpu")
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
                for _, param in params:
                    end = offset + param.numel()
                    if param.grad is not None:
                        grad_squared = param.grad.detach().float().square()
                        fisher[offset:end].add_(
                            grad_squared.reshape(-1).to(device="cpu")
                        )
                        del grad_squared
                    offset = end
            model.zero_grad(set_to_none=True)
            used += 1
            print(
                f"[fisher] {benchmark}: {sample_index}/{len(samples)}",
                end="\r",
                flush=True,
            )

        model.zero_grad(set_to_none=True)
        print(" " * 80, end="\r", flush=True)
        if used == 0:
            del fisher
            raise RuntimeError(
                f"Benchmark {benchmark!r} produced no completion-token gradients."
            )
        fisher.div_(used)
        fisher_path = out / f"fisher_{benchmark}.pt"
        torch.save(fisher, fisher_path)
        metadata["benchmarks"].append(
            {
                "name": benchmark,
                "capability": capability,
                "file": fisher_path.name,
                "probe_samples": len(samples),
                "fisher_samples": used,
            }
        )
        _write_fisher_metadata(out, metadata)
        print(
            f"[fisher] {benchmark} ({capability}): saved {used} samples to "
            f"{fisher_path}",
            flush=True,
        )
        del fisher
        gc.collect()
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()

    model.zero_grad(set_to_none=True)
    del params, model, tokenizer
    gc.collect()
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _check_vectors(vectors: Sequence[torch.Tensor]) -> int:
    if not vectors:
        raise ValueError("At least one vector is required.")
    n = vectors[0].numel()
    if n == 0:
        raise ValueError("Vectors must be nonempty.")
    for vector in vectors:
        if vector.ndim != 1:
            raise ValueError("Fisher vectors must be one-dimensional.")
        if vector.numel() != n:
            raise ValueError("All Fisher vectors must have the same length.")
    return n


def _product_sum(left: torch.Tensor, right: torch.Tensor) -> float:
    # The product stays fp32 to bound temporary memory; reduction is fp64.
    return float(torch.sum(left * right, dtype=torch.float64))


def chunked_cosine(
    left: torch.Tensor,
    right: torch.Tensor,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> float:
    """Cosine similarity without allocating full-size temporary tensors."""
    n = _check_vectors([left, right])
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    dot = left_norm = right_norm = 0.0
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        left_chunk = left[start:end]
        right_chunk = right[start:end]
        dot += _product_sum(left_chunk, right_chunk)
        left_norm += _product_sum(left_chunk, left_chunk)
        right_norm += _product_sum(right_chunk, right_chunk)
    denominator = math.sqrt(max(left_norm, 0.0) * max(right_norm, 0.0))
    return dot / denominator if denominator > 0 else 0.0


def _chunked_raw_log_grams(
    vectors: Sequence[torch.Tensor],
    eps: float = LOG_EPS,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Accumulate raw, log, and centered-log Gram matrices in chunks."""
    n = _check_vectors(vectors)
    if eps <= 0:
        raise ValueError("eps must be positive")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    count = len(vectors)
    raw_gram = np.zeros((count, count), dtype=np.float64)
    log_gram = np.zeros((count, count), dtype=np.float64)
    residual_gram = np.zeros((count, count), dtype=np.float64)

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        raw_chunks = [vector[start:end] for vector in vectors]
        for index, chunk in enumerate(raw_chunks):
            if not bool(torch.isfinite(chunk).all()):
                raise ValueError(f"Fisher vector {index} contains non-finite values")
            if bool((chunk < 0).any()):
                raise ValueError(f"Fisher vector {index} contains negative values")
        log_chunks = []
        for chunk in raw_chunks:
            logged = chunk.to(dtype=torch.float32, device="cpu", copy=True)
            logged.add_(eps).log_()
            log_chunks.append(logged)
        for left_index in range(count):
            for right_index in range(left_index, count):
                raw_value = _product_sum(
                    raw_chunks[left_index], raw_chunks[right_index]
                )
                log_value = _product_sum(
                    log_chunks[left_index], log_chunks[right_index]
                )
                raw_gram[left_index, right_index] += raw_value
                log_gram[left_index, right_index] += log_value
                if left_index != right_index:
                    raw_gram[right_index, left_index] += raw_value
                    log_gram[right_index, left_index] += log_value
        shared_log = torch.zeros_like(log_chunks[0])
        for logged in log_chunks:
            shared_log.add_(logged)
        shared_log.div_(count)
        for logged in log_chunks:
            logged.sub_(shared_log)
        for left_index in range(count):
            for right_index in range(left_index, count):
                residual_value = _product_sum(
                    log_chunks[left_index], log_chunks[right_index]
                )
                residual_gram[left_index, right_index] += residual_value
                if left_index != right_index:
                    residual_gram[right_index, left_index] += residual_value
        del shared_log, log_chunks, raw_chunks
    return raw_gram, log_gram, residual_gram


def _cosine_matrix_from_gram(gram: np.ndarray) -> np.ndarray:
    diagonal = np.maximum(np.diag(gram), 0.0)
    denominator = np.sqrt(np.outer(diagonal, diagonal))
    result = np.zeros_like(gram, dtype=np.float64)
    np.divide(gram, denominator, out=result, where=denominator > 0)
    return np.clip(result, -1.0, 1.0)


def chunked_log_residual_cosine(
    vectors: Sequence[torch.Tensor],
    eps: float = LOG_EPS,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> np.ndarray:
    """Cosines after removing the all-benchmark geometric-mean component."""
    _, _, residual_gram = _chunked_raw_log_grams(vectors, eps, chunk_size)
    return _cosine_matrix_from_gram(residual_gram)


def chunked_topk_indices(
    vector: torch.Tensor,
    fraction: float = TOP_FRACTION,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> torch.Tensor:
    """Return sorted indices of the exact top fraction via streaming top-k."""
    n = _check_vectors([vector])
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    k = max(1, int(math.floor(n * fraction)))
    candidate_values = torch.empty(0, dtype=vector.dtype, device="cpu")
    candidate_indices = torch.empty(0, dtype=torch.int64, device="cpu")
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        chunk = vector[start:end]
        local_k = min(k, chunk.numel())
        values, indices = torch.topk(chunk, local_k, largest=True, sorted=False)
        values = values.to(device="cpu")
        indices = indices.to(device="cpu", dtype=torch.int64).add_(start)
        candidate_values = torch.cat((candidate_values, values))
        candidate_indices = torch.cat((candidate_indices, indices))
        if candidate_values.numel() > k:
            candidate_values, keep = torch.topk(
                candidate_values, k, largest=True, sorted=False
            )
            candidate_indices = candidate_indices[keep]
    return torch.sort(candidate_indices).values


def chunked_jaccard(
    left_indices: torch.Tensor,
    right_indices: torch.Tensor,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> float:
    """Jaccard similarity of two sorted, unique index tensors."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    left = left_indices.to(device="cpu", dtype=torch.int64).reshape(-1)
    right = right_indices.to(device="cpu", dtype=torch.int64).reshape(-1)
    if left.numel() == 0 and right.numel() == 0:
        return 1.0
    intersection = 0
    for start in range(0, left.numel(), chunk_size):
        chunk = left[start : start + chunk_size]
        positions = torch.searchsorted(right, chunk)
        valid = positions < right.numel()
        if bool(valid.any()):
            intersection += int(
                (right[positions[valid]] == chunk[valid]).sum().item()
            )
    union = left.numel() + right.numel() - intersection
    return intersection / union if union else 1.0


def chunked_topk_jaccard(
    left: torch.Tensor,
    right: torch.Tensor,
    fraction: float = TOP_FRACTION,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> float:
    """Jaccard overlap between exact top-fraction coordinate sets."""
    _check_vectors([left, right])
    left_top = chunked_topk_indices(left, fraction, chunk_size)
    right_top = chunked_topk_indices(right, fraction, chunk_size)
    return chunked_jaccard(left_top, right_top, chunk_size)


def _load_tensor_mmap(path: Path) -> torch.Tensor:
    try:
        tensor = torch.load(
            path, map_location="cpu", weights_only=True, mmap=True
        )
    except TypeError as exc:
        raise RuntimeError(
            "This analysis requires a PyTorch version whose torch.load supports "
            "mmap=True, so multi-gigabyte Fisher files remain memory bounded."
        ) from exc
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{path} did not contain a torch.Tensor")
    if tensor.ndim != 1 or tensor.dtype != torch.float32:
        raise ValueError(
            f"{path} must contain one flattened fp32 tensor; got "
            f"shape={tuple(tensor.shape)}, dtype={tensor.dtype}"
        )
    return tensor


def _analysis_inventory(out: Path) -> tuple[list[dict], dict]:
    metadata_path = out / "fisher_meta.json"
    has_metadata = metadata_path.exists()
    if has_metadata:
        metadata = json.loads(metadata_path.read_text())
        by_name = {
            item["name"]: item for item in metadata.get("benchmarks", [])
        }
    else:
        metadata = {"model": out.name, "benchmarks": []}
        by_name = {}
    inventory = []
    for spec in PROBE_REGISTRY:
        if has_metadata and spec.name not in by_name:
            continue
        item = dict(by_name.get(spec.name, {}))
        path = out / item.get("file", f"fisher_{spec.name}.pt")
        if not path.exists():
            continue
        item.update(
            {"name": spec.name, "capability": spec.capability, "path": path}
        )
        inventory.append(item)
    _require_probe_coverage(item["capability"] for item in inventory)
    return inventory, metadata


def _summary_statistics(
    matrix: np.ndarray, capabilities: Sequence[str]
) -> dict[str, float | None]:
    within: list[float] = []
    cross: list[float] = []
    all_pairs: list[float] = []
    for left in range(len(capabilities)):
        for right in range(left + 1, len(capabilities)):
            value = float(matrix[left, right])
            all_pairs.append(value)
            if capabilities[left] == capabilities[right]:
                within.append(value)
            else:
                cross.append(value)
    within_mean = float(np.mean(within)) if within else None
    cross_mean = float(np.mean(cross)) if cross else None
    standard_deviation = float(np.std(all_pairs)) if all_pairs else None
    block_score = None
    if (
        within_mean is not None
        and cross_mean is not None
        and standard_deviation is not None
        and standard_deviation > 0
    ):
        block_score = (within_mean - cross_mean) / standard_deviation
    return {
        "mean_within_capability": within_mean,
        "mean_cross_capability": cross_mean,
        "std_all_pairs": standard_deviation,
        "block_score": block_score,
    }


def _matrix_json(names: Sequence[str], matrix: np.ndarray) -> dict[str, dict]:
    return {
        row_name: {
            column_name: float(matrix[row, column])
            for column, column_name in enumerate(names)
        }
        for row, row_name in enumerate(names)
    }


def _format_number(value: float | None, digits: int = 4) -> str:
    return "n/a" if value is None or not np.isfinite(value) else f"{value:.{digits}f}"


def _markdown_matrix(names: Sequence[str], matrix: np.ndarray) -> list[str]:
    lines = [
        "| benchmark | " + " | ".join(names) + " |",
        "|---|" + "---:|" * len(names),
    ]
    for row, name in enumerate(names):
        values = " | ".join(f"{matrix[row, column]:.4f}" for column in range(len(names)))
        lines.append(f"| {name} | {values} |")
    return lines


def _write_report(
    out: Path,
    model_name: str,
    inventory: Sequence[dict],
    matrices: dict[str, np.ndarray],
    summaries: dict[str, dict[str, float | None]],
) -> str | None:
    names = [item["name"] for item in inventory]
    human_names = {
        "raw_cosine": "Raw Fisher cosine",
        "log_cosine": "Log-Fisher cosine",
        "shared_component_removed_cosine": "Shared-component-removed cosine",
        "top_0.1pct_jaccard": "Top-0.1% coordinate Jaccard",
    }
    positive = [
        key
        for key, summary in summaries.items()
        if summary["mean_within_capability"] is not None
        and summary["mean_cross_capability"] is not None
        and summary["mean_within_capability"] > summary["mean_cross_capability"]
    ]
    scored = [
        (summary["block_score"], key)
        for key, summary in summaries.items()
        if summary["block_score"] is not None and summary["block_score"] > 0
    ]
    strongest = max(scored)[1] if scored else None

    lines = [
        f"# V9 capability-region report — {model_name}",
        "",
        "Benchmarks are ordered by capability (math, code, QA, control). The "
        "shared-component-removed metric subtracts `log(S)`, where `S` is the "
        "coordinatewise geometric mean over every benchmark, including C4.",
        "",
        "## Benchmarks",
        "",
        "| benchmark | capability | Fisher samples |",
        "|---|---|---:|",
    ]
    for item in inventory:
        count = item.get("fisher_samples", "unknown")
        lines.append(f"| {item['name']} | {item['capability']} | {count} |")

    for key, matrix in matrices.items():
        lines.extend(["", f"## {human_names[key]}", ""])
        lines.extend(_markdown_matrix(names, matrix))

    lines.extend(
        [
            "",
            "## Block-structure summary",
            "",
            "Within-capability pairs exclude C4 because control has one "
            "benchmark; cross-capability pairs include all C4 comparisons. "
            "The block score is `(mean within - mean cross) / std(all "
            "off-diagonal pairs)`.",
            "",
            "| metric | mean within | mean cross | std all pairs | block score | block structure? |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for key, summary in summaries.items():
        lines.append(
            f"| {human_names[key]} | "
            f"{_format_number(summary['mean_within_capability'])} | "
            f"{_format_number(summary['mean_cross_capability'])} | "
            f"{_format_number(summary['std_all_pairs'])} | "
            f"{_format_number(summary['block_score'])} | "
            f"{'yes' if key in positive else 'no'} |"
        )
    lines.append("")
    if positive:
        labels = ", ".join(human_names[key] for key in positive)
        lines.append(f"Metrics showing positive block structure: {labels}.")
    else:
        lines.append(
            "No metric shows block structure: none has higher mean similarity "
            "within capabilities than across capabilities."
        )
    if strongest is not None:
        lines.append(
            f"Strongest standardized block structure: {human_names[strongest]} "
            f"(block score {_format_number(summaries[strongest]['block_score'])})."
        )
    report = "\n".join(lines) + "\n"
    (out / "report.md").write_text(report)
    return strongest


def stage_analyze(out: Path, chunk_size: int = DEFAULT_CHUNK_SIZE) -> None:
    """Read saved Fishers and emit pairwise similarity JSON + Markdown."""
    inventory, metadata = _analysis_inventory(out)
    names = [item["name"] for item in inventory]
    capabilities = [item["capability"] for item in inventory]
    vectors = [_load_tensor_mmap(item["path"]) for item in inventory]
    n_parameters = _check_vectors(vectors)
    expected = metadata.get("n_parameters")
    if expected is not None and n_parameters != expected:
        raise ValueError(
            f"Fisher length {n_parameters} does not match metadata {expected}."
        )

    print(
        f"[analyze] {len(vectors)} mmap-backed vectors, "
        f"{n_parameters:,} coordinates each",
        flush=True,
    )
    raw_gram, log_gram, residual_gram = _chunked_raw_log_grams(
        vectors, eps=LOG_EPS, chunk_size=chunk_size
    )
    matrices = {
        "raw_cosine": _cosine_matrix_from_gram(raw_gram),
        "log_cosine": _cosine_matrix_from_gram(log_gram),
        "shared_component_removed_cosine": _cosine_matrix_from_gram(
            residual_gram
        ),
    }

    top_indices = []
    for name, vector in zip(names, vectors):
        print(f"[analyze] selecting top 0.1% coordinates for {name}", flush=True)
        top_indices.append(
            chunked_topk_indices(vector, TOP_FRACTION, chunk_size)
        )
    jaccard = np.eye(len(vectors), dtype=np.float64)
    for left in range(len(vectors)):
        for right in range(left + 1, len(vectors)):
            value = chunked_jaccard(
                top_indices[left], top_indices[right], chunk_size
            )
            jaccard[left, right] = jaccard[right, left] = value
    matrices["top_0.1pct_jaccard"] = jaccard

    summaries = {
        key: _summary_statistics(matrix, capabilities)
        for key, matrix in matrices.items()
    }
    pairs = []
    for left in range(len(names)):
        for right in range(left + 1, len(names)):
            pairs.append(
                {
                    "benchmark_a": names[left],
                    "benchmark_b": names[right],
                    "capability_a": capabilities[left],
                    "capability_b": capabilities[right],
                    **{
                        key: float(matrix[left, right])
                        for key, matrix in matrices.items()
                    },
                }
            )
    strongest = _write_report(
        out,
        str(metadata.get("model", out.name)),
        inventory,
        matrices,
        summaries,
    )
    payload = {
        "version": 9,
        "model": metadata.get("model", out.name),
        "n_parameters": n_parameters,
        "log_eps": LOG_EPS,
        "top_fraction": TOP_FRACTION,
        "benchmarks": names,
        "benchmark_capability": dict(zip(names, capabilities)),
        "matrices": {
            key: _matrix_json(names, matrix) for key, matrix in matrices.items()
        },
        "pairs": pairs,
        "summary": summaries,
        "strongest_block_metric": strongest,
    }
    (out / "similarity.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"[analyze] wrote {out / 'similarity.json'} and {out / 'report.md'}")


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
        "--stage", choices=("fisher", "analyze", "all"), default="all"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="coordinates per analysis chunk (default: 50,000,000)",
    )
    args = parser.parse_args()

    model_name = require_compliant(args.model)
    tag = model_output_tag(args.model, model_name)
    out = OUT_BASE / tag
    out.mkdir(parents=True, exist_ok=True)
    if args.stage in ("fisher", "all"):
        stage_fisher(
            model_name,
            args.device,
            args.n_probe,
            out,
            model_dtype=args.model_dtype,
        )
    if args.stage in ("analyze", "all"):
        stage_analyze(out, chunk_size=args.chunk_size)


if __name__ == "__main__":
    main()
