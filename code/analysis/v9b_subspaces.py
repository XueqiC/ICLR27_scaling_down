#!/usr/bin/env python3
"""V9b: test capability regions using subspaces of per-sample gradients.

The collect stage projects every per-sample gradient independently for each
language weight tensor.  Projection blocks are regenerated deterministically,
so all samples see the same random directions without storing a model-sized
projection matrix.  The analyze stage compares benchmark gradient subspaces
and checks whether individual samples cluster by capability.

Examples:
  python3 analysis/v9b_subspaces.py --model gemma3-1b --stage collect
  python3 analysis/v9b_subspaces.py --model gemma3-1b --stage analyze

Artifacts are written under results/v9b-subspaces/<model_tag>/.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
from typing import Sequence

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
    from .v9_capability_regions import (
        BENCHMARK_CAPABILITY,
        CORE_CAPABILITIES,
        PROBE_REGISTRY,
        PROBE_SEED,
        _format_number,
        _markdown_matrix,
        _matrix_json,
        _require_probe_coverage,
        _summary_statistics,
        build_probe_registry,
    )
except ImportError:  # direct execution: python analysis/v9b_subspaces.py
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
        PROBE_REGISTRY,
        PROBE_SEED,
        _format_number,
        _markdown_matrix,
        _matrix_json,
        _require_probe_coverage,
        _summary_statistics,
        build_probe_registry,
    )


ROOT = Path(__file__).resolve().parents[1]
OUT_BASE = ROOT / "results/v9b-subspaces"
PROJECTION_DIM = 256
PROJECTION_SEED = 0
PROJECTION_CHUNK_SIZE = 16_384
SUBSPACE_RANK = 8
JOINT_COMPONENTS = 10
KMEANS_CLUSTERS = 3
KMEANS_SEED = 0


def projection_block_seed(
    tensor_name: str, chunk_index: int, seed: int = PROJECTION_SEED
) -> int:
    """Return a stable torch seed derived from tensor name and chunk index."""
    if chunk_index < 0:
        raise ValueError("chunk_index must be non-negative")
    payload = f"{seed}\0{tensor_name}\0{chunk_index}".encode("utf-8")
    # Do not use Python's randomized hash().  Torch accepts signed int64 seeds.
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (
        2**63 - 1
    )


def project_gradient(
    gradient: torch.Tensor,
    tensor_name: str,
    projection_dim: int = PROJECTION_DIM,
    chunk_size: int = PROJECTION_CHUNK_SIZE,
    seed: int = PROJECTION_SEED,
) -> torch.Tensor:
    """Project one flattened gradient onto deterministic random directions.

    Each direction is an exactly unit-norm Rademacher vector.  Only one
    ``chunk_size x projection_dim`` block exists at a time, and its seed is a
    stable function of ``(tensor_name, chunk_index)``.  Repeated calls therefore
    use identical projections without retaining projection matrices.
    """
    if projection_dim <= 0:
        raise ValueError("projection_dim must be positive")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    flat = gradient.detach().reshape(-1)
    if flat.numel() == 0:
        raise ValueError("gradient must be nonempty")
    result = torch.zeros(projection_dim, dtype=torch.float32, device=flat.device)
    scale = 1.0 / math.sqrt(flat.numel())
    for chunk_index, start in enumerate(range(0, flat.numel(), chunk_size)):
        end = min(start + chunk_size, flat.numel())
        generator = torch.Generator(device=flat.device)
        generator.manual_seed(projection_block_seed(tensor_name, chunk_index, seed))
        directions = torch.empty(
            (end - start, projection_dim), dtype=torch.float32, device=flat.device
        )
        directions.bernoulli_(0.5, generator=generator).mul_(2.0).sub_(1.0)
        result.add_(torch.matmul(flat[start:end].float(), directions), alpha=scale)
        del directions
    return result


def _sample_projected_gradient(
    params: Sequence[tuple[str, torch.Tensor]],
    projection_dim: int,
    chunk_size: int,
) -> np.ndarray:
    pieces = []
    with torch.no_grad():
        for name, parameter in params:
            if parameter.grad is None:
                projected = torch.zeros(
                    projection_dim, dtype=torch.float32, device=parameter.device
                )
            else:
                projected = project_gradient(
                    parameter.grad,
                    name,
                    projection_dim=projection_dim,
                    chunk_size=chunk_size,
                )
            pieces.append(projected.to(device="cpu", dtype=torch.float32))
    return torch.cat(pieces).numpy()


def _write_feature_metadata(out: Path, metadata: dict) -> None:
    (out / "features_meta.json").write_text(json.dumps(metadata, indent=2) + "\n")


def stage_collect(
    model_name: str,
    device: str,
    n_probe: int,
    out: Path,
    model_dtype: str = "fp32",
    projection_dim: int = PROJECTION_DIM,
    projection_chunk_size: int = PROJECTION_CHUNK_SIZE,
) -> None:
    """Collect one fp32 projected-gradient feature matrix per benchmark."""
    probes = build_probe_registry(n_probe, seed=PROBE_SEED)
    dtype = torch.float32 if model_dtype == "fp32" else torch.bfloat16
    model, tokenizer = load_text_causal_lm(model_name, dtype)
    model.to(device).eval()
    model.requires_grad_(False)
    params = language_weight_parameters(model)
    for _, parameter in params:
        parameter.requires_grad_(True)
    if not params:
        raise RuntimeError("No language weight matrices were found for projection.")

    feature_dim = projection_dim * len(params)
    metadata = {
        "version": "9b",
        "model": model_name,
        "model_dtype": model_dtype,
        "n_probe_requested": n_probe,
        "probe_seed": PROBE_SEED,
        "projection_dim_per_tensor": projection_dim,
        "projection_seed": PROJECTION_SEED,
        "projection_chunk_size": projection_chunk_size,
        "feature_dim": feature_dim,
        "parameter_names": [name for name, _ in params],
        "parameter_numels": [parameter.numel() for _, parameter in params],
        "benchmarks": [],
    }
    _write_feature_metadata(out, metadata)

    for benchmark, samples in probes.items():
        rows: list[np.ndarray] = []
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
            rows.append(
                _sample_projected_gradient(
                    params, projection_dim, projection_chunk_size
                )
            )
            model.zero_grad(set_to_none=True)
            print(
                f"[collect] {benchmark}: {sample_index}/{len(samples)}",
                end="\r",
                flush=True,
            )
        print(" " * 80, end="\r", flush=True)
        if not rows:
            raise RuntimeError(
                f"Benchmark {benchmark!r} produced no completion-token gradients."
            )
        features = np.stack(rows).astype(np.float32, copy=False)
        feature_path = out / f"feats_{benchmark}.npy"
        np.save(feature_path, features, allow_pickle=False)
        metadata["benchmarks"].append(
            {
                "name": benchmark,
                "capability": BENCHMARK_CAPABILITY[benchmark],
                "file": feature_path.name,
                "probe_samples": len(samples),
                "feature_samples": int(features.shape[0]),
            }
        )
        _write_feature_metadata(out, metadata)
        print(
            f"[collect] {benchmark}: saved {features.shape} fp32 features to "
            f"{feature_path}",
            flush=True,
        )
        del rows, features
        gc.collect()
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()

    model.zero_grad(set_to_none=True)
    del params, model, tokenizer
    gc.collect()
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _feature_inventory(out: Path) -> tuple[list[dict], dict]:
    metadata_path = out / "features_meta.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        by_name = {item["name"]: item for item in metadata.get("benchmarks", [])}
    else:
        metadata = {"model": out.name, "benchmarks": []}
        by_name = {}
    inventory = []
    for spec in PROBE_REGISTRY:
        item = dict(by_name.get(spec.name, {}))
        if metadata_path.exists() and spec.name not in by_name:
            continue
        path = out / item.get("file", f"feats_{spec.name}.npy")
        if path.exists():
            item.update(
                {"name": spec.name, "capability": spec.capability, "path": path}
            )
            inventory.append(item)
    _require_probe_coverage(item["capability"] for item in inventory)
    return inventory, metadata


def _load_features(path: Path) -> np.ndarray:
    features = np.load(path, mmap_mode="r", allow_pickle=False)
    if features.ndim != 2 or features.dtype != np.float32:
        raise ValueError(
            f"{path} must contain a 2-D fp32 array; got "
            f"shape={features.shape}, dtype={features.dtype}"
        )
    if features.shape[0] < 2 or features.shape[1] == 0:
        raise ValueError(f"{path} needs at least two rows and one feature")
    if not np.isfinite(features).all():
        raise ValueError(f"{path} contains non-finite features")
    return features


def _centered(features: np.ndarray) -> np.ndarray:
    centered = np.array(features, dtype=np.float32, copy=True)
    centered -= np.mean(centered, axis=0, dtype=np.float32)
    return centered


def _top_right_singular_vectors(
    centered: np.ndarray, rank: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return top right singular vectors using the smaller sample Gram matrix."""
    if centered.ndim != 2 or rank <= 0:
        raise ValueError("centered must be 2-D and rank must be positive")
    gram = np.asarray(centered @ centered.T, dtype=np.float64)
    eigenvalues, left = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    left = left[:, order]
    if eigenvalues.size == 0 or eigenvalues[0] <= 0:
        raise ValueError("feature matrix has no nonzero centered directions")
    tolerance = np.finfo(np.float64).eps * max(centered.shape) * eigenvalues[0]
    keep = np.flatnonzero(eigenvalues > tolerance)[:rank]
    if keep.size == 0:
        raise ValueError("feature matrix has no numerically stable directions")
    singular_values = np.sqrt(eigenvalues[keep])
    right = centered.T @ left[:, keep]
    right = np.asarray(right / singular_values, dtype=np.float64)
    # Repair small loss of orthogonality from fp32 matrix multiplication.
    right, _ = np.linalg.qr(right, mode="reduced")
    return right, singular_values


def principal_angle_similarity(
    left_basis: np.ndarray, right_basis: np.ndarray
) -> float:
    """Mean squared cosine of principal angles between two column bases."""
    left = np.asarray(left_basis, dtype=np.float64)
    right = np.asarray(right_basis, dtype=np.float64)
    if left.ndim != 2 or right.ndim != 2:
        raise ValueError("subspace bases must be two-dimensional")
    if left.shape[0] != right.shape[0]:
        raise ValueError("subspace bases must share their ambient dimension")
    if left.shape[1] == 0 or right.shape[1] == 0:
        raise ValueError("subspace bases must contain at least one vector")
    left_q, _ = np.linalg.qr(left, mode="reduced")
    right_q, _ = np.linalg.qr(right, mode="reduced")
    cosines = np.linalg.svd(left_q.T @ right_q, compute_uv=False)
    return float(np.mean(np.clip(cosines, 0.0, 1.0) ** 2))


def adjusted_rand_index(labels_true: Sequence, labels_pred: Sequence) -> float:
    """Adjusted Rand index, implemented without a scikit-learn dependency."""
    truth = np.asarray(labels_true)
    predicted = np.asarray(labels_pred)
    if truth.ndim != 1 or predicted.ndim != 1 or truth.size != predicted.size:
        raise ValueError("label vectors must be one-dimensional and equally sized")
    if truth.size == 0:
        raise ValueError("label vectors must be nonempty")
    _, truth_inverse = np.unique(truth, return_inverse=True)
    _, predicted_inverse = np.unique(predicted, return_inverse=True)
    contingency = np.zeros(
        (truth_inverse.max() + 1, predicted_inverse.max() + 1), dtype=np.int64
    )
    np.add.at(contingency, (truth_inverse, predicted_inverse), 1)

    def choose_two(values: np.ndarray) -> int:
        return int(np.sum(values * (values - 1) // 2, dtype=np.int64))

    cell_pairs = choose_two(contingency)
    truth_pairs = choose_two(contingency.sum(axis=1))
    predicted_pairs = choose_two(contingency.sum(axis=0))
    total_pairs = int(truth.size * (truth.size - 1) // 2)
    if total_pairs == 0:
        return 1.0
    expected = truth_pairs * predicted_pairs / total_pairs
    maximum = 0.5 * (truth_pairs + predicted_pairs)
    denominator = maximum - expected
    if denominator == 0:
        return 1.0
    return float((cell_pairs - expected) / denominator)


def _kmeans_plus_plus(
    samples: np.ndarray, k: int, rng: np.random.Generator
) -> np.ndarray:
    centers = np.empty((k, samples.shape[1]), dtype=np.float64)
    first = int(rng.integers(samples.shape[0]))
    centers[0] = samples[first]
    closest = np.sum((samples - centers[0]) ** 2, axis=1)
    for index in range(1, k):
        total = float(closest.sum())
        if total <= 0:
            choice = int(rng.integers(samples.shape[0]))
        else:
            choice = int(rng.choice(samples.shape[0], p=closest / total))
        centers[index] = samples[choice]
        distance = np.sum((samples - centers[index]) ** 2, axis=1)
        closest = np.minimum(closest, distance)
    return centers


def kmeans(
    samples: np.ndarray,
    k: int,
    seed: int = KMEANS_SEED,
    n_init: int = 20,
    max_iter: int = 300,
) -> tuple[np.ndarray, float]:
    """Small deterministic NumPy k-means suitable for the probe matrices."""
    data = np.asarray(samples, dtype=np.float64)
    if data.ndim != 2 or data.shape[0] < k or k <= 0:
        raise ValueError("samples must be 2-D with at least k rows")
    if n_init <= 0 or max_iter <= 0 or not np.isfinite(data).all():
        raise ValueError("n_init/max_iter must be positive and samples finite")
    master = np.random.default_rng(seed)
    best_labels = None
    best_inertia = math.inf
    for _ in range(n_init):
        rng = np.random.default_rng(int(master.integers(2**63 - 1)))
        centers = _kmeans_plus_plus(data, k, rng)
        labels = np.full(data.shape[0], -1, dtype=np.int64)
        for _ in range(max_iter):
            distances = np.sum((data[:, None, :] - centers[None, :, :]) ** 2, axis=2)
            updated_labels = np.argmin(distances, axis=1)
            if np.array_equal(updated_labels, labels):
                break
            labels = updated_labels
            for cluster in range(k):
                members = data[labels == cluster]
                if members.size:
                    centers[cluster] = members.mean(axis=0)
                else:
                    nearest = np.min(distances, axis=1)
                    centers[cluster] = data[int(np.argmax(nearest))]
        distances = np.sum((data - centers[labels]) ** 2, axis=1)
        inertia = float(distances.sum())
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()
    assert best_labels is not None
    return best_labels, best_inertia


def _joint_clustering(
    features: Sequence[np.ndarray], capabilities: Sequence[str]
) -> dict:
    all_arrays = []
    core_mask = []
    true_labels = []
    for matrix, capability in zip(features, capabilities):
        all_arrays.append(np.asarray(matrix, dtype=np.float32))
        is_core = capability in CORE_CAPABILITIES
        core_mask.extend([is_core] * matrix.shape[0])
        if is_core:
            true_labels.extend([capability] * matrix.shape[0])
    joint = np.concatenate(all_arrays, axis=0)
    centered = _centered(joint)
    gram = np.asarray(centered @ centered.T, dtype=np.float64)
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    positive = np.maximum(eigenvalues[order], 0.0)
    n_components = min(JOINT_COMPONENTS, centered.shape[0] - 1, centered.shape[1])
    scores = eigenvectors[:, order[:n_components]] * np.sqrt(
        positive[:n_components]
    )
    clustered_scores = scores[np.asarray(core_mask, dtype=bool)]
    cluster_labels, inertia = kmeans(clustered_scores, KMEANS_CLUSTERS)
    return {
        "adjusted_rand_index": adjusted_rand_index(true_labels, cluster_labels),
        "n_samples": int(clustered_scores.shape[0]),
        "n_samples_svd": int(joint.shape[0]),
        "n_samples_clustered": int(clustered_scores.shape[0]),
        "n_components": int(n_components),
        "k": KMEANS_CLUSTERS,
        "kmeans_seed": KMEANS_SEED,
        "inertia": inertia,
        "cluster_sizes": [
            int(np.sum(cluster_labels == cluster))
            for cluster in range(KMEANS_CLUSTERS)
        ],
    }


def _write_report(
    out: Path,
    model_name: str,
    inventory: Sequence[dict],
    similarity: np.ndarray,
    summary: dict,
    ranks: Sequence[int],
    clustering: dict,
) -> None:
    names = [item["name"] for item in inventory]
    lines = [
        f"# V9b gradient-subspace report — {model_name}",
        "",
        f"Each benchmark subspace uses up to the top {SUBSPACE_RANK} right "
        "singular vectors of its centered per-sample projected-gradient matrix. "
        "Similarity is the mean squared cosine of the principal angles.",
        "",
        "## Benchmarks",
        "",
        "| benchmark | capability | samples | effective rank |",
        "|---|---|---:|---:|",
    ]
    for item, rank in zip(inventory, ranks):
        count = item.get("feature_samples", "unknown")
        lines.append(f"| {item['name']} | {item['capability']} | {count} | {rank} |")
    lines.extend(["", "## Principal-angle similarity", ""])
    lines.extend(_markdown_matrix(names, similarity))
    lines.extend(
        [
            "",
            "## Block-structure summary",
            "",
            "The block score is `(mean within - mean cross) / std(all "
            "off-diagonal pairs)`, with C4 included only among cross-capability "
            "pairs.",
            "",
            "| mean within | mean cross | std all pairs | block score |",
            "|---:|---:|---:|---:|",
            f"| {_format_number(summary['mean_within_capability'])} | "
            f"{_format_number(summary['mean_cross_capability'])} | "
            f"{_format_number(summary['std_all_pairs'])} | "
            f"{_format_number(summary['block_score'])} |",
            "",
            "## Joint sample-level clustering",
            "",
            "The joint centering and SVD include every benchmark sample. "
            "C4 scores are then excluded, and the remaining samples in the "
            f"scores in the top {clustering['n_components']} components are "
            f"clustered with k-means (k={clustering['k']}).",
            "",
            f"Adjusted Rand index versus capability: "
            f"**{clustering['adjusted_rand_index']:.4f}**.",
            "",
            f"Cluster sizes: {clustering['cluster_sizes']}; "
            f"inertia: {clustering['inertia']:.6g}.",
        ]
    )
    (out / "report_subspace.md").write_text("\n".join(lines) + "\n")


def stage_analyze(out: Path, rank: int = SUBSPACE_RANK) -> None:
    """Analyze saved feature matrices and write JSON plus Markdown outputs."""
    if rank <= 0:
        raise ValueError("rank must be positive")
    inventory, metadata = _feature_inventory(out)
    names = [item["name"] for item in inventory]
    capabilities = [item["capability"] for item in inventory]
    features = [_load_features(item["path"]) for item in inventory]
    feature_dims = {matrix.shape[1] for matrix in features}
    if len(feature_dims) != 1:
        raise ValueError("All feature matrices must have the same feature dimension")
    expected_dim = metadata.get("feature_dim")
    feature_dim = feature_dims.pop()
    if expected_dim is not None and feature_dim != expected_dim:
        raise ValueError(
            f"Feature dimension {feature_dim} does not match metadata {expected_dim}"
        )

    bases = []
    ranks = []
    for name, matrix in zip(names, features):
        basis, _ = _top_right_singular_vectors(_centered(matrix), rank)
        bases.append(basis)
        ranks.append(int(basis.shape[1]))
        print(f"[analyze] {name}: effective rank {basis.shape[1]}", flush=True)
    similarity = np.eye(len(bases), dtype=np.float64)
    for left in range(len(bases)):
        for right in range(left + 1, len(bases)):
            value = principal_angle_similarity(bases[left], bases[right])
            similarity[left, right] = similarity[right, left] = value
    summary = _summary_statistics(similarity, capabilities)
    clustering = _joint_clustering(features, capabilities)
    pairs = []
    for left in range(len(names)):
        for right in range(left + 1, len(names)):
            pairs.append(
                {
                    "benchmark_a": names[left],
                    "benchmark_b": names[right],
                    "capability_a": capabilities[left],
                    "capability_b": capabilities[right],
                    "mean_cos2_principal_angles": float(similarity[left, right]),
                }
            )
    payload = {
        "version": "9b",
        "model": metadata.get("model", out.name),
        "feature_dim": feature_dim,
        "requested_subspace_rank": rank,
        "benchmarks": names,
        "benchmark_capability": dict(zip(names, capabilities)),
        "effective_ranks": dict(zip(names, ranks)),
        "matrix": _matrix_json(names, similarity),
        "pairs": pairs,
        "summary": summary,
        "joint_clustering": clustering,
    }
    (out / "similarity_subspace.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )
    _write_report(
        out,
        str(metadata.get("model", out.name)),
        inventory,
        similarity,
        summary,
        ranks,
        clustering,
    )
    print(
        f"[analyze] wrote {out / 'similarity_subspace.json'} and "
        f"{out / 'report_subspace.md'}",
        flush=True,
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
        "--stage", choices=("collect", "analyze", "all"), default="all"
    )
    args = parser.parse_args()

    model_name = require_compliant(args.model)
    tag = model_output_tag(args.model, model_name)
    out = OUT_BASE / tag
    out.mkdir(parents=True, exist_ok=True)
    if args.stage in ("collect", "all"):
        stage_collect(
            model_name,
            args.device,
            args.n_probe,
            out,
            model_dtype=args.model_dtype,
        )
    if args.stage in ("analyze", "all"):
        stage_analyze(out)


if __name__ == "__main__":
    main()
