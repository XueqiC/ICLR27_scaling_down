#!/usr/bin/env python3
"""V9 Experiment B2: offline statistical backing for gradient blocks.

The analysis consumes V9b's saved ``feats_<benchmark>.npy`` matrices.  It
bootstraps example rows, permutes benchmark labels over the pooled examples,
and measures leave-one-benchmark-out capability assignment stability.  No
model loading or GPU is required.

Run all completed feature directories with:
  python3 analysis/v9_stats.py
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np

try:
    from .v6_capability_geometry import model_output_tag
    from .v9_capability_regions import (
        CORE_CAPABILITIES,
        _format_number,
        _markdown_matrix,
        _matrix_json,
    )
    from .v9b_subspaces import OUT_BASE, _feature_inventory, _load_features
except ImportError:  # direct execution: python analysis/v9_stats.py
    from v6_capability_geometry import model_output_tag
    from v9_capability_regions import (
        CORE_CAPABILITIES,
        _format_number,
        _markdown_matrix,
        _matrix_json,
    )
    from v9b_subspaces import OUT_BASE, _feature_inventory, _load_features


BOOTSTRAP_RESAMPLES = 1_000
PERMUTATIONS = 2_000
BOOTSTRAP_SEED = 0
PERMUTATION_SEED = 1
RESAMPLE_BATCH_SIZE = 16


def _validate_mean_vectors(mean_vectors: np.ndarray) -> np.ndarray:
    means = np.asarray(mean_vectors)
    if means.ndim != 2 or means.shape[0] < 2 or means.shape[1] == 0:
        raise ValueError("mean_vectors must have at least two rows and one feature")
    if not np.isfinite(means).all():
        raise ValueError("mean_vectors must be finite")
    return means


def raw_cosine_matrix(mean_vectors: np.ndarray) -> np.ndarray:
    """Pairwise raw cosine matrix for per-benchmark mean feature vectors."""
    means = _validate_mean_vectors(mean_vectors).astype(np.float64, copy=False)
    gram = means @ means.T
    norms = np.sqrt(np.maximum(np.diag(gram), 0.0))
    denominator = np.outer(norms, norms)
    result = np.zeros_like(gram, dtype=np.float64)
    np.divide(gram, denominator, out=result, where=denominator > 0)
    return np.clip(result, -1.0, 1.0)


def _pair_indices(n_benchmarks: int) -> list[tuple[int, int]]:
    return [
        (left, right)
        for left in range(n_benchmarks)
        for right in range(left + 1, n_benchmarks)
    ]


def _within_cross_pair_masks(
    capabilities: Sequence[str], pairs: Sequence[tuple[int, int]]
) -> tuple[np.ndarray, np.ndarray]:
    within = np.asarray(
        [capabilities[left] == capabilities[right] for left, right in pairs],
        dtype=bool,
    )
    cross = ~within
    if not bool(within.any()) or not bool(cross.any()):
        raise ValueError("block score requires both within- and cross-capability pairs")
    return within, cross


def within_minus_cross_block_score(
    cosine_matrix: np.ndarray, capabilities: Sequence[str]
) -> float:
    """Mean within-capability cosine minus mean cross-capability cosine."""
    matrix = np.asarray(cosine_matrix, dtype=np.float64)
    if matrix.shape != (len(capabilities), len(capabilities)):
        raise ValueError("cosine_matrix shape must match capabilities")
    pairs = _pair_indices(len(capabilities))
    within, cross = _within_cross_pair_masks(capabilities, pairs)
    values = np.asarray([matrix[left, right] for left, right in pairs])
    return float(values[within].mean() - values[cross].mean())


def permutation_p_value(
    observed_score: float, permutation_scores: Sequence[float]
) -> float:
    """One-sided randomization p-value with the finite-sample +1 correction."""
    null = np.asarray(permutation_scores, dtype=np.float64)
    if null.ndim != 1 or null.size == 0 or not np.isfinite(null).all():
        raise ValueError("permutation_scores must be a nonempty finite vector")
    if not math.isfinite(observed_score):
        raise ValueError("observed_score must be finite")
    exceedances = int(np.count_nonzero(null >= observed_score))
    return (exceedances + 1) / (null.size + 1)


def _pair_cosines_from_batched_means(
    means: Sequence[np.ndarray], pairs: Sequence[tuple[int, int]]
) -> np.ndarray:
    if not means:
        raise ValueError("at least one benchmark mean batch is required")
    batch_size = means[0].shape[0]
    values = np.zeros((batch_size, len(pairs)), dtype=np.float64)
    norms = [
        np.sqrt(np.maximum(np.einsum("bd,bd->b", mean, mean, dtype=np.float64), 0.0))
        for mean in means
    ]
    for pair_index, (left, right) in enumerate(pairs):
        numerator = np.einsum("bd,bd->b", means[left], means[right], dtype=np.float64)
        denominator = norms[left] * norms[right]
        np.divide(
            numerator,
            denominator,
            out=values[:, pair_index],
            where=denominator > 0,
        )
    return np.clip(values, -1.0, 1.0)


def _scores_from_pair_values(
    pair_values: np.ndarray, within: np.ndarray, cross: np.ndarray
) -> np.ndarray:
    return pair_values[:, within].mean(axis=1) - pair_values[:, cross].mean(axis=1)


def _bootstrap_pair_cosines(
    features: Sequence[np.ndarray],
    capabilities: Sequence[str],
    n_resamples: int,
    seed: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    if n_resamples <= 0 or batch_size <= 0:
        raise ValueError("n_resamples and batch_size must be positive")
    pairs = _pair_indices(len(features))
    within, cross = _within_cross_pair_masks(capabilities, pairs)
    pair_values = np.empty((n_resamples, len(pairs)), dtype=np.float64)
    scores = np.empty(n_resamples, dtype=np.float64)
    rng = np.random.default_rng(seed)
    for start in range(0, n_resamples, batch_size):
        end = min(start + batch_size, n_resamples)
        current_batch = end - start
        means = []
        for matrix in features:
            n_samples = matrix.shape[0]
            draws = rng.integers(
                0, n_samples, size=(current_batch, n_samples), endpoint=False
            )
            counts = np.zeros((current_batch, n_samples), dtype=np.float32)
            rows = np.repeat(np.arange(current_batch), n_samples)
            np.add.at(counts, (rows, draws.reshape(-1)), 1.0 / n_samples)
            means.append(counts @ np.asarray(matrix))
        batch_pairs = _pair_cosines_from_batched_means(means, pairs)
        pair_values[start:end] = batch_pairs
        scores[start:end] = _scores_from_pair_values(batch_pairs, within, cross)
    return pair_values, scores


def _permutation_block_scores(
    features: Sequence[np.ndarray],
    capabilities: Sequence[str],
    n_permutations: int,
    seed: int,
    batch_size: int,
) -> np.ndarray:
    """Shuffle benchmark labels over the capability-agnostic pooled rows."""
    if n_permutations <= 0 or batch_size <= 0:
        raise ValueError("n_permutations and batch_size must be positive")
    sample_counts = [matrix.shape[0] for matrix in features]
    pooled = np.concatenate([np.asarray(matrix) for matrix in features], axis=0)
    n_samples = pooled.shape[0]
    n_benchmarks = len(features)
    pairs = _pair_indices(n_benchmarks)
    within, cross = _within_cross_pair_masks(capabilities, pairs)
    scores = np.empty(n_permutations, dtype=np.float64)
    rng = np.random.default_rng(seed)
    for start in range(0, n_permutations, batch_size):
        end = min(start + batch_size, n_permutations)
        current_batch = end - start
        assignments = np.zeros(
            (current_batch * n_benchmarks, n_samples), dtype=np.float32
        )
        for batch_index in range(current_batch):
            permutation = rng.permutation(n_samples)
            cursor = 0
            for benchmark_index, count in enumerate(sample_counts):
                row = batch_index * n_benchmarks + benchmark_index
                assignments[row, permutation[cursor : cursor + count]] = 1.0 / count
                cursor += count
        flat_means = assignments @ pooled
        means_3d = flat_means.reshape(current_batch, n_benchmarks, pooled.shape[1])
        means = [means_3d[:, index, :] for index in range(n_benchmarks)]
        batch_pairs = _pair_cosines_from_batched_means(means, pairs)
        scores[start:end] = _scores_from_pair_values(batch_pairs, within, cross)
    return scores


def _nearest_centroid_lobo_details(
    features_by_benchmark: Sequence[np.ndarray] | np.ndarray,
    capabilities: Sequence[str],
    benchmark_names: Sequence[str] | None = None,
) -> dict:
    if isinstance(features_by_benchmark, np.ndarray):
        array = np.asarray(features_by_benchmark)
        if array.ndim == 2:
            features = [row[None, :] for row in array]
        elif array.ndim == 3:
            features = [array[index] for index in range(array.shape[0])]
        else:
            raise ValueError("features_by_benchmark must be 2-D or 3-D")
    else:
        features = [np.asarray(matrix) for matrix in features_by_benchmark]
    if len(features) < 2:
        raise ValueError("LOBO needs at least two benchmark feature matrices")
    feature_dims = set()
    for matrix in features:
        if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
            raise ValueError("each benchmark feature matrix must be nonempty and 2-D")
        if not np.isfinite(matrix).all():
            raise ValueError("benchmark feature matrices must be finite")
        feature_dims.add(matrix.shape[1])
    if len(feature_dims) != 1:
        raise ValueError("benchmark feature matrices must share a feature dimension")

    labels = np.asarray(capabilities, dtype=object)
    if labels.ndim != 1 or labels.size != len(features):
        raise ValueError("capabilities must match the benchmark feature matrices")
    unique_labels = list(dict.fromkeys(str(label) for label in labels))
    if len(unique_labels) < 2:
        raise ValueError("nearest-centroid LOBO needs at least two capabilities")
    for label in unique_labels:
        if int(np.count_nonzero(labels == label)) < 2:
            raise ValueError(
                f"capability {label!r} needs at least two benchmarks for LOBO"
            )
    names = (
        list(benchmark_names)
        if benchmark_names is not None
        else [str(index) for index in range(len(features))]
    )
    if len(names) != len(features):
        raise ValueError("benchmark_names must match the benchmark feature matrices")

    predictions = []
    correct = 0
    total_samples = 0
    for held_out, held_out_features in enumerate(features):
        centroids = []
        for label in unique_labels:
            training = [
                matrix
                for index, (matrix, capability) in enumerate(zip(features, labels))
                if index != held_out and capability == label
            ]
            centroids.append(
                np.concatenate(training, axis=0).mean(axis=0, dtype=np.float64)
            )
        centroids_array = np.stack(centroids)
        queries = held_out_features.astype(np.float64, copy=False)
        numerator = queries @ centroids_array.T
        denominator = np.outer(
            np.linalg.norm(queries, axis=1), np.linalg.norm(centroids_array, axis=1)
        )
        similarities = np.full_like(numerator, -np.inf, dtype=np.float64)
        np.divide(
            numerator,
            denominator,
            out=similarities,
            where=denominator > 0,
        )
        predicted_indices = np.argmax(similarities, axis=1)
        predicted_labels = [unique_labels[int(index)] for index in predicted_indices]
        truth = str(labels[held_out])
        benchmark_correct = sum(label == truth for label in predicted_labels)
        correct += benchmark_correct
        total_samples += queries.shape[0]
        predictions.append(
            {
                "held_out_benchmark": names[held_out],
                "true_capability": truth,
                "n_samples": queries.shape[0],
                "n_correct": benchmark_correct,
                "accuracy": benchmark_correct / queries.shape[0],
                "predicted_capability_counts": {
                    label: predicted_labels.count(label) for label in unique_labels
                },
            }
        )
    return {
        "method": "cosine nearest centroid; train on sample rows from all other benchmarks",
        "n_correct": correct,
        "n_samples": total_samples,
        "n_benchmarks": len(features),
        "accuracy": correct / total_samples,
        "predictions": predictions,
    }


def nearest_centroid_lobo_accuracy(
    features_by_benchmark: Sequence[np.ndarray] | np.ndarray,
    capabilities: Sequence[str],
) -> float:
    """Sample accuracy when every benchmark is held out in turn."""
    return float(
        _nearest_centroid_lobo_details(features_by_benchmark, capabilities)["accuracy"]
    )


def _write_report(out: Path, payload: dict) -> None:
    names = payload["benchmarks"]
    bootstrap = payload["example_bootstrap"]
    permutation = payload["permutation_test"]
    lobo = payload["leave_one_benchmark_out"]
    lines = [
        f"# V9 Experiment B2 statistics — {payload['model']}",
        "",
        "Raw cosine is computed between per-benchmark mean projected-gradient "
        "vectors. The block score is mean within-capability cosine minus mean "
        "cross-capability cosine; C4 contributes only cross-capability pairs.",
        "",
        "## Observed raw cosine",
        "",
    ]
    matrix = np.asarray(
        [
            [payload["observed_raw_cosine"][row][column] for column in names]
            for row in names
        ]
    )
    lines.extend(_markdown_matrix(names, matrix))
    block = bootstrap["block_score"]
    lines.extend(
        [
            "",
            "## Block inference",
            "",
            "| observed block score | bootstrap 95% CI | permutations | exceedances | p-value |",
            "|---:|---:|---:|---:|---:|",
            f"| {_format_number(block['observed'])} | "
            f"[{_format_number(block['ci95'][0])}, {_format_number(block['ci95'][1])}] | "
            f"{permutation['permutations']:,} | {permutation['exceedance_count']:,} | "
            f"{permutation['p_value']:.6g} |",
            "",
            "## Pairwise example-bootstrap intervals",
            "",
            "| benchmark A | benchmark B | raw cosine | bootstrap 95% CI |",
            "|---|---|---:|---:|",
        ]
    )
    for pair in bootstrap["pairwise_raw_cosine"]:
        lines.append(
            f"| {pair['benchmark_a']} | {pair['benchmark_b']} | "
            f"{pair['observed']:.4f} | [{pair['ci95'][0]:.4f}, "
            f"{pair['ci95'][1]:.4f}] |"
        )
    lines.extend(
        [
            "",
            "## Leave-one-benchmark-out capability stability",
            "",
            f"Held-out sample accuracy: **{lobo['n_correct']}/"
            f"{lobo['n_samples']} = {lobo['accuracy']:.2%}** across "
            f"{lobo['n_benchmarks']} benchmarks.",
            "",
            "| held-out benchmark | true | samples | correct | accuracy |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for prediction in lobo["predictions"]:
        lines.append(
            f"| {prediction['held_out_benchmark']} | "
            f"{prediction['true_capability']} | "
            f"{prediction['n_samples']} | {prediction['n_correct']} | "
            f"{prediction['accuracy']:.2%} |"
        )
    (out / "stats.md").write_text("\n".join(lines) + "\n")


def stage_stats(
    out: Path,
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES,
    permutations: int = PERMUTATIONS,
    bootstrap_seed: int = BOOTSTRAP_SEED,
    permutation_seed: int = PERMUTATION_SEED,
    batch_size: int = RESAMPLE_BATCH_SIZE,
) -> dict:
    """Analyze one completed V9b feature directory and write B2 artifacts."""
    inventory, metadata = _feature_inventory(out)
    names = [item["name"] for item in inventory]
    capabilities = [item["capability"] for item in inventory]
    features = [_load_features(item["path"]) for item in inventory]
    feature_dims = {matrix.shape[1] for matrix in features}
    if len(feature_dims) != 1:
        raise ValueError("All feature matrices must have the same feature dimension")
    feature_dim = feature_dims.pop()
    expected_dim = metadata.get("feature_dim")
    if expected_dim is not None and feature_dim != expected_dim:
        raise ValueError(
            f"Feature dimension {feature_dim} does not match metadata {expected_dim}"
        )
    means = np.stack([np.mean(matrix, axis=0, dtype=np.float64) for matrix in features])
    observed_cosine = raw_cosine_matrix(means)
    observed_block = within_minus_cross_block_score(observed_cosine, capabilities)

    pairs = _pair_indices(len(names))
    bootstrap_pairs, bootstrap_scores = _bootstrap_pair_cosines(
        features,
        capabilities,
        bootstrap_resamples,
        bootstrap_seed,
        batch_size,
    )
    pairwise_bootstrap = []
    for pair_index, (left, right) in enumerate(pairs):
        ci = np.percentile(bootstrap_pairs[:, pair_index], [2.5, 97.5])
        pairwise_bootstrap.append(
            {
                "benchmark_a": names[left],
                "benchmark_b": names[right],
                "capability_a": capabilities[left],
                "capability_b": capabilities[right],
                "observed": float(observed_cosine[left, right]),
                "ci95": [float(ci[0]), float(ci[1])],
            }
        )
    block_ci = np.percentile(bootstrap_scores, [2.5, 97.5])

    permutation_scores = _permutation_block_scores(
        features,
        capabilities,
        permutations,
        permutation_seed,
        batch_size,
    )
    exceedance_count = int(np.count_nonzero(permutation_scores >= observed_block))
    null_ci = np.percentile(permutation_scores, [2.5, 97.5])

    core_indices = [
        index
        for index, capability in enumerate(capabilities)
        if capability in CORE_CAPABILITIES
    ]
    lobo = _nearest_centroid_lobo_details(
        [features[index] for index in core_indices],
        [capabilities[index] for index in core_indices],
        [names[index] for index in core_indices],
    )
    payload = {
        "version": "9-stats",
        "experiment": "B2",
        "model": metadata.get("model", out.name),
        "feature_dim": feature_dim,
        "benchmarks": names,
        "benchmark_capability": dict(zip(names, capabilities)),
        "sample_counts": {
            name: int(matrix.shape[0]) for name, matrix in zip(names, features)
        },
        "observed_raw_cosine": _matrix_json(names, observed_cosine),
        "observed_block_score": observed_block,
        "example_bootstrap": {
            "resamples": bootstrap_resamples,
            "seed": bootstrap_seed,
            "resampling_unit": "sample rows independently within each benchmark",
            "pairwise_raw_cosine": pairwise_bootstrap,
            "block_score": {
                "observed": observed_block,
                "ci95": [float(block_ci[0]), float(block_ci[1])],
            },
        },
        "permutation_test": {
            "permutations": permutations,
            "seed": permutation_seed,
            "null": "benchmark labels shuffled across all capability-agnostic pooled sample rows, preserving benchmark sample counts",
            "alternative": "observed block score is larger",
            "observed_block_score": observed_block,
            "exceedance_count": exceedance_count,
            "p_value": permutation_p_value(observed_block, permutation_scores),
            "null_mean": float(np.mean(permutation_scores)),
            "null_std": float(np.std(permutation_scores)),
            "null_ci95": [float(null_ci[0]), float(null_ci[1])],
        },
        "leave_one_benchmark_out": lobo,
    }
    (out / "stats.json").write_text(json.dumps(payload, indent=2) + "\n")
    _write_report(out, payload)
    print(f"[stats] wrote {out / 'stats.json'} and {out / 'stats.md'}", flush=True)
    return payload


def _feature_directories(models: Sequence[str] | None) -> list[Path]:
    if models:
        directories = [OUT_BASE / model_output_tag(model, model) for model in models]
    else:
        directories = (
            sorted(
                path
                for path in OUT_BASE.iterdir()
                if path.is_dir() and any(path.glob("feats_*.npy"))
            )
            if OUT_BASE.exists()
            else []
        )
    if not directories:
        raise RuntimeError(f"No V9b feature directories found under {OUT_BASE}")
    return directories


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        dest="models",
        action="append",
        help=(
            "model registry tag, raw model id, or V9b output tag; repeat as "
            "needed (default: every directory with feature files)"
        ),
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=BOOTSTRAP_RESAMPLES)
    parser.add_argument("--permutations", type=int, default=PERMUTATIONS)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--permutation-seed", type=int, default=PERMUTATION_SEED)
    parser.add_argument("--batch-size", type=int, default=RESAMPLE_BATCH_SIZE)
    args = parser.parse_args()

    successes = 0
    failures = []
    for out in _feature_directories(args.models):
        try:
            stage_stats(
                out,
                bootstrap_resamples=args.bootstrap_resamples,
                permutations=args.permutations,
                bootstrap_seed=args.bootstrap_seed,
                permutation_seed=args.permutation_seed,
                batch_size=args.batch_size,
            )
            successes += 1
        except (OSError, RuntimeError, ValueError) as exc:
            failures.append(f"{out.name}: {exc}")
            print(f"WARNING: skipping {out}: {exc}", flush=True)
    if failures and successes == 0:
        raise RuntimeError(
            "No complete feature directory was analyzed: " + "; ".join(failures)
        )


if __name__ == "__main__":
    main()
