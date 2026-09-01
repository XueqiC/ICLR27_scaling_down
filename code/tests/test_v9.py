import json

import numpy as np
import torch

from analysis import v9_capability_regions as regions


def _numpy_cosine(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right)))


def test_chunked_cosine_matches_numpy_reference():
    rng = np.random.default_rng(7)
    left = rng.normal(size=137).astype(np.float32)
    right = rng.normal(size=137).astype(np.float32)

    actual = regions.chunked_cosine(
        torch.from_numpy(left), torch.from_numpy(right), chunk_size=19
    )

    np.testing.assert_allclose(
        actual,
        _numpy_cosine(left.astype(np.float64), right.astype(np.float64)),
        rtol=1e-6,
        atol=1e-7,
    )


def test_chunked_log_residual_cosine_matches_numpy_reference():
    rng = np.random.default_rng(11)
    arrays = [
        rng.lognormal(mean=offset, sigma=0.8, size=113).astype(np.float32)
        for offset in (-0.4, 0.1, 0.7, 1.0)
    ]
    eps = 1e-12
    logs = np.stack(
        [np.log(array.astype(np.float64) + eps) for array in arrays]
    )
    residuals = logs - logs.mean(axis=0, keepdims=True)
    expected = np.zeros((len(arrays), len(arrays)), dtype=np.float64)
    for left in range(len(arrays)):
        for right in range(len(arrays)):
            expected[left, right] = _numpy_cosine(
                residuals[left], residuals[right]
            )

    actual = regions.chunked_log_residual_cosine(
        [torch.from_numpy(array) for array in arrays],
        eps=eps,
        chunk_size=17,
    )

    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)


def test_chunked_topk_jaccard_matches_numpy_reference():
    # Unique values make the exact top-k sets independent of tie-breaking.
    left = np.arange(101, dtype=np.float32)
    right = np.roll(left, 17) + np.linspace(0.0, 0.1, left.size, dtype=np.float32)
    fraction = 0.2
    k = max(1, int(np.floor(left.size * fraction)))
    left_top = set(np.argpartition(left, -k)[-k:].tolist())
    right_top = set(np.argpartition(right, -k)[-k:].tolist())
    expected = len(left_top & right_top) / len(left_top | right_top)

    actual = regions.chunked_topk_jaccard(
        torch.from_numpy(left),
        torch.from_numpy(right),
        fraction=fraction,
        chunk_size=13,
    )

    np.testing.assert_allclose(actual, expected, rtol=0, atol=0)


def test_chunked_jaccard_of_sorted_indices_matches_numpy_reference():
    left = torch.tensor([1, 4, 8, 12, 21, 34], dtype=torch.int64)
    right = torch.tensor([0, 4, 9, 12, 22, 34, 55], dtype=torch.int64)
    expected = 3 / 10

    assert regions.chunked_jaccard(left, right, chunk_size=2) == expected


def test_analyze_stage_writes_json_and_four_report_matrices(tmp_path):
    rng = np.random.default_rng(23)
    benchmark_meta = []
    for offset, spec in enumerate(regions.PROBE_REGISTRY):
        vector = torch.from_numpy(
            rng.lognormal(mean=offset / 20, sigma=0.5, size=97).astype(np.float32)
        )
        filename = f"fisher_{spec.name}.pt"
        torch.save(vector, tmp_path / filename)
        benchmark_meta.append(
            {
                "name": spec.name,
                "capability": spec.capability,
                "file": filename,
                "fisher_samples": 3,
            }
        )
    metadata = {
        "version": 9,
        "model": "synthetic/model",
        "n_parameters": 97,
        "benchmarks": benchmark_meta,
    }
    (tmp_path / "fisher_meta.json").write_text(json.dumps(metadata))

    regions.stage_analyze(tmp_path, chunk_size=13)

    payload = json.loads((tmp_path / "similarity.json").read_text())
    assert payload["benchmark_capability"]["gsm8k"] == "math"
    assert payload["benchmark_capability"]["c4"] == "control"
    assert set(payload["matrices"]) == {
        "raw_cosine",
        "log_cosine",
        "shared_component_removed_cosine",
        "top_0.1pct_jaccard",
    }
    report = (tmp_path / "report.md").read_text()
    assert report.count("## ") == 6
    assert "Block-structure summary" in report
