import json
import sys

import numpy as np
import pytest
import torch

from analysis import v9_capability_regions as regions


def _numpy_cosine(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right)))


def test_chunked_cosine_matches_numpy_reference():
    rng = np.random.default_rng(7)
    left = rng.normal(size=137).astype(np.float32)
    right = rng.normal(size=137).astype(np.float32)

    actual = regions.chunked_cosine(
        torch.from_numpy(left),
        torch.from_numpy(right),
        chunk_size=19,
        device="cpu",
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
        device="cpu",
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
        device="cpu",
    )

    np.testing.assert_allclose(actual, expected, rtol=0, atol=0)


def test_chunked_jaccard_of_sorted_indices_matches_numpy_reference():
    left = torch.tensor([1, 4, 8, 12, 21, 34], dtype=torch.int64)
    right = torch.tensor([0, 4, 9, 12, 22, 34, 55], dtype=torch.int64)
    expected = 3 / 10

    assert regions.chunked_jaccard(
        left, right, chunk_size=2, device="cpu"
    ) == expected


def test_streaming_topk_indices_matches_full_reference():
    # Unique shuffled values exercise both threshold passes and chunk merges.
    rng = np.random.default_rng(41)
    values = rng.permutation(503).astype(np.float32)
    fraction = 0.127
    k = max(1, int(np.floor(values.size * fraction)))
    expected = torch.sort(torch.topk(torch.from_numpy(values), k).indices).values

    actual = regions.chunked_topk_indices(
        torch.from_numpy(values),
        fraction=fraction,
        chunk_size=37,
        device="cpu",
    )

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_streaming_topk_indices_breaks_threshold_ties_by_source_order():
    values = torch.tensor([5.0, 1.0, 5.0, 2.0, 5.0, 3.0])

    actual = regions.chunked_topk_indices(
        values, fraction=2 / 6, chunk_size=2, device="cpu"
    )

    torch.testing.assert_close(actual, torch.tensor([0, 2]), rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_and_cpu_chunked_reductions_agree():
    rng = np.random.default_rng(43)
    vectors = [
        torch.from_numpy(
            rng.lognormal(mean=offset, sigma=0.7, size=521).astype(np.float32)
        )
        for offset in (-0.3, 0.0, 0.4, 0.9)
    ]

    cpu_grams = regions._chunked_raw_log_grams(
        vectors, chunk_size=47, device="cpu"
    )
    cuda_grams = regions._chunked_raw_log_grams(
        vectors, chunk_size=47, device="cuda"
    )
    for cpu_gram, cuda_gram in zip(cpu_grams, cuda_grams):
        np.testing.assert_allclose(
            cuda_gram, cpu_gram, rtol=1e-5, atol=1e-5
        )

    cpu_cosine = regions.chunked_cosine(
        vectors[0], vectors[1], chunk_size=47, device="cpu"
    )
    cuda_cosine = regions.chunked_cosine(
        vectors[0], vectors[1], chunk_size=47, device="cuda"
    )
    np.testing.assert_allclose(cuda_cosine, cpu_cosine, rtol=1e-5, atol=1e-5)

    cpu_top = [
        regions.chunked_topk_indices(
            vector, fraction=0.1, chunk_size=47, device="cpu"
        )
        for vector in vectors[:2]
    ]
    cuda_top = [
        regions.chunked_topk_indices(
            vector, fraction=0.1, chunk_size=47, device="cuda"
        )
        for vector in vectors[:2]
    ]
    for cpu_indices, cuda_indices in zip(cpu_top, cuda_top):
        torch.testing.assert_close(
            cuda_indices.cpu(), cpu_indices, rtol=0, atol=0
        )
    cpu_jaccard = regions.chunked_jaccard(
        cpu_top[0], cpu_top[1], chunk_size=11, device="cpu"
    )
    cuda_jaccard = regions.chunked_jaccard(
        cuda_top[0], cuda_top[1], chunk_size=11, device="cuda"
    )
    np.testing.assert_allclose(cuda_jaccard, cpu_jaccard, rtol=0, atol=0)


def test_analyze_only_inventory_lists_every_missing_fisher(tmp_path):
    torch.save(torch.ones(7), tmp_path / "fisher_gsm8k.pt")

    with pytest.raises(FileNotFoundError) as error:
        regions._analysis_inventory(tmp_path, require_all=True)

    message = str(error.value)
    assert "--analyze-only" in message
    assert "fisher_math500.pt" in message
    assert "fisher_c4.pt" in message
    assert "fisher_gsm8k.pt" not in message


def test_analyze_only_cli_skips_fisher_and_enables_strict_inventory(
    tmp_path, monkeypatch
):
    analyze_calls = []
    monkeypatch.setattr(regions, "OUT_BASE", tmp_path)
    monkeypatch.setattr(regions, "require_compliant", lambda model: model)
    monkeypatch.setattr(regions, "model_output_tag", lambda requested, resolved: requested)
    monkeypatch.setattr(
        regions,
        "stage_fisher",
        lambda *args, **kwargs: pytest.fail("analyze-only ran Fisher computation"),
    )
    monkeypatch.setattr(
        regions,
        "stage_analyze",
        lambda *args, **kwargs: analyze_calls.append((args, kwargs)),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "v9_capability_regions.py",
            "--model",
            "synthetic-model",
            "--analyze-only",
            "--analyze-device",
            "cpu",
            "--analyze-chunk",
            "123",
        ],
    )

    regions.main()

    assert len(analyze_calls) == 1
    _, kwargs = analyze_calls[0]
    assert kwargs == {
        "chunk_size": 123,
        "analyze_device": "cpu",
        "require_all": True,
    }


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

    regions.stage_analyze(tmp_path, chunk_size=13, analyze_device="cpu")

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
