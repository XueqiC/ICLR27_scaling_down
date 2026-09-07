import numpy as np

from analysis import v9_stats
from analysis.v9_capability_regions import PROBE_REGISTRY


def test_permutation_p_value_counts_ties_and_uses_plus_one_correction():
    null_scores = np.array([0.1, 0.5, 0.7, 0.49])

    # Two null statistics meet or exceed the observed statistic.
    assert v9_stats.permutation_p_value(0.5, null_scores) == 3 / 5


def test_nearest_centroid_lobo_accuracy_on_clustered_benchmarks():
    rng = np.random.default_rng(9)
    centers = [
        np.array([5.0, 0.0, 0.0]),
        np.array([5.0, 0.0, 0.0]),
        np.array([0.0, 5.0, 0.0]),
        np.array([0.0, 5.0, 0.0]),
        np.array([0.0, 0.0, 5.0]),
        np.array([0.0, 0.0, 5.0]),
    ]
    features_by_benchmark = [
        center + rng.normal(scale=0.05, size=(7, 3)) for center in centers
    ]
    capabilities = ["math", "math", "code", "code", "qa", "qa"]

    assert (
        v9_stats.nearest_centroid_lobo_accuracy(features_by_benchmark, capabilities)
        == 1.0
    )


def test_stage_stats_writes_offline_artifacts(tmp_path):
    rng = np.random.default_rng(31)
    centers = {
        "math": np.array([4.0, 0.0, 0.0, 0.0]),
        "code": np.array([0.0, 4.0, 0.0, 0.0]),
        "qa": np.array([0.0, 0.0, 4.0, 0.0]),
        "control": np.array([0.0, 0.0, 0.0, 4.0]),
    }
    for spec in PROBE_REGISTRY:
        features = centers[spec.capability] + rng.normal(scale=0.1, size=(6, 4))
        np.save(
            tmp_path / f"feats_{spec.name}.npy",
            features.astype(np.float32),
            allow_pickle=False,
        )

    payload = v9_stats.stage_stats(
        tmp_path,
        bootstrap_resamples=20,
        permutations=30,
        batch_size=7,
    )

    assert payload["example_bootstrap"]["resamples"] == 20
    assert payload["permutation_test"]["permutations"] == 30
    assert payload["leave_one_benchmark_out"]["accuracy"] == 1.0
    assert (tmp_path / "stats.json").exists()
    assert (tmp_path / "stats.md").exists()
