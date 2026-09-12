import numpy as np
import torch
from scipy.linalg import subspace_angles

from analysis import v9b_subspaces as subspaces
from analysis import v9c_ablation as ablation


def test_principal_angle_similarity_matches_scipy_reference():
    rng = np.random.default_rng(41)
    left, _ = np.linalg.qr(rng.normal(size=(17, 5)))
    right, _ = np.linalg.qr(rng.normal(size=(17, 4)))

    expected = float(np.mean(np.cos(subspace_angles(left, right)) ** 2))
    actual = subspaces.principal_angle_similarity(left, right)

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_project_gradient_is_stable_by_tensor_name_and_chunk():
    gradient = torch.linspace(-1.0, 1.0, 29)

    first = subspaces.project_gradient(
        gradient, "layers.0.weight", projection_dim=7, chunk_size=5
    )
    second = subspaces.project_gradient(
        gradient, "layers.0.weight", projection_dim=7, chunk_size=5
    )
    other_name = subspaces.project_gradient(
        gradient, "layers.1.weight", projection_dim=7, chunk_size=5
    )

    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert not torch.equal(first, other_name)


def test_residual_log_fisher_selection_removes_other_capability_top_sets():
    # Math and code both rank coordinate 3 in their top two, so it must be
    # removed from both exclusive sets.  QA's two top coordinates are unique.
    log_fishers = {
        "math_a": [10, 0, 0, 10, 0, 0, 0, 0],
        "math_b": [10, 0, 0, 10, 0, 0, 0, 0],
        "code_a": [0, 10, 0, 10, 0, 0, 0, 0],
        "code_b": [0, 10, 0, 10, 0, 0, 0, 0],
        "qa_a": [0, 0, 10, 0, 10, 0, 0, 0],
        "qa_b": [0, 0, 10, 0, 10, 0, 0, 0],
    }
    fishers = {
        name: torch.exp(torch.tensor(values, dtype=torch.float32))
        for name, values in log_fishers.items()
    }
    capabilities = {name: name.split("_", maxsplit=1)[0] for name in log_fishers}

    selected = ablation.select_exclusive_residual_top_coordinates(
        fishers,
        capabilities,
        top_fraction=0.25,
        chunk_size=3,
    )

    assert selected["math"].tolist() == [0]
    assert selected["code"].tolist() == [1]
    assert selected["qa"].tolist() == [2, 4]


def test_magnitude_matched_sampler_preserves_quantile_bin_counts():
    capability_coordinates = torch.arange(40, dtype=torch.int64)
    weights = torch.cat(
        [
            torch.arange(1, 41, dtype=torch.float32),
            torch.arange(1, 41, dtype=torch.float32).repeat(5),
        ]
    )

    sampled = ablation.sample_magnitude_matched_coordinates(
        weights,
        capability_coordinates,
        n_bins=20,
        seed=17,
        chunk_size=19,
    )

    reference = weights[capability_coordinates].abs()
    target_bins = ablation.magnitude_quantile_bin_indices(
        reference, reference, n_bins=20
    )
    sampled_bins = ablation.magnitude_quantile_bin_indices(
        weights[sampled].abs(), reference, n_bins=20
    )
    torch.testing.assert_close(
        torch.bincount(sampled_bins, minlength=20),
        torch.bincount(target_bins, minlength=20),
        rtol=0,
        atol=0,
    )
    assert sampled.numel() == capability_coordinates.numel()
    assert not bool(torch.isin(sampled, capability_coordinates).any())


def test_adjusted_rand_index_known_partitions():
    truth = [0, 0, 1, 1]

    assert subspaces.adjusted_rand_index(truth, truth) == 1.0
    assert subspaces.adjusted_rand_index(truth, [1, 1, 0, 0]) == 1.0
    np.testing.assert_allclose(
        subspaces.adjusted_rand_index(truth, [0, 1, 0, 1]),
        -0.5,
        rtol=0,
        atol=1e-15,
    )
