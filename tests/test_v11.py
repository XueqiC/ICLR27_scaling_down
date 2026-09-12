import numpy as np
import torch

from analysis import v11_geometry_damage as damage


def test_residual_log_vectors_matches_hand_computation():
    logs = torch.tensor(
        [
            [0.0, 3.0, 6.0, -3.0],
            [3.0, 3.0, 0.0, 0.0],
            [6.0, 0.0, 3.0, 3.0],
        ],
        dtype=torch.float64,
    )
    fishers = [torch.exp(row).to(torch.float32) for row in logs]
    expected = logs - logs.mean(dim=0, keepdim=True)

    actual = damage.residual_log_vectors(fishers, eps=1e-12)

    torch.testing.assert_close(
        torch.stack(actual).to(torch.float64), expected, rtol=1e-6, atol=2e-7
    )
    torch.testing.assert_close(
        torch.stack(actual).sum(dim=0), torch.zeros(4), rtol=0, atol=5e-7
    )


def test_mass_retention_matches_hand_computed_top_region():
    # At top_fraction=0.2, the largest excluded value is 7, so positions 0
    # and 1 (values 9 and 8) form the pruned top region.  One of the four
    # dense residual coordinates remains in that region.
    pruned_fisher = torch.tensor(
        [9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0]
    )
    dense_region = torch.tensor([0, 2, 5, 9])

    actual = damage.mass_retention(
        dense_region, pruned_fisher, top_fraction=0.2, chunk_size=3
    )

    assert actual == 0.25


def test_cliff_crossing_summary_uses_first_ordered_exceedance():
    curve = [
        {
            "density": 0.9,
            "delta_loss": 0.2,
            "residual_cosine": 0.96,
            "log_fisher_cosine": 0.98,
            "mass_retention": 0.91,
        },
        {
            "density": 0.8,
            "delta_loss": 1.2,
            "residual_cosine": 0.82,
            "log_fisher_cosine": 0.90,
            "mass_retention": 0.70,
        },
        {
            "density": 0.7,
            "delta_loss": 2.0,
            "residual_cosine": 0.60,
            "log_fisher_cosine": 0.78,
            "mass_retention": 0.45,
        },
    ]

    summary = damage.summarize_cliff_crossing(curve, threshold=1.0)

    assert summary["crossed"] is True
    assert summary["density"] == 0.8
    assert summary["delta_loss"] == 1.2
    assert summary["geometric_at_cliff"] == {
        "residual_cosine": 0.82,
        "log_fisher_cosine": 0.90,
        "mass_retention": 0.70,
    }


def test_cliff_crossing_summary_reports_no_crossing():
    curve = [
        {
            "density": 0.9,
            "delta_loss": 1.0,
            "residual_cosine": 0.95,
            "log_fisher_cosine": 0.97,
            "mass_retention": 0.90,
        }
    ]

    summary = damage.summarize_cliff_crossing(curve, threshold=1.0)

    assert summary["crossed"] is False
    assert summary["density"] is None
    assert summary["geometric_at_cliff"] is None
