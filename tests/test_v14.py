import numpy as np

from analysis import v14_fitting as fitting


def test_deleted_mass_interpolates_inside_spectrum_bins():
    counts = np.array([2, 2])
    mass = np.array([1.0, 3.0])

    assert fitting.deleted_mass_fraction(counts, mass, density=1.0) == 0.0
    np.testing.assert_allclose(
        fitting.deleted_mass_fraction(counts, mass, density=0.75), 0.125
    )
    np.testing.assert_allclose(
        fitting.deleted_mass_fraction(counts, mass, density=0.5), 0.25
    )
    np.testing.assert_allclose(
        fitting.deleted_mass_fraction(counts, mass, density=0.25), 0.625
    )
    assert fitting.deleted_mass_fraction(counts, mass, density=0.0) == 1.0


def test_saturating_recovery_form_is_anchored_and_monotone():
    budgets = np.array([0.0, 500_000.0, 2_000_000.0, 8_000_000.0])
    damage = fitting.saturating_recovery_form(
        budgets,
        initial_damage=6.0,
        residual_fraction=0.2,
        d0=1_000_000.0,
        beta=0.75,
    )

    assert damage[0] == 6.0
    assert np.all(np.diff(damage) < 0.0)
    assert np.all(damage >= 6.0 * 0.2)


def test_leave_one_bit_out_splitter_uses_declared_targets():
    assert fitting.leave_one_bit_out_splits([3, 4, 6, 8]) == [
        ([8, 6, 3], 4),
        ([8, 6, 4], 3),
    ]


def test_sign_accuracy_counter_counts_cells_and_wilson_interval():
    result = fitting.sign_accuracy_counter(
        predicted=[1.0, -2.0, 3.0, -4.0],
        measured=[0.5, -0.1, -0.3, -0.2],
    )

    assert result["n_cells"] == 4
    assert result["n_correct"] == 3
    assert result["accuracy"] == 0.75
    assert result["ci95"][0] < result["accuracy"] < result["ci95"][1]
