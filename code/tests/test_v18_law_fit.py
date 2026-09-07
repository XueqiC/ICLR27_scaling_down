from __future__ import annotations

import math

import numpy as np
import pytest

from analysis import v18_law_fit as v18


def _base_row(model: str, family: str, size: float, **extra) -> dict:
    return {
        "model": model,
        "family": family,
        "N0": size,
        "L_c0": 1.5,
        "capability": "math",
        **extra,
    }


def test_heldout_splitters_keep_groups_disjoint() -> None:
    rows = [
        _base_row("a-small", "a", 1.0, density=0.8, bits=8),
        _base_row("a-large", "a", 4.0, density=0.5, bits=4),
        _base_row("b-small", "b", 2.0, density=0.7, bits=8),
        _base_row("b-large", "b", 8.0, density=0.4, bits=4),
    ]
    shallow = v18.split_shallow_to_deep(rows, cutoff=0.6)[0]
    assert set(shallow["train_indices"]).isdisjoint(shallow["test_indices"])
    assert {rows[i]["density"] for i in shallow["train_indices"]} == {0.8, 0.7}

    family = v18.leave_one_family_out_splits(rows)
    assert {fold["held_out"] for fold in family} == {"a", "b"}
    for fold in family:
        assert {rows[i]["family"] for i in fold["test_indices"]} == {fold["held_out"]}
        assert fold["held_out"] not in {rows[i]["family"] for i in fold["train_indices"]}

    largest = v18.leave_largest_model_out_splits(rows)
    assert {fold["held_out"] for fold in largest} == {"a-large", "b-large"}
    for fold in largest:
        assert set(fold["train_indices"]).isdisjoint(fold["test_indices"])

    bit = v18.leave_one_bit_out_splits(rows)
    assert {fold["held_out"] for fold in bit} == {"int8", "int4"}
    assert v18.leave_one_bit_out_splits((8, 6, 4, 3))[0] == ([6, 4, 3], 8)


def _synthetic_pruning_rows() -> list[dict]:
    rows = []
    for family_index, family in enumerate(("a", "b")):
        for model_index, size in enumerate((1e9, 4e9)):
            for capability_index, capability in enumerate(v18.CAPABILITIES):
                for density in (0.9, 0.8, 0.7, 0.6, 0.5):
                    sparsity = 1.0 - density
                    first = -0.01 * (capability_index + 1) * sparsity
                    mass = sparsity ** (1.0 + 0.1 * capability_index)
                    observed = first + (0.8 + 0.1 * family_index) * mass ** 1.4
                    rows.append({
                        "model": f"{family}{model_index}", "family": family,
                        "N0": size, "L_c0": 1.2 + 0.1 * capability_index,
                        "capability": capability, "density": density,
                        "observed": observed, "first_order": first,
                        "deleted_mass": mass,
                        "removed_weight_norm": math.sqrt(sparsity),
                    })
    return rows


@pytest.mark.parametrize("candidate", v18.PRUNING_CANDIDATES)
def test_each_pruning_candidate_fits_and_predicts_synthetic(candidate: str) -> None:
    rows = _synthetic_pruning_rows()
    fit = v18.fit_pruning_candidate(rows, candidate)
    assert fit["status"] == "ok"
    predictions = [v18.predict_pruning_candidate(fit, row) for row in rows]
    assert all(value is not None and math.isfinite(value) for value in predictions)


def _synthetic_quantization_rows() -> list[dict]:
    rows = []
    for family_index, family in enumerate(("a", "b")):
        for model_index, size in enumerate((1e9, 4e9)):
            for capability_index, capability in enumerate(v18.CAPABILITIES):
                for bits in (8, 6, 4, 3):
                    smooth = 0.02 * (capability_index + 1) * (
                        math.exp(0.9 * (4 - bits)) - math.exp(0.9 * (4 - 16))
                    )
                    cliff = (0.3 + 0.05 * family_index) / (1 + math.exp((bits - 3.4) / 0.2))
                    rows.append({
                        "model": f"{family}{model_index}", "family": family,
                        "N0": size, "L_c0": 1.0, "capability": capability,
                        "bits": bits, "observed": smooth + cliff,
                    })
    return rows


@pytest.mark.parametrize("candidate", v18.QUANTIZATION_CANDIDATES)
def test_each_quantization_candidate_fits_and_predicts_synthetic(candidate: str) -> None:
    rows = _synthetic_quantization_rows()
    fit = v18.fit_quantization_candidate(rows, candidate)
    assert fit["status"] == "ok"
    predictions = [v18.predict_quantization_candidate(fit, row) for row in rows]
    assert all(value is not None and math.isfinite(value) for value in predictions)


def test_distillation_candidate_recovers_floor_and_exponent() -> None:
    rows = []
    expected = {"math": (-0.1, 0.2), "code": (0.05, 0.1), "qa": (-1.5, -0.05)}
    for model_index, ratio in enumerate((0.01, 0.04, 0.16, 0.45)):
        for capability, (floor, alpha) in expected.items():
            rows.append({"model": f"s{model_index}", "capability": capability,
                         "r_storage": ratio,
                         "observed": floor - alpha * math.log(ratio)})
    fit = v18.fit_distillation_law(rows)
    assert fit["status"] == "ok"
    for capability, (floor, alpha) in expected.items():
        assert fit["parameters"][capability]["floor"] == pytest.approx(floor)
        assert fit["parameters"][capability]["alpha"] == pytest.approx(alpha)
    assert v18.predict_distillation_law(fit, rows[0]) == pytest.approx(rows[0]["observed"])


@pytest.mark.parametrize("candidate", v18.RECOVERY_CANDIDATES)
def test_each_recovery_candidate_fits_its_synthetic_curve(candidate: str) -> None:
    tokens = np.geomspace(1e5, 6.4e7, 16)
    parameters = {
        "monotonic_saturation": {"r": 0.2, "D0": 2e6, "beta": 0.7},
        "change_point": {"a": -0.3, "c": 0.5, "tau": 1.5},
        "early_recovery_late_penalty": {"D0": 1e6, "beta": 0.8,
                                         "kappa": 0.003, "p": 1.2},
    }[candidate]
    values = v18.recovery_curve(candidate, tokens, parameters)
    rows = [{"tokens": float(token), "normalized_damage": float(value)}
            for token, value in zip(tokens, values)]
    fit = v18.fit_recovery_candidate(rows, candidate)
    assert fit["status"] == "ok"
    predicted = np.asarray([v18.predict_recovery_candidate(fit, row) for row in rows])
    assert np.mean(np.abs(predicted - values)) < 2e-3


def test_prediction_metrics_are_heldout_only_and_have_intervals() -> None:
    rows = _synthetic_pruning_rows()
    splits = v18.leave_one_family_out_splits(rows)
    result = v18.evaluate_splits(
        rows, splits,
        lambda train: v18.fit_pruning_candidate(train, "density_only"),
        v18.predict_pruning_candidate,
        "synthetic-heldout",
    )
    assert result["metrics"]["n"] == len(rows)
    assert result["metrics"]["mae_ci95"][0] <= result["metrics"]["mae"]
    assert result["metrics"]["mae_ci95"][1] >= result["metrics"]["mae"]


def test_paired_bootstrap_uses_common_heldout_cells() -> None:
    candidate = {"records": [
        {"held_out": "f", "model": "a", "capability": "math", "density": 0.6,
         "observed": 1.0, "predicted": 0.9},
        {"held_out": "f", "model": "b", "capability": "math", "density": 0.6,
         "observed": 2.0, "predicted": 1.9},
    ]}
    baseline = {"records": [
        {"held_out": "f", "model": "a", "capability": "math", "density": 0.6,
         "observed": 1.0, "predicted": 0.5},
        {"held_out": "f", "model": "b", "capability": "math", "density": 0.6,
         "observed": 2.0, "predicted": 1.5},
        {"held_out": "f", "model": "extra", "capability": "math", "density": 0.6,
         "observed": 3.0, "predicted": 0.0},
    ]}

    result = v18.paired_bootstrap_mae_difference(
        candidate, baseline, n_resamples=100, label="test-paired"
    )

    assert result["n_paired"] == 2
    assert result["difference"] == pytest.approx(-0.4)
    assert result["ci95"] == pytest.approx((-0.4, -0.4))
    assert result["includes_zero"] is False
