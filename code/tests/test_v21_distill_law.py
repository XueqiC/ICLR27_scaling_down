from __future__ import annotations

import math

import pytest

from analysis import v21_distill_law as v21


def _size_rows(
    family: str,
    models_and_ratios: tuple[tuple[str, float], ...],
    parameters: dict[str, tuple[float, float]],
) -> list[dict]:
    rows = []
    for model, ratio in models_and_ratios:
        for capability, (floor, alpha) in parameters.items():
            rows.append(
                {
                    "row_id": f"{family}|{model}|{capability}",
                    "family": family,
                    "model": model,
                    "capability": capability,
                    "r_storage": ratio,
                    "observed": floor - alpha * math.log(ratio),
                    "qa_annotation": v21.QA_CAVEAT if capability == "qa" else "",
                }
            )
    return rows


def test_shared_cross_family_path_predicts_qwen_without_refit() -> None:
    gemma_parameters = {
        "math": (-0.1, 0.2),
        "code": (0.05, 0.1),
        "qa": (-1.5, 0.15),
    }
    gemma = _size_rows(
        "gemma3",
        (("g0", 0.01), ("g1", 0.04), ("g2", 0.16), ("g3", 0.45)),
        gemma_parameters,
    )
    qwen = _size_rows(
        "qwen3",
        (("q0", 0.15), ("q1", 0.425)),
        {capability: (floor - 0.7, alpha) for capability, (floor, alpha) in gemma_parameters.items()},
    )

    first = v21.evaluate_shared_no_refit(gemma, qwen)
    perturbed_qwen = [
        {**row, "observed": float(row["observed"]) + 1000.0} for row in qwen
    ]
    second = v21.evaluate_shared_no_refit(gemma, perturbed_qwen)

    assert first["fit"]["train_families"] == ["gemma3"]
    assert first["fit"]["train_models"] == ["g0", "g1", "g2", "g3"]
    assert first["fit"]["n_train"] == 12
    assert not (
        set(first["fit"]["train_row_ids"])
        & {row["row_id"] for row in qwen}
    )
    assert [row["predicted"] for row in first["records"]] == pytest.approx(
        [row["predicted"] for row in second["records"]]
    )
    for capability, (floor, alpha) in gemma_parameters.items():
        assert first["fit"]["parameters"][capability]["floor"] == pytest.approx(floor)
        assert first["fit"]["parameters"][capability]["alpha"] == pytest.approx(alpha)


def test_shared_path_rejects_qwen_rows_in_gemma_training_set() -> None:
    parameters = {capability: (0.0, 0.1) for capability in v21.CAPABILITIES}
    gemma = _size_rows("gemma3", (("g0", 0.1), ("g1", 0.4)), parameters)
    qwen = _size_rows("qwen3", (("q0", 0.15), ("q1", 0.425)), parameters)

    with pytest.raises(ValueError, match="must all be gemma3"):
        v21.evaluate_shared_no_refit(gemma + qwen[:1], qwen)


def test_hierarchical_intercept_uses_only_other_qwen_size() -> None:
    parameters = {capability: (0.0, 0.2) for capability in v21.CAPABILITIES}
    gemma = _size_rows(
        "gemma3", (("g0", 0.01), ("g1", 0.04), ("g2", 0.16), ("g3", 0.45)), parameters
    )
    qwen = _size_rows(
        "qwen3",
        (("q0", 0.15), ("q1", 0.425)),
        {capability: (-0.75, 0.2) for capability in v21.CAPABILITIES},
    )
    shared = v21.evaluate_shared_no_refit(gemma, qwen)
    hierarchical = v21.evaluate_hierarchical_intercept(shared["fit"], qwen)

    assert hierarchical["metrics"]["mae"] == pytest.approx(0.0, abs=1e-12)
    assert hierarchical["metrics"]["sign_accuracy"] == 1.0
    assert hierarchical["added_qwen_parameters_per_capability"] == 1
    for fold in hierarchical["folds"]:
        assert set(fold["train_row_ids"]).isdisjoint(fold["test_row_ids"])
        assert fold["held_out_model"] not in fold["calibration_models"]


def test_family_specific_exponent_is_not_given_fabricated_heldout_error() -> None:
    parameters = {capability: (0.0, 0.2) for capability in v21.CAPABILITIES}
    gemma = _size_rows(
        "gemma3", (("g0", 0.01), ("g1", 0.04), ("g2", 0.16), ("g3", 0.45)), parameters
    )
    qwen = _size_rows(
        "qwen3", (("q0", 0.15), ("q1", 0.425)), parameters
    )
    shared = v21.evaluate_shared_no_refit(gemma, qwen)
    diagnostic = v21.family_specific_diagnostic(shared["fit"], qwen)

    assert diagnostic["status"] == "NON-DECISIONAL_UNDERIDENTIFIED_HELDOUT"
    assert diagnostic["heldout_metrics"]["n"] == 0
    assert diagnostic["heldout_metrics"]["mae"] is None
    assert diagnostic["in_sample_metrics"]["mae"] == pytest.approx(0.0, abs=1e-12)


def test_existing_artifacts_load_declared_ladders_and_no_refit_audit() -> None:
    summary = v21.build_summary()
    size = summary["source_referenced_size_law"]
    data = summary["data_ladder"]

    assert len(size["rows"]) == 6 * len(v21.CAPABILITIES)
    assert len(data["rows"]) == 4 * len(v21.CAPABILITIES)
    assert size["shared_no_refit"]["fit"]["train_families"] == ["gemma3"]
    assert size["shared_no_refit"]["metrics"]["n"] == 6
    assert size["hierarchical_intercept"]["metrics"]["n"] == 6
    assert size["family_specific_exponent"]["heldout_metrics"]["n"] == 0
    assert size["verdict"] == "hierarchical"
    assert all(
        data["monotonicity"][capability]["is_monotone"] is False
        for capability in v21.CAPABILITIES
    )
