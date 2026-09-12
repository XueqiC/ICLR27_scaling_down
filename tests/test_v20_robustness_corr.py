from __future__ import annotations

import math

import numpy as np
import pytest

from analysis import v20_robustness_corr as v20


def _curve(method: str, damages: list[float]) -> list[dict]:
    coordinates = [1.0, 0.8, 0.6, 0.3] if method == "pruning" else [16, 8, 4, 3]
    key = "density" if method == "pruning" else "bits"
    return [
        {key: coordinate, "raw_coordinate": coordinate, "delta_L_c": damage}
        for coordinate, damage in zip(coordinates, damages)
    ]


def test_curve_robustness_rewards_low_damage_and_censors_no_crossing() -> None:
    robust = v20.curve_robustness(
        _curve("pruning", [0.0, 0.02, 0.1, 0.3]), "pruning"
    )
    fragile = v20.curve_robustness(
        _curve("pruning", [0.0, 0.8, 2.0, 20.0]), "pruning"
    )

    assert robust["auc_capped_damage"] < fragile["auc_capped_damage"]
    assert robust["cliff_censored"] is True
    assert fragile["cliff_censored"] is False
    assert robust["cliff_survival"] > fragile["cliff_survival"]


def test_cluster_correlation_reports_perfect_monotone_pair() -> None:
    rows = [
        {
            "model": f"m{index}", "family": f"f{index % 2}",
            "pruning_score": float(index),
            "quantization_score": float(2 * index + 1),
        }
        for index in range(6)
    ]

    result = v20.correlation_summary(rows, 100, label="test-perfect")

    assert result["pearson"] == pytest.approx(1.0)
    assert result["spearman"] == pytest.approx(1.0)
    assert result["bootstrap_success"] > 0


def test_family_interaction_lomo_marks_two_size_family_unidentified() -> None:
    rows = []
    sizes = {"gemma3": [1.0, 4.0, 12.0], "olmo3": [7.0, 32.0], "qwen3": [0.6, 1.7, 4.0]}
    for family, family_sizes in sizes.items():
        for model_index, size in enumerate(family_sizes):
            for capability in v20.CAPABILITIES:
                rows.append({
                    "model": f"{family}-{model_index}",
                    "family": family,
                    "capability": capability,
                    "log2_size_over_4b": math.log2(size / 4.0),
                    "degradation": 1.0 + 0.1 * math.log2(size / 4.0),
                })

    metrics, _ = v20._leave_one_model_out(rows, tuple(range(8)))

    assert set(metrics["skipped_models"]) == {"olmo3-0", "olmo3-1"}
    assert metrics["coverage"] == pytest.approx(18 / 24)

