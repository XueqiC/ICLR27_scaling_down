from __future__ import annotations

import math

import numpy as np
import pytest

from analysis import v19_links as v19


def test_sigmoid_fit_recovers_synthetic_link() -> None:
    losses = np.linspace(0.3, 3.0, 20)
    rows = [
        {
            "model": f"m{index % 5}",
            "family": f"f{index % 2}",
            "loss": float(loss),
            "accuracy": float(v19.sigmoid_accuracy(loss, 0.8, 3.5, 1.4)),
        }
        for index, loss in enumerate(losses)
    ]

    fit = v19.fit_sigmoid(rows)

    assert fit["status"] == "ok"
    parameters = fit["parameters"]
    assert parameters["A_max"] == pytest.approx(0.8, abs=2e-3)
    assert parameters["k"] == pytest.approx(3.5, abs=2e-2)
    assert parameters["L_half"] == pytest.approx(1.4, abs=2e-3)


def test_fixed_eval_validation_rejects_archive_and_old_decoding(tmp_path) -> None:
    payload = {
        "version": 15,
        "model": "gemma3-4b",
        "checkpoint": "gemma3-4b/dense__easy",
        "accuracy_benchmark_mode": "easy",
        "accuracy_benchmark": v19.EXPECTED_BENCHMARKS,
        "probe_source": "analysis.v15_accuracy_link.build_easy_probes",
        "measurement_samples": {"math": 64, "code": 64, "qa": 64},
        "decoding": (
            "greedy, domain-delimiter stopping, "
            "post-hoc first-block truncation"
        ),
        "prune_density": None,
    }
    path = tmp_path / "dense__easy" / "accuracy.json"
    path.parent.mkdir()
    v19.validate_fixed_easy_artifact(
        path, payload, model="gemma3-4b", checkpoint="dense"
    )

    old = dict(payload, decoding="greedy")
    with pytest.raises(ValueError, match="predates the stop-sequence fix"):
        v19.validate_fixed_easy_artifact(path, old)

    archived = tmp_path / "_zeroshot_floor" / "dense__easy" / "accuracy.json"
    with pytest.raises(ValueError, match="forbidden"):
        v19.validate_fixed_easy_artifact(archived, payload)


def test_cliff_coincidence_distinguishes_loss_crossing_from_accuracy_floor() -> None:
    rows = []
    for outcome in v19.OUTCOMES:
        for density, loss, accuracy in (
            (1.0, 1.0, 0.8),
            (0.8, 1.2, 0.7),
            (0.7, 2.2, 0.2),
            (0.6, 4.0, 0.0),
        ):
            rows.append({
                "model": "gemma3-4b", "family": "gemma3",
                "outcome": outcome, "density": density, "loss": loss,
                "accuracy": accuracy, "n": 64,
            })

    result = v19.cliff_coincidence(rows)

    for outcome in v19.OUTCOMES:
        summary = result["summary"][outcome]
        assert summary["n_observed_loss_cliffs"] == 1
        assert summary["n_loss_cliffs_at_accuracy_floor"] == 0
        assert summary["n_density_coincidences"] == 0

