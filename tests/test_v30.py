"""V30 shape, reference, model-pairing, and held-out leakage checks.

Synthetic fixtures verify statistical machinery; reported results use saved JSON.
"""
from __future__ import annotations

import copy
import json
import sys

import numpy as np
import pytest

from analysis import prediction_audit as audit
from analysis import v30_quant_shape_candidates as v30


def synthetic_rows(eta=2.9, n_models=5):
    rows = []
    for i in range(n_models):
        for j, cap in enumerate(audit.CAPABILITIES):
            amplitude = (-1 if cap == "qa" else 1) * (0.2 + i*.1 + j*.05)
            for bit in v30.BITS:
                y = amplitude * float(v30.normalized_shape(bit, "shared_eta", eta=eta))
                rows.append({"row_id": f"fixture-{i}|{cap}|{bit}", "model": f"fixture-{i}",
                             "capability": cap, "bit": bit, "b_ref": 16, "reference_key": "dense",
                             "reference_loss": 1.5-i*.1, "nominal_size_b": i+1.,
                             "family": "a" if i < 3 else "b", "observed": y, "loss": 1.5-i*.1+y})
    return rows


def test_eta_two_reproduces_fixed_four_shape_and_ratio():
    for ref in (8, 16):
        bits = np.array([3, 4, 5, 6, 8, ref])
        assert v30.shape(bits, "shared_eta", ref, 2.) == pytest.approx(
            256 * v30.shape(bits, "fixed_4_power", ref), rel=1e-12, abs=1e-14)
    assert float(v30.normalized_shape(5, "shared_eta", eta=2.)) == pytest.approx(.25, abs=1e-7)


def test_actual_step_ratio_uses_signed_quantizer_range():
    ratio = float(v30.normalized_shape(5, "actual_step"))
    assert ratio == pytest.approx((7/15)**2, abs=1e-7)
    assert ratio == pytest.approx(.218, abs=.0003)
    assert ratio != pytest.approx(.25, abs=.001)
    assert -np.log2(ratio) == pytest.approx(2.20, abs=.001)


@pytest.mark.parametrize("candidate", v30.CANDIDATES)
@pytest.mark.parametrize("ref", (8, 16))
def test_shapes_vanish_at_reference(candidate, ref):
    assert float(v30.shape(ref, candidate, ref, 2.9)) == 0.
    assert float(v30.normalized_shape(4, candidate, ref, 2.9)) == pytest.approx(1.)


def test_reference_preference_and_missing_anchor():
    assert v30.reference({"16": {}, "dense": {}, "8": {}}) == ("16", 16)
    assert v30.reference({"dense": {}, "8": {}}) == ("dense", 16)
    assert v30.reference({"8": {}}) == ("8", 8)
    with pytest.raises(ValueError, match="reference"):
        v30.reference({"4": {}})


def test_shared_eta_recovers_one_exponent_across_signed_capability_curves():
    rows = synthetic_rows()
    profile = v30.ShapeProfile(rows)
    fit = profile.fit_eta()
    assert fit["eta"] == pytest.approx(2.9, abs=2e-6)
    assert fit["sse"] < 1e-10
    labels, _ = profile.evaluate("shared_eta", fit["eta"])
    for (model, cap), amplitude in zip(profile.keys, labels):
        target = next(r["observed"] for r in rows if r["model"] == model and r["capability"] == cap and r["bit"] == 4)
        assert amplitude == pytest.approx(target, abs=1e-6)
    boot = v30.eta_analysis(rows, 25)
    assert boot["eta_ci95"] == pytest.approx([2.9, 2.9], abs=2e-6)
    assert len(boot["excluded_eta_values"]) == 2


def test_missing_bits_are_masked_and_single_bit_models_do_not_identify_eta():
    rows = [r for r in synthetic_rows() if r["model"] != "fixture-0" or r["bit"] == 4]
    profile = v30.ShapeProfile(rows)
    assert profile.fit_eta()["eta"] == pytest.approx(2.9, abs=2e-6)
    assert profile.fit_eta(weights=[1, 0, 0, 0, 0]) is None
    boot = v30.eta_analysis([r for r in rows if r["model"] in ("fixture-0", "fixture-1")], 100)
    assert boot["unidentified_bootstrap_draws"] > 0
    assert boot["identified_bootstrap_draws"] + boot["unidentified_bootstrap_draws"] == 100


def test_zero_response_does_not_identify_an_exponent():
    rows = synthetic_rows()
    for row in rows:
        row["observed"] = 0.
    with pytest.raises(ValueError, match="Eta requires"):
        v30.ShapeProfile(rows).fit_eta()


def test_eta_bootstrap_refits_whole_models_with_all_capabilities():
    rows = synthetic_rows()
    for row in rows:
        model_index = int(row["model"].split("-")[-1])
        row["observed"] = (model_index + 1) * float(v30.normalized_shape(
            row["bit"], "shared_eta", eta=2.5 + model_index*.3))
    result = v30.eta_analysis(rows, 25)
    profile = v30.ShapeProfile(rows)
    grid = profile.eta_grid()
    draws = np.random.default_rng(v30.SEED).integers(5, size=(25, 5))
    refits = [profile.fit_eta(np.bincount(draw, minlength=5), grid)["eta"] for draw in draws]
    assert result["eta_ci95"] == pytest.approx(audit.interval(refits))
    assert result["eta_ci95"][1] - result["eta_ci95"][0] > .1


def test_lomo_refits_shared_eta_and_mapping_without_any_held_out_outcomes():
    rows = synthetic_rows()
    before, folds_before = v30.lomo(rows)
    changed = copy.deepcopy(rows)
    for r in changed:
        if r["model"] == "fixture-0":
            r["observed"] = 99. + r["bit"]
            r["loss"] = r["reference_loss"] + r["observed"]
    after, folds_after = v30.lomo(changed)
    assert folds_before[0] == folds_after[0]
    assert [r["predictions"] for r in before if r["model"] == "fixture-0"] == [
        r["predictions"] for r in after if r["model"] == "fixture-0"]
    for fold in folds_before:
        assert fold["held_out"] not in fold["train_models"]
        assert not set(fold["train_row_ids"]) & set(fold["test_row_ids"])
        for cap in audit.CAPABILITIES:
            fit = fold["fits"]["shared_eta"]
            assert fit["eta"] == pytest.approx(2.9, abs=2e-6)
            assert fold["held_out"] not in fit["mappings"][cap]["train_models"]
    assert all(r["calibration_points_used"] == 0 for r in before)


def test_predictor_uses_only_basic_target_inputs():
    rows = synthetic_rows()
    train = [r for r in rows if r["model"] != "fixture-0"]
    target = rows[0]
    for candidate in v30.CANDIDATES:
        fit = v30.fit_candidate(train, candidate)
        basic = v30.basic_input(target)
        expected = v30.predict_candidate(fit, basic, "math", 5, 16)
        polluted = {**basic, "observed": 1e30, "loss": 1e30, "delta4": 1e30, "amplitude": 1e30}
        assert v30.predict_candidate(fit, polluted, "math", 5, 16) == expected
        with pytest.raises(ValueError, match="leaked"):
            v30.predict_candidate(fit, v30.basic_input(train[0]), "math", 5, 16)


def test_paired_test_uses_differences_without_dividing_by_observed_delta4():
    rows = synthetic_rows(eta=2.)
    for r in rows:
        if r["model"] == "fixture-0":
            r["observed"] = 0.
    result = v30.paired_analysis(rows, 100)
    for cap in (*audit.CAPABILITIES, "pooled"):
        tests = result["by_capability"][cap]["tests"]
        assert abs(tests["ratio_0.250"]["mean_paired_difference"]) < 1e-7
        assert tests["ratio_0.250"]["calibration_points_used"] == 1
    paired = result["rows"]
    draws = audit.bootstrap_means([r["paired_differences"]["ratio_0.218"] for r in paired],
                                  [r["model"] for r in paired], n_boot=100, seed=v30.SEED)
    assert result["by_capability"]["pooled"]["tests"]["ratio_0.218"]["ci95"] == audit.interval(draws[:, 0])


def test_actual_step_exact_paired_residual_is_zero():
    rows = synthetic_rows()
    for r in rows:
        r["observed"] = .7 * float(v30.normalized_shape(r["bit"], "actual_step"))
    result = v30.paired_analysis(rows, 25)
    for cap in result["by_capability"].values():
        exact = cap["tests"]["actual_step_exact"]
        assert exact["mean_paired_difference"] == pytest.approx(0., abs=1e-12)
        assert exact["calibrated_prediction_mae"] == pytest.approx(0., abs=1e-12)
        assert cap["tests"]["ratio_0.250"]["reject_fixed_ratio"]


def test_loader_never_reads_reserved_models_and_uses_own_panel_reference(tmp_path, monkeypatch):
    monkeypatch.setattr(v30, "ROOT", tmp_path)
    for model in ("gemma3-1b", "gemma3-4b", "gemma3-12b", "Qwen--Qwen3-8B", "Qwen3-14B"):
        path = tmp_path / model / "quant_losses.json"
        path.parent.mkdir()
        if "Qwen" in model:
            path.write_text("invalid JSON: reserved data must not be read")
        else:
            path.write_text(json.dumps({str(bit): {c: float(bit) for c in audit.CAPABILITIES}
                                        for bit in (8, 4, 5)}))
    rows, paths, inventory, excluded = v30.load_panel(tmp_path)
    assert len(paths) == len(inventory) == 3
    assert {r["model"] for r in excluded} == v30.RESERVED_MODELS
    assert all(r["b_ref"] == 8 and r["reference_loss"] == 8 for r in rows)
    assert all(r["bit"] != 8 for r in rows)
    assert all(r["observed"] == r["bit"]-8 for r in rows)


def test_reported_mae_gain_and_amplitude_have_correct_sign_and_common_cells():
    records, _ = v30.lomo(synthetic_rows())
    result = v30.summarize_predictions(records, 50)
    assert result["observed_response_amplitude"] == result["metrics"]["zero_change"]["mae"]
    assert result["n_cells"] == len(records)
    for name, metric in result["metrics"].items():
        error = np.mean([abs(r["observed"]-r["predictions"][name]) for r in records])
        assert metric["mae"] == pytest.approx(error)
        assert metric["improvement_over_zero_change"] == pytest.approx(result["observed_response_amplitude"]-error)


def test_dry_run_writes_no_artifacts(monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        pytest.fail("dry-run attempted an output write")
    monkeypatch.setattr(v30, "build_summary", lambda n_boot: {})
    monkeypatch.setattr(v30, "render", lambda summary: "fixture report")
    monkeypatch.setattr(audit, "write_outputs", forbidden)
    monkeypatch.setattr(sys, "argv", [v30.__file__, "--dry-run", "--bootstrap", "2"])
    v30.main()
    assert "DRY RUN" in capsys.readouterr().out
