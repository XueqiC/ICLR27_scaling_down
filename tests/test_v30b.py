"""Regional pairing, range leakage, incomplete coverage, and pre-check integrity."""
from __future__ import annotations

import copy
import hashlib
import sys

import numpy as np
import pytest

from analysis import prediction_audit as audit
from analysis import v30_quant_shape_candidates as v30
from analysis import v30b_quant_regions as regions


def synthetic_rows(n_models=4):
    rows = []
    for i in range(n_models):
        for j, cap in enumerate(audit.CAPABILITIES):
            for bit in v30.BITS:
                delta = (-1 if cap == "qa" else 1) * (.2 + i * .1 + j * .05) * float(
                    v30.normalized_shape(bit, "shared_eta", eta=2.6 + i * .2))
                rows.append({"row_id": f"fixture-{i}|{cap}|{bit}", "model": f"fixture-{i}",
                             "capability": cap, "bit": bit, "b_ref": 16, "reference_key": "dense",
                             "reference_loss": 1.5 - i * .1, "nominal_size_b": i + 1.,
                             "family": "a" if i < 2 else "b", "observed": delta, "loss": 1.5 - i * .1 + delta})
    return rows


@pytest.fixture(scope="module")
def result():
    # Saved-data regression and report exercise without output writes or model runs.
    before = hashlib.sha256(regions.SOURCE.read_bytes()).hexdigest()
    summary = regions.build_summary(n_boot=60)
    assert hashlib.sha256(regions.SOURCE.read_bytes()).hexdigest() == before
    return summary


@pytest.mark.parametrize("reference", [8, 16])
def test_actual_step_is_full_quantizer_curve_not_local_exponent(reference):
    bits = np.array([2, 3, 4, 5, 6, 8, reference])
    expected = (2. ** (bits - 1) - 1) ** -2 - (2. ** (reference - 1) - 1) ** -2
    assert regions.shape(bits, "actual_step", reference) == pytest.approx(expected, abs=1e-14)
    assert regions.shape(reference, "actual_step", reference) == 0
    actual3 = regions.shape(3, "actual_step", reference) / regions.shape(4, "actual_step", reference)
    surrogate3 = v30.normalized_shape(3, "shared_eta", reference, eta=2.2)
    assert abs(actual3 - surrogate3) > .8


def test_regions_partition_all_cells_without_censoring_signed_changes(result):
    for panel in result["panels"].values():
        scores = panel["regions"]
        partition = [row_id for region in regions.REGIONS[:-1] for row_id in scores[region]["row_ids"]]
        assert len(partition) == len(set(partition)) == len(panel["rows"])
        assert set(partition) == set(scores["full_domain"]["row_ids"])
        assert any(r["observed"] < 0 for r in panel["rows"])
        for r in panel["rows"]:
            assert sum(regions.in_region(r["bit"], group) for group in regions.REGIONS[:-1]) == 1
    assert result["panels"]["shape512"]["regions"]["collapse"]["by_capability"] == {}
    assert "olmo3-32b" in result["panels"]["broad"]["regions"]["collapse"]["models"]
    assert result["olmo3_32b_int3_exception"]["math"] == pytest.approx(.6824050486336266)


def test_headline_is_unchanged_and_all_region_metrics_use_paired_models(result):
    broad = result["panels"]["broad"]
    metrics = broad["regions"]["full_domain"]["by_capability"]["pooled"]["metrics"]
    assert metrics["shared_eta"]["mae"] == pytest.approx(1.04509, abs=5e-6)
    assert metrics["zero_change"]["mae"] == pytest.approx(2.77695, abs=5e-6)
    for region, scores in broad["regions"].items():
        rows = [r for r in broad["rows"] if regions.in_region(r["bit"], region)]
        names = ("zero_change", *regions.CANDIDATES)
        errors = np.array([[abs(r["observed"] - r["predictions"][c]) for c in names] for r in rows])
        draws = audit.bootstrap_means(errors, [r["model"] for r in rows], n_boot=result["n_boot"])
        for j, name in enumerate(names):
            m = scores["by_capability"]["pooled"]["metrics"][name]
            assert m["n"] == len(rows)
            assert m["mae"] == pytest.approx(errors[:, j].mean())
            assert m["mae_ci95"] == pytest.approx(audit.interval(draws[:, j]))
            assert m["improvement_ci95"] == pytest.approx(audit.interval(draws[:, 0] - draws[:, j]))
            assert m["improvement_over_zero_change"] == pytest.approx(errors[:, 0].mean() - m["mae"])


def test_region_gain_contributions_sum_to_full_domain_gain(result):
    broad = result["panels"]["broad"]
    for candidate in regions.CANDIDATES:
        contributions = broad["gain_decomposition"]
        total = sum(part[candidate]["full_domain_gain_contribution"] for part in contributions.values())
        metric = broad["regions"]["full_domain"]["by_capability"]["pooled"]["metrics"][candidate]
        assert total == pytest.approx(metric["improvement_over_zero_change"])
        assert sum(p[candidate]["fraction_of_net_gain"] for p in contributions.values()) == pytest.approx(1.)


def test_local_lomo_excludes_all_held_out_outcomes_and_keeps_all_test_bits():
    rows = synthetic_rows()
    before, folds_before = regions.lomo_fit_range(rows)
    changed = copy.deepcopy(rows)
    for row in changed:
        if row["model"] == "fixture-0":
            row["observed"] = 999 + row["bit"]
            row["loss"] = row["reference_loss"] + row["observed"]
    after, folds_after = regions.lomo_fit_range(changed)
    assert folds_before[0] == folds_after[0]
    assert [r["predictions"] for r in before if r["model"] == "fixture-0"] == [
        r["predictions"] for r in after if r["model"] == "fixture-0"]
    assert {r["bit"] for r in after} == set(v30.BITS)
    for fold in folds_before:
        assert all(int(key.split("|")[-1]) in (4, 5) for key in fold["train_row_ids"])
        assert fold["held_out"] not in fold["train_models"]
        assert not set(fold["train_row_ids"]) & set(fold["test_row_ids"])
    # Deep training outcomes cannot affect local fits, either.
    deep = copy.deepcopy(rows)
    for row in deep:
        if row["bit"] not in (4, 5):
            row["observed"] += 1000
    _, deep_folds = regions.lomo_fit_range(deep)
    assert deep_folds == folds_before


def test_eta_range_difference_uses_paired_refits():
    rows = synthetic_rows()
    for row in rows:
        if row["bit"] == 3:
            row["observed"] *= 3
    result = regions.eta_range_comparison(rows, 20)
    profiles = [v30.ShapeProfile(rows), v30.ShapeProfile([r for r in rows if r["bit"] in (4, 5)])]
    grids = [p.eta_grid() for p in profiles]
    samples = []
    for draw in np.random.default_rng(regions.SEED).integers(4, size=(20, 4)):
        weights = np.bincount(draw, minlength=4)
        fits = [p.fit_eta(weights, g)["eta"] for p, g in zip(profiles, grids)]
        samples.append(fits[0] - fits[1])
    assert result["all_minus_4_5"]["ci95"] == pytest.approx(audit.interval(samples))


def test_missing_five_commands_and_scoring_vs_training_sensitivity(result):
    fill = result["fill_missing_5bit"]
    assert fill["missing_models"] == ["gemma3-270m", "gemma3-4b", "olmo3-7b"]
    assert fill["commands_executed"] is False
    for model, command in zip(fill["missing_models"], fill["commands"]):
        assert f"--model {model} " in command
        assert "--bits 5 " in command and "--n-probe 512 " in command
        assert "--model-dtype bf16 " in command
        assert command.endswith("--output-base results/v10-quant-shape512-fill5")
    s = result["missing_model_sensitivity"]
    full = lambda key: s[key]["full_domain"]["by_capability"]["pooled"]
    assert full("with_missing_models")["n_cells"] == 33
    assert full("without_missing_models_fixed_v30_predictions")["n_cells"] == 24
    assert full("without_missing_models_refitted_lomo")["n_models"] == 4
    assert s["five_bit"]["all_seven_training_models"]["n_cells"] == 12
    assert s["five_bit"]["four_complete_training_models"]["n_cells"] == 12
    for fold in s["refitted_folds"]:
        assert not set(fill["missing_models"]) & set(fold["train_models"])
        assert len(fold["train_models"]) == 3


def test_snapshot_validation_catches_tampering():
    raw = synthetic_rows()
    records, folds = v30.lomo(raw)
    saved = {"rows": records, "folds": folds}
    regions.verify_snapshot(saved, raw)
    changed = copy.deepcopy(saved)
    changed["rows"][0]["predictions"]["actual_step"] += .1
    with pytest.raises(ValueError, match="prediction changed"):
        regions.verify_snapshot(changed, raw)
    with pytest.raises(ValueError, match="coverage changed"):
        regions.verify_snapshot(saved, raw[:-1])


def test_precheck_has_no_invented_target_observations_or_measurement_se(result):
    pre = result["qwen3_14b_precheck"]
    assert pre["target_reference_losses"] is None
    assert pre["target_quantized_outcomes_used"] == pre["target_compressed_calibration_points"] == 0
    assert pre["verdict"] == "INCONCLUSIVE_FOR_ACTUAL_TARGET"
    assert result["dev_measurement_error_proxy"]["measurement_se"] is None
    scale = result["dev_measurement_error_proxy"]["by_capability"]["math"]["4"]
    assert scale["n_models"] == 7
    assert scale["mean_absolute_discrepancy"] == pytest.approx(.025704762823290563)
    assert not any("14b" in p.lower() or "v28" in p for p in result["input_sha256"])
    for protocol in pre["protocols"].values():
        assert not set(v30.RESERVED_MODELS) & set(protocol["train_models"])
        assert len(protocol["cells"]) == 6
        for cell in protocol["cells"]:
            for candidate in regions.CANDIDATES:
                eq = cell["prediction_affine_in_reference_loss"][candidate]
                predicted = eq["intercept"] + eq["reference_loss_coefficient"] * cell["reference_loss_scenario"]
                assert cell["predicted_delta_loss"][candidate] == pytest.approx(predicted)
            gaps = cell["pairwise_gaps"]
            assert len(gaps) == 3
            assert cell["pairwise_screening_verdicts"].keys() == gaps.keys()
            for pair, value in gaps.items():
                a, b = pair.split("__")
                assert value == pytest.approx(abs(cell["predicted_delta_loss"][a] - cell["predicted_delta_loss"][b]))


def test_precheck_supplied_dense_input_and_allowlist(result):
    rows = result["panels"]["broad"]["rows"]
    noise = result["dev_measurement_error_proxy"]
    dense = {"math": .9, "code": 1.4, "qa": 6.}
    pre = regions.discrimination_precheck(rows, noise, dense)
    assert pre["target_reference_losses"] == dense
    for protocol in pre["protocols"].values():
        for cell in protocol["cells"]:
            for candidate, fit in protocol["frozen_fits"].items():
                row = {"model": "qwen3-14b", "family": "qwen3", "nominal_size_b": 14.,
                       "reference_loss": dense[cell["capability"]], "capability": cell["capability"],
                       "bit": cell["bit"], "b_ref": 16}
                assert cell["predicted_delta_loss"][candidate] == pytest.approx(regions.predict(fit, row))
    with pytest.raises(ValueError, match="only math, code, qa"):
        regions.discrimination_precheck(rows, noise, {**dense, "4": 7.})
    with pytest.raises(ValueError, match="Nonfinite"):
        regions.discrimination_precheck(rows, noise, {**dense, "qa": float("nan")})


def test_screen_does_not_call_near_zero_candidates_distinguishable():
    near_zero = dict(zip(regions.CANDIDATES, [.001, -.001, .002]))
    assert regions.screen_predictions(near_zero, .02)["screening_verdict"] == "INCONCLUSIVE_NEAR_ZERO"
    separated = dict(zip(regions.CANDIDATES, [.1, .3, .5]))
    assert regions.screen_predictions(separated, .02)["screening_verdict"] == "ALL_PAIRS_ABOVE_PROXY"
    assert regions.absolute_affine_range(-1., 1., 0., 2.) == [0., 1.]


def test_report_and_dry_run_preserve_freeze_labels_and_do_not_write(result, monkeypatch, capsys):
    report = regions.render(result)
    for phrase in ("CONDITIONAL SHAPE TRANSFER", "Full domain", "OLMo3-32B", "not executed",
                   "INCONCLUSIVE_FOR_ACTUAL_TARGET", "η≈2.20", "measurement standard error"):
        assert phrase in report
    monkeypatch.setattr(regions, "build_summary", lambda *args: result)
    def forbidden(*args, **kwargs):
        pytest.fail("Dry run wrote an artifact")
    monkeypatch.setattr(audit, "write_outputs", forbidden)
    monkeypatch.setattr(regions.Path, "write_text", forbidden)
    monkeypatch.setattr(sys, "argv", [regions.__file__, "--bootstrap", "2", "--dry-run"])
    regions.main()
    assert "DRY RUN" in capsys.readouterr().out
