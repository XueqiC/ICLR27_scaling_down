"""Leakage, paired comparison, boundary and provenance checks for V24–V26.

Synthetic arrays here are unit-test fixtures only; empirical reports always
load the saved input artifacts.
"""
from __future__ import annotations

import copy
import hashlib
import sys

import numpy as np
import pytest

from analysis import prediction_audit as audit
from analysis import v24_quant_baselines as quant
from analysis import v25_distill_delta as distill
from analysis import v26_loss_validity_prediction as validity


def test_bootstrap_preserves_pairing_and_unequal_cluster_sizes():
    # Both columns share every sampled row: their difference is exactly seven.
    values = np.array([[0, 7], [2, 9], [20, 27]])
    draws = audit.bootstrap_means(values, ["a", "a", "b"], n_boot=500)
    assert draws[:, 1] - draws[:, 0] == pytest.approx(np.full(500, 7.))
    assert set(np.round(draws[:, 0], 6)) == {1., round(22/3, 6), 20.}
    assert np.array_equal(draws, audit.bootstrap_means(values, ["a", "a", "b"], n_boot=500))


def test_quantization_5bit_never_enters_q_calibration():
    loss = {"dense": {"math": 1.}}
    loss.update({str(b): {"math": 1 + 200 * quant.shape(b)} for b in (8, 6, 5, 4)})
    assert quant.calibrated_q(loss, "math") == pytest.approx(200)
    loss["5"]["math"] = 999.
    assert quant.calibrated_q(loss, "math") == pytest.approx(200)


def test_dense_only_predictor_cannot_read_test_calibration_or_outcome():
    rows, _, _ = quant.load_rows()
    rows = [r for r in rows if r["capability"] == "code"]
    train, test = rows[:-1], rows[-1:]
    before, _ = quant.dense_predict(train, test)
    altered = [{**test[0], "q_calibrated": 1e20, "observed": 1e10,
                "delta4": 1e10, "delta6": 1e10, "law_frozen": 1e10}]
    after, _ = quant.dense_predict(train, altered)
    assert after == pytest.approx(before)
    with pytest.raises(ValueError, match="leaked"):
        quant.dense_predict(train + test, test)


def test_quant_comparators_cover_identical_frozen_cells():
    summary = quant.build_summary(100)
    assert len(summary["rows"]) == 36
    for row in summary["rows"]:
        assert set(row["predictions"]) == set(quant.NAMES)
        assert row["predictions"]["linear_4_6bit"] == pytest.approx((row["delta4"] + row["delta6"]) / 2)
        assert row["predictions"]["law_frozen"] == row["law_frozen"]
        assert abs(row["law_frozen"] - row["law_reconstructed"]) <= .000051
    for fold in summary["dense_only_folds"]:
        assert not set(fold["train_row_ids"]) & set(fold["test_row_ids"])
        assert all(not r.startswith(fold["held_out"] + "|") for r in fold["train_row_ids"])


@pytest.mark.parametrize("candidate", distill.CANDIDATES[2:])
def test_transfer_responses_zero_at_zero_data(candidate):
    rows = [{"N_S": n, "D": 0} for n in distill.SIZES.values()]
    assert np.all(distill.design(rows, candidate) == 0)


@pytest.mark.parametrize("key", ["model", "D"])
def test_distill_outcomes_held_out_on_entire_axis(key):
    rows, _ = distill.load_rows()
    rows = [r for r in rows if r["capability"] == "math"]
    split = next(audit.splits(rows, key))
    train, test = ([rows[i] for i in split[k]] for k in ("train_indices", "test_indices"))
    assert not {r[key] for r in train} & {r[key] for r in test}
    for candidate in distill.CANDIDATES:
        before, _ = distill.fit_predict(train, test, candidate)
        altered = [{**r, "observed": 1e20, "post_loss": 1e20} for r in test]
        after, _ = distill.fit_predict(train, altered, candidate)
        assert before == pytest.approx(after)
    with pytest.raises(ValueError, match="separately"):
        distill.fit_predict(train, [{**test[0], "capability": "qa"}], "mean")


def test_delta_and_total_loss_errors_equal_and_training_modes_not_hidden():
    summary = distill.build_summary(100)
    assert len(summary["rows"]) == 36
    for protocol in distill.PROTOCOLS:
        for cap in audit.CAPABILITIES:
            full = summary["analyses"]["full_grid"][protocol][cap]
            lora = summary["analyses"]["lora_only"][protocol][cap]
            assert len(full["records"]) == 12
            assert len(lora["records"]) == 11
            assert all(r["training_mode"] == "lora" for r in lora["records"])
            for row in full["records"]:
                for c in distill.CANDIDATES:
                    assert row["predictions_loss"][c] - row["post_loss"] == pytest.approx(
                        row["predictions_delta"][c] - row["observed"])


def rank_one_rows():
    rows = []
    for m in range(4):
        for d in (.9, .8, .7, .6):
            damage = (m + 1) * (1 - d)**2
            rows.append({"row_id": f"m{m}|{d}", "model": f"m{m}", "density": d,
                         "primary": dict(zip(audit.CAPABILITIES, [damage, 2*damage, 3*damage])),
                         "secondary": dict(zip(audit.CAPABILITIES, [4*damage, 5*damage, 6*damage]))})
    return rows


def test_general_damage_high_correlation_has_no_same_capability_gain():
    rows = rank_one_rows()
    result = validity.evaluate(rows, "math", "leave_model_and_density_out", 100)
    assert result["same_over_cross_gain"] == pytest.approx(0, abs=1e-12)
    assert result["metrics"]["math"]["mae"] == pytest.approx(0, abs=1e-12)
    assert result["metrics"]["cross_selected"]["mae"] == pytest.approx(0, abs=1e-12)


def test_joint_holdout_excludes_both_axes_and_never_selects_on_test_targets():
    rows = rank_one_rows()
    for fold in validity.folds(rows, "leave_model_and_density_out"):
        train, test = ([rows[i] for i in fold[k]] for k in ("train_indices", "test_indices"))
        assert not {r["model"] for r in train} & {r["model"] for r in test}
        assert not {r["density"] for r in train} & {r["density"] for r in test}
    before = validity.evaluate(rows, "qa", "leave_model_out", 100)
    altered = copy.deepcopy(rows)
    for r in altered:
        if r["model"] == "m0":
            r["secondary"]["qa"] += 100
    after = validity.evaluate(altered, "qa", "leave_model_out", 100)
    f0 = next(f for f in before["folds"] if f["held_out"] == "m0")
    f1 = next(f for f in after["folds"] if f["held_out"] == "m0")
    assert f0 == f1
    p0 = [r["predictions"] for r in before["records"] if r["model"] == "m0"]
    p1 = [r["predictions"] for r in after["records"] if r["model"] == "m0"]
    assert p0 == p1


def test_loss_item_pairing_rejects_changed_spans():
    rows, _, paths = validity.load_rows()
    path = audit.ROOT / rows[0]["source_path"]
    p = audit.read_json(path)["capabilities"]["math"]["primary"]
    dense = audit.read_json(path.parent.parent / "dense/loss_validity.json")["capabilities"]["math"]["primary"]
    delta = validity.paired_item_deltas(p, dense)
    assert delta.mean() == pytest.approx(p["delta_L_c"])
    bad = copy.deepcopy(p)
    bad["items"][0]["n_tokens"] += 1
    with pytest.raises(ValueError, match="Unpaired"):
        validity.paired_item_deltas(bad, dense)


def test_noise_bootstrap_preserves_shared_item_covariance():
    item_rows = []
    # Item heterogeneity identical across densities cancels on centering.
    for cap in audit.CAPABILITIES:
        for role in ("primary", "secondary"):
            for d in (.9, .8, .7, .6):
                item_rows.append({"model": "olmo-fixture", "capability": cap, "role": role,
                                  "density": d, "item_ids": list(range(8)),
                                  "deltas": np.arange(8) + 1-d})
    result = validity.noise_audit(item_rows, 100)
    assert all(r["centered_sampling_noise_rms"] < 1e-12 for r in result)
    assert all(r["centered_signal_rms"] > .1 for r in result)


@pytest.mark.parametrize("module", [quant, distill, validity])
def test_dry_run_performs_no_artifact_or_report_writes(module, monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        pytest.fail("dry-run attempted an output write")
    monkeypatch.setattr(audit, "write_outputs", forbidden)
    monkeypatch.setattr(audit, "replace_section", forbidden)
    monkeypatch.setattr(module, "REPORT", module.ROOT / "docs" / module.REPORT.name)
    before = hashlib.sha256(module.REPORT.read_bytes()).hexdigest()
    monkeypatch.setattr(sys, "argv", [module.__file__, "--dry-run", "--bootstrap", "50"])
    module.main()
    assert "DRY RUN" in capsys.readouterr().out
    assert hashlib.sha256(module.REPORT.read_bytes()).hexdigest() == before


def test_source_mutation_aborts_output(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    path = tmp_path / "input.json"
    path.write_text("{}")
    summary = {"input_sha256": audit.provenance([path])}
    path.write_text('{"changed": true}')
    with pytest.raises(RuntimeError, match="Input changed"):
        audit.write_outputs(summary, "report", tmp_path / "out")
    assert not (tmp_path / "out").exists()
