"""V28 CPU tests. Synthetic observations below are fixtures, never report data."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from analysis import v28_new_source_prediction as v28


def synthetic_panel(arm):
    rows = []
    for i in range(6):
        coordinates = v28.PRUNE_DEV if arm == "pruning" else (*v28.QUANT_DEV, 5)
        a = (-1 if i == 0 else 1) * (0.1 + i*.07)
        rows.append({"model": f"fixture-{i}", "family": "a" if i < 3 else "b",
                     "N0": (i+1)*1e9, "dense_loss": 1.5-i*.1, "capability": "qa",
                     "deltas": {str(c): a*v28.shape(arm, c, 2.0) for c in coordinates}})
    return rows


@pytest.mark.parametrize("arm", ["pruning", "quantization"])
def test_coefficient_labels_recover_signed_known_law(arm):
    rows = synthetic_panel(arm)
    fit = v28.fit_arm(rows, arm)
    if arm == "pruning":
        assert fit["gamma"] == pytest.approx(2., abs=2e-5)
    for i, r in enumerate(rows):
        expected = (-1 if i == 0 else 1) * (.1+i*.07)
        assert fit["development_coefficient_labels"][r["model"]] == pytest.approx(expected, rel=1e-5)
    assert fit["shape_fit_sse"] < 1e-10


@pytest.mark.parametrize("arm", ["pruning", "quantization"])
def test_lomo_refits_every_coefficient_and_preprocessor_without_target(arm):
    rows = synthetic_panel(arm)
    before = v28.lomo_arm(rows, arm)
    changed = copy.deepcopy(rows)
    changed[0]["deltas"] = {key: 9999. for key in changed[0]["deltas"]}
    after = v28.lomo_arm(changed, arm)
    f0 = before["folds"][0]; f1 = after["folds"][0]
    assert f0["fit"] == f1["fit"]  # Includes gamma, coefficient labels, scaling, ridge.
    assert f0["mode_A_target_inputs"] == f1["mode_A_target_inputs"]
    assert f0["mode_B_extra_input"] != f1["mode_B_extra_input"]
    assert [r["A"] for r in before["records"] if r["model"] == "fixture-0"] == [
        r["A"] for r in after["records"] if r["model"] == "fixture-0"]
    for fold in before["folds"]:
        assert fold["held_out"] not in fold["fit"]["mapping"]["train_models"]
        assert len(fold["fit"]["mapping"]["train_models"]) == 5
    train = rows[1:]
    assert f0["fit"]["mapping"]["center"] == pytest.approx(
        np.mean([[np.log(r["N0"]/1e9), r["dense_loss"]] for r in train], axis=0))


def test_unknown_family_has_zero_correction_and_pending_dense_formula_is_exact():
    rows = synthetic_panel("quantization")
    fit = v28.fit_arm(rows[:-1], "quantization")["mapping"]
    target = v28.basic_input(rows[-1]); target["family"] = "unseen"
    form = v28.mapping_formula(fit, target)
    beta = fit["coefficients"]
    expected = beta[0]+beta[1]*(np.log(target["N0"]/1e9)-fit["center"][0])/fit["scale"][0]
    expected += beta[2]*(target["dense_loss"]-fit["center"][1])/fit["scale"][1]
    assert v28.predict_mapping(fit, target) == pytest.approx(expected)
    assert form["intercept"]+form["dense_slope"]*target["dense_loss"] == pytest.approx(expected)
    with pytest.raises(ValueError, match="leaked"):
        v28.predict_mapping(fit, rows[0])


@pytest.mark.parametrize("arm", ["pruning", "quantization"])
def test_modes_enforce_zero_vs_exactly_one_disjoint_calibration_point(arm):
    rows = synthetic_panel(arm)
    fit = v28.fit_arm(rows[:-1], arm)
    target = v28.basic_input(rows[-1])
    coords = v28.PRUNE_TEST if arm == "pruning" else v28.QUANT_TEST
    c = v28.CALIBRATION[arm]
    point = {"coordinate": c, "loss": target["dense_loss"]+rows[-1]["deltas"][str(c)]}
    a = v28.predict_arm(fit, target, coords)
    polluted = {**target, "q": 1e30, "observed": 1e30, "deltas": {str(c): 1e30}}
    assert a == v28.predict_arm(fit, polluted, coords)
    with pytest.raises(ValueError, match="Mode A forbids"):
        v28.predict_arm(fit, target, coords, calibration=point)
    with pytest.raises(ValueError, match="exactly one"):
        v28.predict_arm(fit, target, coords, "B")
    with pytest.raises(ValueError, match="exactly one"):
        v28.predict_arm(fit, target, coords, "B", {**point, "another_point": 1.})
    with pytest.raises(ValueError, match="pre-specified disjoint"):
        v28.predict_arm(fit, target, coords, "B", {**point, "coordinate": coords[0]})
    b = v28.predict_arm(fit, target, coords, "B", point)
    assert b == pytest.approx([rows[-1]["deltas"][str(c)] for c in coords], rel=1e-5)
    assert v28.predict_arm(fit, target, [1.0] if arm == "pruning" else [16]) == pytest.approx([0.])


def test_quant_5bit_is_never_a_development_coefficient_label():
    rows = synthetic_panel("quantization")
    before = v28.fit_arm(rows, "quantization")
    for r in rows:
        r["deltas"]["5"] = -1e30
    assert before == v28.fit_arm(rows, "quantization")


@pytest.fixture(scope="module")
def frozen():
    metadata = v28.audit.read_json(v28.METADATA)
    source = {"model": "Qwen3-8B", "hf_id": "Qwen/Qwen3-8B-Base", "family": "qwen3",
              "N0": metadata["qwen3_8b"]["N0"], "training_stage": "Base", "revision": None,
              "pretraining_tokens": None}
    student = {"model": "gemma3-12b", "hf_id": "google/gemma-3-12b-pt", "budget": 300, "revision": None}
    return v28.freeze(source, student)


def measurements(frozen):
    dense = {"protocol_id": v28.PROTOCOL_ID}
    calibration = {"protocol_id": v28.PROTOCOL_ID}
    for who in ("source", "student"):
        dense[who] = {"model": frozen[who]["model"], "hf_id": frozen[who]["hf_id"],
                      "revision": "synthetic-test-revision", "losses": {"math": 1., "code": 2., "qa": 3.}}
    for arm, c in v28.CALIBRATION.items():
        who = "student" if arm == "distillation" else "source"
        calibration[arm] = {**copy.deepcopy(dense[who]), "coordinate": c}
        calibration[arm]["losses"] = {cap: value+.03 for cap, value in dense[who]["losses"].items()}
    return dense, calibration


def save_json(path, value):
    path.write_text(json.dumps(value))
    return path


def test_real_freeze_keeps_missing_inputs_null_and_uses_correct_non_embedding_n0(frozen):
    assert frozen["source"]["N0"] == 6_945_767_424
    assert len(frozen["metadata"]["models"]) == 12
    assert frozen["metadata"]["models"]["gemma3-270m"]["N0"] == 100_270_080
    assert len(frozen["predictions"]) == 36
    assert frozen["freeze_sha256"] == v28.seal(frozen)
    for row in frozen["predictions"]:
        assert row["L_predicted"] is None
        assert row["target_compressed_points_used"] == 0
        if row["arm"] != "distillation" or row["mode"] == "B":
            assert row["delta_L"] is None
        else:
            assert np.isfinite(row["delta_L"])
        assert np.isfinite(row["interval_recipe"]["offsets"]).all()
    for arm in ("pruning", "quantization"):
        for r in frozen["methods"][arm]["by_capability"].values():
            assert len(r["lomo"]["folds"]) == 12
            assert all(len(f["fit"]["mapping"]["train_models"]) == 11 for f in r["lomo"]["folds"])
            assert all(p["coordinate"] != v28.CALIBRATION[arm] for p in r["lomo"]["records"])


def test_bind_inputs_cannot_refit_or_read_development_and_calibration_cannot_change_A(frozen, tmp_path, monkeypatch):
    dense, calibration = measurements(frozen)
    dp = save_json(tmp_path/"dense.json", dense); cp = save_json(tmp_path/"cal.json", calibration)
    def forbidden(*args, **kwargs):
        pytest.fail("Attempted to fit/read dev data during frozen application")
    monkeypatch.setattr(v28, "load_development", forbidden)
    monkeypatch.setattr(v28, "fit_arm", forbidden)
    monkeypatch.setattr(v28, "fit_transfer", forbidden)
    monkeypatch.setattr(v28, "shape", forbidden)
    monkeypatch.setattr(v28, "mapping_formula", forbidden)
    before = v28.bind_inputs(frozen, dp)
    after = v28.bind_inputs(frozen, dp, cp)
    assert [r for r in before["predictions"] if r["mode"] == "A"] == [r for r in after["predictions"] if r["mode"] == "A"]
    assert all(r["delta_L"] is not None for r in after["predictions"])
    assert after["refit"] is False
    for r in after["predictions"]:
        if r["mode"] == "B":
            assert r["target_compressed_points_used"] == 1
        who = "student" if r["arm"] == "distillation" else "source"
        assert r["L_predicted"]-r["delta_L"] == pytest.approx(dense[who]["losses"][r["capability"]])
    # The other two arms' calibration does not enter pruning Mode B either.
    calibration["quantization"]["losses"]["math"] = 100.
    save_json(cp, calibration)
    changed = v28.bind_inputs(frozen, dp, cp)
    assert [r for r in after["predictions"] if r["arm"] == "pruning"] == [r for r in changed["predictions"] if r["arm"] == "pruning"]


@pytest.mark.parametrize("mutation,kind,match", [
    (lambda d, c: d.update(compressed={}), "dense", "Unexpected"),
    (lambda d, c: d["source"].update(q_calibrated=42), "dense", "requires exactly"),
    (lambda d, c: d.update(protocol_id="different"), "dense", "protocol"),
    (lambda d, c: d["source"].update(hf_id="Qwen/Qwen3-4B"), "dense", "identity"),
    (lambda d, c: d["source"]["losses"].update(math=float("nan")), "dense", "finite"),
    (lambda d, c: c["quantization"].update(coordinate=4), "calibration", "pre-specified"),
    (lambda d, c: c["pruning"].update(extra_point={}), "calibration", "requires exactly"),
])
def test_strict_measurement_input_separation(frozen, tmp_path, mutation, kind, match):
    dense, calibration = measurements(frozen)
    mutation(dense, calibration)
    path = save_json(tmp_path/"inputs.json", dense if kind == "dense" else calibration)
    with pytest.raises(ValueError, match=match):
        v28.read_measurements(path, frozen, kind)


def test_revision_mismatch_and_freeze_tampering_are_rejected(frozen, tmp_path):
    dense, calibration = measurements(frozen)
    calibration["quantization"]["revision"] = "another-checkpoint"
    with pytest.raises(ValueError, match="revisions differ"):
        v28.bind_inputs(frozen, save_json(tmp_path/"dense.json", dense), save_json(tmp_path/"cal.json", calibration))
    altered = copy.deepcopy(frozen)
    altered["methods"]["pruning"]["by_capability"]["math"]["fit"]["gamma"] += .1
    with pytest.raises(ValueError, match="integrity"):
        v28.bind_inputs(altered)


def test_prebound_dense_inputs_persist_and_cannot_be_replaced(frozen, tmp_path):
    dense, cal = measurements(frozen)
    prebound = copy.deepcopy(frozen)
    prebound["measurement_inputs"] = {"dense": {k: copy.deepcopy(v) for k, v in dense.items() if k != "protocol_id"}}
    prebound["freeze_sha256"] = v28.seal(prebound)
    result = v28.bind_inputs(prebound, calibration_path=save_json(tmp_path/"cal.json", cal))
    assert all(r["L_predicted"] is not None for r in result["predictions"])
    dense["source"]["losses"]["qa"] = 99.
    with pytest.raises(ValueError, match="Cannot change"):
        v28.bind_inputs(prebound, save_json(tmp_path/"dense.json", dense))


def test_alias_overlap_and_tested_student_are_rejected_before_fit(frozen):
    source = copy.deepcopy(frozen["source"])
    source["hf_id"] = "Qwen--Qwen3-4B"
    with pytest.raises(ValueError, match="overlaps"):
        v28.freeze(source, frozen["student"])
    with pytest.raises(ValueError, match="already tested"):
        v28.freeze(frozen["source"], {**frozen["student"], "budget": 600})


def test_distillation_data_dependence_recipe_filter_and_student_grouping(frozen):
    distill = frozen["methods"]["distillation"]
    assert len(distill["excluded_non_lora_row_ids"]) == 3
    for cap, result in distill["by_capability"].items():
        assert v28.transfer_shape(cap, 0) == 0.
        assert ("log" in result["fit"]["form"]) == (cap == "code")
        for fold in result["lomo"]["folds"]:
            assert all(not row.startswith(fold["held_out"]+"|") for row in fold["fit"]["train_row_ids"])
        assert all(r["coordinate"] != 150 for r in result["lomo"]["records"])


def test_dry_run_prints_conditional_table_and_never_writes(monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        pytest.fail("Dry run attempted a write")
    monkeypatch.setattr(v28, "write_new", forbidden)
    monkeypatch.setattr(Path, "write_text", forbidden)
    result = v28.main(["--dry-run"])
    out = capsys.readouterr().out
    assert "DRY RUN" in out and "NO-compression-calibration" in out and "FEW-compression-calibration" in out
    assert "L_dense" in out and "LOMO" in out
    assert result["source"]["hf_id"] == "Qwen/Qwen3-8B"
    assert "identity_note" in result["source"]


def test_cpu_import_and_exclusive_freeze_write(tmp_path):
    process = subprocess.run([sys.executable, "-c", "import sys; from analysis import v28_new_source_prediction; assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules"], cwd=v28.ROOT, capture_output=True, text=True)
    assert process.returncode == 0, process.stderr
    path = tmp_path/"frozen.json"
    v28.write_new(path, {"first": True})
    with pytest.raises(FileExistsError):
        v28.write_new(path, {"replacement": True})
    assert json.loads(path.read_text()) == {"first": True}
