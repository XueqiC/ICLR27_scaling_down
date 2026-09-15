"""CPU-only provenance, arithmetic, trajectory-unit, and failure tests for A7."""
import ast
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import pytest

from analysis import a7_closeout_audit as a7


@pytest.fixture(scope="module")
def artifacts():
    return {name: json.loads((a7.ROOT / path).read_text()) for name, path in a7.SOURCES.items()}


@pytest.fixture(scope="module")
def audit():
    return a7.build_audit()


def test_gate_uses_per_fold_selection_and_frozen_metrics_not_decision_table(artifacts):
    source = deepcopy(artifacts["a2"])
    expected = a7.check_gate(source)
    source["decision_table"] = {"poison": "must never be read"}
    assert a7.check_gate(source) == expected
    del source["decision_table"]
    assert a7.check_gate(source) == expected
    # Rename an inner-selected baseline in one held fold; the audit must follow it.
    result = next(r for r in source["results"] if r["distribution"].startswith("training_probe:MBPP:"))
    fold = result["folds"][0]
    m = fold["metrics"]["I_T"]
    old = fold["selection"]["baseline"]["I_T"]
    fold["selection"]["baseline"]["I_T"] = "synthetic_inner_baseline"
    m["inner_selected_baselines"] = {"synthetic_inner_baseline": m["inner_selected_baselines"][old]}
    m["all_model_mae"]["synthetic_inner_baseline"] = m["baseline_mae"]
    row = a7.check_gate(source)["inner_selected"][0]
    assert row["baseline"] == "synthetic_inner_baseline"
    assert row["frozen_error_interval"] == m["frozen_prediction_error_interval"]
    assert row["paired_gain"] == m["paired_improvement"]


def test_gate_all_primary_folds_diagnostic_and_unscorable_are_preserved(artifacts, audit):
    gate = audit["checks"]["gate"]
    assert len(gate["inner_selected"]) == len(gate["outer_oracle_diagnostic"]) == 24
    for inner, oracle in zip(gate["inner_selected"], gate["outer_oracle_diagnostic"]):
        assert inner["distribution"].startswith(("training_probe:MBPP:", "2wiki_new:"))
        result = next(r for r in artifacts["a2"]["results"]
                      if r["distribution"] == inner["distribution"] and r["split"] == inner["split"])
        fold = next(f for f in result["folds"] if f["held"] == inner["held"])
        if inner["status"] != "scored":
            assert inner["capability"] == "qa" and inner["split"] == "largest_budget"
            assert inner["primary"] is inner["baseline"] is inner["paired_gain"] is None
            continue
        metric = fold["metrics"][inner["target"]]
        assert inner["baseline"] == fold["selection"]["baseline"][inner["target"]]
        assert oracle["baseline"] == metric["strongest_observed_baseline"]
        assert oracle["baseline_mae"] == metric["strongest_baseline_mae"]
        assert oracle["frozen_error_interval"] == metric["interval_over_strongest"]
    assert [(r["scorable_folds"], r["positive_gain_folds"], r["positive_gain_clear_interval_folds"])
            for r in gate["conclusions"]] == [(12, 3, 1), (11, 4, 0)]
    assert not any(r["verdict_changed"] for r in gate["conclusions"])


@pytest.mark.parametrize("gain,interval,flags", [
    (1, [0, 2], (False, False)), (1, [0.1, 2], (True, True)),
    (-1, [-2, -0.1], (True, False)), (-1, [-2, 0], (False, False)),
    (1, None, (None, False)),
])
def test_gate_clear_interval_is_strict_and_sign_aware(gain, interval, flags):
    assert a7.interval_flags(gain, interval) == flags


def test_gate_rejects_inconsistent_per_fold_baseline(artifacts):
    result = deepcopy(artifacts["a2"]["results"][0])
    result["folds"][0]["selection"]["baseline"]["I_T"] = "wrong"
    with pytest.raises(ValueError, match="inner selection"):
        a7.gate_row(result, result["folds"][0], "I_T")


def test_accounting_exact_sums_reproduce_every_file_count_and_revision(artifacts, audit):
    for row in audit["checks"]["accounting"]["rows"]:
        source = artifacts["counts"]["students"][row["student"]]
        for key in ("total_parameters", "embedding_parameters", "vision_and_projector_parameters",
                    "non_embedding_parameters", "revision", "embedding_tensors"):
            assert row[key] == source[key]
        assert row["total_parameters"] - row["embedding_parameters"] - row["vision_and_projector_parameters"] == row["non_embedding_parameters"]
        assert row["non_embedding_parameters"] == artifacts["a2"]["nonembedding_counts"][row["student"]]
    # Change all corresponding stored counts coherently: displayed values must follow.
    counts, a2 = deepcopy(artifacts["counts"]), deepcopy(artifacts["a2"])
    row = counts["students"]["gemma3-4b"]
    row["total_parameters"] += 2_000_000
    row["non_embedding_parameters"] += 2_000_000
    a2["nonembedding_counts"]["gemma3-4b"] += 2_000_000
    changed = a7.check_accounting(counts, a2)
    assert f"{row['total_parameters']/1e9:.3f}B" in changed["note"]
    assert f"{row['non_embedding_parameters']/1e9:.3f}B" in changed["note"]
    row["vision_and_projector_parameters"] += 1
    with pytest.raises(ValueError, match="does not reconcile"):
        a7.check_accounting(counts, a2)


def test_seed_pair_is_one_correlated_trajectory_and_ignores_sigma_summaries(artifacts, audit):
    pair = audit["checks"]["seed_pair"]
    assert pair["run_pair_count"] == 1
    assert pair["shared_checkpoint_count"] == 14
    assert pair["independent_checkpoint_replicates"] is False
    assert "Descriptive only" in pair["interpretation"]
    assert "not 14 independent replicates" in pair["interpretation"]
    assert "generally conservative" in pair["interpretation"]
    assert "sigma_delta" not in pair and "sigma_I" not in pair
    seed = deepcopy(artifacts["seed"])
    seed["summary"] = {"sigma_delta": "invalid", "sigma_I": "invalid", "n": 999999}
    assert a7.check_seed_pair(seed, artifacts["corners"]) == pair
    del seed["summary"]
    assert a7.check_seed_pair(seed, artifacts["corners"]) == pair
    for row, raw in zip(pair["trajectory"], seed["per_checkpoint"]):
        assert row["checkpoint"] == raw[0]
        assert row["T_supervised"] == raw[1]
        assert row["differences_seed1_minus_seed0"] == raw[2]
    assert [r["update"] for r in pair["corner_checkpoints"]] == [36, 74]
    for row in pair["corner_checkpoints"]:
        raw = next(r for r in seed["per_checkpoint"] if r[0] == row["checkpoint"])
        assert row["differences_seed1_minus_seed0"] == raw[2]


def test_seed_pair_missing_corner_or_mismatched_budget_fails(artifacts):
    seed = deepcopy(artifacts["seed"])
    seed["per_checkpoint"][3][1] += 1
    with pytest.raises(ValueError, match="corner mismatch"):
        a7.check_seed_pair(seed, artifacts["corners"])
    seed = deepcopy(artifacts["seed"])
    seed["per_checkpoint"].pop(3)
    with pytest.raises(ValueError, match="membership/order"):
        a7.check_seed_pair(seed, artifacts["corners"])


def rectangular_coordinates():
    # u_low=1, u_high=3; v_low=2, v_high=5 at T_ref=1.
    return {"1": (math.expm1(1), math.expm1(2)), "2": (math.expm1(1), math.expm1(5)),
            "3": (math.expm1(3), math.expm1(2)), "4": (math.expm1(3), math.expm1(5))}


@pytest.mark.parametrize("z", [-100, 0, 2, 100])
def test_f_int_matches_hand_computation_and_cancels_student_terms(z):
    # k=.5: I=.5*(3-1)*(2-5)=-3, independent of a,a',b,b',z.
    value = a7.structure_second_difference("F_int", [7, -11, 13, 17, 0.5], rectangular_coordinates(), z, 1)
    assert value["I"] == pytest.approx(-3, abs=1e-14)
    assert value["budget_mismatch_term"] == value["reuse_mismatch_term"] == 0
    assert value["is_exact_rectangle"]


@pytest.mark.parametrize("name,p", [("F_log", None), ("F_curv", 0), ("F_curv", 1),
                                   ("F_curv", -1), ("F_curv", 0.37), ("F_curv", 1e-9)])
def test_additive_forms_are_exactly_zero_on_rectangle(name, p):
    result = a7.structure_second_difference(name, [7, -11, 13, 17], rectangular_coordinates(), 1.25, 1, p)
    assert result["I"] == 0.0


@pytest.mark.parametrize("name,p", [("F_log", None), ("F_curv", 0.7), ("F_int", None)])
def test_achieved_mismatches_are_evaluated_instead_of_forced_to_zero(name, p):
    coords = {"1": (3, 2), "2": (3.1, 5), "3": (11, 2.2), "4": (11.3, 5.2)}
    coefficients = [7, -11, 13, 17] + ([0.5] if name == "F_int" else [])
    z, t_ref = 1.25, 2
    values = {}
    for i, (t, e) in coords.items():
        u, v = math.log1p(t/t_ref), math.log1p(e)
        reuse = ((1+e)**p-1)/p if name == "F_curv" else v
        values[i] = (7-11*z)*u + (13+17*z)*reuse + (0.5*u*v if name == "F_int" else 0)
    expected = values["3"] - values["4"] - values["1"] + values["2"]
    result = a7.structure_second_difference(name, coefficients, coords, z, t_ref, p)
    assert result["I"] == pytest.approx(expected, abs=1e-12)
    assert result["I"] != 0 and not result["is_exact_rectangle"]


def test_real_predictions_use_full_development_fits_exact_distribution_and_scaler(artifacts, audit):
    predictions = audit["checks"]["predictions"]
    assert len(predictions["rows"]) == 6
    for row in predictions["rows"]:
        assert row["status"] == "COMPLETE"
        source = next(r for r in artifacts["a2"]["parameter_intervals"]
                      if r["capability"] == row["capability"] and r["distribution"] == row["distribution"])
        if row["capability"] == "qa":
            assert row["distribution"].startswith("training_probe:2WikiMultihopQA:")
        for name, pred in row["predictions"].items():
            frozen = source["fits"][name]["fit"]
            for key in pred["fit"]:
                assert pred["fit"][key] == frozen[key]
            scaler = frozen["standardizer"]
            student = artifacts["corners"]["students"][row["student"]]
            raw_descriptor = (math.log(artifacts["a2"]["nonembedding_counts"][row["student"]])
                              if scaler["descriptor"] == "log_parameters" else
                              student["corners"]["1"]["responses"][row["capability"]]["initial_loss"])
            assert pred["z"] == (raw_descriptor-scaler["mean"])/scaler["scale"]
            assert pred["absolute_error"] == abs(pred["I"] - row["measured_I"])
            assert not pred["is_exact_rectangle"]
        assert row["additive_rectangle_prediction"] == 0
        assert row["registered_noise"] == artifacts["corners"]["preregistration"]["readouts"][row["capability"]]["noise_on_I"]


@pytest.mark.parametrize("missing", ["coefficients", "standardizer", "lambda_"])
def test_missing_frozen_parameter_stops_rows_without_estimation(artifacts, missing):
    a2 = deepcopy(artifacts["a2"])
    param = next(p for p in a2["parameter_intervals"] if p["distribution"].startswith("training_probe:MBPP:"))
    del param["fits"]["F_int"]["fit"][missing]
    result = a7.check_predictions(a2, artifacts["plan"], artifacts["corners"])
    assert result["status"] == "INCOMPLETE"
    stopped = [r for r in result["rows"] if r["status"] == "STOPPED"]
    assert len(stopped) == 2
    assert all(r["capability"] == "code" and "no parameter was estimated" in r["reason"] for r in stopped)
    assert all("predictions" not in r for r in stopped)


def test_changed_structure_form_fails_before_prediction(artifacts):
    a2 = deepcopy(artifacts["a2"])
    a2["protocol"]["structures"]["F_int"] += "+extra"
    with pytest.raises(ValueError, match="form changed"):
        a7.check_predictions(a2, artifacts["plan"], artifacts["corners"])


def copy_inputs(tmp_path, artifacts):
    for name, path in a7.SOURCES.items():
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(artifacts[name]))


def run_cli(root):
    return subprocess.run([sys.executable, "-B", str(Path(a7.__file__).resolve()), "--root", str(root)],
                          capture_output=True, text=True, timeout=30)


def test_cli_writes_only_requested_reports_preserves_inputs_and_prints_tables(tmp_path, artifacts):
    copy_inputs(tmp_path, artifacts)
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp_path.rglob("*") if p.is_file()}
    result = run_cli(tmp_path)
    assert result.returncode == 0, result.stderr
    expected = {tmp_path / a7.OUT / "summary.json", tmp_path / a7.OUT / "summary.md", tmp_path / a7.DOC}
    assert {p for p in tmp_path.rglob("*") if p.is_file()} - set(before) == expected
    assert all(hashlib.sha256(p.read_bytes()).hexdigest() == digest for p, digest in before.items())
    summary = json.loads((tmp_path / a7.OUT / "summary.json").read_text())
    report = (tmp_path / a7.OUT / "summary.md").read_text()
    assert result.stdout == report == (tmp_path / a7.DOC).read_text()
    assert summary["status"] == "COMPLETE"
    assert all(f"## {i}." in report for i in range(1, 5))
    assert "DIAGNOSTIC" in report and "**Update 36" in report and "**Update 74" in report
    assert "failed to reject additivity" in report
    assert "F_log achieved" in report and "Additive rectangle 0" in report
    # Prohibit affirmative conclusions without putting them in the audit prose.
    for forbidden in ("additivity " + "holds", "the response is " + "additive",
                      "failure is attributed to " + "the form of the individual terms"):
        assert forbidden not in report


@pytest.mark.parametrize("failure", ["missing_file", "accounting", "missing_fit"])
def test_cli_nonzero_on_failure(tmp_path, artifacts, failure):
    if failure != "missing_file":
        changed = deepcopy(artifacts)
        if failure == "accounting":
            changed["counts"]["students"]["gemma3-4b"]["non_embedding_parameters"] += 1
        else:
            changed["a2"]["parameter_intervals"][0]["fits"]["F_int"]["fit"].pop("coefficients")
        copy_inputs(tmp_path, changed)
    result = run_cli(tmp_path)
    assert result.returncode != 0
    assert "FAILED" in result.stdout + result.stderr
    if failure == "missing_fit":
        data = json.loads((tmp_path / a7.OUT / "summary.json").read_text())
        assert data["checks"]["predictions"]["status"] == "INCOMPLETE"
        assert "STOPPED" in result.stdout


def test_audit_has_only_standard_library_imports_and_zero_new_work(audit):
    tree = ast.parse(Path(a7.__file__).read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module.split(".")[0])
    assert imported <= {"__future__", "argparse", "hashlib", "json", "math", "pathlib", "sys"}
    assert audit["device"] == "cpu"
    assert audit["new_measurements"] == audit["fits_performed"] == audit["new_candidates"] == 0
    assert audit["training_run"] is audit["resampling_performed"] is False
