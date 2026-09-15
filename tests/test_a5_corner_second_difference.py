"""CPU-only mathematical, provenance, decision, and CLI failure checks for A5."""
from copy import deepcopy
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import pytest

from analysis import a5_corner_second_difference as a5


@pytest.mark.parametrize("h", [
    lambda e: Fraction(7),
    lambda e: e,
    lambda e: e * e,
    lambda e: Fraction(1, 1) / (1 + e),
    lambda e: Fraction(e ** 0.37),  # A freely chosen, noninteger exponent.
    lambda e: Fraction(math.log(e)),
])
def test_additive_cancellation_is_exact_for_arbitrary_reuse_curvature(h):
    # Rational arithmetic isolates the algebra from floating-point rounding.
    # h may be irrational: reuse the same exact representation at each budget.
    coords = {1: (3, 2), 2: (3, 5), 3: (11, 2), 4: (11, 5)}
    for coefficient in (Fraction(0), Fraction(13, 7), Fraction(-9, 2)):
        responses = {i: Fraction(t ** 3 + 2 * t, 7) + coefficient * h(e)
                     for i, (t, e) in coords.items()}
        assert a5.second_difference(responses) == 0


def test_sign_convention_and_interaction():
    assert a5.second_difference({1: 2, 2: 3, 3: 11, 4: 5}) == 7
    for corner, sign in ((1, -1), (2, 1), (3, 1), (4, -1)):
        assert a5.second_difference({i: int(i == corner) for i in range(1, 5)}) == sign
    assert a5.second_difference({1: 3 * 2, 2: 3 * 5, 3: 11 * 2, 4: 11 * 5}) == -24
    with pytest.raises(ValueError, match="four corner"):
        a5.second_difference({1: 0, 2: 0, 3: 0})


def test_constants_come_from_amendment_not_original_pooled_table():
    document = (a5.ROOT / a5.PREREG).read_text()
    parsed = a5.parse_prereg(document)
    assert [(r["noise_on_I"], r["predicted_disagreement"]) for r in parsed["readouts"].values()] == [
        (0.1608, 1.5546), (0.0088, 0.0215), (0.0254, 0.0144)]
    changed = document.replace("0.1608", "0.1234").replace("1.5546", "1.2345")
    row = a5.parse_prereg(changed)["readouts"]["qa"]
    assert row["noise_on_I"] == 0.1234
    assert row["twice_noise"] == 0.2468
    assert row["predicted_disagreement"] == 1.2345
    assert [(r["noise_on_I"], r["matched_seed_pairs"]) for r in
            parsed["fresh_sample_readouts"].values()] == [(0.3823, 2), (0.1440, 2), (0.8360, 2)]
    # Changing the obsolete capability table must have no effect.
    assert a5.parse_prereg(document.replace("0.1856", "999999")) == parsed
    for number in ("0.1608", "1.5546", "0.0088", "0.0215", "0.0254", "0.0144",
                   "0.3823", "0.1440", "0.8360"):
        assert number not in Path(a5.__file__).read_text()
    with pytest.raises(ValueError, match="amendment"):
        a5.parse_prereg(document.split("## Amendment")[0])
    with pytest.raises(ValueError, match="Missing distribution"):
        a5.parse_prereg(document.replace("| MBPP probe |", "| removed |"))


@pytest.mark.parametrize("values,status", [
    ((0.25, 0.5), "REJECTED"),
    ((-0.25, -0.5), "REJECTED"),
    ((0.25, -0.5), "SIZE_DEPENDENT_INTERACTION"),
    ((0.125, -0.125), "SIZE_DEPENDENT_INTERACTION"),
    ((0.25, 0.25), "SURVIVES"),  # Rejection is strict: exactly 2*noise is inside.
    ((0.125, 0), "SURVIVES"),
    ((0.5, 0.125), "UNRESOLVED"),
    ((0.5, 0), "UNRESOLVED"),
])
def test_registered_decision_rules_without_student_averaging(values, status):
    # First two rows use the smaller noise so both values exceed its band.
    noise = 0.0625 if values in ((0.25, 0.5), (-0.25, -0.5)) else 0.125
    result = a5.registered_decision(dict(zip(a5.STUDENTS, values)), noise)
    assert result["status"] == status


def test_decision_needs_both_students_and_mbpp_never_supports_additivity():
    for cap in a5.READOUTS:
        result = a5.registered_decision({"gemma3-1b": 200}, 0.1, underpowered=cap == "code")
        assert result["status"] == "PENDING"
        assert "needs both students" in result["explanation"]
    result = a5.registered_decision(dict.fromkeys(a5.STUDENTS, 0), 0.1, underpowered=True)
    assert result["status"] == "UNDERPOWERED"
    assert "not support for additivity" in result["explanation"]
    result = a5.registered_decision(dict.fromkeys(a5.STUDENTS, 1), 0.1, underpowered=True)
    assert result["status"] == "REJECTED"
    assert "underpowered" in result["explanation"]


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


@pytest.fixture
def experiment(tmp_path):
    """Synthetic responses at the real registered geometry, with distinct baselines.

    Only A3's frozen geometry/slope data and A1 probe metadata are reused. No
    measured corner response enters the fixture or any expected statistic.
    """
    stored = json.loads((a5.ROOT / a5.PLAN).read_text())
    plan = {k: deepcopy(stored[k]) for k in (
        "status", "launchable", "design", "pools", "protocol", "students", "residual_contamination")}
    for student in a5.STUDENTS:
        specs = a5.corner_specs(plan, student)
        # Copy metadata only, for exact equality with A3's distribution hashes.
        source = json.loads((a5.ROOT / specs[2]["baseline_source"]).read_text())
        metadata = {k: source[k] for k in (
            "resolved_student", "probe_source", "probe_seed", "probe_half", "n_probe_requested",
            "loss_definition", "measurement_benchmarks", "measurement_samples", "measurement_tokens")}
        for number, spec in specs.items():
            pool = plan["pools"][spec["pool"]]
            initial = {"qa": 10. + ord(spec["pool"]), "math": 20. + ord(spec["pool"]),
                       "code": 30. + ord(spec["pool"])}
            payload = dict(metadata, student=student, trajectory_run_id=spec["run_id"],
                           updates=0, completion_tokens_seen=0, processed_tokens=0,
                           data_pool_sha256=pool["data_pool_sha256"], data_sampling_seed=pool["data_seed"],
                           n_per_domain=pool["n_per_domain"], training_seed=0, post_training=initial)
            payload.update({k: plan["protocol"][k] for k in (
                "teacher", "recipe", "training_mode", "schedule_tokens", "learning_rate")})
            write_json(tmp_path / spec["baseline_source"], payload)
            delta = {1: 0.5, 2: 0.25, 3: 2., 4: 0.75}[number]
            if student == "gemma3-4b":
                delta *= -2
            payload.update(updates=spec["update"], completion_tokens_seen=spec["T_supervised"],
                           processed_tokens=spec["processed_tokens"],
                           post_training={cap: loss + delta for cap, loss in initial.items()},
                           delta=dict.fromkeys(a5.READOUTS, delta))
            write_json(tmp_path / spec["source"], payload)
        old = plan["students"][student]["already_measured_corner_2"]
        old["sha256"] = hashlib.sha256((tmp_path / old["source"]).read_bytes()).hexdigest()
    write_json(tmp_path / a5.PLAN, plan)
    prereg = tmp_path / a5.PREREG
    prereg.parent.mkdir(parents=True, exist_ok=True)
    prereg.write_text((a5.ROOT / a5.PREREG).read_text())
    return tmp_path, plan


def test_complete_run_uses_each_own_initial_and_frozen_sensitivities(experiment):
    root, plan = experiment
    result = a5.build(root)
    assert result["status"] == "COMPLETE"
    for student, expected in zip(a5.STUDENTS, (1., -2.)):
        entry = result["students"][student]
        assert entry["status"] == "COMPLETE"
        assert entry["fresh_sample_check"]["status"] == "PENDING"
        assert entry["fresh_sample_check"]["missing_corners"] == [1, 2, 3, 4]
        for cap, row in entry["readouts"].items():
            assert row["I"] == expected
            assert len({r["initial_loss"] for r in row["raw_responses"].values()}) == 3
            assert row["interval"] == [expected - 2 * row["noise_on_I"], expected + 2 * row["noise_on_I"]]
            contamination = row["contamination"]
            assert contamination["total_contamination_nats"] is None
            assert contamination["abs_I_vs_total_contamination"] is None
            assert "do not sum them" in contamination["limitation"]
            channel = next(c for c in plan["residual_contamination"]["channels"]
                           if c["student"] == student and c["capability"] == cap)
            for edge, sensitivity in contamination["residuals"].items():
                old = channel["residuals"][edge]
                assert sensitivity["max_absolute_sensitivity_nats"] == old["max_absolute_sensitivity_nats"]
                assert sensitivity["signed_range_nats"] == [-old["signed_sensitivity_range_nats"][1],
                                                            -old["signed_sensitivity_range_nats"][0]]
                assert sensitivity["sign_for_registered_I"] == -old["second_difference_sign"]
        for edge in a5.EDGES:
            assert Fraction(entry["residual_mismatches"][edge]["fraction_exact"]) == Fraction(
                plan["design"]["mismatches"][edge]["fraction_exact"])
    assert all(d["status"] == "SIZE_DEPENDENT_INTERACTION" for d in result["decisions"].values())
    assert all(result[k] is False for k in ("model_weights_loaded", "training_run", "new_evaluation_run"))
    assert result["fits_performed"] == 0
    markdown = a5.render(result)
    assert "underpowered" in markdown and "Unimplementable total" in markdown
    assert "Four raw responses" in markdown


def test_build_uses_prereg_values_at_runtime(experiment):
    root, _ = experiment
    path = root / a5.PREREG
    path.write_text(path.read_text().replace("0.1608", "0.6234").replace("1.5546", "9.8765"))
    row = a5.build(root)["students"]["gemma3-1b"]["readouts"]["qa"]
    assert row["noise_on_I"] == 0.6234
    assert row["predicted_disagreement"] == 9.8765
    assert row["exceeds_twice_noise"] is False
    assert row["interval"] == [1. - 2 * 0.6234, 1. + 2 * 0.6234]


def cli(root, *args):
    return subprocess.run([sys.executable, "-B", "-O", str(Path(a5.__file__)),
                           "--root", str(root), *args], capture_output=True, text=True)


def test_missing_default_pending_but_explicit_missing_fails_with_nonzero_exit(experiment):
    root, plan = experiment
    missing = a5.corner_specs(plan, "gemma3-4b")[3]["source"]
    (root / missing).unlink()
    result = cli(root)
    assert result.returncode == 0, result.stderr
    assert "PENDING" in result.stdout
    summary = json.loads((root / a5.OUT / "summary.json").read_text())
    assert summary["status"] == "PENDING"
    assert summary["students"]["gemma3-1b"]["readouts"]["qa"]["I"] == 1
    assert summary["students"]["gemma3-4b"]["missing_checkpoints"] == [missing]
    assert not summary["students"]["gemma3-4b"]["readouts"]
    assert all(d["status"] == "PENDING" for d in summary["decisions"].values())
    assert "needs both students and cannot be reached yet" in (root / a5.REPORT).read_text()
    assert (root / a5.REPORT).read_text() == (root / a5.OUT / "summary.md").read_text()
    result = cli(root, "--students", "gemma3-4b")
    assert result.returncode != 0
    assert "missing checkpoint" in result.stderr
    assert missing in result.stderr
    summary = json.loads((root / a5.OUT / "summary.json").read_text())
    assert summary["status"] == "FAILED"
    assert summary["students"]["gemma3-4b"]["status"] == "PENDING"


def test_explicit_complete_student_succeeds_but_no_joint_decision(experiment):
    root, _ = experiment
    result = cli(root, "--students", "gemma3-1b")
    assert result.returncode == 0, result.stderr
    summary = json.loads((root / a5.OUT / "summary.json").read_text())
    assert summary["students"]["gemma3-1b"]["status"] == "COMPLETE"
    assert summary["students"]["gemma3-4b"]["status"] == "NOT_REQUESTED"
    assert summary["status"] == "PENDING"


@pytest.mark.parametrize("key,new_value", [
    ("updates", 37), ("completion_tokens_seen", 50559), ("processed_tokens", 170152),
    ("data_pool_sha256", "wrong pool"), ("student", "gemma3-4b"),
    ("probe_seed", 27), ("delta", {"qa": 99, "math": 99, "code": 99}),
    ("post_training", {"qa": float("nan"), "math": 1., "code": 1.}),
])
def test_present_but_invalid_checkpoint_hard_fails_even_in_default_mode(experiment, key, new_value):
    root, plan = experiment
    path = root / a5.corner_specs(plan, "gemma3-1b")[1]["source"]
    payload = json.loads(path.read_text())
    payload[key] = new_value
    write_json(path, payload)
    result = cli(root)
    assert result.returncode != 0
    assert "FAILED:" in result.stderr
    summary = json.loads((root / a5.OUT / "summary.json").read_text())
    assert summary["status"] == "FAILED"
    assert summary["students"]["gemma3-1b"]["status"] == "FAILED"
    assert "corner 1" in summary["failures"][0]
    assert "gemma3-1b corner 1" in (root / a5.REPORT).read_text()


def test_neighbour_and_endpoint_are_never_substituted(experiment):
    root, plan = experiment
    spec = a5.corner_specs(plan, "gemma3-1b")[1]
    path = root / spec["source"]
    payload = json.loads(path.read_text())
    for substitute in (path.parent.parent / "update-00000037/eval.json", path.parents[2] / "eval.json"):
        write_json(substitute, payload)
    path.unlink()
    result = cli(root, "--students", "gemma3-1b")
    assert result.returncode != 0
    assert spec["source"] in result.stderr


def test_missing_own_initial_is_not_replaced_by_embedded_dense(experiment):
    root, plan = experiment
    spec = a5.corner_specs(plan, "gemma3-1b")[3]
    (root / spec["baseline_source"]).unlink()
    path = root / spec["source"]
    payload = json.loads(path.read_text())
    payload["dense"] = {"qa": 1., "code": 1., "math": 1.}
    write_json(path, payload)
    result = cli(root, "--students", "gemma3-1b")
    assert result.returncode != 0
    assert spec["baseline_source"] in result.stderr


def test_bad_json_and_missing_prereg_fail_nonzero(experiment):
    root, plan = experiment
    path = root / a5.corner_specs(plan, "gemma3-1b")[3]["source"]
    path.write_text('{"partial":')
    assert cli(root).returncode != 0
    (root / a5.PREREG).unlink()
    assert cli(root).returncode != 0
    assert "FAILED" in (root / a5.REPORT).read_text()


def test_pool_a_checksum_discrepancy_is_reported(experiment):
    root, plan = experiment
    path = root / a5.corner_specs(plan, "gemma3-1b")[2]["source"]
    path.write_text(path.read_text() + "\n")
    result = cli(root)
    assert result.returncode != 0
    assert "checksum discrepancy" in result.stderr


def test_mutable_inputs_fail_audit(tmp_path):
    path = tmp_path / "input.json"
    write_json(path, {"value": 1})
    inputs = a5.Inputs(tmp_path)
    inputs.read("input.json")
    write_json(path, {"value": 2})
    with pytest.raises(ValueError, match="Input changed"):
        inputs.verify()


def test_fresh_endpoint_is_never_substituted(experiment):
    root, plan = experiment
    spec = a5.corner_specs(plan, "gemma3-1b")[2]
    write_json(root / "results/v99-scope/a5-corners-gemma3-1b-poolA/fixture/update-00000152.json",
               {"trajectory_run_id": spec["run_id"], "updates": 152, "losses": {"2wiki_new": 42}})
    result = a5.build(root)
    assert result["students"]["gemma3-1b"]["fresh_sample_check"]["status"] == "PENDING"
    assert all("update-00000152" not in p for p in result["source_sha256"])


def write_scope_corners(root, plan):
    """Independent corner mapping and synthetic fresh deltas, unlike the probes."""
    paths = {}
    deltas = {"2wiki_new": (-1., 0.5, -0.25, 1.5),
              "triviaqa": (0.5, -0.25, 1.5, -1.), "musique": (2., 3., 11., 5.)}
    for student, scale in zip(a5.STUDENTS, (1, -2)):
        specs = a5.corner_specs(plan, student)
        for number, (pool, update) in enumerate((("B", 36), ("A", 38), ("C", 74), ("B", 74)), 1):
            spec = specs[number]
            path = root / (f"results/v99-scope/a5-corners-{student}-pool{pool}"
                           f"/synthetic-run-hash/update-{update:08d}.json")
            write_json(path, dict(status="complete", student=student,
                                 trajectory_run_id=spec["run_id"], updates=update,
                                 actual_supervised_tokens=spec["T_supervised"],
                                 delta_from_update_0={key: scale * values[number - 1]
                                                      for key, values in deltas.items()}))
            paths[student, number] = path
    return paths


def test_secondary_matches_hand_computed_contrast_and_uses_amended_noise(experiment):
    root, plan = experiment
    primary = a5.build(root)
    paths = write_scope_corners(root, plan)
    result = a5.build(root)
    assert result["status"] == "COMPLETE"
    assert result["decisions"] == primary["decisions"]
    for student, scale in zip(a5.STUDENTS, (1, -2)):
        assert result["students"][student]["readouts"] == primary["students"][student]["readouts"]
        secondary = result["students"][student]["fresh_sample_check"]
        assert secondary["status"] == "COMPLETE"
        # Hand arithmetic: -.25 - 1.5 + 1 + .5 = -.25;
        # 1.5 + 1 - .5 - .25 = 1.75; 11 - 5 - 2 + 3 = 7.
        for key, expected, noise in (("2wiki_new", -0.25, 0.3823),
                                     ("triviaqa", 1.75, 0.1440), ("musique", 7., 0.8360)):
            row = secondary["readouts"][key]
            assert row["I"] == scale * expected
            assert row["role"] == "SECONDARY" and row["matched_seed_pairs"] == 2
            assert row["noise_on_I"] == noise and row["twice_noise"] == 2 * noise
            assert row["interval"] == [scale * expected - 2 * noise, scale * expected + 2 * noise]
        assert all(c["status"] == "VERIFIED" and
                   c["actual_supervised_tokens"] == c["predicted_supervised_tokens"]
                   for c in secondary["corners"].values())
    assert all(str(p.relative_to(root)) in result["source_sha256"] for p in paths.values())
    assert "Noise estimate is weak: only two matched seed pairs" in a5.render(result)
    prereg = root / a5.PREREG
    prereg.write_text(prereg.read_text().replace("0.3823", "0.1234"))
    changed = a5.build(root)
    row = changed["students"]["gemma3-1b"]["fresh_sample_check"]["readouts"]["2wiki_new"]
    assert row["noise_on_I"] == 0.1234 and row["twice_noise"] == 0.2468
    assert row["interval"] == [-0.25 - 0.2468, -0.25 + 0.2468]
    assert changed["decisions"] == primary["decisions"]


@pytest.mark.parametrize("corner", [1, 2, 3, 4])
def test_missing_scope_corner_only_leaves_secondary_pending(experiment, corner):
    root, plan = experiment
    paths = write_scope_corners(root, plan)
    complete = a5.build(root)
    paths["gemma3-1b", corner].unlink()
    run = cli(root, "--students", *a5.STUDENTS)
    assert run.returncode == 0, run.stderr
    result = json.loads((root / a5.OUT / "summary.json").read_text())
    assert result["status"] == "COMPLETE" and not result["failures"]
    assert result["decisions"] == complete["decisions"]
    for student, expected in zip(a5.STUDENTS, (1., -2.)):
        assert result["students"][student]["status"] == "COMPLETE"
        assert result["students"][student]["readouts"]["qa"]["I"] == expected
    secondary = result["students"]["gemma3-1b"]["fresh_sample_check"]
    assert secondary["status"] == "PENDING" and secondary["missing_corners"] == [corner]
    assert not secondary["readouts"]
    assert result["students"]["gemma3-4b"]["fresh_sample_check"]["status"] == "COMPLETE"
    assert "SECONDARY PENDING" in run.stdout
    assert "cannot be reached yet" not in run.stdout


def test_scope_budget_discrepancy_is_reported_without_changing_primary(experiment):
    root, plan = experiment
    paths = write_scope_corners(root, plan)
    complete = a5.build(root)
    path = paths["gemma3-1b", 2]
    payload = json.loads(path.read_text())
    expected = payload["actual_supervised_tokens"]
    payload["actual_supervised_tokens"] += 1
    write_json(path, payload)
    run = cli(root, "--students", *a5.STUDENTS)
    assert run.returncode != 0
    discrepancy = f"actual_supervised_tokens discrepancy: predicted {expected}, observed {expected + 1}"
    assert discrepancy in run.stderr and discrepancy in (root / a5.REPORT).read_text()
    result = json.loads((root / a5.OUT / "summary.json").read_text())
    assert result["status"] == "FAILED" and result["decisions"] == complete["decisions"]
    entry = result["students"]["gemma3-1b"]
    assert entry["status"] == "COMPLETE" and entry["readouts"]["qa"]["I"] == 1.
    assert entry["fresh_sample_check"]["status"] == "FAILED"
    assert not entry["fresh_sample_check"]["readouts"]
    assert not entry["fresh_sample_check"]["corners"]["2"]["responses"]
    assert "cannot be reached yet" not in run.stdout
