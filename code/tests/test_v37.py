"""Actual token accounting, whole-group holdouts, signed prediction and paired seeds."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

from analysis import v37_reuse_sufficiency as v37


@pytest.fixture(scope="module")
def loaded():
    return v37.load_trajectories()


@pytest.fixture(scope="module")
def summary():
    return v37.build_summary()


@pytest.mark.parametrize("tokens,pool,expected", [(0, 67027, 0), (67027, 67027, 1),
                                                 (1000000, 500000, 2), (65476, 65476, 1)])
def test_reuse_mapping(tokens, pool, expected):
    assert v37.reuse_count(tokens, pool) == expected


@pytest.mark.parametrize("tokens,pool", [(-1, 1), (1, 0), (1, -1), (float("nan"), 1),
                                        (1, float("inf"))])
def test_reuse_mapping_rejects_invalid_accounting(tokens, pool):
    with pytest.raises(ValueError):
        v37.reuse_count(tokens, pool)


def test_actual_tokens_and_per_run_log_not_nominal_milestones(loaded):
    rows, runs, hashes = loaded
    assert len(rows) == 36 and len(runs) == 15 and len(hashes) == 66
    assert all(r["processed_tokens"] > 0 for r in rows)
    assert len({r["row_id"] for r in rows}) == len(rows)
    logs = {r["trajectory_id"]: r for r in runs}
    for r in rows:
        raw_log = json.loads((v37.ROOT / logs[r["trajectory_id"]]["train_log"]).read_text())
        assert r["D_U"] == raw_log["unique_data_pool_tokens"]
        assert r["E"] == r["processed_tokens"] / raw_log["unique_data_pool_tokens"]
        assert r["E"] != r["milestone"] / raw_log["unique_data_pool_tokens"]
        assert "/trajectory/" in r["row_id"]
    overshoot, = [r for r in rows if r["student"] == "gemma3-1b" and r["variant"] == "uxseenE"
                  and r["seed"] == 0 and r["milestone"] == 62775]
    assert overshoot["E"] == 1 and overshoot["processed_tokens"] == 67027
    assert {r["D_U"] for r in runs} == {67027, 533869, 65476, 521533}
    assert any(v < 0 for r in rows for v in r["delta"].values())
    assert any(v > 0 for r in rows for v in r["delta"].values())
    for relative, digest in hashes.items():
        assert hashlib.sha256((v37.ROOT / relative).read_bytes()).hexdigest() == digest


@pytest.fixture
def input_tree(tmp_path, loaded):
    for relative in loaded[2]:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(v37.ROOT / relative, target)
    return tmp_path


def test_missing_checkpoint_does_not_fall_back_to_nearest(input_tree):
    path = input_tree / "results/v12-distill/gemma3-1b/gpt-5.6-luna_full_75_uxseenE/trajectory/update-00000027"
    shutil.rmtree(path)
    with pytest.raises(ValueError, match="Missing baseline or requested milestones"):
        v37.load_trajectories(input_tree)


def test_pool_accounting_mismatch_is_not_silently_replaced_by_constant(input_tree):
    path = input_tree / "results/v12-distill/gemma3-1b/gpt-5.6-luna_full_75_uxseenE_seed1/train_log.json"
    data = json.loads(path.read_text())
    data["unique_data_pool_tokens"] = 99999
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="D_U disagrees with train_log"):
        v37.load_trajectories(input_tree)


def test_signed_delta_is_validated_against_loss_endpoint(input_tree):
    path = input_tree / "results/v12-distill/Qwen3-4B/gpt-5.6-luna_full_75_uxseen/trajectory/update-00000014/eval.json"
    data = json.loads(path.read_text())
    data["delta"]["math"] = abs(data["delta"]["math"])
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="post minus dense"):
        v37.load_trajectories(input_tree)


@pytest.mark.parametrize("scheme,n_folds", [("student_pool", 4), ("student", 2)])
def test_folds_hold_every_seed_checkpoint_and_run_variant(loaded, scheme, n_folds):
    rows = loaded[0]
    folds = v37.make_folds(rows, scheme)
    assert len(folds) == n_folds
    assert sorted(i for f in folds for i in f["test_indices"]) == list(range(36))
    for f in folds:
        train = [rows[i] for i in f["train_indices"]]
        test = [rows[i] for i in f["test_indices"]]
        assert set(f["train_indices"]).isdisjoint(f["test_indices"])
        assert len(train) + len(test) == len(rows)
        assert {r["trajectory_id"] for r in train}.isdisjoint(r["trajectory_id"] for r in test)
        assert {r["seed"] for r in test} == {0, 1, 2}
        if scheme == "student":
            assert {r["student"] for r in train}.isdisjoint(r["student"] for r in test)
            assert {r["pool"] for r in test} == {75, 600}
        else:
            assert {(r["student"], r["pool"]) for r in test} == {tuple(f["held_out"])}
        if f["held_out"] == ["gemma3-1b", 75]:
            assert {r["variant"] for r in test} == {"uxseen", "uxseenE"}
            assert len(test) == 12


def test_fold_rejects_bad_scheme_or_empty_training(loaded):
    with pytest.raises(ValueError, match="Unknown"):
        v37.make_folds(loaded[0], "checkpoint")
    with pytest.raises(ValueError, match="nonempty"):
        v37.make_folds([loaded[0][0]], "student")


def test_checkpoint_counts_do_not_overweight_a_student_pool(loaded):
    rows = loaded[0]
    w = v37.balanced_weights(rows)
    assert w.sum() == pytest.approx(1)
    for student in v37.STUDENTS:
        for pool in v37.POOLS:
            assert sum(a for r, a in zip(rows, w) if (r["student"], r["pool"]) == (student, pool)) == pytest.approx(.25)
            for seed in v37.SEEDS:
                assert sum(a for r, a in zip(rows, w) if (r["student"], r["pool"], r["seed"]) == (
                    student, pool, seed)) == pytest.approx(1/12)


@pytest.mark.parametrize("with_volume", [False, True])
def test_unconstrained_signed_prediction_and_zero_identity(loaded, with_volume):
    rows = copy.deepcopy(loaded[0])
    # A sign-changing known response surface: negative at low E, positive at high E.
    for r in rows:
        x, z = np.log1p(r["E"]), np.log(r["D_U"] / 100000)
        r["delta"]["math"] = -2*x + x*x + (.3*x*z - .1*x*x*z if with_volume else 0)
    fit = v37.fit_response(rows, "math", with_volume)
    predicted = v37.predict_response(fit, v37.prediction_inputs(rows))
    assert predicted == pytest.approx([r["delta"]["math"] for r in rows], abs=1e-10)
    assert min(predicted) < 0 < max(predicted)
    assert v37.predict_response(fit, [{"E": 0, "D_U": 70000}, {"E": 0, "D_U": 530000}]) == pytest.approx([0, 0])
    with pytest.raises(ValueError, match="only E and D_U"):
        v37.predict_response(fit, [{"E": 1, "D_U": 70000, "target_offset": -2}])


def test_e_only_is_independent_of_volume_and_models_are_nested():
    inputs = [{"E": 1., "D_U": 65000}, {"E": 1., "D_U": 530000}]
    a = v37.design_matrix(inputs, False)
    b = v37.design_matrix(inputs, True)
    np.testing.assert_array_equal(a[0], a[1])
    np.testing.assert_array_equal(a, b[:, :a.shape[1]])
    assert not np.array_equal(b[0], b[1])


@pytest.mark.parametrize("scheme", v37.SCHEMES)
def test_target_outcomes_cannot_affect_their_held_out_predictions(loaded, scheme):
    rows = copy.deepcopy(loaded[0])
    before = v37.cross_validate(rows, scheme)
    held_out = before["folds"][0]
    ids = set(held_out["test_row_ids"])
    for r in rows:
        if r["row_id"] in ids:
            for cap in v37.CAPS:
                r["delta"][cap] += 1000
    after = v37.cross_validate(rows, scheme)
    for cap in v37.CAPS:
        first = [r for r in before["capabilities"][cap]["predictions"] if r["row_id"] in ids]
        second = [r for r in after["capabilities"][cap]["predictions"] if r["row_id"] in ids]
        assert [r["predictions"] for r in first] == [r["predictions"] for r in second]
        assert first[0]["observed"] != second[0]["observed"]
        assert before["folds"][0]["capabilities"][cap]["fits"] == after["folds"][0]["capabilities"][cap]["fits"]


def test_checkpoint_contrasts_keep_pairing_and_n_three():
    values = {0: [0., 10.], 1: [10., 20.], 2: [20., 30.]}
    a = v37.paired_checkpoint_summaries(values)
    # All high-minus-low contrasts are 10 despite large marginal checkpoint noise.
    assert a["high_minus_low"]["ci95"] == [10, 10]
    assert a["mean_over_checkpoints"]["mean"] == 15
    half = 4.302652729911275 * 10 / np.sqrt(3)
    assert a["mean_over_checkpoints"]["ci95"] == pytest.approx([15-half, 15+half])
    assert all(v["n_seeds"] == 3 for v in a.values())
    # Crossing signs must not vanish from the magnitude diagnostic.
    b = v37.paired_checkpoint_summaries({0: [-1, 1], 1: [-1, 1], 2: [-1, 1]})
    assert b["mean_over_checkpoints"]["mean"] == 0
    assert b["mean_absolute_over_checkpoints"]["mean"] == 1
    with pytest.raises(ValueError, match="both paired checkpoints"):
        v37.paired_checkpoint_summaries({0: [1]})


def test_single_seed_has_no_fabricated_confidence_interval():
    a = v37.paired_interval({0: .01})
    assert a["ci95"] is None
    assert not v37.interval_within_tolerance(a)


def test_prediction_mae_and_bootstrap_preserve_pairing_and_equal_cells():
    records = []
    for student in ("a", "b"):
        for seed in range(3):
            # Different pools have unequal checkpoint counts but equal weight.
            for pool, error, n in ((75, 2., 4), (600, 8., 2)):
                for _ in range(n):
                    records.append({"student": student, "seed": seed, "pool": pool,
                                    "observed": -10-seed,
                                    "predictions": {"E_only": -10-seed+error, "E_log_D": -10-seed+error-1}})
    a = v37.summarize_predictions(records)
    assert a["mae_E_only"] == 5  # Not the snapshot-weighted value 4.
    assert a["mae_E_log_D"] == 4
    assert a["mae_reduction_with_log_D"] == 1
    assert a["paired_seed_bootstrap_ci95"]["mae_reduction_with_log_D"] == [1, 1]
    assert a["n_student_seed_blocks"] == 6 and a["n_bootstrap_resamples"] == 729
    assert a["per_student_seed_t_intervals"]["a"]["mae_reduction_with_log_D"]["n_seeds"] == 3
    # Duplicating all checkpoints leaves paired intervals and scores unchanged.
    b = v37.summarize_predictions(records + records)
    a.pop("n_snapshots")
    b.pop("n_snapshots")
    assert a == b


def test_matched_residuals_reproduce_prior_values_and_pair_high_low(summary):
    expected = {"gemma3-1b": {"math": [-.083, -.089], "code": [-.096, -.127], "qa": [.250, -.619]},
                "Qwen3-4B": {"math": [-.009, -.043], "code": [.015, -.134], "qa": [.531, -.773]}}
    for student in v37.STUDENTS:
        m = summary["matched_E"][student]
        for cap in v37.CAPS:
            low, high = [m["nominal_matched_E"][p][cap] for p in ("low", "high")]
            assert [low["mean"], high["mean"]] == pytest.approx(expected[student][cap], abs=.0006)
            assert low["n_seeds"] == high["n_seeds"] == 3
            changes = m["paired_across_checkpoints"][cap]["high_minus_low"]
            assert changes["seed_values"] == pytest.approx({s: high["seed_values"][s]-low["seed_values"][s]
                                                           for s in low["seed_values"]})
        assert max(p["relative_E_mismatch"] for p in m["pairs"]) > 0


def test_interpolation_is_actual_e_diagnostic_not_nominal_relabelling(summary):
    # Qwen low checkpoint is exactly E=1 in every seed. Interpolate from the zero
    # baseline to actual large-pool E<1: delta_small * E_large - delta_large.
    m = summary["matched_E"]["Qwen3-4B"]
    lookup = {r["row_id"]: r for r in summary["data"]["snapshots"]}
    for pair in m["pairs"]:
        if pair["pair"] != "low":
            continue
        small, large = lookup[pair["U75_source"]], lookup[pair["U600_source"]]
        assert small["E"] == 1
        for cap in v37.CAPS:
            actual = m["linear_interpolation_sensitivity"]["low"][cap]["seed_values"][str(pair["seed"])]
            assert actual == pytest.approx(small["delta"][cap]*large["E"]-large["delta"][cap])


def test_sufficiency_requires_both_equivalence_and_no_prediction_gain(summary):
    matched, cv = copy.deepcopy(summary["matched_E"]), copy.deepcopy(summary["held_out"])
    for student in v37.STUDENTS:
        for pair in ("low", "high"):
            for cap in v37.CAPS:
                matched[student]["nominal_matched_E"][pair][cap]["ci95"] = [-.02, .02]
    for scheme in v37.SCHEMES:
        for cap in v37.CAPS:
            cv[scheme]["capabilities"][cap]["metrics"]["mae_reduction_with_log_D"] = -.01
    assert all(r["choose_E_only"] for r in v37.decide_sufficiency(matched, cv).values())
    # Includes zero, but is wider than the pre-specified tolerance.
    matched["gemma3-1b"]["nominal_matched_E"]["low"]["math"]["ci95"] = [-.06, .02]
    # Even a small held-out volume gain fails the point-error rule.
    cv["student"]["capabilities"]["code"]["metrics"]["mae_reduction_with_log_D"] = .001
    decision = v37.decide_sufficiency(matched, cv)
    assert not decision["math"]["choose_E_only"]
    assert not decision["code"]["choose_E_only"]
    assert decision["qa"]["choose_E_only"]
    assert v37.TOLERANCE_NAT == .05
    assert not any(r["choose_E_only"] for r in summary["decisions"].values())


def test_summary_oof_metrics_have_all_observations_and_identifiable_fits(summary):
    assert summary["protocol"]["target_calibration_outcomes_used_for_prediction"] == 0
    for scheme in v37.SCHEMES:
        cv = summary["held_out"][scheme]
        for cap in v37.CAPS:
            records = cv["capabilities"][cap]["predictions"]
            assert len(records) == len({r["row_id"] for r in records}) == 36
            assert v37.summarize_predictions(records) == cv["capabilities"][cap]["metrics"]
        for fold in cv["folds"]:
            assert set(fold["test_row_ids"]).isdisjoint(fold["train_row_ids"])
            for cap in v37.CAPS:
                assert fold["capabilities"][cap]["fits"]["E_only"]["rank"] == 2
                assert fold["capabilities"][cap]["fits"]["E_log_D"]["rank"] == 4
        assert summary["basis_sensitivity"][scheme]["status"] == "unavailable"
        assert "rank deficient" in summary["basis_sensitivity"][scheme]["reason"]


def test_cli_reproduces_artifacts_cpu_only_without_torch(tmp_path, summary):
    output, report = tmp_path / "summary.json", tmp_path / "report.md"
    code = ("import sys; from analysis import v37_reuse_sufficiency as v; "
            "v.main(sys.argv[1:]); assert 'torch' not in sys.modules")
    subprocess.run([sys.executable, "-c", code, "--output", str(output), "--report", str(report)],
                   cwd=v37.ROOT, check=True, capture_output=True, text=True)
    assert json.loads(output.read_text()) == summary
    assert report.read_text() == v37.render_report(summary)
    saved = v37.ROOT / "results/v37-reuse-sufficiency/summary.json"
    assert json.loads(saved.read_text()) == summary
    assert (v37.ROOT / "paper/docs/REUSE_SUFFICIENCY.md").read_text() == report.read_text()
