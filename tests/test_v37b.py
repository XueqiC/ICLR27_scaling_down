"""Exposure accounting, anchored interaction, leakage guards and paired inference."""
from __future__ import annotations

import copy
import hashlib
import json
import shutil
import subprocess
import sys

import numpy as np
import pytest

from analysis import v37b_reuse_interaction as v


@pytest.fixture(scope="module")
def loaded():
    return v.load_data()


@pytest.fixture(scope="module")
def summary():
    return v.build_summary()


@pytest.mark.parametrize("tokens,pool,expected", [(0, 67027, 0), (67027, 67027, 1),
                                                 (1067738, 533869, 2), (126303, 65476, 126303/65476)])
def test_e_mapping_uses_processed_input_not_supervised_tokens(tokens, pool, expected):
    assert v.reuse_count(tokens, pool) == expected


@pytest.mark.parametrize("tokens,pool", [(-1, 2), (2, 0), (2, -1), (float("nan"), 2),
                                        (2, float("inf"))])
def test_invalid_e_mapping_rejected(tokens, pool):
    with pytest.raises(ValueError):
        v.reuse_count(tokens, pool)


def test_complete_roster_and_snapshot_exposures_are_log_milestones(loaded):
    rows, runs, logs, hashes = loaded
    assert (len(rows), len(runs), len(hashes)) == (36, 15, 66)
    for r in rows:
        log = logs[r["trajectory_id"]]
        milestone, = [p for p in log["trajectory"] if p["requested_token_milestones"] == [r["milestone"]]]
        e = r["exposure"]
        assert r["E"] == r["processed_tokens"]/log["unique_data_pool_tokens"]
        assert r["processed_tokens"] != r["milestone"]
        assert e["optimizer_steps_so_far"] == milestone["updates"]
        assert e["completion_tokens_seen"] == milestone["completion_tokens_seen"]
        assert e["interpolation"]["exact_logged_update"]
        assert e["schedule_fraction"] == milestone["updates"]/log["total_updates_planned"]
        assert e["optimizer_steps_so_far"] < log["optimizer_steps"]
    for relative, digest in hashes.items():
        assert hashlib.sha256((v.ROOT / relative).read_bytes()).hexdigest() == digest
    assert any(r["delta"][c] < 0 for r in rows for c in v.CAPS)
    assert any(r["delta"][c] > 0 for r in rows for c in v.CAPS)


def test_diagnostic_full_runs_exactly_verified_and_not_used_as_checkpoints(summary):
    diagnostic = summary["confound"]["gemma_full_run_diagnostic"]
    small, large = diagnostic["75"], diagnostic["600"]
    assert (small["optimizer_steps"], small["completion_tokens_seen"]) == (28, 39134)
    assert (large["optimizer_steps"], large["completion_tokens_seen"]) == (224, 316782)
    assert small["final_E"] == large["final_E"] == 2
    pair, = [p for p in summary["matched_E_pairs"] if p["student"] == "gemma3-1b"
             and p["seed"] == 0 and p["E_level"] == "matched_high"]
    assert pair["nominal_U75"]["optimizer_steps_so_far"] == 27
    assert pair["nominal_U600"]["optimizer_steps_so_far"] == 209
    assert pair["nominal_U75"]["completion_tokens_seen"] == 38177
    assert pair["nominal_U600"]["completion_tokens_seen"] == 295457


def test_adjacent_step_interpolation_not_full_run_proportions(loaded):
    log = next(iter(loaded[2].values()))
    a, b = log["loss_curve"][10:12]
    t = a["processed_tokens"] + .25*(b["processed_tokens"]-a["processed_tokens"])
    e = v.interpolate_exposure(log, t)
    assert e["optimizer_steps_so_far"] == pytest.approx(11.25)
    assert e["completion_tokens_seen"] == pytest.approx(a["completion_tokens_seen"] + .25*b["completion_tokens"])
    assert e["interpolation"]["bracket_steps"] == [11, 12]
    assert e["interpolation"]["fraction"] == pytest.approx(.25)
    assert not e["interpolation"]["exact_logged_update"]
    assert e["completion_tokens_seen"] != pytest.approx(log["completion_tokens_seen"]*t/log["processed_tokens"])
    origin = v.interpolate_exposure(log, 0)
    assert origin["optimizer_steps_so_far"] == origin["completion_tokens_seen"] == 0
    end = v.interpolate_exposure(log, log["processed_tokens"])
    assert end["optimizer_steps_so_far"] == log["optimizer_steps"]
    for invalid in (-1, log["processed_tokens"]+1, float("nan")):
        with pytest.raises(ValueError, match="extrapolation"):
            v.interpolate_exposure(log, invalid)


@pytest.mark.parametrize("damage", ["increment", "step", "total", "milestone", "schedule"])
def test_corrupt_train_log_fails_closed(loaded, damage):
    log = copy.deepcopy(next(iter(loaded[2].values())))
    if damage == "increment":
        log["loss_curve"][1]["completion_tokens_seen"] += 1
    elif damage == "step":
        log["loss_curve"][1]["step"] = 1
    elif damage == "total":
        log["optimizer_steps"] += 1
    elif damage == "milestone":
        log["trajectory"][0]["completion_tokens_seen"] += 1
    else:
        log["warmup_steps"] += 1
    with pytest.raises(ValueError):
        v.validate_train_log(log)


def test_snapshot_cannot_disagree_with_matching_log(tmp_path, loaded):
    for relative in loaded[3]:
        dst = tmp_path / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(v.ROOT / relative, dst)
    path = tmp_path / loaded[0][0]["row_id"]
    snap = json.loads(path.read_text())
    snap["completion_tokens_seen"] += 100
    path.write_text(json.dumps(snap))
    with pytest.raises(ValueError, match="Snapshot exposure disagrees"):
        v.load_data(tmp_path)


def test_exact_e_pairs_preserve_identity_and_report_collinearity(summary):
    assert len(summary["matched_E_pairs"]) == 12
    for p in summary["matched_E_pairs"]:
        a, b = p["aligned_U75"], p["aligned_U600"]
        assert a["E"] == pytest.approx(b["E"], abs=1e-14)
        assert a["E"] == pytest.approx(p["E"])
        assert b["processed_tokens"]/a["processed_tokens"] == pytest.approx(b["D_U"]/a["D_U"])
        nominal = p["nominal_U75"]
        assert a["processed_tokens"]-nominal["processed_tokens"] == pytest.approx(
            (p["E"]-nominal["E"])*a["D_U"])
        assert 7.8 < b["optimizer_steps_so_far"]/a["optimizer_steps_so_far"] < 8.2
        assert 7.8 < b["completion_tokens_seen"]/a["completion_tokens_seen"] < 8.5
    conf = summary["confound"]
    assert conf["max_abs_logT_minus_logE_minus_logD"] < 1e-12
    corr = np.array(conf["within_pair_centered_log_correlation"])
    assert corr[0, 1] == pytest.approx(1)
    assert np.min(corr) > .9999
    assert conf["centered_standardized_rank"] == 3
    assert conf["condition_number"] is None  # Singular, never JSON Infinity.
    # Gemma seed 1's low checkpoint undershoots the large pool's actual E;
    # alignment can move forward as well as backward within the saved grid.
    assert any(p["aligned_U75"]["processed_tokens"] > p["nominal_U75"]["processed_tokens"]
               for p in summary["matched_E_pairs"])


def test_signed_outcome_interpolation_uses_actual_large_e_and_no_extrapolation(loaded, summary):
    rows = {r["row_id"]: r for r in loaded[0]}
    pair, = [p for p in summary["matched_E_pairs"] if p["student"] == "Qwen3-4B"
             and p["seed"] == 0 and p["E_level"] == "matched_low"]
    small, large = rows[pair["U75_source"]], rows[pair["U600_source"]]
    assert small["E"] == 1
    for c in v.CAPS:
        assert pair["residual"][c] == pytest.approx(small["delta"][c]*large["E"]-large["delta"][c])
    changed = copy.deepcopy(loaded[0])
    for r in changed:
        if r["row_id"] == pair["U600_source"]:
            r["E"] = 1000
    with pytest.raises(ValueError, match="extrapolation"):
        v.matched_pairs(changed, loaded[2])


def test_one_coefficient_interaction_is_nested_and_anchored():
    inputs = [{"E": e, "D_U": d} for e in (0, 1, 2) for d in (65000, 530000)]
    base = v.design_matrix(inputs, "E_only")
    inter = v.design_matrix(inputs, "interaction")
    assert base.shape == (6, 2) and inter.shape == (6, 3)
    np.testing.assert_array_equal(inter[:, :2], base)
    np.testing.assert_array_equal(inter[:2], np.zeros((2, 3)))
    assert inter[:, 2] == pytest.approx([r["E"]*np.log(r["D_U"]/v.D_REF) for r in inputs])
    assert inter[4, 2] == pytest.approx(2*inter[2, 2])  # k linear in E.
    np.testing.assert_array_equal(base[2], base[3])
    assert inter[2, 2] != inter[3, 2]
    additive = v.design_matrix(inputs, "additive_diagnostic")
    assert np.any(additive[:2] != 0)  # Documented violation, not an anchored candidate.
    with pytest.raises(ValueError, match="only E and D_U"):
        v.design_matrix([{"E": 1, "D_U": 65000, "target_delta": 1}], "interaction")
    with pytest.raises(ValueError, match="Invalid"):
        v.design_matrix([{"E": 1, "D_U": 0}], "interaction")


@pytest.mark.parametrize("model", v.MODELS)
def test_exact_signed_response_recovery(loaded, model):
    rows = copy.deepcopy([r for r in loaded[0] if r["student"] == v.STUDENTS[0]])
    inputs = v.v37.prediction_inputs(rows)
    coefficient = np.array([-2, 1] + ([.05] if model != "E_only" else []))
    y = v.design_matrix(inputs, model) @ coefficient
    for r, target in zip(rows, y):
        r["delta"]["math"] = target
    fit = v.fit_response(rows, "math", model)
    assert fit["coefficients"] == pytest.approx(coefficient, abs=1e-10)
    assert v.predict_response(fit, inputs) == pytest.approx(y, abs=1e-10)
    assert min(y) < 0 < max(y)


def test_rank_deficiency_and_cross_student_pooling_rejected(loaded):
    with pytest.raises(ValueError, match="each student"):
        v.fit_response(loaded[0], "math", "interaction")
    with pytest.raises(ValueError, match="Rank deficient"):
        v.fit_response([loaded[0][0]]*3, "math", "interaction")


def test_folds_hold_whole_nominal_e_levels_across_seeds_and_pools(loaded):
    rows = loaded[0]
    folds = v.make_folds(rows)
    assert len(folds) == 8
    assert sorted(i for f in folds for i in f["test_indices"]) == list(range(36))
    for f in folds:
        train, test = [[rows[i] for i in f[k]] for k in ("train_indices", "test_indices")]
        assert {r["student"] for r in train+test} == {f["student"]}
        assert {v.e_level(r) for r in test} == {f["held_out_E_level"]}
        assert f["held_out_E_level"] not in {v.e_level(r) for r in train}
        assert {r["seed"] for r in test} == set(v.SEEDS)
        assert set(f["train_indices"]).isdisjoint(f["test_indices"])
        if f["held_out_E_level"] in v.LEVELS[:2]:
            assert {r["pool"] for r in test} == {75, 600}
        # Same trajectories at OTHER levels are deliberately allowed and disclosed.
        assert {r["trajectory_id"] for r in train} & {r["trajectory_id"] for r in test}
    altered = copy.deepcopy(rows)
    for r in altered:
        r["E"] += r["seed"] * 1000
        r["delta"] = {c: 1e6 for c in v.CAPS}
    assert v.make_folds(altered) == folds  # No outcome or actual-E-jitter split.
    with pytest.raises(ValueError, match="all four"):
        v.make_folds(rows[:1])


def test_held_e_level_outcomes_never_enter_interaction_fits(loaded):
    rows = copy.deepcopy(loaded[0])
    before = v.cross_validate(rows)
    fold = before["folds"][0]
    ids = set(fold["test_row_ids"])
    for r in rows:
        if r["row_id"] in ids:
            r["delta"] = {c: r["delta"][c]+10000 for c in v.CAPS}
    after = v.cross_validate(rows)
    assert before["folds"][0]["fits"] == after["folds"][0]["fits"]
    for left, right in zip(before["predictions"], after["predictions"]):
        if left["row_id"] in ids:
            assert left["predictions"] == right["predictions"]
            assert left["observed"] != right["observed"]


def test_paired_metrics_balance_pools_and_keep_n_three():
    records = []
    for seed in v.SEEDS:
        for pool, error, n in ((75, 2, 4), (600, 8, 2)):
            for _ in range(n):
                records.append({"seed": seed, "pool": pool, "observed": -10-seed,
                                "predictions": {"E_only": -10-seed+error,
                                                "interaction": -10-seed+error-1,
                                                "additive_diagnostic": -10-seed+error-.5}})
    m = v.prediction_metrics(records)
    assert m["mae_E_only"]["mean"] == 5
    assert m["mae_interaction"]["mean"] == 4
    assert m["gain_over_E_only"]["ci95"] == [1, 1]
    assert m["gain_over_additive"]["mean"] == .5
    assert m["gain_over_E_only"]["n_seeds"] == 3
    doubled = v.prediction_metrics(records+records)
    m.pop("n_snapshots")
    doubled.pop("n_snapshots")
    assert doubled == m


def synthetic_pairs(beta=0.2, flip_high=False):
    return [{"pair_id": f"s/{seed}/{level}", "student": "s", "seed": seed, "E_level": level,
             "exposure_predictors": {v.EXPOSURES[0]: -2, v.EXPOSURES[1]: -(i+1)},
             "residual": {"math": beta*-(i+1)*(-1 if flip_high and i == 1 else 1)}}
            for seed in v.SEEDS for i, level in enumerate(v.LEVELS[:2])]


def test_exposure_slope_recovers_signed_gap_out_of_level():
    pairs = synthetic_pairs()
    r = v.exposure_regression(pairs, "s", "math", v.EXPOSURES[1])
    assert r["full_data_slope_descriptive"] == pytest.approx(.2)
    assert r["cross_level_SSE_fraction_explained"] == pytest.approx(1)
    assert r["largely_explained"] and r["supported_gain_over_constant"]
    assert r["metrics"]["mae_exposure"]["mean"] == pytest.approx(0)
    assert all(ci["n_seeds"] == 3 for ci in r["corrected_residual_by_level"].values())
    for fold in r["folds"]:
        assert set(fold["train_pair_ids"]).isdisjoint(fold["test_pair_ids"])
        assert len(fold["train_pair_ids"]) == len(fold["test_pair_ids"]) == 3
    pairs[0]["residual"]["math"] += 100
    after = v.exposure_regression(pairs, "s", "math", v.EXPOSURES[1])
    assert after["folds"][0]["slope"] == r["folds"][0]["slope"]


def test_high_fit_fraction_or_sign_changing_gaps_do_not_prove_sufficiency():
    # High descriptive fit fraction, but a fixed gap cannot predict growth.
    r = v.exposure_regression(synthetic_pairs(), "s", "math", v.EXPOSURES[0])
    assert r["full_data_uncentered_R2_descriptive"] > .8
    assert not r["largely_explained"]
    flip = v.exposure_regression(synthetic_pairs(flip_high=True), "s", "math", v.EXPOSURES[1])
    assert flip["cross_level_SSE_fraction_explained"] < 0
    assert not flip["largely_explained"]


def test_prior_provenance_corrects_additive_premise(summary):
    prior = summary["v37_provenance"]
    assert prior["status"] == "verified local artifact"
    assert "b3*x*z+b4*x^2*z" in prior["actual_model"]
    assert "not an additive" in prior["note"]
    assert v.TOLERANCE_NAT == .05 and v.LARGELY_EXPLAINED_FRACTION == .5


def test_summary_decisions_use_primary_paired_ci_and_fixed_exposure_rule(summary):
    cv = summary["held_out_interaction"]
    assert len(cv["predictions"]) == 108
    assert len({(r["row_id"], r["capability"]) for r in cv["predictions"]}) == 108
    for s in v.STUDENTS:
        for c in v.CAPS:
            m = cv["metrics"][s][c]["matched_levels_primary"]
            d = summary["decisions"][s][c]
            assert d["interaction_gain_supported"] == (m["gain_over_E_only"]["ci95"][0] > 0)
            assert m["n_snapshots"] == 12
            for x in v.EXPOSURES:
                r = summary["residual_vs_exposure"][s][c][x]
                meets = r["cross_level_SSE_fraction_explained"] >= .5 and all(
                    v.v37.interval_within_tolerance(ci, .05) for ci in r["corrected_residual_by_level"].values())
                assert r["largely_explained"] == meets
                assert (x in d["exposure_models_meeting_rule"]) == meets
    assert summary["decisions"]["gemma3-1b"]["math"]["exposure_models_meeting_rule"] == [v.EXPOSURES[0]]


def test_cli_artifacts_reproducible_without_gpu_imports(tmp_path, summary):
    output, report = tmp_path / "summary.json", tmp_path / "report.md"
    code = ("import sys; from analysis import v37b_reuse_interaction as v; "
            "v.main(sys.argv[1:]); assert 'torch' not in sys.modules; "
            "assert 'transformers' not in sys.modules")
    subprocess.run([sys.executable, "-c", code, "--output", str(output), "--report", str(report)],
                   cwd=v.ROOT, check=True, capture_output=True, text=True)
    assert json.loads(output.read_text()) == summary
    assert report.read_text() == v.render_report(summary)
    assert json.loads((v.ROOT / "results/v37b-reuse-interaction/summary.json").read_text()) == summary
    assert (v.ROOT / "docs/REUSE_INTERACTION.md").read_text() == report.read_text()
