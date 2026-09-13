"""CPU-only synthetic checks for the fixed V95 matrix intervention protocol."""
from copy import deepcopy
from dataclasses import replace
from itertools import product
import math

import numpy as np
import pytest

from analysis import v95_matrix_intervention as v95


def point(run="a", t=100, du=100, delta=0.0, student="gemma3-270m", seed=41,
          u=66, cap="math", distribution="fixture", initial=2.0, n=100_000_000):
    return v95.Point(f"{student}/{run}/seed{seed}", student, u, seed, cap, distribution,
                     int(t), int(t * 3), t, du, initial + delta, initial, n)


def test_fixed_registry_contains_only_the_seven_requested_forms():
    assert v95.CANDIDATES == ("zero", "constant", "T_only", "reuse_only", "two_dimensional",
                               "student_log_parameters", "student_initial_loss")
    assert v95.BOOTSTRAPS == 5000
    assert v95.T_REF == v95.D_REF == 100_000
    with pytest.raises(ValueError, match="Unregistered"):
        v95.fit_candidate([point()], "saturation")


def test_i1_inclusive_halving_window_uses_actual_completion_counts():
    points = [point(t=t) for t in (0, 100, 169, 170, 200, 230, 231, 400)]
    # Processed counts are deliberately unrelated to completion ratios.
    points = [replace(p, processed_tokens=1000 + i) for i, p in enumerate(points)]
    pairs = v95.construct_pairs(points)
    expected = {(a.T, b.T) for a in points for b in points if a.T > 0 and 1.7 <= b.T / a.T <= 2.3}
    assert {(p.first.T, p.second.T) for p in pairs} == expected
    assert {(100, 170), (100, 230), (200, 400)} <= expected
    assert (100, 169) not in expected and (100, 231) not in expected
    assert all(p.target == "I1" and len(p.clusters) == 1 for p in pairs)


def test_i2_inclusive_symmetric_match_tolerance_and_same_seed_student_distribution():
    a = point("a", t=100, delta=0.7)
    valid = [point("b", t=t, du=300, u=198, delta=0.1) for t in (90, 91, 100, 110, 111)]
    invalid = [point("seed", u=198, du=300, seed=42),
               point("student", u=198, du=300, student="gemma3-1b"),
               point("distribution", u=198, du=300, distribution="other"),
               point("capability", u=198, du=300, cap="qa"),
               point("same_u", u=66, du=101), point("zero", t=0, u=198, du=300)]
    pairs = [p for p in v95.construct_pairs([a, *valid]) if p.target == "I2"]
    assert [p.second.T for p in sorted(pairs, key=lambda p: p.second.T)] == [91, 100, 110]
    assert all(p.first == a and p.actual == pytest.approx(-0.6) for p in pairs)
    for b in invalid:
        assert v95.construct_pairs([a, b]) == []
    # Both directions of the exact max/min = 1.10 boundary are accepted.
    assert len(v95.construct_pairs([point(t=110), point("b", t=100, u=198, du=300)])) == 1
    assert v95.construct_pairs([point(t=111), point("b", t=100, u=198, du=300)]) == []


def test_pair_orientation_all_pool_contrasts_and_no_cross_trajectory_budget_effects():
    points = [point(str(u), u=u, du=u, t=t, delta=u / 1000 - t / 100)
              for u in v95.POOLS for t in (0, 100, 200, 400, 800)]
    pairs = v95.construct_pairs(points)
    assert sum(p.target == "I1" for p in pairs) == 9
    assert sum(p.target == "I2" for p in pairs) == 12
    assert {(p.first.U, p.second.U) for p in pairs if p.target == "I2"} == {(66, 198), (66, 594), (198, 594)}
    assert all(p.first.T > 0 and p.second.T > 0 for p in pairs)
    assert all(p.first.trajectory == p.second.trajectory for p in pairs if p.target == "I1")


@pytest.mark.parametrize("candidate", v95.CANDIDATES)
def test_every_candidate_fits_delta_then_differences_own_predictions(candidate):
    points = [point(str(u), t=t, u=u, du=u, delta=.2 + .3 * math.log1p(t / 100) - .01 * u,
                    initial=2 + u / 100, n=1_000_000 * u)
              for u in (66, 198) for t in (0, 100, 200)]
    model = v95.fit_candidate(points, candidate)
    pairs = v95.construct_pairs(points)
    expected = [model.predict([p.second])[0] - model.predict([p.first])[0] for p in pairs]
    np.testing.assert_allclose(v95.predicted_intervention(model, pairs), expected)
    design = v95.features(points, candidate, model.standardizer)
    np.testing.assert_allclose(model.coefficients, np.linalg.lstsq(design, [p.delta for p in points], rcond=None)[0])
    if candidate in ("zero", "constant"):
        np.testing.assert_array_equal(expected, 0)
    if candidate == "constant":
        assert model.coefficients == pytest.approx([np.mean([p.delta for p in points])])
        assert model.coefficients[0] != 0


def test_intercept_cancels_and_near_matched_budget_predictions_use_both_actual_Ts():
    pair = v95.Pair("I2", point(t=100_000, du=50_000), point("b", t=109_000, du=150_000, u=198))
    model = v95.Fit("two_dimensional", [50, 3, -.4])
    expected = 3 * (np.log1p(1.09) - np.log1p(1)) - .4 * np.log(3)
    assert v95.predicted_intervention(model, [pair])[0] == pytest.approx(expected)
    model.coefficients[0] = -100
    assert v95.predicted_intervention(model, [pair])[0] == pytest.approx(expected)
    t_model = v95.Fit("T_only", [3])
    assert v95.predicted_intervention(t_model, [pair])[0] != 0


@pytest.mark.parametrize("candidate", v95.CANDIDATES[-2:])
def test_descriptor_standardization_uses_training_fold_only(candidate):
    train = [point("a", initial=1, n=10), point("b", initial=3, n=1000, student="gemma3-1b")]
    held = [point("c", initial=1000, n=10**20, student="gemma3-4b")]
    model = v95.fit_candidate(train, candidate)
    scaler = model.standardizer
    raw = v95.descriptor_values(train, candidate)
    assert scaler.mean == pytest.approx(raw.mean())
    assert scaler.scale == pytest.approx(raw.std(ddof=0))
    assert set(scaler.training_point_ids) == {p.id for p in train}
    assert not set(scaler.training_point_ids) & {p.id for p in held}
    before = deepcopy(scaler)
    z = scaler.transform(held, candidate)
    np.testing.assert_allclose(z, (v95.descriptor_values(held, candidate) - raw.mean()) / raw.std())
    assert abs(z[0]) > 5
    model.predict(held)
    assert model.standardizer == before
    # The coefficient scales the budget term only, not intercept or data term.
    model.coefficients = [2, 3, 4, 5]
    p = held[0]
    expected = 2 + (3 + 5 * z[0]) * np.log1p(p.T / v95.T_REF) + 4 * np.log(p.D_U / v95.D_REF)
    assert model.predict(held)[0] == pytest.approx(expected)


def test_constant_training_descriptor_has_unit_scale_without_using_held_out_values():
    training = [point(t=t, initial=2) for t in (0, 100, 200)]
    scaler = v95.Standardizer.fit(training, "student_initial_loss")
    assert scaler.mean == 2 and scaler.training_std == 0 and scaler.scale == 1
    np.testing.assert_array_equal(scaler.transform(training, "student_initial_loss"), 0)
    np.testing.assert_array_equal(scaler.transform([point(initial=9)], "student_initial_loss"), [7])


def test_bootstrap_repeats_entire_trajectories_not_individual_checkpoints():
    pairs = [v95.Pair("I1", point("a", t=100), point("a", t=200)),
             v95.Pair("I1", point("a", t=200), point("a", t=400)),
             v95.Pair("I1", point("b", t=100), point("b", t=200))]
    clusters = sorted({c for p in pairs for c in p.clusters})
    weights = v95.pair_weights(pairs, np.array([[2, 0], [1, 1], [0, 2]]), clusters)
    np.testing.assert_array_equal(weights, [[2, 2, 0], [1, 1, 1], [0, 0, 2]])
    errors = np.array([[0], [2], [10]])
    result = v95.bootstrap_statistics(pairs, errors)
    assert result["clusters"] == 2
    assert result["mae_draws"].shape == (5000, 1)
    assert set(result["mae_draws"][:, 0]) == {1.0, 4.0, 10.0}
    assert result["empty_pair_resamples_redrawn"] == 0
    np.testing.assert_array_equal(result["mae_draws"], v95.bootstrap_statistics(pairs, errors)["mae_draws"])


def test_cross_pool_bootstrap_resamples_both_endpoint_clusters_and_shared_edges():
    a, b, c = point("a"), point("b", u=198, du=300), point("c", u=594, du=900)
    pairs = [v95.Pair("I2", a, b), v95.Pair("I2", a, c), v95.Pair("I2", b, c)]
    clusters = [a.trajectory, b.trajectory, c.trajectory]
    weights = v95.pair_weights(pairs, np.array([[2, 1, 0], [1, 1, 1], [0, 2, 1]]), clusters)
    np.testing.assert_array_equal(weights, [[2, 0, 0], [1, 1, 1], [0, 0, 2]])
    result = v95.bootstrap_statistics(pairs, np.array([[1, -1], [4, -4], [10, -10]]))
    assert result["clusters"] == 3 and result["empty_pair_resamples_redrawn"] > 0
    assert result["mae_draws"].shape == (5000, 2)
    assert set(result["mae_draws"][:, 0]) == {1.0, 4.0, 5.0, 10.0}
    np.testing.assert_array_equal(result["mae_draws"][:, 0], result["mae_draws"][:, 1])
    np.testing.assert_array_equal(result["bias_draws"][:, 0], -result["bias_draws"][:, 1])


def comparison(gain, width):
    return {"estimate": gain, "width": width, "ci95": [gain - width / 2, gain + width / 2]}


@pytest.mark.parametrize("gain,width,passes", [(.11, .10, True), (.10, .10, False),
                                                (.075, .10, False), (-.2, .10, False), (0, 0, False)])
def test_reading_rule_requires_strictly_more_than_full_paired_interval_width(gain, width, passes):
    assert v95.clears_reading_rule(comparison(gain, width)) is passes


def metric_row(candidate, target, gain=.2, width=.1, split=v95.SPLITS[0], scope="pooled"):
    return {"candidate": candidate, "target": target, "split": split, "scope": scope,
            "mae_improvement_over_baseline": comparison(gain, width)}


def test_reading_rule_same_candidate_both_targets_only_pooled_LOSO():
    rows = [metric_row("T_only", "I1"), metric_row("reuse_only", "I2")]
    assert v95.capability_verdict(rows)["verdict"] == "NOT PREDICTABLE"
    rows += [metric_row("T_only", "I2", split=v95.SPLITS[1]), metric_row("T_only", "I2", scope="held_out=gemma3-4b")]
    assert v95.capability_verdict(rows)["verdict"] == "NOT PREDICTABLE"
    rows.append(metric_row("T_only", "I2", gain=.1, width=.1))
    assert v95.capability_verdict(rows)["verdict"] == "NOT PREDICTABLE"
    rows[-1] = metric_row("T_only", "I2")
    verdict = v95.capability_verdict(rows)
    assert verdict["verdict"] == "PREDICTABLE" and verdict["passing_candidates"] == ["T_only"]


def matrix_fixture():
    return [point(f"u{u}", t=t, u=u, du=u * 100, student=st, seed=seed,
                  initial=2 + i / 10, n=10 ** (8 + i), delta=.1 * i + .01 * seed + np.log1p(t / 100_000))
            for (i, st), u, seed, t in product(enumerate(v95.STUDENTS), v95.POOLS, v95.SEEDS, (0, 100_000, 200_000))]


def test_folds_withhold_complete_students_or_seed_across_every_pool():
    points = matrix_fixture()
    for split in v95.SPLITS:
        for fold, training, held in v95.split_folds(points, split):
            assert not {p.trajectory for p in training} & {p.trajectory for p in held}
            if split == v95.SPLITS[0]:
                assert {p.student for p in held} == {fold}
                assert fold not in {p.student for p in training}
                assert len({p.trajectory for p in held}) == 6
            else:
                assert {p.seed for p in held} == {int(fold)}
                assert not {p.seed for p in held} & {p.seed for p in training}
                assert {p.student for p in training} == {p.student for p in held} == set(v95.STUDENTS)
                assert len({p.trajectory for p in held}) == 9


def test_evaluation_never_fits_pair_differences_and_uses_same_model_at_both_endpoints(monkeypatch):
    points = matrix_fixture()
    calls = []

    def checked_fit(training, candidate):
        assert all(isinstance(p, v95.Point) for p in training)
        assert any(p.T == 0 for p in training)
        calls.append((candidate, {p.trajectory for p in training}))
        return v95.Fit("constant", [np.mean([p.delta for p in training])])

    monkeypatch.setattr(v95, "fit_candidate", checked_fit)
    pairs = v95.construct_pairs(points)
    predictions, fits, folds = v95.evaluate(points, pairs)
    assert len(calls) == len(fits) == 5 * len(v95.CANDIDATES)
    assert len(predictions) == 2 * len(pairs)
    for row in predictions:
        assert all(value == 0 for value in row["predicted_intervention"].values())
        assert row["predicted_delta_first"] == row["predicted_delta_second"]
    assert all(set(f["training_trajectories"]).isdisjoint(f["held_out_trajectories"]) for f in folds)


def test_metrics_paired_gain_bias_cluster_counts_and_one_baseline():
    points = matrix_fixture()
    pairs = v95.construct_pairs(points)
    predictions, _, _ = v95.evaluate(points, pairs)
    metrics = v95.score_predictions(predictions, pairs, resamples=50)
    for row in metrics:
        expected_clusters = 18 if row["scope"] == "pooled" else (6 if row["split"] == v95.SPLITS[0] else 9)
        for key in ("mae", "signed_bias", "mae_improvement_over_baseline"):
            assert row[key]["clusters"] == expected_clusters
            assert row[key]["width"] == row[key]["ci95"][1] - row[key]["ci95"][0]
        if row["candidate"] in ("zero", "constant"):
            assert row["mae_improvement_over_baseline"]["estimate"] == 0
            assert row["mae_improvement_over_baseline"]["ci95"] == [0, 0]
            assert not row["clears_target_reading_rule"]
    selected = [r for r in predictions if r["split"] == v95.SPLITS[0] and r["target"] == "I1"]
    errors = np.array([r["predicted_intervention"]["T_only"] - r["actual"] for r in selected])
    row = next(r for r in metrics if r["split"] == v95.SPLITS[0] and r["scope"] == "pooled"
               and r["target"] == "I1" and r["candidate"] == "T_only")
    assert row["mae"]["estimate"] == pytest.approx(np.abs(errors).mean())
    assert row["signed_bias"]["estimate"] == pytest.approx(errors.mean())


def test_intervals_always_report_cluster_count_and_plain_below_six_statement():
    for clusters in (3, 6):
        value = v95.interval(.2, [.1, .2, .3], clusters)
        text = v95.format_interval(value)
        assert f"clusters={clusters}" in text
        assert ("below six trajectory clusters" in text) == (clusters < 6)
        assert value["below_six_clusters"] == (clusters < 6)


def log_fixture():
    rows = []
    for step in range(1, 9):
        rows.append({"step": step, "epoch": (step + 1) // 2, "tokens": 30, "completion_tokens": 10,
                     "processed_tokens": 30 * step, "seen_tokens": 30 * step, "tokens_seen": 30 * step,
                     "completion_tokens_seen": 10 * step, "unique_data_pool_tokens": 60,
                     "unique_data_tokens": min(60, 30 * step), "unique_examples_seen": min(2, step)})
    return {"status": "trained", "stopped_after_trajectory": True, "trajectory_tokens_unreached": [],
            "training_examples": 2, "unique_data_pool_examples": 2, "unique_data_pool_tokens": 60,
            "updates": 8, "optimizer_steps": 8, "processed_tokens": 240, "seen_tokens": 240, "tokens_seen": 240,
            "completion_tokens_seen": 80, "loss_curve": rows,
            "trajectory": [{"updates": s, "processed_tokens": 30*s, "completion_tokens_seen": 10*s,
                            "requested_token_milestones": [999_999]} for s in (2, 4, 6, 8)]}


def test_token_accounting_uses_logged_completion_counts_and_exact_complete_pool_pass():
    log = log_fixture()
    counts, du, first_epoch_end = v95.token_accounting(log)
    assert counts == {i: (i * 30, i * 10) for i in range(9)}
    assert du == 20 and du != log["unique_data_pool_tokens"]
    assert first_epoch_end == 2


@pytest.mark.parametrize("change", ["missing_update", "bad_completion", "bad_processed", "partial_pool", "duplicate_pool", "snapshot_mismatch"])
def test_token_accounting_rejects_inconsistent_or_unrecoverable_counts(change):
    log = log_fixture()
    if change == "missing_update":
        log["loss_curve"].pop(2)
    elif change == "bad_completion":
        log["loss_curve"][2]["completion_tokens_seen"] += 1
    elif change == "bad_processed":
        log["loss_curve"][2]["processed_tokens"] += 1
    elif change == "partial_pool":
        log["loss_curve"][1]["unique_examples_seen"] = 1
    elif change == "duplicate_pool":
        log["training_examples"] += 1
    else:
        log["trajectory"][0]["completion_tokens_seen"] += 1
    with pytest.raises(ValueError):
        v95.token_accounting(log)


def test_raw_response_curves_keep_seeds_actual_counts_and_own_baselines():
    points = [point(t=t, seed=seed, initial=seed, delta=-t / 1000) for seed in v95.SEEDS for t in (0, 101, 203)]
    curves = v95.response_curves(points)
    assert len(curves) == 2 and [c["seed"] for c in curves] == [41, 42]
    for curve in curves:
        assert [p["T"] for p in curve["points"]] == [0, 101, 203]
        assert [p["delta"] for p in curve["points"]] == pytest.approx([0, -.101, -.203])
        assert curve["initial_loss"] == curve["seed"]
    table = v95.response_curve_table({"response_curves": curves})
    assert "101 : -0.101000" in table and "203 : -0.203000" in table


def test_verdict_block_states_exact_verdict_and_one_baseline():
    summary = {"verdicts": {cap: {"verdict": "NOT PREDICTABLE", "passing_candidates": []} for cap in v95.CAPABILITIES}}
    text = v95.verdict_block(summary)
    assert "identically zero" in text and "ONE baseline" in text
    assert text.count("**NOT PREDICTABLE**") == 3
