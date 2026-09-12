"""CPU-only checks for V89 retrospective development-set intervention analysis."""
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from analysis import v89_intervention_retro as v89


def point(run="a", t=100, du=100, delta=0.0, student="gemma3-270m", seed=11,
          cap="math", distribution="fixture", update=None):
    update = int(t) if update is None else update
    return v89.Point(f"{student}/{run}@{update}:{cap}:{distribution}", f"{student}/{run}",
                     student, seed, cap, distribution, update, t, du, delta,
                     2 + delta, 2, f"{run}/eval.json", f"{run}/initial.json")


def test_i1_constructs_every_ordered_ratio_inside_inclusive_window():
    points = [point(t=t) for t in (0, 100, 169, 170, 200, 230, 231, 400)]
    pairs = v89.construct_pairs(points)
    actual = {(p.first.T, p.second.T) for p in pairs if p.target == "I1"}
    expected = {(a.T, b.T) for a in points for b in points
                if a.T > 0 and 1.7 <= b.T / a.T <= 2.3}
    assert actual == expected
    assert {(100, 170), (100, 200), (100, 230), (200, 400)} <= actual
    assert (100, 169) not in actual and (100, 231) not in actual
    assert all(p.second.T > p.first.T > 0 for p in pairs)


def test_i2_all_shared_t_matches_same_student_different_du_and_distribution():
    a = point("a", 100, 100, delta=0.7)
    points = [a, point("b", 90, 200), point("b", 100, 200, delta=0.1),
              point("b", 110, 200, delta=0.2), point("b", 111, 200),
              point("same_du", 100, 100), point("other_j", 100, 200, distribution="other"),
              point("other_cap", 100, 200, cap="code"),
              point("other_student", 100, 200, student="gemma3-1b")]
    pairs = [p for p in v89.construct_pairs(points) if p.target == "I2" and p.first.id == a.id]
    assert [(p.second.T, p.actual) for p in pairs] == [(100, pytest.approx(-0.6)), (110, pytest.approx(-0.5))]
    assert all(p.second.D_U > p.first.D_U for p in pairs)
    # Equal nominal pool size is not required: D_U is the specified intervention axis.
    assert v89.construct_pairs([a, point("different_seed", 100, 101, seed=12)])[0].target == "I2"
    assert v89.construct_pairs([point("a", 0, 100), point("b", 0, 200)]) == []


@pytest.mark.parametrize("candidate", v89.CANDIDATES)
def test_difference_of_own_predictions_for_all_candidates(candidate):
    points = [point("a", t, 100, delta=0.3 + np.log1p(t / 100)) for t in (0, 100, 200)]
    points += [point("b", t, 200, delta=-0.2 + np.log1p(t / 100)) for t in (0, 100, 200)]
    fit = v89.fit_candidate(points, candidate, resamples=30)
    pairs = v89.construct_pairs(points)
    expected = np.array([fit.predict([p.second])[0] - fit.predict([p.first])[0] for p in pairs])
    np.testing.assert_allclose(v89.predicted_intervention(fit, pairs), expected)
    if candidate in ("constant", "zero"):
        np.testing.assert_array_equal(expected, 0)
    if candidate == "constant":
        assert fit.coefficients == pytest.approx([np.mean([p.delta for p in points])])
        assert fit.coefficients[0] != 0


def test_intercept_cancels_and_i2_uses_actual_budget_at_each_endpoint():
    pair = v89.Pair("I2", point("a", 100, 100), point("b", 109, 200))
    model = v89.Fit("two_dimensional", [50.0, 3.0, -0.4])
    expected = 3 * (np.log1p(109 / v89.T_REF) - np.log1p(100 / v89.T_REF)) - 0.4 * np.log(2)
    assert v89.predicted_intervention(model, [pair])[0] == pytest.approx(expected)
    shifted = v89.Fit("two_dimensional", [-100.0, 3.0, -0.4])
    assert v89.predicted_intervention(shifted, [pair])[0] == pytest.approx(expected)


def test_cluster_bootstrap_weights_all_pairs_in_a_trajectory_together():
    pairs = [v89.Pair("I1", point("a", 100), point("a", 200)),
             v89.Pair("I1", point("a", 200), point("a", 400)),
             v89.Pair("I1", point("b", 100), point("b", 200))]
    clusters = sorted({c for p in pairs for c in p.clusters})
    weights = v89.pair_weights(pairs, np.array([[2, 0], [1, 1], [0, 2]]), clusters)
    np.testing.assert_array_equal(weights, [[2, 2, 0], [1, 1, 1], [0, 0, 2]])
    result = v89.bootstrap_statistics(pairs, np.array([[0], [2], [10]]))
    # Unequal trajectory lengths: exactly the three possible WHOLE-trajectory draws.
    assert set(result["mae_draws"][:, 0]) == {1.0, 4.0, 10.0}
    assert result["clusters"] == 2 and len(result["mae_draws"]) == 5000
    assert result["empty_pair_resamples_redrawn"] == 0
    again = v89.bootstrap_statistics(pairs, np.array([[0], [2], [10]]))
    np.testing.assert_array_equal(result["mae_draws"], again["mae_draws"])


def test_i2_bootstrap_resamples_both_endpoint_trajectories_and_shared_edges():
    a, b, c = point("a", du=100), point("b", du=200), point("c", du=300)
    pairs = [v89.Pair("I2", a, b), v89.Pair("I2", a, c), v89.Pair("I2", b, c)]
    weights = v89.pair_weights(pairs, np.array([[2, 1, 0], [1, 1, 1], [0, 2, 1]]),
                              [a.trajectory, b.trajectory, c.trajectory])
    np.testing.assert_array_equal(weights, [[2, 0, 0], [1, 1, 1], [0, 0, 2]])
    result = v89.bootstrap_statistics(pairs, np.array([[1], [4], [10]]), resamples=5000)
    assert set(result["mae_draws"][:, 0]) == {1.0, 4.0, 5.0, 10.0}
    assert result["empty_pair_resamples_redrawn"] > 0
    assert len(result["mae_draws"]) == 5000 and result["clusters"] == 3


def comparison(gain, width):
    return {"estimate": gain, "width": width, "ci95": [gain - width / 2, gain + width / 2]}


def test_reading_rule_strict_full_interval_width_and_both_baselines():
    assert v89.clears_reading_rule({"zero": comparison(0.11, 0.10), "constant": comparison(0.3, 0.2)})
    assert not v89.clears_reading_rule({"zero": comparison(0.10, 0.10), "constant": comparison(0.3, 0.2)})
    # Positive CI and beating a half-width are insufficient: FULL width is required.
    assert not v89.clears_reading_rule({"zero": comparison(0.075, 0.10), "constant": comparison(0.3, 0.2)})
    assert not v89.clears_reading_rule({"zero": comparison(0.11, 0.10), "constant": comparison(-0.1, 0.2)})
    assert not v89.clears_reading_rule({"zero": comparison(0.11, 0.10)})


def metric_row(candidate, target, gain=0.2, width=0.1, split=v89.SPLITS[0]):
    return {"candidate": candidate, "target": target, "student": "all", "split": split,
            "improvement_over_baselines": {b: comparison(gain, width) for b in ("zero", "constant")}}


def test_capability_rule_same_candidate_both_targets_only_student_split():
    rows = [metric_row("T_only", "I1"), metric_row("reuse_only", "I2")]
    assert v89.capability_verdict(rows)["signal"] == "absent"
    rows.append(metric_row("T_only", "I2", split=v89.SPLITS[1]))
    assert v89.capability_verdict(rows)["signal"] == "absent"
    rows.append(metric_row("T_only", "I2"))
    result = v89.capability_verdict(rows)
    assert result["signal"] == "present" and result["passing_candidates"] == ["T_only"]


def test_pool_seed_folds_withhold_all_sizes_and_repeats_and_never_cross_fit_endpoints():
    points = [point(f"u{du}_s{seed}_{repeat}", 100, du, student=st, seed=seed)
              for st in v89.STUDENTS for seed in (11, 12) for du in (100, 200) for repeat in (0, 1)]
    for name, train, held in v89.split_folds(points, v89.SPLITS[1]):
        assert len({p.student for p in train + held}) == 1
        assert len({p.seed for p in held}) == 1
        assert not ({p.seed for p in held} & {p.seed for p in train})
        assert len(held) == 4
        pairs = v89.construct_pairs(held)
        assert all(p.first.seed == p.second.seed for p in pairs)
    for name, train, held in v89.split_folds(points, v89.SPLITS[0]):
        assert {p.student for p in held} == {name}
        assert name not in {p.student for p in train}


def test_evaluation_fits_delta_and_predicts_two_points_with_one_model(monkeypatch):
    points = [point(f"u{du}_s{seed}", t, du, student=st, seed=seed,
                    delta=seed + t / 100 + du / 100, cap="math")
              for st in v89.STUDENTS for seed in (11, 12) for du in (100, 200) for t in (100, 200)]
    observed = []

    def checked_fit(train, candidate, *args):
        assert all(isinstance(p, v89.Point) for p in train)
        assert all(p.delta == p.seed + p.T / 100 + p.D_U / 100 for p in train)
        observed.append((candidate, set(p.student for p in train), set(p.seed for p in train)))
        return v89.Fit("constant", [np.mean([p.delta for p in train])])

    monkeypatch.setattr(v89, "fit_candidate", checked_fit)
    pairs = v89.construct_pairs(points)
    predictions, _, folds = v89.evaluate(points, pairs, resamples=10, progress=lambda *a, **k: None)
    assert len(predictions) > 0 and len(observed) > 0
    assert all(value == 0 for row in predictions for value in row["predicted_intervention"].values())
    assert all(set(row["training_trajectories"]).isdisjoint(row["held_out_trajectories"]) for row in folds)
    counts = v89.pair_counts(points, pairs, predictions)
    seed_i2 = [r for r in counts if r["split"] == v89.SPLITS[1] and r["target"] == "I2"]
    assert all(r["not_evaluated_pairs"] > 0 and r["not_evaluated_reason"] for r in seed_i2)


def test_tau_zero_signal_profile_reaches_boundaries_and_is_not_identified():
    points = [point(run, t, du, delta=0) for run, du in (("a", 100), ("b", 200)) for t in (0, 100, 200, 400)]
    for candidate in ("saturation_p1", "saturation_p2"):
        fit = v89.fit_candidate(points, candidate, resamples=50)
        profile = fit.profile
        assert profile["ci95"] == [100, 400]
        assert profile["reaches_lower_boundary"] and profile["reaches_upper_boundary"]
        assert not profile["identified_inside_observed_T_range"]
        assert profile["status"] == "not identified"
        assert profile["clusters"] == 2 and profile["below_six_clusters"]


def test_tau_profile_can_identify_an_interior_synthetic_timescale():
    points = [point(f"run{i}", t, 100 * (i + 1),
                    delta=0.4 * np.expm1(-t / 400) + 0.02 * np.log1p(t / (100 * (i + 1))) ** 2)
              for i in range(8) for t in (0, 100, 200, 400, 800, 1600)]
    fit = v89.fit_candidate(points, "saturation_p2", resamples=100)
    assert fit.tau == pytest.approx(400)
    assert fit.profile["identified_inside_observed_T_range"]
    assert fit.profile["ci95"] == pytest.approx([400, 400])
    assert fit.profile["clusters"] == 8


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def filesystem_fixture(root, initial=True):
    write_json(root / v89.REGISTER, {"pools": {"pool": {"U": 75, "data_seed": 11, "D_U_completion": 123}}})
    for student in v89.STUDENTS:
        (root / "results/v12-distill" / student).mkdir(parents=True)
    run = root / "results/v12-distill/gemma3-270m/fixture"
    meta = {"student": "gemma3-270m", "n_per_domain": 75, "data_seed": 11, "data_sampling_seed": 11,
            "teacher": "gpt-5.6-luna", "recipe": "full", "probe_source": "fixture", "probe_seed": 0,
            "probe_half": "odd", "n_probe_requested": 2, "loss_definition": "nats/token",
            "measurement_benchmarks": dict.fromkeys(v89.CAPABILITIES, "fixture"),
            "measurement_samples": dict.fromkeys(v89.CAPABILITIES, 1),
            "measurement_tokens": dict.fromkeys(v89.CAPABILITIES, 10),
            "dense": dict.fromkeys(v89.CAPABILITIES, 2.0)}
    write_json(run / "train_log.json", {**meta, "updates": 2, "completion_tokens_seen": 210,
               "planned_T_completion": 999999, "tokens_seen": 1234567,
               "loss_curve": [{"step": 1, "completion_tokens": 100, "completion_tokens_seen": 100},
                              {"step": 2, "completion_tokens": 110, "completion_tokens_seen": 210}]})
    if initial:
        write_json(run / "trajectory/update-00000000/eval.json", {**meta, "updates": 0,
                    "post_training": dict.fromkeys(v89.CAPABILITIES, 3.0), "completion_tokens_seen": 0})
    write_json(run / "trajectory/update-00000001/eval.json", {**meta, "updates": 1,
               "post_training": dict.fromkeys(v89.CAPABILITIES, 3.1), "completion_tokens_seen": 100})
    write_json(run / "eval.json", {**meta, "updates": 2,
               "post_training": dict.fromkeys(v89.CAPABILITIES, 3.2), "completion_tokens_seen": 210})
    return run


@pytest.mark.parametrize("initial", [True, False])
def test_loader_uses_own_initial_or_dense_and_actual_logged_completion_tokens(tmp_path, initial):
    filesystem_fixture(tmp_path, initial)
    points, audit, exclusions, distributions, inputs = v89.load_data(tmp_path)
    final = next(p for p in points if p.update == 2 and p.capability == "math")
    assert final.delta == pytest.approx(0.2 if initial else 1.2)
    assert final.T == 210 and final.D_U == 123 and final.E == pytest.approx(210 / 123)
    assert audit[0]["status"] == "included" and not exclusions
    assert len(distributions) == 1
    assert all(len(d) == 64 for d in inputs.digests.values())
    inputs.verify()


def test_loader_audits_unregistered_runs_and_token_mismatch(tmp_path):
    run = filesystem_fixture(tmp_path)
    old = run.parent / "unregistered"
    for name in ("eval.json", "train_log.json"):
        data = json.loads((run / name).read_text())
        data["data_seed"] = None
        write_json(old / name, data)
    checkpoint = run / "trajectory/update-00000001/eval.json"
    data = json.loads(checkpoint.read_text())
    data["completion_tokens_seen"] = 999999  # Cannot replace actual T with a planned milestone.
    write_json(checkpoint, data)
    points, audits, exclusions, _, inputs = v89.load_data(tmp_path)
    assert not any(p.update == 1 for p in points)
    assert any("disagrees with actual train_log" in e["reason"] for e in exclusions)
    dropped = [a for a in audits if a["status"] == "dropped"]
    assert len(dropped) == 1 and "No pool entry" in dropped[0]["reasons"][0]
    assert str((old / "eval.json").relative_to(tmp_path)) in inputs.digests


def test_duplicate_checkpoint_is_explicit_and_never_counted_twice(tmp_path):
    run = filesystem_fixture(tmp_path)
    data = json.loads((run / "eval.json").read_text())
    write_json(run / "trajectory/update-00000002/eval.json", data)
    points, _, exclusions, _, _ = v89.load_data(tmp_path)
    assert len([p for p in points if p.update == 2]) == 3
    assert len(exclusions) == 1 and "Duplicate update 2" in exclusions[0]["reason"]


def test_inconsistent_register_or_log_fails_explicitly():
    with pytest.raises(ValueError, match="Conflicting registered pool"):
        v89.registered_pools({"a": {"U": 75, "data_seed": 11, "D_U_completion": 100},
                              "b": {"U": 75, "data_seed": 11, "D_U_completion": 200}})
    with pytest.raises(ValueError, match="Conflicting completion token counts"):
        v89.completion_by_update({"updates": 1, "completion_tokens_seen": 999,
                                  "loss_curve": [{"step": 1, "completion_tokens": 100}]})


def test_output_rejects_symlink_before_touching_other_results(tmp_path):
    sentinel = tmp_path / "results/frozen.json"
    write_json(sentinel, {"frozen": True})
    destination = tmp_path / v89.OUT_REL
    destination.mkdir()
    (destination / "summary.json").symlink_to(sentinel)
    before = sentinel.read_bytes()
    with pytest.raises(ValueError, match="Symlink output file"):
        v89.write_outputs({}, tmp_path)
    assert sentinel.read_bytes() == before


def test_interval_always_carries_cluster_count_and_low_cluster_warning():
    value = v89.interval(2.0, np.arange(10), 3)
    assert value["clusters"] == 3 and value["below_six_clusters"]
    assert "n=3 trajectories; BELOW SIX" in v89.ci_text(value)
    assert not v89.interval(2.0, np.arange(10), 6)["below_six_clusters"]


def test_analysis_is_cpu_numpy_only_and_no_environment_device_setting():
    import ast
    source = Path(v89.__file__).read_text()
    tree = ast.parse(source)
    imported = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    imported |= {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    assert not any(name and name.split(".")[0] in {"torch", "transformers", "cupy", "jax"} for name in imported)
    assert "CUDA_VISIBLE_DEVICES" not in source
