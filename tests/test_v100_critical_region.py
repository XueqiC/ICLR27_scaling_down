"""CPU-only checks of matching, excluded runs, bootstrap units and one-F semantics."""
from copy import deepcopy
from dataclasses import replace
from itertools import product
from pathlib import Path

import numpy as np
import pytest

from analysis import v100_critical_region as v


def point(student="s1", u=66, seed=41, t=100000, delta=0., cap="math", initial=2., n=10**7):
    return v.old.Point(f"{student}/pool{u}/seed{seed}", student, u, seed, cap, "fixture",
                       int(t), int(3*t), t, u*250, initial+delta, initial, n)


@pytest.mark.parametrize("tolerance,offset", [(.01, 1000), (.003, 300)])
def test_matching_inclusive_boundary_and_rejects_next_token(tolerance, offset):
    a = point(t=100000)
    boundary = point(u=132, seed=51, t=100000+offset)
    outside = point(u=198, seed=42, t=100001+offset)
    pairs = v.matched_pairs([outside, boundary, a], tolerance)
    assert any(p.first == a and p.second == boundary for p in pairs)
    assert not any(p.first == a and p.second == outside for p in pairs)
    for p in pairs:
        v.validate_matched_pair(p, tolerance)
    assert v.matched_pairs([boundary, a], tolerance) == v.matched_pairs([a, boundary], tolerance)


def test_both_tolerances_enumerate_all_matches_without_ordinal_or_seed_alignment():
    a = point(t=100000)
    b = point(u=132, seed=51, t=100200)
    c = point(u=132, seed=52, t=100700)
    zero = point(u=594, t=0)
    wrong_cap = replace(b, capability="code")
    wrong_student = point(student="other", u=198, t=100000)
    wrong_distribution = replace(b, distribution="foreign")
    same_pool = point(seed=42, t=100000)
    points = [a, b, c, zero, wrong_cap, wrong_student, wrong_distribution, same_pool]
    wide, tight = v.matched_pairs(points, .01), v.matched_pairs(points, .003)
    assert len(wide) == 4 and len(tight) == 2
    assert {p.id for p in tight} < {p.id for p in wide}
    assert all(p.first.U == 66 and p.second.U == 132 for p in wide)
    assert all(p.first.seed != p.second.seed for p in wide)
    assert v.matched_pairs(points, 0) == []
    with pytest.raises(AssertionError, match="Duplicate"):
        v.matched_pairs([a, a], .01)


def directories():
    selected = [Path(s)/f"gpt-5.6-luna_full_{u}_matrix2_lora_dseed{seed}"
                for s, u, seed in product(v.old.STUDENTS, v.old.POOLS, v.old.SEEDS)]
    selected += [Path(s)/f"gpt-5.6-luna_full_132_critical_lora_dseed{seed}"
                 for s, seed in product(v.CRITICAL_STUDENTS, v.CRITICAL_SEEDS)]
    discarded = [Path(f"old{i}")/f"gpt-5.6-luna_full_66_matrix_lora_dseed{41+i%2}" for i in range(5)]
    return selected, discarded


def test_exclusion_assertion_rejects_discarded_directory_even_with_22_runs():
    selected, discarded = directories()
    v.assert_controlled_directories(selected, discarded)
    with pytest.raises(AssertionError, match="Excluded"):
        v.assert_controlled_directories([*selected[:-1], discarded[0]], discarded)
    with pytest.raises(AssertionError, match="five"):
        v.assert_controlled_directories(selected, discarded[:-1])
    with pytest.raises(AssertionError, match="22"):
        v.assert_controlled_directories(selected[:-1], discarded)
    with pytest.raises(AssertionError, match="four critical"):
        v.assert_controlled_directories([*selected[:-1], selected[-1].with_name(
            "gpt-5.6-luna_full_198_critical_lora_dseed52")], discarded)


@pytest.fixture(scope="module")
def real_data():
    return v.load_data()


def test_saved_cohort_token_accounting_and_tolerance_counts(real_data):
    points, provenance, inputs = real_data
    assert len(points) == 450
    assert provenance["directory_assertion"]["excluded_count"] == 5
    assert len({p.trajectory for p in points}) == 22
    assert all(sum(p.trajectory == trajectory and p.capability == "math" for p in points) == 15
               for trajectory in {p.trajectory for p in points if p.U == 132})
    for path in inputs.digests:
        assert "_matrix_lora_dseed" not in path
    for tolerance, per_student in ((.01, (15, 27, 27)), (.003, (6, 9, 9))):
        pairs = v.matched_pairs(points, tolerance)
        for student, expected in zip(v.old.STUDENTS, per_student):
            for cap in v.old.CAPABILITIES:
                assert sum(p.first.student == student and p.first.capability == cap for p in pairs) == expected
        if tolerance == .003:
            assert {v.budget_band(p) for p in pairs if 132 in (p.first.U, p.second.U)} == {"200k"}
    assert v.matched_pairs(points, 0) == []
    inputs.verify()


def test_dense_accounting_does_not_skip_checkpoints_after_four(real_data):
    _, _, inputs = real_data
    name = next(n for n in inputs.digests if "critical" in n and n.endswith("train_log.json"))
    log = inputs.read(inputs.root/name)
    damaged = deepcopy(log)
    damaged["trajectory"][10]["completion_tokens_seen"] += 1
    with pytest.raises(ValueError, match="Dense trajectory accounting"):
        v.critical_token_accounting(damaged)
    with pytest.raises(ValueError, match="fourteen"):
        v.critical_token_accounting({**log, "trajectory": log["trajectory"][:-1]})


def drifting_pairs():
    points = []
    for student, slope in (("s1", 2.), ("s2", 4.), ("s3", 6.)):
        for t in (25000, 50000, 200000):
            a, b = point(student=student, t=t), point(student=student, t=t, u=198, seed=42)
            bracket = np.log1p(b.E)-np.log1p(a.E)
            b = replace(b, loss=b.initial_loss+bracket*(.3+slope*np.log1p(t/v.joint.T_REF)))
            points += [a, b]
    return points, v.matched_pairs(points, .003)


@pytest.mark.parametrize("tolerance", v.TOLERANCES)
def test_slope_test_uses_only_matched_pairs_and_rejects_unmatched_poison(tolerance):
    points, pairs = drifting_pairs()
    result = v.slope_test(pairs, tolerance)
    assert result["slope"]["estimate"] == pytest.approx(4.)
    assert result["used_pairs"] == 9 and result["informative_dyads"] == 3
    assert result["slope"]["clusters"] == 6 and result["valid_draws"] == 5000
    poison = point(student="s1", u=594, seed=52, t=120000, delta=1e10)
    same = v.slope_test(v.matched_pairs([*points, poison], tolerance), tolerance)
    assert same == result
    unmatched = replace(pairs[0], second=replace(pairs[0].second, T=1.02*pairs[0].first.T))
    with pytest.raises(AssertionError, match="Unmatched"):
        v.slope_test([*pairs, unmatched], tolerance)
    with pytest.raises(AssertionError, match="matched pool"):
        v.slope_test([replace(pairs[0], target="I1")], tolerance)


def test_slope_never_uses_changes_in_pool_or_seed_composition_as_budget_drift():
    _, pairs = drifting_pairs()
    singleton = v.old.Pair("I2_size", point(student="late", t=200000),
                          point(student="late", u=594, seed=52, t=200000, delta=1e9))
    original = v.slope_test(pairs, .003)
    result = v.slope_test([*pairs, singleton], .003)
    assert result["slope"] == original["slope"]
    assert singleton.id not in result["used_pair_ids"]
    assert result["single_band_pairs_excluded"] == 1
    assert v.slope_test([singleton], .003)["slope"] is None
    assert v.slope_test([], .003)["slope"] is None


def test_whole_trajectory_bootstrap_keeps_all_dyad_checkpoints_together():
    _, pairs = drifting_pairs()
    weights, clusters, _ = v.bootstrap_weights(pairs)
    assert weights.shape == (5000, 9) and len(clusters) == 6
    for ids in ([i for i, p in enumerate(pairs) if p.first.student == s] for s in ("s1", "s2", "s3")):
        np.testing.assert_array_equal(weights[:, ids[0]], weights[:, ids[1]])
        np.testing.assert_array_equal(weights[:, ids[1]], weights[:, ids[2]])


def fitting_points():
    return [point(student=student, u=u, seed=seed, t=t, initial=1+s*.3, n=10**(6+s))
            for (s, student), u, seed, t in product(enumerate(v.old.STUDENTS), v.POOLS, (41, 42),
                                                   (0, 25000, 50000, 100000, 200000))]


@pytest.mark.parametrize("candidate", v.LAW_CANDIDATES)
def test_single_fitted_parameter_set_generates_all_three_prediction_types(candidate):
    points = fitting_points()
    descriptor = candidate.split("_", 1)[1]
    standardizer = v.old.Standardizer.fit(points, v.joint.descriptor_candidate(descriptor))
    coefficients = np.array([.2, -.03, .12, .04] + ([.08] if candidate.startswith("interaction_") else []))
    values = v.joint.design(points, candidate, standardizer) @ coefficients
    fit = v.fit_response([replace(p, loss=p.initial_loss+y) for p, y in zip(points, values)], candidate)
    np.testing.assert_allclose(fit.coefficients, coefficients, atol=1e-12)
    assert fit.rank == len(coefficients)
    a = point(t=50000, initial=1.3, n=10**7)
    b, c = replace(a, T=100000), replace(a, U=594, D_U=594*250, trajectory="larger_pool")
    i1, i2 = v.old.Pair("I1", a, b), v.old.Pair("I2_size", a, c)
    before = deepcopy(fit)
    predictions = fit.response([a, b, c])
    np.testing.assert_allclose(fit.intervention([i1, i2]),
                               [predictions[1]-predictions[0], predictions[2]-predictions[0]])
    np.testing.assert_array_equal(fit.budget_doubling([a]), fit.intervention([i1]))
    np.testing.assert_array_equal(fit.fixed_T_pool_change([i2]), fit.intervention([i2]))
    np.testing.assert_array_equal(fit.response([replace(a, T=0)]), [0.])
    z = fit.standardizer.transform([a], v.joint.descriptor_candidate(descriptor))[0]
    expected_ratio = coefficients[2]+coefficients[3]*z
    if len(coefficients) == 5:
        expected_ratio += coefficients[4]*np.log1p(a.T/v.joint.T_REF)
    assert fit.intervention([i2])[0]/(np.log1p(c.E)-np.log1p(a.E)) == pytest.approx(expected_ratio)
    assert fit == before


def test_pool_holdout_never_scores_an_in_training_endpoint():
    points = fitting_points()
    pairs = v.matched_pairs(points, .003)
    for split, fold, training, held in v.response_folds(points):
        training_ids, held_ids = {p.id for p in training}, {p.id for p in held}
        assert training_ids.isdisjoint(held_ids)
        selected = [p for p in pairs if p.first.id in held_ids and p.second.id in held_ids]
        if split == v.SPLIT_POOL:
            assert not selected
            assert all(p.U != 132 for p in training)
        elif split == v.SPLIT_TWO_POOLS:
            reference = int(fold.split("+")[1])
            assert selected and all(p.U not in (132, reference) for p in training)
            assert all({p.first.U, p.second.U} == {132, reference} for p in selected)


def test_reading_rule_requires_full_width_and_both_targets():
    assert not v.old.clears_reading_rule({"estimate": .1, "width": .1})
    assert not v.old.clears_reading_rule({"estimate": .06, "width": .1})
    assert v.old.clears_reading_rule({"estimate": .10001, "width": .1})
    rows = [{"split": v.SPLIT_STUDENT, "scope": "pooled", "tolerance": .01,
             "capability": "math", "candidate": "joint_log_parameters", "target": target,
             "clears_zero_rule": passed} for target, passed in (("I1", True), ("I2_size", False))]
    verdict = next(r for r in v.decisions(rows) if r["split"] == v.SPLIT_STUDENT and r["tolerance"] == .01
                   and r["capability"] == "math" and r["candidate"] == "joint_log_parameters")
    assert verdict["status"] == "did not meet the pre-registered threshold"
