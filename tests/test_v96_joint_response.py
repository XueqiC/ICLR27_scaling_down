"""CPU-only checks of V96 cohort, accounting, leakage and one-response semantics."""
from copy import deepcopy
from dataclasses import replace
from itertools import product
from pathlib import Path

import numpy as np
import pytest

from analysis import v96_joint_response as v


def point(student="s1", u=66, seed=41, t=100000, delta=0.0, cap="math", initial=2.0, n=10000000):
    return v.old.Point(f"{student}/pool{u}/seed{seed}", student, u, seed, cap, "fixture",
                       int(t), int(t*3), t, u*250, initial+delta, initial, n)


def curves():
    return [point(student=f"s{s}", u=u, seed=seed, t=t, initial=1+s*.3, n=10**(6+s),
                  delta=.2*np.log1p(t/v.T_REF)+(.1+s*.05)*np.log1p(t/(u*250)))
            for s, u, seed, t in product(range(3), v.old.POOLS, v.old.SEEDS, (0, 50000, 100000, 200000))]


def directories():
    selected = [Path(s)/f"gpt-5.6-luna_full_{u}_matrix2_lora_dseed{seed}"
                for s, u, seed in product(v.old.STUDENTS, v.old.POOLS, v.old.SEEDS)]
    discarded = [Path(f"old{i}")/f"gpt-5.6-luna_full_66_matrix_lora_dseed{41+i%2}" for i in range(5)]
    return selected, discarded


def test_excluded_directory_assertion_rejects_discarded_launch_even_with_18_runs():
    selected, discarded = directories()
    v.assert_controlled_directories(selected, discarded)
    with pytest.raises(AssertionError, match="Excluded"):
        v.assert_controlled_directories([*selected[:-1], discarded[0]], discarded)
    with pytest.raises(AssertionError, match="18"):
        v.assert_controlled_directories(selected[:-1], discarded)
    with pytest.raises(AssertionError, match="five"):
        v.assert_controlled_directories(selected, discarded[:-1])


def test_real_cohort_assertion_and_no_discarded_contents_read():
    audit = v.directory_audit(v.ROOT)
    assert audit["selected_count"] == 18 and audit["excluded_count"] == 5
    assert audit["passed"] and not audit["excluded_contents_read"]


def test_missing_supervised_domain_shares_never_replaced_with_example_or_eval_shares():
    payload = {"completion_tokens_seen": 1000, "measurement_tokens": {"math": 1, "code": 2, "qa": 3},
               "trace_counts": {c: {"training_rows": 10} for c in v.old.CAPABILITIES}}
    assert v.recorded_domain_shares(payload) == {c: None for c in v.old.CAPABILITIES}
    payload["completion_tokens_seen_by_domain"] = {"math": 200, "code": 300, "qa": 500}
    assert v.recorded_domain_shares(payload) == {"math": .2, "code": .3, "qa": .5}
    payload["completion_tokens_seen_by_domain"]["math"] += 1
    with pytest.raises(ValueError, match="disagrees"):
        v.recorded_domain_shares(payload)


def test_t_only_exact_zero_on_fixed_T_even_if_DU_seed_and_other_accounting_change():
    fit = v.fit_response(curves(), "T_only")
    first = point(t=100000)
    second = replace(first, trajectory="other", U=594, D_U=170000, seed=42,
                     processed_tokens=900000, update=87)
    pair = v.old.Pair("I2_size", first, second)
    assert fit.intervention([pair])[0] == 0.0
    assert fit.fixed_T_pool_change([pair])[0] == 0.0
    mismatched = replace(pair, second=replace(second, T=107000))
    assert fit.intervention([mismatched])[0] != 0.0
    assert fit.fixed_T_pool_change([mismatched])[0] == 0.0
    expected = fit.coefficients[0]*(np.log1p(1.07)-np.log1p(1.0))
    assert fit.intervention([mismatched])[0] == pytest.approx(expected)


def test_mean_effect_uses_only_training_capability_target_and_both_endpoints():
    a = point(student="train", t=50000, delta=.1)
    b = replace(a, T=100000, update=100000, loss=a.initial_loss+.5)
    c = replace(b, T=200000, update=200000, loss=a.initial_loss+.7)
    held = point(student="held", u=198, delta=1000)
    foreign_cap = replace(a, capability="qa", loss=-999)
    pairs = [v.old.Pair("I1", a, b), v.old.Pair("I1", b, c),
             v.old.Pair("I1", a, held), v.old.Pair("I2_size", a, c),
             v.old.Pair("I1", foreign_cap, replace(foreign_cap, loss=1000))]
    model = v.fit_mean_effect([a, b, c, foreign_cap], pairs, "math", "I1")
    assert model.value == pytest.approx(.3)
    assert model.value != 0
    assert len(model.training_pair_ids) == 2
    assert model.training_trajectories == (a.trajectory,)
    poisoned = [replace(p, second=replace(p.second, loss=-1e12)) if p.second == held else p for p in pairs]
    assert v.fit_mean_effect([a, b, c], poisoned, "math", "I1").value == model.value
    assert v.fit_mean_effect([a, b, c], pairs, "math", "I2_seed") is None


def test_equivalence_check_against_all_seven_actual_V95_families():
    result = v.equivalence_check({"candidates": v.old.FORMULAS})
    assert result["performed_before_fitting"]
    assert result["equivalent_candidates"] == []
    assert len(result["comparisons"]) == 14
    for row in result["comparisons"]:
        assert not row["equivalent"]
        joint, earlier, union = row["ranks_joint_old_union"]
        assert joint == 4 and (joint != earlier or union > joint)
    registry = dict(v.old.FORMULAS, student_initial_loss="changed")
    with pytest.raises(AssertionError, match="registry changed"):
        v.equivalence_check({"candidates": registry})


def test_equivalence_test_recognizes_actual_reparameterizations():
    a = np.array([[1., 0.], [1., 2.], [3., 1.], [2., -1.]])
    b = a @ np.array([[2., 1.], [-1., 3.]])
    assert v.same_column_space(a, b)[0]
    assert not v.same_column_space(a, np.column_stack((a, [2, 4, 1, 8])))[0]


@pytest.mark.parametrize("descriptor", v.DESCRIPTORS)
def test_one_fitted_parameter_set_generates_response_doubling_and_pool_change(descriptor):
    data = curves()
    name = f"joint_{descriptor}"
    scaler = v.old.Standardizer.fit(data, v.descriptor_candidate(descriptor))
    coefficients = np.array([.2, -.03, .12, .04])
    y = v.design(data, name, scaler)@coefficients
    training = [replace(p, loss=p.initial_loss+float(delta)) for p, delta in zip(data, y)]
    model = v.fit_response(training, name)
    np.testing.assert_allclose(model.coefficients, coefficients, atol=1e-12)
    a = point(t=50000, initial=1.3, n=10**7)
    b = replace(a, T=100000)
    c = replace(a, U=594, D_U=594*250, trajectory="different_pool")
    budget, pool = v.old.Pair("I1", a, b), v.old.Pair("I2_size", a, c)
    before = deepcopy(model)
    responses = model.response([a, b, c])
    np.testing.assert_allclose(model.intervention([budget, pool]),
                               [responses[1]-responses[0], responses[2]-responses[0]], atol=1e-14)
    np.testing.assert_array_equal(model.budget_doubling([a]), model.intervention([budget]))
    np.testing.assert_array_equal(model.fixed_T_pool_change([pool]), model.intervention([pool]))
    z = model.standardizer.transform([a], v.descriptor_candidate(descriptor))[0]
    bracket = np.log1p(c.E)-np.log1p(a.E)
    assert model.intervention([pool])[0] == pytest.approx((coefficients[2]+coefficients[3]*z)*bracket)
    assert model == before


@pytest.mark.parametrize("candidate", v.CANDIDATES)
def test_all_response_candidates_anchor_zero_without_extra_parameters(candidate):
    model = v.fit_response(curves(), candidate)
    assert v.T_REF == 100000.0 and v.BOOTSTRAPS == 5000
    np.testing.assert_array_equal(model.response([replace(p, T=0) for p in curves()]), np.zeros(len(curves())))
    if candidate.startswith("joint_"):
        assert len(model.coefficients) == model.rank == 4
    if candidate.startswith("surface_"):
        assert len(model.coefficients) == 7


def test_descriptor_scaler_and_mean_are_not_recomputed_from_heldout_outcomes():
    training = [p for p in curves() if p.student != "s2"]
    model = v.fit_response(training, "joint_initial_loss")
    assert set(model.training_point_ids) == {p.id for p in training}
    assert model.standardizer.mean == pytest.approx(np.mean([p.initial_loss for p in training]))
    a = point(student="held", initial=3., delta=1.)
    b = replace(a, loss=1e10)
    np.testing.assert_array_equal(model.response([a]), model.response([b]))


def test_extra_interaction_cannot_be_fitted_without_stable_residual_gate():
    with pytest.raises(ValueError, match="stable budget"):
        v.fit_response(curves(), "interaction_log_parameters")
    fit = v.fit_response(curves(), "interaction_log_parameters", stable_residual=True)
    assert len(fit.coefficients) == 5


@pytest.mark.parametrize("descriptor", v.DESCRIPTORS)
def test_only_permitted_extra_term_has_the_implied_budget_dependent_pool_ratio(descriptor):
    training = curves()
    scaler = v.old.Standardizer.fit(training, v.descriptor_candidate(descriptor))
    name = f"interaction_{descriptor}"
    coefficients = np.array([.2, -.01, .05, .03, .12])
    values = v.design(training, name, scaler)@coefficients
    fitted = v.fit_response([replace(p, loss=p.initial_loss+float(y)) for p, y in zip(training, values)],
                            name, stable_residual=True)
    np.testing.assert_allclose(fitted.coefficients, coefficients, atol=1e-12)
    for t in (50000, 100000, 200000):
        a = point(t=t, initial=1.3, n=10**7)
        b = replace(a, D_U=150000, U=594, trajectory="larger")
        pair = v.old.Pair("I2_size", a, b)
        z = fitted.standardizer.transform([a], v.descriptor_candidate(descriptor))[0]
        ratio = fitted.intervention([pair])[0]/(np.log1p(b.E)-np.log1p(a.E))
        assert ratio == pytest.approx(coefficients[2]+coefficients[3]*z+coefficients[4]*np.log1p(t/v.T_REF))


def test_conditional_extension_does_no_fit_if_residual_gate_fails():
    diagnostic = {"interaction_gate": {"stable_observed_residual_by_capability": {c: False for c in v.old.CAPABILITIES}}}
    result = v.conditional_extension([], [], [], diagnostic)
    assert not result["refitted"]


def test_whole_trajectory_bootstrap_keeps_checkpoints_together():
    a = point(t=50000)
    b = replace(a, T=100000, update=100000)
    c = point(u=198, t=50000)
    d = replace(c, T=100000, update=100000)
    pairs = [v.old.Pair("I2_size", a, c), v.old.Pair("I2_size", b, d)]
    weights, n = v.cluster_weights(pairs)
    assert n == 2 and weights.shape == (5000, 2)
    np.testing.assert_array_equal(weights[:, 0], weights[:, 1])


def test_seed_only_contrast_has_no_fold_with_both_endpoints_heldout():
    a, b = point(seed=41), point(seed=42)
    for _, _, held in v.old.split_folds([a, b], v.old.SPLITS[1]):
        assert not {a.id, b.id}.issubset({p.id for p in held})
