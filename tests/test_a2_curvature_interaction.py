"""CPU-only tests for A2's algebra, endpoint identity and leakage risks."""
from dataclasses import replace
import subprocess
import sys

import numpy as np
import pytest

from analysis import a2_curvature_interaction as a2


@pytest.fixture
def points():
    rows = []
    for student, n in (("small", 100), ("middle", 700), ("large", 3000)):
        for seed in (41, 42, 51, 52):
            for rung in (66, 132, 198, 594):
                run = f"{student}/{rung}/{seed}"
                for T in (25000, 50000, 100000, 200000):
                    D = rung*251 + seed
                    u, v = np.log1p(T/a2.T_REF), np.log1p(T/D)
                    # A nontrivial known curved response, no GPU/training.
                    z = np.log(n)-np.log(700)
                    y = (.1+.01*z)*u + (.07-.005*z)*a2.h_p(T/D, .65)
                    rows.append(a2.Point((run, str(T), "code", "synthetic"), run, student,
                                         f"pool/{rung}/{seed}", seed, rung, "synthetic", T, D,
                                         2+np.log(n)/10, float(y), n))
    return rows


def test_h_p_zero_limit_and_linear_reuse():
    e = np.array([0., 1e-12, .2, 1., 20., 100.])
    np.testing.assert_array_equal(a2.h_p(e, 0), np.log1p(e))
    for p in (-1e-10, 1e-10):
        np.testing.assert_allclose(a2.h_p(e, p), np.log1p(e), rtol=3e-10, atol=1e-14)
    np.testing.assert_allclose(a2.h_p(e, 1), e, rtol=1e-13, atol=1e-13)
    a2.numerical_invariants()


def test_curvature_at_zero_is_exactly_log_response(points):
    scaler = a2.Scaler.fit(points, "log_parameters")
    log = a2.design(points, "F_log", scaler)
    curv = a2.design(points, "F_curv", scaler, 0)
    np.testing.assert_array_equal(log, curv)
    coef = np.array([.1, -.02, -.3, .05])
    np.testing.assert_array_equal(log@coef, curv@coef)


@pytest.mark.parametrize("name", [*a2.STRUCTURES, *a2.RESPONSE_BASELINES])
def test_all_response_predictors_are_zero_at_zero_budget(points, name):
    fit = a2.fit_response(points, name, profile=False)
    anchors = [replace(p, T=0) for p in points[:5]]
    np.testing.assert_array_equal(fit.predict(anchors), np.zeros(len(anchors)))


def test_budget_only_exact_I_U_zero_and_mismatch_is_not_fixed_T(points):
    fit = a2.fit_response(points, "budget_only")
    first = points[0]
    second = replace(first, D=first.D*3, run="another-pool")
    pair = a2.Pair("exact", "I_U", first, second, .01, "low", .1)
    assert fit.intervention([pair])[0] == 0
    mismatch = replace(pair, second=replace(second, T=first.T+1000))
    assert fit.intervention([mismatch])[0] != 0
    assert fit.intervention([mismatch], fixed_T=True)[0] == 0


def test_constant_has_zero_positive_budget_interventions(points):
    fit = a2.fit_response(points, "constant")
    pair = a2.Pair("doubling", "I_T", points[0], points[1], None, "low", .1)
    assert fit.intervention([pair])[0] == 0


def test_student_initial_descriptor_is_fixed_across_data_intervention(points):
    fit = a2.fit_response(points, "F_int", "initial_loss")
    first = points[0]
    second = replace(first, D=first.D*2, run="new-pool")
    pair = a2.Pair("pool", "I_U", first, second, .01, "low", .1)
    changed = replace(pair, second=replace(second, initial_loss=900))
    np.testing.assert_array_equal(fit.intervention([pair]), fit.intervention([changed]))


@pytest.mark.parametrize("split", a2.SPLITS)
def test_outer_values_never_enter_standardisation_or_p(points, split):
    held, train, test = a2.make_folds(points, split)[0]
    fit = a2.fit_response(train, "F_curv")
    changed = {p.key: replace(p, n=10**30, initial_loss=999, y=-1e8) for p in test}
    altered = [changed.get(p.key, p) for p in points]
    held2, train2, test2 = a2.make_folds(altered, split)[0]
    other = a2.fit_response(train2, "F_curv")
    assert held == held2 and test2 != test
    assert fit.scaler == other.scaler
    assert fit.p == other.p and fit.profile == other.profile
    np.testing.assert_array_equal(fit.coefficients, other.coefficients)
    assert set(fit.train_keys).isdisjoint(p.key for p in test)
    assert fit.scaler.students == tuple(sorted({p.student for p in train}))


def test_inner_selection_uses_only_its_training_rows(points):
    _, train, test = a2.make_folds(points, "student")[0]
    # Two inner students => fixed settings, no evaluated-student tuning.
    selection = a2.select_inside(train, [], "student", {})
    assert selection["primary"] == "F_log"
    assert selection["descriptor"] == "log_parameters"
    assert selection["lambda_"] == a2.FIXED_LAMBDA
    assert selection["inner_folds"] == 2
    inner_train = a2.make_folds(train, "data_rung", inner=True)[0][1]
    f = a2.fit_response(inner_train, "F_curv", "initial_loss", profile=False)
    assert set(f.train_keys).issubset(p.key for p in train)
    assert set(f.train_keys).isdisjoint(p.key for p in test)


def test_one_fitted_predictor_supplies_both_endpoints(points, monkeypatch):
    fit = a2.fit_response(points, "F_int")
    p = a2.Pair("pair", "I_T", points[0], points[1], None, "low", points[1].y-points[0].y)
    expected = fit.predict([p.second])[0] - fit.predict([p.first])[0]
    calls = []
    original = a2.Fit.predict

    def spy(self, rows):
        calls.append((id(self), self.fit_id, tuple(x.key for x in rows)))
        return original(self, rows)

    monkeypatch.setattr(a2.Fit, "predict", spy)
    assert fit.intervention([p])[0] == expected
    assert len(calls) == 2
    assert {x[0] for x in calls} == {id(fit)}
    assert {x[1] for x in calls} == {fit.fit_id}
    assert {calls[0][2], calls[1][2]} == {(p.first.key,), (p.second.key,)}


def test_dense_checkpoint_trajectories_have_equal_total_weight(points):
    sparse = [p for p in points if p.rung != 132 or p.T == 25000]
    weights = a2.response_weights(sparse)
    totals = [sum(w for p,w in zip(sparse, weights) if p.run == run) for run in {p.run for p in sparse}]
    np.testing.assert_allclose(totals, np.full(len(totals), 1/len(totals)))


def test_training_mean_requires_both_pair_endpoints(points):
    _, train, test = a2.make_folds(points, "largest_budget")[0]
    a, b = next(p for p in train if p.T==100000), next(p for p in test if p.T==200000)
    pair = a2.Pair("boundary", "I_T", a, b, None, "high", b.y-a.y)
    assert a2.pair_subset([pair], train) == []
    assert a2.pair_subset([pair], train, test) == [pair]
    assert a2.mean_effects([pair])["I_T"] != 0


def test_no_interpolation_extrapolation_or_student_transfer(points):
    train = [p for p in points if p.student == "small" and p.T<200000]
    query = [replace(train[0], T=1e8), replace(train[0], student="unseen")]
    assert np.isnan(a2.interpolate(train, query)).all()


def test_real_canonical_inventory_is_read_without_external_files(monkeypatch):
    opened = []
    original = a2.Path.read_bytes

    def read(path):
        opened.append(str(path))
        return original(path)

    monkeypatch.setattr(a2.Path, "read_bytes", read)
    points, pairs, _, _ = a2.load_inputs()
    assert {a2.Path(p).name for p in opened} == {"development_table.csv", "summary.json"}
    assert len(points) == 582
    assert all(p.n is None for p in points)
    assert all(p.first.T != p.second.T for p in pairs if p.kind == "I_U")
    inventory = a2.four_corners(points)
    assert all(r["exact_rectangles"] == 0 for r in inventory["coverage"])
    with pytest.raises(ValueError, match="non-embedding parameter counts"):
        a2.fit_response([p for p in points if p.T>0 and p.key[2]=="code"], "F_log")


def test_nonzero_exit_on_failure_even_under_python_optimisation():
    result = subprocess.run([sys.executable, "-B", "-O", str(a2.Path(a2.__file__)), "--interval-draws", "0"],
                            capture_output=True, text=True, cwd=a2.ROOT)
    assert result.returncode != 0
    assert "A2 failed" in result.stderr
    assert "At least 100" in result.stderr
