"""A4 fixed-median residual fit, nested selection, and published V92 integrity."""
from copy import deepcopy
from dataclasses import replace
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from analysis import a4_shrunk_source_correction as a4
from analysis import v92_input_comparison as v92


def panel(arm="pruning"):
    rows = []
    for size, step, config in itertools.product(range(3), range(3), v92.COMMON[arm]):
        if arm == "pruning":
            compression = {"density": float(config)}
        elif arm == "per_channel_quantization":
            compression = {"bits": float(config)}
        else:
            bits, group = config.split("_g")
            compression = {"bits": float(bits[1:]), "group_size": float(group)}
        raw = {"N0": 100. * (size + 1), "D0": 1000. * (step + 1), **compression,
               "L0": 2 + .1 * size + .05 * step, "B": .2 * size - .1 * step,
               "V": .5 + .02 * size + .01 * step ** 2, "W": 3 + .2 * size + .1 * step}
        state = f"size{size}-step{step}"
        query = v92.Query(f"{arm}|{state}|{config}", state, f"size{size}", "math", arm, config,
                          {k: v92.Feature(v, v92.ORIGINS[k]) for k, v in raw.items()})
        strength = compression["density"] if arm == "pruning" else 1 / compression["bits"]
        target = (1 - strength) ** 2 * (1 + .5 * size) + .03 * step
        rows.append(v92.Observation(query, target))
    return rows


@pytest.mark.parametrize("arm", v92.ARMS)
@pytest.mark.parametrize("scheme", a4.SCHEMES)
def test_infinite_shrinkage_is_bit_exact_median_and_zero_paired_gain(arm, scheme):
    _, training, testing = next(v92.make_folds(panel(arm), scheme))
    baseline = v92.fit(training, "K0", "median_curve")
    frozen = deepcopy(baseline)
    model = a4.fit(training, median=baseline, alpha=float("inf"))
    queries = [r.query for r in testing]
    actual, expected = a4.predict(model, queries), v92.predict(baseline, queries)
    np.testing.assert_array_equal(actual, expected)
    assert model["coefficients"] == [0.] * len(model["coefficients"])
    assert model["median"] == baseline == frozen
    effect = v92.paired(testing, expected, actual)
    assert effect["estimate"] == 0
    assert all(v == 0 for v in effect["source_means"].values())
    if effect["n_clusters"] > 1:
        assert effect["ci95"] == [0., 0.]


def test_all_coefficients_including_intercept_shrink_and_zero_limit_is_residual_ols():
    # The old ridge solver's unpenalized intercept would incorrectly stay at 4.
    x = np.column_stack((np.ones(6), [-2., -1., 0., 0., 1., 2.]))
    residual = np.full(6, 4.)
    assert a4.correction_coefficients(x, residual, 1)[0] == pytest.approx(2.)
    assert a4.correction_coefficients(x, residual, 1000)[0] == pytest.approx(4 / 1001)
    np.testing.assert_array_equal(a4.correction_coefficients(x, residual, "infinity"), [0., 0.])
    np.testing.assert_allclose(a4.correction_coefficients(x, residual, 0),
                               np.linalg.lstsq(x, residual, rcond=None)[0], atol=1e-14)
    for bad in (-1, float("nan"), float("-inf")):
        with pytest.raises(ValueError, match="Invalid shrinkage"):
            a4.correction_coefficients(x, residual, bad)


@pytest.mark.parametrize("scheme", a4.SCHEMES)
def test_shrinkage_and_every_fitted_quantity_use_outer_training_only(scheme):
    rows = panel()
    held, training, testing = next(v92.make_folds(rows, scheme))
    group = "size" if scheme == a4.SCHEMES[1] else "state"
    expected = a4.fit(training, group)
    test_ids = {r.query.row_id for r in testing}
    poisoned = [replace(r, target=-1e20, query=replace(r.query, inputs={
        k: replace(f, value=f.value + 1e10) for k, f in r.query.inputs.items()}))
        if r.query.row_id in test_ids else r for r in rows]
    held_again, train_again, _ = next(v92.make_folds(poisoned, scheme))
    assert held_again == held
    assert a4.fit(train_again, group) == expected
    train_ids = {r.query.row_id for r in training}
    for fold in expected["selection"]["folds"]:
        inner_ids = set(fold["median"]["training_ids"])
        validation_ids = set(fold["validation_ids"])
        assert inner_ids == set(fold["standardizer"]["training_ids"])
        assert inner_ids.isdisjoint(test_ids | validation_ids)
        assert validation_ids.isdisjoint(test_ids)
        assert inner_ids | validation_ids == train_ids
        inner_rows = [r for r in training if r.query.row_id in inner_ids]
        inner_groups = {getattr(r.query, group) for r in inner_rows}
        val_groups = {getattr(r.query, group) for r in training if r.query.row_id in validation_ids}
        assert inner_groups.isdisjoint(val_groups)
        np.testing.assert_allclose(fold["standardizer"]["center"],
                                   v92.design([r.query for r in inner_rows], a4.BUDGET).mean(axis=0))


@pytest.mark.parametrize("group", ("state", "size"))
def test_inner_scores_match_independent_frozen_median_residual_calculation(group):
    scheme = a4.SCHEMES[0 if group == "state" else 1]
    _, training, _ = next(v92.make_folds(panel(), scheme))
    selected, audit = a4.choose_shrinkage(training, group)
    scores = []
    for alpha in a4.ALPHAS:
        source_errors = {}
        for held in sorted({getattr(r.query, group) for r in training}):
            tr = [r for r in training if getattr(r.query, group) != held]
            va = [r for r in training if getattr(r.query, group) == held]
            # Independent construction, original y only; no outer median.
            medians = {config: np.median([r.target for r in tr if r.query.config == config])
                       for config in v92.COMMON["pruning"]}
            raw = np.array([[f.value for f in r.query.inputs.values()] for r in tr])
            rawv = np.array([[f.value for f in r.query.inputs.values()] for r in va])
            center, scale = raw.mean(axis=0), raw.std(axis=0)
            scale = np.where(scale > 1e-14 * np.maximum(1, abs(center)), scale, 1.)
            x = np.column_stack((np.ones(len(tr)), (raw - center) / scale))
            xv = np.column_stack((np.ones(len(va)), (rawv - center) / scale))
            residual = np.array([r.target - medians[r.query.config] for r in tr])
            beta = np.zeros(x.shape[1]) if np.isinf(alpha) else np.linalg.solve(
                x.T @ x + len(tr) * alpha * np.eye(x.shape[1]), x.T @ residual)
            predictions = [medians[r.query.config] for r in va] + xv @ beta
            for row, p in zip(va, predictions):
                source_errors.setdefault(row.query.state, []).append(abs(row.target - p))
        scores.append(np.mean([np.mean(v) for v in source_errors.values()]))
    np.testing.assert_allclose(audit["mae_nats"], scores, atol=1e-12, rtol=1e-10)
    best = min(range(len(scores)), key=lambda i: (scores[i], -a4.ALPHAS[i]))
    assert selected == a4.ALPHAS[best]


def test_median_is_fit_once_per_fold_to_original_targets_and_never_to_residuals(monkeypatch):
    rows = panel()
    original = {r.query.row_id: r.target for r in rows}
    real_fit = v92.fit
    calls = []

    def watched(training, budget, form, *args, **kwargs):
        assert budget == "K0" and form == "median_curve"
        assert all(r.target == original[r.query.row_id] for r in training)
        result = real_fit(training, budget, form, *args, **kwargs)
        calls.append((result, deepcopy(result)))
        return result

    monkeypatch.setattr(v92, "fit", watched)
    model = a4.fit(rows)
    assert len(calls) == 1 + len({r.query.state for r in rows})
    assert all(actual == frozen for actual, frozen in calls)
    expected_residual = np.array([r.target for r in rows]) - v92.predict(
        model["median"], [r.query for r in rows])
    np.testing.assert_array_equal(model["residual_targets"], expected_residual)
    assert not np.array_equal(expected_residual, [r.target for r in rows])


def test_exact_inner_ties_choose_infinity():
    rows = [replace(r, target=float(r.query.config)) for r in panel()]
    selected, audit = a4.choose_shrinkage(rows)
    assert selected == float("inf")
    assert audit["mae_nats"] == [0.] * len(a4.ALPHAS)


def test_budget_refuses_new_or_compressed_inputs_even_at_infinity():
    rows = panel()
    model = a4.fit(rows, alpha="infinity")
    q = rows[0].query
    bad = replace(q, inputs={**q.inputs, "orthogonal_residual_variance": v92.Feature(1., "compressed_model")})
    with pytest.raises(ValueError, match="Budget separation"):
        a4.predict(model, [bad])
    bad = replace(q, inputs={**q.inputs, "B": v92.Feature(1., "compressed_model")})
    with pytest.raises(ValueError, match="Pre-compression provenance"):
        a4.predict(model, [bad])


@pytest.fixture(scope="module")
def published():
    return a4.read_published()[0]


@pytest.fixture(scope="module")
def real_panel():
    return v92.load_data()


def test_baseline_numbers_are_read_and_preserved_not_recomputed(published, real_panel, monkeypatch):
    saved = deepcopy(published)
    old = saved["evaluations"]["pruning"][a4.SCHEMES[0]]["capabilities"]["math"]
    # A recomputed MAE object would lose this publication marker.
    old["fits"]["K0/median_curve"]["mae_nats"]["read_through_marker"] = "published"

    def forbidden(*args, **kwargs):
        raise AssertionError("A4 must not rerun V92's evaluation or summary")

    monkeypatch.setattr(v92, "evaluate", forbidden)
    monkeypatch.setattr(v92, "build_summary", forbidden)
    result = a4.evaluate(real_panel[0], "pruning", "math", a4.SCHEMES[0], saved)
    assert result["median_mae_nats"] == old["fits"]["K0/median_curve"]["mae_nats"]
    assert result["median_mae_nats"]["read_through_marker"] == "published"
    assert result["median_predictions"] == old["fits"]["K0/median_curve"]["predictions"]
    for form in v92.LINEAR_FORMS:
        assert result["published_whole_response_mae_nats"][form] == old["fits"][f"{a4.BUDGET}/{form}"]["mae_nats"]


def test_real_panel_and_provenance_match_publication_and_drift_fails(published, real_panel):
    rows, _, provenance = real_panel
    a4.validate_panel(rows, provenance, published)
    assert len(rows) == len([r for r in published["observations"] if r["primary"]]) == 459
    changed = [replace(rows[0], target=rows[0].target + 1), *rows[1:]]
    with pytest.raises(ValueError, match="observation changed"):
        a4.validate_panel(changed, provenance, published)
    bad_provenance = deepcopy(provenance)
    bad_provenance["input_sha256"][next(iter(bad_provenance["input_sha256"]))] = "changed"
    with pytest.raises(ValueError, match="input hashes changed"):
        a4.validate_panel(rows, bad_provenance, published)


def test_corrupted_saved_baseline_fails_instead_of_silently_replacing_it(published, real_panel):
    saved = deepcopy(published)
    old = saved["evaluations"]["pruning"][a4.SCHEMES[0]]["capabilities"]["math"]
    old["fits"]["K0/median_curve"]["folds"][0]["model"]["anchors"]["0.6"] += 1
    with pytest.raises(ValueError, match="Published median mismatch"):
        a4.evaluate(real_panel[0], "pruning", "math", a4.SCHEMES[0], saved)


@pytest.fixture(scope="module")
def result():
    return a4.build_summary()


def test_full_run_reuses_published_numbers_folds_and_source_cluster_scoring(result, published, real_panel):
    rows = {r.query.row_id: r for r in real_panel[0]}
    assert len(result["evaluations"]) == 18
    assert result["protocol"]["copied_v92_functions"] == []
    assert result["protocol"]["alpha_grid"] == [1., 10., 100., 1000., 10000., 1000000., "infinity"]
    assert result["published_v92"]["sha256"] == a4.sha256(a4.ROOT / a4.PUBLISHED)
    for e in result["evaluations"]:
        old = published["evaluations"][e["arm"]][e["scheme"]]["capabilities"][e["capability"]]
        median = old["fits"]["K0/median_curve"]
        assert e["row_order"] == old["row_order"]
        assert e["median_mae_nats"] == median["mae_nats"]
        assert e["median_predictions"] == median["predictions"]
        assert e["infinite_shrinkage"]["mae_nats"] == median["mae_nats"]
        assert e["infinite_shrinkage"]["paired_gain_nats"]["estimate"] == 0
        assert e["infinite_shrinkage"]["paired_gain_nats"]["ci95"] == [0., 0.]
        ordered = [rows[row_id] for row_id in e["row_order"]]
        assert e["paired_gain_nats"] == v92.paired(ordered, median["predictions"], e["predictions"])
        assert e["paired_gain_nats"]["n_clusters"] == median["mae_nats"]["n_clusters"] == 9
        for fold, saved_fold in zip(e["folds"], median["folds"]):
            assert fold["test_ids"] == a4.published_ids(published, saved_fold, "test")
            assert fold["model"]["training_ids"] == a4.published_ids(published, saved_fold["model"], "training")
        assert e["positive_gain_ci_excludes_zero"] == (e["paired_gain_nats"]["ci95"][0] > 0)
    # Strict JSON encoding must never write a nonstandard Infinity or NaN token.
    assert json.loads(json.dumps(result, allow_nan=False))["decision"] == result["decision"]
    body = a4.markdown(result)
    assert result["earlier_corrections"] in body
    assert "not a demonstration that the residual is random noise" in body
    assert "not a pre-compression predictor because computing it needs the compressed model" in body
    if result["decision"]["status"] == "CLOSED":
        assert "This branch is CLOSED." in body


def test_fixed_majority_gate_does_not_count_size_only_wins_or_zero_intervals():
    evaluations = [{"arm": arm, "capability": cap, "scheme": scheme,
                    "positive_gain_ci_excludes_zero": scheme == a4.SCHEMES[1],
                    "maximum_shrinkage_folds": 0, "n_folds": 3}
                   for scheme in a4.SCHEMES for arm in v92.ARMS for cap in v92.CAPS]
    assert a4.decision(evaluations)["status"] == "CLOSED"
    for e in evaluations[:4]:
        e["positive_gain_ci_excludes_zero"] = True
    assert a4.decision(evaluations)["status"] == "CLOSED"
    evaluations[4]["positive_gain_ci_excludes_zero"] = True
    assert a4.decision(evaluations)["status"] == "MAJORITY_CRITERION_MET"
    for e in evaluations:
        e["maximum_shrinkage_folds"] = 2
    d = a4.decision(evaluations)
    assert d["status"] == "CLOSED"
    assert d["selection_reading"] == "The data prefers the median curve."


def test_cli_exits_nonzero_on_integrity_failure_and_writes_nothing(tmp_path):
    # A readable but invalid publication exercises the validation failure path.
    path = tmp_path / a4.PUBLISHED
    path.parent.mkdir(parents=True)
    path.write_text('{"version": "corrupted"}')
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OPENBLAS_NUM_THREADS": "1",
           "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    process = subprocess.run([sys.executable, "-B", str(Path(a4.__file__)), "--root", str(tmp_path)],
                             capture_output=True, text=True, env=env, timeout=30)
    assert process.returncode != 0
    assert "Expected published V92 summary" in process.stderr
    assert not (tmp_path / a4.OUT).exists()
    assert not (tmp_path / a4.REPORT).exists()
