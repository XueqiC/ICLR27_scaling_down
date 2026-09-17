"""V92 input provenance, nested holdouts, source-free baselines and fixed gate."""
from dataclasses import replace
from collections import Counter
import itertools
import json
from pathlib import Path

import numpy as np
import pytest

from analysis import v92_input_comparison as v92


def panel():
    rows = []
    for si, step, density in itertools.product(range(3), range(3), (.6, .7, .8, .9)):
        state = f"size{si}-step{step}"
        raw = {"N0": (si + 1) * 100., "D0": (step + 1) * 1000., "density": density,
               "L0": 2 + .1 * si + .05 * step, "B": .2 * si - .1 * step,
               "V": .5 + .02 * si + .01 * step ** 2, "W": 3 + .2 * si + .1 * step}
        query = v92.Query(f"{state}|{density}", state, f"size{si}", "math", "pruning", str(density),
                          {k: v92.Feature(v, v92.ORIGINS[k]) for k, v in raw.items()})
        target = (1 - density) ** 2 * (1 + .5 * si) + .03 * step
        rows.append(v92.Observation(query, target))
    return rows


@pytest.mark.parametrize("budget", v92.BUDGETS)
@pytest.mark.parametrize("quantity", ("compressed_loss", "delta_L", "eps_hat", "rounding_error",
                                      "quantization_statistics", "compressed_V", "logit_displacement"))
def test_budget_refuses_compressed_model_quantity(budget, quantity):
    query = panel()[0].query
    poisoned = replace(query, inputs={**query.inputs, quantity: v92.Feature(1., "compressed_model")})
    with pytest.raises(ValueError, match="Budget separation"):
        v92.design([poisoned], budget)


@pytest.mark.parametrize("key", ("N0", "D0", "density", "L0", "B", "V", "W"))
def test_renaming_compressed_quantity_cannot_evade_provenance_check(key):
    query = panel()[0].query
    query = replace(query, inputs={**query.inputs, key: v92.Feature(1., "compressed_model")})
    with pytest.raises(ValueError, match="Pre-compression provenance"):
        v92.budget_inputs(query, "dense_statistics")


def test_budgets_are_strict_nested_inputs_without_response_or_hidden_transforms():
    query = panel()[0].query
    k0 = v92.budget_inputs(query, "K0")
    anchor = v92.budget_inputs(query, "dense_anchor")
    stats = v92.budget_inputs(query, "dense_statistics")
    assert set(k0) == {"N0", "D0", "density"}
    assert set(anchor) - set(k0) == {"L0"}
    assert set(stats) - set(anchor) == {"B", "V", "W"}
    with pytest.raises(ValueError, match="Budget separation"):
        v92.validate_inputs(anchor, v92.fields("pruning", "K0"))
    assert "target" not in query.__dataclass_fields__
    np.testing.assert_array_equal(v92.design([query], "K0"), [[100., 1000., .6]])
    changed = replace(query, inputs={**query.inputs, **{
        k: v92.Feature(1e10, v92.ORIGINS[k]) for k in ("L0", "B", "V", "W")}})
    np.testing.assert_array_equal(v92.design([query], "K0"), v92.design([changed], "K0"))


@pytest.mark.parametrize("scheme", ("leave_one_source_state_out", "leave_one_size_out"))
@pytest.mark.parametrize("form", ("ols", "ridge"))
def test_outer_held_out_values_and_outcomes_never_affect_scaler_fit_or_penalty(scheme, form):
    rows = panel()
    held, train, test = next(v92.make_folds(rows, scheme))
    inner_group = "size" if scheme == "leave_one_size_out" else "state"
    expected = v92.fit(train, "dense_statistics", form, inner_group)
    held_ids = {r.query.row_id for r in test}
    poisoned = []
    for row in rows:
        if row.query.row_id in held_ids:
            query = replace(row.query, inputs={key: replace(value, value=value.value + 1e9)
                                               for key, value in row.query.inputs.items()})
            row = replace(row, query=query, target=-1e20)
        poisoned.append(row)
    new_held, new_train, _ = next(v92.make_folds(poisoned, scheme))
    assert held == new_held
    actual = v92.fit(new_train, "dense_statistics", form, inner_group)
    assert expected == actual
    assert set(actual["standardizer"]["training_ids"]).isdisjoint(held_ids)
    np.testing.assert_allclose(actual["standardizer"]["center"],
                               v92.design([r.query for r in train], "dense_statistics").mean(axis=0))
    if form == "ridge":
        for inner in actual["selection"]["folds"]:
            validation = set(inner["validation_ids"])
            training = set(inner["standardizer"]["training_ids"])
            assert training.isdisjoint(validation | held_ids)
            assert validation.isdisjoint(held_ids)
            assert training | validation == set(actual["training_ids"])
            own_rows = [r.query for r in train if r.query.row_id in training]
            np.testing.assert_allclose(inner["standardizer"]["center"],
                                       v92.design(own_rows, "dense_statistics").mean(axis=0))


def test_inner_penalty_scores_match_independent_training_only_ridge_calculation():
    rows = panel()
    _, training, _ = next(v92.make_folds(rows, "leave_one_size_out"))
    chosen, audit = v92.choose_penalty(training, "dense_anchor", "size")
    scores = []
    for alpha in v92.ALPHAS:
        source_errors = {}
        for held in {r.query.size for r in training}:
            tr = [r for r in training if r.query.size != held]
            va = [r for r in training if r.query.size == held]
            raw = np.array([[r.query.inputs[k].value for k in ("N0", "D0", "density", "L0")] for r in tr])
            rawv = np.array([[r.query.inputs[k].value for k in ("N0", "D0", "density", "L0")] for r in va])
            center, scale = raw.mean(axis=0), raw.std(axis=0)
            scale[scale == 0] = 1
            x = np.column_stack((np.ones(len(tr)), (raw - center) / scale))
            xv = np.column_stack((np.ones(len(va)), (rawv - center) / scale))
            penalty = np.diag([0., *([alpha * len(tr)] * 4)])
            beta = np.linalg.solve(x.T @ x + penalty, x.T @ [r.target for r in tr])
            for row, error in zip(va, abs(xv @ beta - [r.target for r in va])):
                source_errors.setdefault(row.query.state, []).append(error)
        scores.append(np.mean([np.mean(e) for e in source_errors.values()]))
    np.testing.assert_allclose(audit["mae"], scores, rtol=1e-10, atol=1e-12)
    assert chosen == v92.ALPHAS[min(range(len(scores)), key=lambda i: (scores[i], -v92.ALPHAS[i]))]


def test_median_is_source_free_and_each_training_source_has_one_vote():
    rows = panel()
    train = [r for r in rows if r.query.state != rows[0].query.state]
    model = v92.fit(train, "K0", "median_curve")
    query = rows[0].query
    other = replace(query, state="unseen-source", size="unseen-size", inputs={
        key: replace(value, value=value.value + 99999) if key != "density" else value
        for key, value in query.inputs.items()})
    predictions = v92.predict(model, [query, other])
    assert predictions[0] == predictions[1]
    assert predictions[0] == np.median([r.target for r in train if r.query.config == query.config])
    duplicate_source = [r for r in train if r.query.state == train[0].query.state] * 20
    assert v92.fit(train + duplicate_source, "K0", "median_curve")["anchors"] == model["anchors"]
    for budget in v92.BUDGETS:
        np.testing.assert_array_equal(v92.predict(v92.fit(train, budget, "median_curve"), [query]), predictions[:1])
        np.testing.assert_array_equal(v92.predict(v92.fit(train, budget, "delivered"), [query]), predictions[:1])


@pytest.mark.parametrize("gain,ci,expected", [(.125, [0., .25], False), (.25, [0., .25], False),
                                            (.25001, [0., .25], True), (.04, [.01, .07], False),
                                            (0., [0., 0.], False), (-.1, [-.2, -.1], False)])
def test_reading_rule_uses_strict_full_interval_width(gain, ci, expected):
    effect = {"estimate": gain, "ci95": ci, "interval_width": ci[1] - ci[0], "n_clusters": 9}
    assert v92.reading_rule(effect, scheme="leave_one_source_state_out", lower_form="ols", upper_form="ols",
                            lower_budget="dense_anchor", upper_budget="dense_statistics") is expected


@pytest.mark.parametrize("change", ({"scheme": "unseen_strength_same_sources"}, {"scheme": "leave_one_size_out"},
                                    {"upper_form": "ridge"}, {"lower_budget": "K0"}))
def test_within_source_or_different_form_or_wrong_budget_cannot_clear_rule(change):
    effect = {"estimate": 1., "interval_width": .1, "n_clusters": 9}
    kwargs = dict(scheme="leave_one_source_state_out", lower_form="ols", upper_form="ols",
                  lower_budget="dense_anchor", upper_budget="dense_statistics")
    assert not v92.reading_rule(effect, **{**kwargs, **change})


def test_cluster_interval_pairs_state_means_not_pseudoreplicated_rows():
    queries = [r.query for r in panel()[:5]]  # four rows of state A, one of B
    result = v92.clustered([1., 1., 1., 1., 3.], queries)
    assert result["n_clusters"] == 2
    assert result["estimate"] == 2.
    assert result["ci95"] == [1., 3.]
    rows = [v92.Observation(q, 0.) for q in queries]
    effect = v92.paired(rows, [2.] * 5, [1.] * 5)
    assert effect["estimate"] == 1.
    assert effect["interval_width"] == 0.


@pytest.mark.parametrize("scheme", ("unseen_strength_same_sources", "unseen_strength_new_source"))
def test_unseen_strength_is_absent_from_every_training_source(scheme):
    rows = panel()
    test_ids = []
    for _, train, test in v92.make_folds(rows, scheme):
        assert {r.query.strength for r in train}.isdisjoint({r.query.strength for r in test})
        if scheme.endswith("new_source"):
            assert {r.query.state for r in train}.isdisjoint({r.query.state for r in test})
        test_ids.extend(r.query.row_id for r in test)
    assert sorted(test_ids) == sorted(r.query.row_id for r in rows)


def test_median_interpolates_without_using_held_strength_target():
    rows = panel()
    train = [r for r in rows if r.query.strength != .7]
    query = next(r.query for r in rows if r.query.strength == .7)
    model = v92.fit(train, "K0", "median_curve")
    expected = .5 * (model["anchors"]["0.6"] + model["anchors"]["0.8"])
    assert v92.predict(model, [query])[0] == pytest.approx(expected)


def test_fragility_uses_signed_change_and_fair_ties():
    query = panel()[0].query
    rows = [v92.Observation(replace(query, capability=cap, row_id=cap), target)
            for cap, target in zip(v92.CAPS, (-5., -.2, -.1))]
    zero = v92.fragility(rows, dict.fromkeys(v92.CAPS, 0.))
    assert zero["top1_accuracy"]["estimate"] == pytest.approx(1 / 3)
    correct = v92.fragility(rows, dict(zip(v92.CAPS, (-5., -.2, -.1))))
    assert correct["top1_accuracy"]["estimate"] == 1
    assert correct["regret_nats"]["estimate"] == 0
    assert correct["cells"][0]["actual_most_fragile"] == ["qa"]


@pytest.mark.parametrize("snapshot", ("current", "legacy"))
def test_real_loader_uses_only_requested_sources_and_common_development_grid(snapshot, tmp_path):
    root = v92.ROOT
    artifact = root / "results/v92-input-comparison/summary.json"
    if snapshot == "legacy":
        # Replay the published mirror against its own registered input files.
        mirror = root / "data_mirror"
        artifact = mirror / "v92-input-comparison-sixstate-historical/summary.json"
    published = json.loads(artifact.read_text())
    if snapshot == "legacy":
        root = tmp_path
        for relative in published["provenance"]["input_sha256"]:
            destination = root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.symlink_to(mirror / Path(relative).relative_to("results"))

    rows, extra, provenance = v92.load_data(root)
    primary = [r for r in published["observations"] if r["primary"]]
    assert len(primary) == (459 if snapshot == "current" else 378)
    assert len(provenance["input_sha256"]) == (54 if snapshot == "current" else 51)
    additional = [r for r in published["observations"] if not r["primary"]]
    assert Counter(r.query.arm for r in rows) == Counter(r["arm"] for r in primary)
    assert {r.query.row_id for r in rows} == {r["id"] for r in primary}
    for arm in v92.ARMS:
        assert {r.query.state for r in rows if r.query.arm == arm} == {
            r["state"] for r in primary if r["arm"] == arm}
    assert len(extra) == len(additional)
    assert provenance["missing_states"] == published["provenance"]["missing_states"]
    assert {r.query.config for r in extra} == {r["config"] for r in additional}
    assert all(r.query.config in v92.COMMON[r.query.arm] for r in rows)
    assert not any("freeze" in p or "confirm" in p for p in provenance["input_sha256"])
    assert set(provenance["input_sha256"]) == set(published["provenance"]["input_sha256"])
    if snapshot == "legacy":
        # Preserve missing-state coverage even when the public mirror catches up.
        relative = next(p for p in provenance["input_sha256"] if "/v54-quant-group/" in p)
        (root / relative).unlink()  # Only the temporary fixture symlink.
        reduced, _, missing = v92.load_data(root)
        state = Path(relative).parent.name
        expected_missing = [*provenance["missing_states"],
            {"arm": "grouped_quantization", "state": state, "path": relative}]
        assert sorted(missing["missing_states"], key=lambda r: r["path"]) == sorted(
            expected_missing, key=lambda r: r["path"])
        assert len(reduced) == len(rows) - 3 * len(v92.COMMON["grouped_quantization"])
        assert not any(r.query.arm == "grouped_quantization" and r.query.state == state for r in reduced)
        pruning = next(p for p in provenance["input_sha256"] if "/v6-capability-geometry/" in p)
        (root / pruning).unlink()
        with pytest.raises(ValueError, match="Required pruning panel missing"):
            v92.load_data(root)


def test_delivered_respects_locked_new_source_domain():
    rows = panel()
    model = v92.fit(rows, "dense_anchor", "delivered")
    query = rows[0].query
    outside = replace(query, config="0.55", inputs={**query.inputs,
                       "density": v92.Feature(.55, "compression_configuration")})
    assert v92.predict(model, [outside]) == [None]
    for arm in ("grouped_quantization", "per_channel_quantization"):
        keys = {k: v for k, v in query.inputs.items() if k != "density"}
        keys["bits"] = v92.Feature(3., "compression_configuration")
        if arm == "grouped_quantization":
            keys["group_size"] = v92.Feature(64., "compression_configuration")
        q = replace(query, arm=arm, config="b3_g64" if arm == "grouped_quantization" else "3", inputs=keys)
        model = {"budget": "dense_anchor", "form": "delivered", "arm": arm,
                 "anchors": {"b4_g64": .2, "b5_g64": .1} if arm == "grouped_quantization" else {"4": .2, "6": .1}}
        assert v92.predict(model, [q]) == [None]


def test_additive_input_form_decomposition_and_identical_scoring_rows():
    rows = []
    for cap, offset in zip(v92.CAPS, (0., .1, -.1)):
        for row in panel():
            rows.append(replace(row, query=replace(row.query, capability=cap,
                                                  row_id=row.query.row_id + "|" + cap), target=row.target + offset))
    result = v92.evaluate(rows, "leave_one_source_state_out")
    for cap, cr in result["capabilities"].items():
        assert len(cr["row_order"]) == 36
        assert len(set(cr["row_order"])) == 36
        assert all(f["coverage"] == 1 and f["mae_nats"]["n_clusters"] == 9 for f in cr["fits"].values())
        for d in cr["gain_decomposition"]:
            assert d["total"]["estimate"] == pytest.approx(
                d["input_at_fixed_median"]["estimate"] + d["form_at_upper_budget"]["estimate"])
            assert d["total"]["estimate"] == pytest.approx(
                d["input_at_fixed_linear"]["estimate"] + d["form_at_lower_budget"]["estimate"])
        for gain in cr["input_gains"]:
            if gain["form"] not in v92.LINEAR_FORMS:
                assert gain["estimate"] == gain["interval_width"] == 0


def test_compact_audit_keeps_order_and_exact_membership():
    summary = {"observations": [{"id": "a"}, {"id": "b"}, {"id": "c"}], "protocol": {},
               "evaluations": {"model": {"training_ids": ["b", "a"], "test_ids": ["c"],
                               "standardizer": {"training_ids": ["b", "a"]},
                               "selection": [{"training_ids": ["a"], "validation_ids": ["b"]}]}}}
    v92.compact_audit_rows(summary)
    model, sets = summary["evaluations"]["model"], summary["audit_row_sets"]
    assert model["training_rows_ref"] == model["standardizer"]["training_rows_ref"]
    assert sets[model["training_rows_ref"]] == [1, 0]
    assert sets[model["test_rows_ref"]] == [2]
    assert sets[model["selection"][0]["validation_rows_ref"]] == [1]
