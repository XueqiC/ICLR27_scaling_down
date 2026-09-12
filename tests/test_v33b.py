"""Measured dominance, policy fairness, and whole-model paired uncertainty."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from analysis import v33b_selection_pareto as pareto


def vector(value):
    return dict.fromkeys(pareto.CAPS, value)


def candidate(key, method, cost, actual, predicted=0.):
    delta = vector(actual) if np.isscalar(actual) else actual
    return {"id": key, "method": method, "nominal_storage": cost,
            "actual_loss": {c: 2. + delta[c] for c in pareto.CAPS},
            "actual_delta": delta, "delta_hat": vector(predicted)}


def candidates():
    return [candidate("dense", "dense", 1., 0.),
            candidate("prune", "pruning", .4, .08, .7),
            candidate("quant", "quantization", .25, .6, .05)]


def panel(name="fixture", budgets=(.1, .8)):
    rows = candidates()
    predictions = pareto.v33.prediction_view(rows)
    return {"model": name, "cohort": "development_lomo", "candidates": rows,
            "source_dense": vector(2.),
            "rows": [pareto.v33.score_budget(rows, predictions, vector(t)) for t in budgets]}


def test_measured_frontier_survives_complete_predictor_rejection():
    rows = candidates()
    rows[1]["delta_hat"] = vector(1e9)
    result = pareto.pareto_frontier(rows)
    assert set(result["frontier_ids"]) == {"dense", "prune", "quant"}
    assert result["arms"]["pruning"]["appears_on_actual_frontier"]
    for row in rows:
        row["delta_hat"] = vector(-1e9)
    assert pareto.pareto_frontier(rows) == result


def test_dominance_requires_all_capabilities_and_retains_identical_ties():
    a = candidate("a", "pruning", .4, {"math": .1, "code": .3, "qa": .1})
    b = candidate("b", "quantization", .3, {"math": .2, "code": .2, "qa": .2})
    assert not pareto.dominates(a, b) and not pareto.dominates(b, a)
    twin = {**a, "id": "twin"}
    bad = candidate("bad", "pruning", .5, .5)
    result = pareto.pareto_frontier([a, b, twin, bad])
    assert set(result["frontier_ids"]) == {"a", "b", "twin"}
    dominated = result["candidates"][-1]
    assert dominated["dominated_by"] == ["a", "b", "twin"]
    assert dominated["cross_arm_dominators"] == ["b"]
    # Equal cost and strictly better losses suffice; identical objectives do not.
    assert pareto.dominates(a, {**bad, "nominal_storage": .4})
    assert not pareto.dominates(a, twin)


def test_frontier_and_any_budget_optimality_are_distinct():
    rows = [candidate("dense", "dense", 1., 0.),
            candidate("negative", "pruning", .5, -.2),
            candidate("cheap", "quantization", .25, -.1),
            candidate("loss_dominated_tie", "pruning", .25, .3)]
    result = pareto.pareto_frontier(rows)
    lookup = {r["id"]: r for r in result["candidates"]}
    assert lookup["negative"]["on_frontier"]
    assert not lookup["negative"]["cost_optimal_at_any_nonnegative_budget_including_ties"]
    assert lookup["negative"]["minimal_nonnegative_budget"] == vector(0.)
    assert not lookup["loss_dominated_tie"]["on_frontier"]
    assert lookup["loss_dominated_tie"]["cost_optimal_at_any_nonnegative_budget_including_ties"]
    assert result["arms"]["dense"]["all_configurations_dominated"]


def test_any_budget_witness_finds_region_absent_from_sampled_grid():
    source = panel(budgets=(0., 1.))
    replay = pareto.replay_panel(source)
    assert replay["selector_behavior"]["pruning"]["oracle_budgets"] == 0
    row = next(r for r in replay["pareto"]["candidates"] if r["id"] == "prune")
    assert row["cost_optimal_at_any_nonnegative_budget_including_ties"]
    assert row["minimal_nonnegative_budget"] == pytest.approx(vector(.08 - pareto.TOL))
    assert row["cheaper_at_minimal_budget"] == []


def test_policies_share_actual_scoring_and_fixed_distillation_has_no_oracle_rescue():
    source = panel(budgets=(.1,))
    distill = candidate("distill", "distillation", 1., .2)
    distill.pop("delta_hat")  # No prediction is invented for this fixed baseline.
    result = pareto.replay_panel(source, distill)
    policies = result["rows"][0]["policies"]
    assert policies["oracle"]["selected"] == "prune"
    assert policies["oracle"]["constraint_satisfied"]
    for name in ("selector", "quant_only", "always_cheapest"):
        assert policies[name]["selected"] == "quant"
        assert not policies[name]["constraint_satisfied"]
        assert policies[name]["nominal_cost_gap_vs_oracle"] == pytest.approx(-.15)
    assert policies["always_dense"]["constraint_satisfied"]
    assert policies["distill_only"]["selected"] == "distill"
    assert not policies["distill_only"]["constraint_satisfied"]
    assert policies["distill_only"]["nominal_storage"] == 1.
    assert result["metrics"]["exclusions/pruning"] == 1.
    source["rows"][0]["policies"]["selector"]["selected"] = "dense"
    with pytest.raises(ValueError, match="Saved V33 decision"):
        pareto.replay_panel(source)


def test_quant_only_uses_frozen_predictions_with_dense_fallback():
    source = panel(budgets=(0.,))
    result = pareto.replay_panel(source)
    assert result["rows"][0]["policies"]["quant_only"]["selected"] == "dense"


def test_matching_uses_uniform_mixtures_and_exact_achieved_rate():
    raised = pareto.match_reliability(.6, .3, .1, .1, .8)
    assert raised["base_policy_probability"] == pytest.approx(.5)
    assert raised["nominal_storage"] == pytest.approx(.65)
    assert raised["constraint_satisfaction"] == pytest.approx(.8)
    assert raised["anchor"] == "always_dense"
    lowered = pareto.match_reliability(1., .4, .1, .1, .8)
    assert lowered["base_policy_probability"] == pytest.approx(7/9)
    assert lowered["nominal_storage"] == pytest.approx(1/3)
    assert lowered["constraint_satisfaction"] == pytest.approx(.8)
    assert lowered["anchor"] == "always_cheapest"
    assert pareto.match_reliability(.6, .3, .1, .1, 1.)["nominal_storage"] == 1.
    assert np.isnan(pareto.match_reliability(.6, .3, .1, .1, .05)["nominal_storage"])
    equal = pareto.match_reliability(.1, .2, .1, .1, .1)
    assert equal["base_policy_probability"] == 1. and equal["nominal_storage"] == .2
    for target in (-.1, 1.1, float("nan")):
        with pytest.raises(ValueError, match="Target satisfaction"):
            pareto.match_reliability(.6, .3, .1, .1, target)


def test_cluster_bootstrap_pairs_ratios_contrasts_and_recomputes_matching(monkeypatch):
    models = [pareto.replay_panel(panel(name)) for name in ("a", "b")]
    for i, model in enumerate(models):
        model["n_budgets"] = (10, 1000)[i]  # Size must not determine cluster mass.
        totals = model["metric_numerator_denominator"]
        for policy in ("selector", "quant_only"):
            totals[f"{policy}/constraint_satisfaction"] = [(.2, .6)[i], 1.]
            totals[f"{policy}/nominal_storage"] = [(.2, .4)[i], 1.]
        totals["exclusions/pruning"] = [float(i), float(1 + 2*i)]
        totals["selector_arm/pruning/constraint_satisfaction"] = [0., 0.]
    original = pareto.audit.bootstrap_means
    calls = []

    def inspect(values, groups, **kwargs):
        draws = original(values, groups, **kwargs)
        calls.append((np.asarray(values), groups, draws))
        return draws

    monkeypatch.setattr(pareto.audit, "bootstrap_means", inspect)
    result = pareto.aggregate(models, n_boot=1000, targets=(.8,))
    assert len(calls) == 1 and calls[0][1] == ["a", "b"]
    assert calls[0][0].shape[0] == 2
    assert result["metrics"]["selector/constraint_satisfaction"]["estimate"] == .4
    exclusion = result["metrics"]["exclusions/pruning"]
    assert exclusion["estimate"] == .25  # Ratio of totals, not mean(0, 1/3).
    assert exclusion["ci95"] == pytest.approx([0., 1/3])
    undefined = result["metrics"]["selector_arm/pruning/constraint_satisfaction"]
    assert undefined["estimate"] is None and undefined["undefined_bootstrap_draws"] == 1000
    gap = result["matched_reliability"][0]["selector_minus_baseline_nominal_storage"]["quant_only"]
    assert gap["estimate"] == 0. and gap["ci95"] == [0., 0.]
    # Reconstruct the nonlinear matching statistic from the SAME model draws.
    keys = list(models[0]["metric_numerator_denominator"])
    draws = calls[0][2].reshape(1000, len(keys), 2)
    s = draws[:, keys.index("selector/constraint_satisfaction"), 0]
    c = draws[:, keys.index("selector/nominal_storage"), 0]
    expected_cost = 1. - .2 * (1. - c) / (1. - s)
    matched = result["matched_reliability"][0]["policies"]["selector"]["nominal_storage"]
    assert matched["ci95"] == pytest.approx(pareto.audit.interval(expected_cost))
    assert matched["estimate"] == pytest.approx(1 - .2*.7/.6)
    # Duplicating every within-model budget changes no estimates or intervals.
    once = pareto.aggregate([pareto.replay_panel(panel())], n_boot=20)
    duplicate = panel()
    duplicate["rows"] *= 7
    many = pareto.aggregate([pareto.replay_panel(duplicate)], n_boot=20)
    for key, value in once["metrics"].items():
        other = many["metrics"][key]
        assert other["estimate"] == (pytest.approx(value["estimate"]) if value["estimate"] is not None else None)
        assert other["ci95"] == (pytest.approx(value["ci95"]) if value["ci95"] is not None else None)
        assert other["undefined_bootstrap_draws"] == value["undefined_bootstrap_draws"]
    json.dumps(result, allow_nan=False)


@pytest.fixture(scope="module")
def saved_summary():
    return pareto.build_summary(n_boot=200)


def test_existing_artifacts_reproduce_v33_and_true_frontier(saved_summary):
    saved = pareto.audit.read_json(pareto.SOURCE)
    main = saved_summary["cohorts"]["development_lomo"]
    assert (main["n_models"], main["n_budgets"]) == (12, 2210)
    for key in ("selector/constraint_satisfaction", "selector/nominal_storage",
                "selector/nominal_cost_gap_vs_oracle", "exclusions/pruning", "exclusions/quantization"):
        assert main["metrics"][key]["estimate"] == pytest.approx(saved["cohorts"]["development_lomo"]["metrics"][key]["estimate"])
    assert main["selector_behavior"]["pruning"] == {
        "selected_budgets": 31, "satisfying_selected_budgets": 0, "oracle_budgets": 81}
    assert main["actual_frontier"]["pruning"]["frontier_configurations"] == 24
    assert len(main["actual_frontier"]["pruning"]["models_on_frontier"]) == 6
    assert len(main["actual_frontier"]["pruning"]["models_all_configurations_dominated"]) == 6
    assert main["metrics"]["oracle/constraint_satisfaction"]["estimate"] == 1.
    for cohort in saved_summary["cohorts"].values():
        for row in cohort["matched_reliability"]:
            for policy in row["policies"].values():
                satisfaction = policy["constraint_satisfaction"]
                assert satisfaction["estimate"] == pytest.approx(row["target_constraint_satisfaction"])
                assert satisfaction["ci95"] == pytest.approx([row["target_constraint_satisfaction"]]*2)
    assert len(saved_summary["separate_offline_transfer"]) == 1
    json.dumps(saved_summary, allow_nan=False)


def test_distillation_is_same_source_subset_with_no_missing_model_imputation(saved_summary):
    covered = saved_summary["cohorts"]["distillation_covered_subset"]
    assert covered["n_models"] == 8
    assert len(saved_summary["distillation"]["missing_models"]) == 4
    assert "distill_only" not in saved_summary["cohorts"]["development_lomo"]["policies"]
    for model in saved_summary["distillation_subset_models"]:
        distill = model["candidates"][-1]
        assert distill["method"] == "distillation" and distill["nominal_storage"] == 1.
        assert f"/{model['model']}/" in distill["source_path"]
        raw = pareto.audit.read_json(pareto.ROOT / distill["source_path"])
        for cap in pareto.CAPS:
            assert distill["actual_delta"][cap] == pytest.approx(raw["post_training"][cap] - model["source_dense"][cap])
        assert model["metrics"]["distill_only/nominal_storage"] == 1.
    report = pareto.render(saved_summary)
    for text in ("24 measured Pareto-frontier configurations", "No latency claims", "no new runs",
                 "bootstrap", "8 models", "not size reduction", "Matched satisfaction"):
        assert text in report


def test_provenance_and_measured_endpoint_mismatch_are_rejected(monkeypatch):
    original = pareto.audit.read_json
    source = original(pareto.SOURCE)
    bad = copy.deepcopy(source)
    key = next(iter(bad["input_sha256"]))
    bad["input_sha256"][key] = "wrong"
    monkeypatch.setattr(pareto.audit, "read_json", lambda p: bad if p == pareto.SOURCE else original(p))
    with pytest.raises(ValueError, match="provenance mismatch"):
        pareto.checked_source()
    bad = copy.deepcopy(source)
    bad["models"][0]["candidates"][0]["actual_loss"]["qa"] += 1.
    with pytest.raises(ValueError, match="measured loss/cost mismatch"):
        pareto.checked_source()


def test_dry_run_never_writes_and_import_never_loads_gpu(monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        pytest.fail("Dry run attempted a write or predictor evaluation")

    monkeypatch.setattr(Path, "write_text", forbidden)
    monkeypatch.setattr(pareto.audit, "write_outputs", forbidden)
    monkeypatch.setattr(pareto.v33, "frozen_prediction", forbidden)
    monkeypatch.setattr(pareto.v33, "load_panels", forbidden)
    pareto.main(["--dry-run", "--bootstrap", "2"])
    assert "DRY RUN: no files written" in capsys.readouterr().out
    process = subprocess.run([sys.executable, "-c", "import sys; from analysis import v33b_selection_pareto; "
                              "assert not {'torch', 'transformers', 'cupy'} & set(sys.modules)"],
                             cwd=pareto.ROOT, capture_output=True, text=True)
    assert process.returncode == 0, process.stderr
