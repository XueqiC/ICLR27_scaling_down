"""Selection, frozen-input isolation, actual scoring and paired-model uncertainty."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from analysis import v33_loss_constrained_selection as v33


def vector(value):
    return dict.fromkeys(v33.CAPS, value)


def candidate(key, method, nominal_storage, predicted, actual):
    return {"id": key, "method": method, "nominal_storage": nominal_storage,
            "delta_hat": vector(predicted), "actual_delta": vector(actual)}


def fixture_candidates():
    # Quantization is optimistically unsafe; conservative pruning loses its
    # genuinely feasible, cheapest actual region. Dense must still be available.
    return [candidate("dense", "dense", 1., 0., 0.),
            candidate("pruning:0.4", "pruning", .4, .7, .08),
            candidate("quantization:4", "quantization", .25, .05, .6)]


def test_selection_is_scored_on_actual_all_capability_constraints_and_oracle():
    candidates = fixture_candidates()
    row = v33.score_budget(candidates, v33.prediction_view(candidates), vector(.1))
    chosen = row["policies"]["selector"]
    assert chosen["selected"] == "quantization:4"
    assert chosen["constraint_satisfied"] is False
    assert chosen["violated_capabilities"] == list(v33.CAPS)
    assert row["oracle"] == "pruning:0.4"
    assert chosen["nominal_cost_gap_vs_oracle"] == pytest.approx(-.15)
    assert row["wrongly_excluded"] == ["pruning:0.4"]
    assert row["exclusion_counts"]["all_compressed"] == {
        "actually_feasible": 1, "wrongly_excluded": 1, "fraction": 1.}
    assert row["exclusion_counts"]["quantization"]["fraction"] is None
    assert row["policies"]["always_dense"]["constraint_satisfied"] is True
    assert row["policies"]["always_dense"]["nominal_cost_gap_vs_oracle"] == pytest.approx(.6)


def test_capability_conjunction_signed_changes_boundary_and_nominal_ties():
    candidates = fixture_candidates()
    candidates[1]["delta_hat"] = {"math": -.5, "code": -.5, "qa": .11}
    candidates[2]["delta_hat"] = vector(.11)
    # An average/summed constraint would incorrectly permit pruning.
    assert v33.select(v33.prediction_view(candidates), vector(.1)) == "dense"
    candidates[1]["delta_hat"]["qa"] = .1
    assert v33.select(v33.prediction_view(candidates), vector(.1)) == "pruning:0.4"
    candidates[2]["delta_hat"] = vector(.1)
    candidates[2]["nominal_storage"] = .4
    # Tie-breaking must be stable under candidate order and actual-loss changes.
    for order in (candidates, list(reversed(candidates))):
        assert v33.select(v33.prediction_view(order), vector(.1)) == "pruning:0.4"
    candidates[1]["actual_delta"] = vector(1e6)
    assert v33.select(v33.prediction_view(candidates), vector(.1)) == "pruning:0.4"


@pytest.mark.parametrize("tau", [{"math": .1}, vector(-.1), vector(float("nan")), vector(float("inf"))])
def test_invalid_budgets_are_rejected(tau):
    with pytest.raises(ValueError):
        v33.select(v33.prediction_view(fixture_candidates()), tau)


def test_budget_cartesian_sweep_keeps_negative_observations_and_zero_dense_anchor():
    candidates = fixture_candidates()
    candidates[1]["actual_delta"] = {"math": -.5, "code": .2, "qa": 2.}
    before = copy.deepcopy(candidates)
    axes, budgets = v33.budget_grid(candidates, (0., .5, 1.))
    assert candidates == before
    assert axes["math"][0] == 0.
    assert len(budgets) == np.prod([len(a) for a in axes.values()])
    assert vector(0.) in budgets
    assert {"math": max(axes["math"]), "code": 0., "qa": max(axes["qa"])} in budgets
    for tau in budgets:
        assert v33.feasible(vector(0.), tau)
    with pytest.raises(ValueError, match="quantiles"):
        v33.budget_grid(candidates, [1.1])


def test_empty_feasible_compressed_denominator_is_undefined_and_dense_is_safe():
    candidates = fixture_candidates()
    row = v33.score_budget(candidates, v33.prediction_view(candidates), vector(0.))
    assert row["oracle"] == row["policies"]["selector"]["selected"] == "dense"
    assert row["exclusion_counts"]["all_compressed"] == {
        "actually_feasible": 0, "wrongly_excluded": 0, "fraction": None}


def test_domination_is_a_result_and_oracle_optimal_ties_are_retained():
    candidates = fixture_candidates()
    candidates[1]["delta_hat"] = vector(999.)
    candidates[2]["actual_delta"] = vector(.08)
    candidates[2]["nominal_storage"] = .4
    replay = v33.replay_panel({"model": "fixture", "candidates": candidates})
    assert replay["domination"]["pruning"]["never_selected"] is True
    assert replay["domination"]["pruning"]["oracle_never_selected"] is False
    assert replay["domination"]["quantization"]["oracle_never_selected"] is True
    assert replay["domination"]["quantization"]["never_actually_nominal_cost_optimal_including_ties"] is False
    assert replay["metrics"]["exclusions/pruning"] == 1.


def test_ratio_bootstrap_uses_whole_models_and_pairs_baselines(monkeypatch):
    models = [v33.replay_panel({"model": name, "candidates": fixture_candidates()}) for name in ("a", "b")]
    # Deliberately unequal grid sizes: equal model budget mass, not 10:1000.
    models[0]["n_budgets"], models[1]["n_budgets"] = 10, 1000
    for i, model in enumerate(models):
        totals = model["metric_numerator_denominator"]
        totals["selector/constraint_satisfaction"] = [float(i), 1.]
        totals["always_cheapest/constraint_satisfaction"] = [float(i), 1.]
        totals["exclusions/pruning"] = [float(i), float(1+2*i)]
    called = []
    original = v33.audit.bootstrap_means

    def check_pairing(values, groups, **kwargs):
        called.append((np.asarray(values).shape, groups))
        return original(values, groups, **kwargs)

    monkeypatch.setattr(v33.audit, "bootstrap_means", check_pairing)
    result = v33.aggregate(models, 1000)
    assert len(called) == 1 and called[0][1] == ["a", "b"]
    assert result["metrics"]["selector/constraint_satisfaction"]["estimate"] == .5
    # Ratio of paired totals = 1/(1+3), not average(0/1,1/3).
    exclusion = result["metrics"]["exclusions/pruning"]
    assert exclusion["estimate"] == .25
    assert exclusion["ci95"] == pytest.approx([0., 1/3])
    gain = result["improvement_vs_baselines"]["always_cheapest"]["constraint_satisfaction_gain"]
    assert gain["estimate"] == 0. and gain["ci95"] == [0., 0.]


def test_undefined_conditional_nominal_gap_bootstrap_is_explicit():
    model = v33.replay_panel({"model": "fixture", "candidates": fixture_candidates()})
    model["metric_numerator_denominator"]["selector/nominal_cost_gap_when_satisfied"] = [0., 0.]
    result = v33.aggregate([model], 20)["metrics"]["selector/nominal_cost_gap_when_satisfied"]
    assert result == {"estimate": None, "ci95": None, "n_models": 1, "undefined_bootstrap_draws": 20}
    json.dumps(result, allow_nan=False)


@pytest.fixture(scope="module")
def saved():
    return v33.audit.read_json(v33.V28), v33.audit.read_json(v33.V30)


def test_saved_freeze_integrity_and_canonical_training_leak_are_rejected(saved):
    frozen28, frozen30 = saved
    v33.validate_freezes(frozen28, frozen30)
    tampered = copy.deepcopy(frozen28)
    tampered["methods"]["pruning"]["by_capability"]["qa"]["fit"]["gamma"] += 1.
    with pytest.raises(ValueError, match="integrity"):
        v33.validate_freezes(tampered, frozen30)
    with pytest.raises(ValueError, match="leaked"):
        v33.assert_held_out({"train_models": ["Qwen--Qwen3-4B"]}, "Qwen3-4B")
    with pytest.raises(ValueError, match="exactly one"):
        v33.held_out_fit([], "Qwen3-4B")


def test_missing_v30_shape_falls_back_to_saved_v28_dev_lomo(saved, monkeypatch):
    frozen28, frozen30 = saved
    changed = copy.deepcopy(frozen30)
    for fold in changed["panels"]["broad"]["folds"]:
        fold["fits"].pop(v33.QUANT_SHAPE)
    model = "Qwen3-0.6B"
    rows = frozen28["methods"]["quantization"]["by_capability"]
    dense = {c: next(f for f in rows[c]["lomo"]["folds"] if f["held_out"] == model)
             ["mode_A_target_inputs"]["dense_loss"] for c in v33.CAPS}

    def forbidden(*args, **kwargs):
        pytest.fail("Fallback tried to refit")

    monkeypatch.setattr(v33.v28, "fit_arm", forbidden)
    predictions, sources = v33.frozen_prediction(
        frozen28, changed, model=model, info=frozen28["metadata"]["models"][model],
        method="quantization", coordinate=5, method_dense=dense, source_dense=dense,
        cohort="development_lomo")
    for cap in v33.CAPS:
        expected = next(r["A"] for r in rows[cap]["lomo"]["records"]
                        if r["model"] == model and r["coordinate"] == 5)
        assert predictions[cap] == pytest.approx(expected)
        assert "fallback" in sources[cap] and "dev-LOMO" in sources[cap]


def test_imported_saved_lomo_predictions_match_without_refitting(saved, monkeypatch):
    frozen28, frozen30 = saved

    def forbidden(*args, **kwargs):
        pytest.fail("Replay tried to fit/read development outcomes")

    for name in ("fit_arm", "fit_mapping", "lomo_arm", "load_development"):
        monkeypatch.setattr(v33.v28, name, forbidden)
    for name in ("fit_candidate", "fit_mapping", "lomo", "load_panel"):
        monkeypatch.setattr(v33.v30, name, forbidden)
    panels, _, _, _ = v33.load_panels()
    for panel in panels:
        if panel["cohort"] != "development_lomo":
            continue
        model = panel["model"]
        for cap in v33.CAPS:
            records = frozen28["methods"]["pruning"]["by_capability"][cap]["lomo"]["records"]
            for row in (r for r in records if r["model"] == model):
                prediction = v33.candidates_by_id(panel["candidates"], f"pruning:{row['coordinate']:g}")
                assert prediction["delta_hat"][cap] == pytest.approx(row["A"])
        records = frozen30["panels"]["broad"]["rows"]
        for row in (r for r in records if r["model"] == v33.v30.canonical(model)):
            prediction = v33.candidates_by_id(panel["candidates"], f"quantization:{row['bit']}")
            offset = panel["quant_dense_minus_source_dense"][row["capability"]]
            assert prediction["delta_hat"][row["capability"]] == pytest.approx(
                row["predictions"][v33.QUANT_SHAPE]+offset)


def test_mutating_scored_outcomes_cannot_change_frozen_predictions_or_fixed_budget_decisions(saved):
    frozen28, frozen30 = saved
    model = "Qwen3-0.6B"
    paths = {"pruning": v33.ROOT/f"results/v6-capability-geometry/{model}/prune_losses.json",
             "quantization": v33.ROOT/f"results/v10-quantization/Qwen--{model}/quant_losses.json"}
    tables = {k: v33.audit.read_json(p) for k, p in paths.items()}
    args = (model, frozen28["metadata"]["models"][model])
    before = v33.make_panel(*args, tables, paths, frozen28, frozen30, "development_lomo")
    changed = copy.deepcopy(tables)
    for table in changed.values():
        for key, values in table.items():
            if key not in ("1.0", "dense") and not key.startswith("_"):
                values.update(vector(1e5))
    after = v33.make_panel(*args, changed, paths, frozen28, frozen30, "development_lomo")
    assert v33.prediction_view(before["candidates"]) == v33.prediction_view(after["candidates"])
    tau = vector(100.)
    scored = [v33.score_budget(p["candidates"], v33.prediction_view(p["candidates"]), tau)
              for p in (before, after)]
    assert scored[0]["policies"]["selector"]["selected"] == scored[1]["policies"]["selector"]["selected"]
    assert scored[0]["policies"]["selector"]["constraint_satisfied"] is True
    assert scored[1]["policies"]["selector"]["constraint_satisfied"] is False
    assert scored[1]["oracle"] == "dense"


def test_real_panel_reference_nominal_costs_inventory_and_no_shape_filtering():
    panels, hashes, excluded, _ = v33.load_panels()
    assert len([p for p in panels if p["cohort"] == "development_lomo"]) == 12
    assert len({p["model"] for p in panels}) == len(panels)
    assert any("qwen3-14b" == r["model"] for r in excluded)
    assert all("archive" not in path and "shape512" not in path for path in hashes)
    for panel in panels:
        for row in panel["candidates"]:
            assert row["calibration_points_used"] == 0
            assert row["actual_delta"] == pytest.approx({
                c: row["actual_loss"][c]-panel["source_dense"][c] for c in v33.CAPS})
            if row["method"] == "pruning":
                assert row["nominal_storage"] == row["nominal_active_parameter_ratio"] == row["coordinate"]
                assert row["extrapolated_shape"] == (row["coordinate"] < .6)
            elif row["method"] == "quantization":
                assert row["nominal_storage"] == row["coordinate"]/16
                assert row["nominal_active_parameter_ratio"] == 1.
    qwen = next(p for p in panels if p["model"] == "Qwen3-0.6B")
    assert v33.candidates_by_id(qwen["candidates"], "pruning:0.3")["extrapolated_shape"]
    assert qwen["measurement_caveats"][0]["key"] == "_5bit_meta"
    with pytest.raises(ValueError, match="Missing source"):
        v33.losses_at({"8": vector(0.)}, "dense")


def test_full_replay_invariants_and_report_four_columns():
    summary = v33.build_summary(n_boot=40)
    assert summary["evaluation"] == "OFFLINE REPLAY; not prospective"
    assert summary["refit"] is False and summary["calibration_points_used"] == 0
    assert summary["distillation"]["included"] is False
    for model in summary["models"]:
        predictions = v33.prediction_view(model["candidates"])
        for row in model["rows"]:
            assert row["policies"]["always_dense"]["constraint_satisfied"]
            assert row["policies"]["selector"]["selected"] == v33.select(predictions, row["tau"])
            if row["policies"]["selector"]["constraint_satisfied"]:
                assert row["policies"]["selector"]["nominal_cost_gap_vs_oracle"] >= 0.
            assert set(row["wrongly_excluded"]) == set(row["actually_feasible"]) & set(row["predicted_infeasible"])
            assert row["oracle"] in row["actually_feasible"]
    text = v33.render(summary)
    assert "| Response amplitude context | Improvement vs always-cheapest and always-dense | Calibration points used | Which candidate excluded |" in text
    assert "No latency claims" in text and "not prospective" in text
    json.dumps(summary, allow_nan=False)


def test_dry_run_writes_nothing_and_real_write_rechecks_provenance(monkeypatch, tmp_path, capsys):
    def forbidden(*args, **kwargs):
        pytest.fail("Dry run attempted a write")

    with monkeypatch.context() as patch:
        patch.setattr(Path, "write_text", forbidden)
        patch.setattr(v33.audit, "write_outputs", forbidden)
        v33.main(["--dry-run", "--bootstrap", "2"])
    assert "DRY RUN: no files written" in capsys.readouterr().out
    summary = {"input_sha256": {"analysis/v33_loss_constrained_selection.py": "not-current"}}
    with pytest.raises(RuntimeError, match="Input changed"):
        v33.audit.write_outputs(summary, "report", tmp_path)
    assert not (tmp_path/"summary.json").exists()


def test_cpu_only_fresh_import():
    process = subprocess.run([sys.executable, "-c", "import sys; from analysis import v33_loss_constrained_selection; "
                              "assert not {'torch', 'transformers', 'cupy'} & set(sys.modules)"],
                             cwd=v33.ROOT, capture_output=True, text=True, check=False)
    assert process.returncode == 0, process.stderr
