"""Information isolation, genuine L1 bounds, raw endpoints and paired uncertainty."""
from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from analysis import v35_info_budget as v35


@pytest.fixture(scope="module")
def loaded():
    return v35.load_panels()


@pytest.fixture(scope="module")
def summary():
    return v35.build_summary(n_boot=200)


def basic(panel, cap="math"):
    return {"model": panel["model"], "N0": panel["N0"], "family": panel["family"],
            "dense_loss": panel["dense"][cap]}


def test_whole_analysis_cannot_refit_development(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Development fitting is forbidden in V35")
    for module, names in ((v35.v28, ("fit_arm", "fit_mapping", "fit_transfer", "lomo_arm", "freeze")),
                          (v35.v30, ("fit_candidate", "fit_mapping", "lomo", "ShapeProfile")),
                          (v35.v25, ("fit_predict", "evaluate"))):
        for name in names:
            monkeypatch.setattr(module, name, forbidden)
    result = v35.build_summary(n_boot=10)
    assert result["refit_development"] is False
    assert result["new_model_runs"] == 0


@pytest.mark.parametrize("arm", v35.ARMS)
def test_k0_and_k1_do_not_see_scored_outcomes(loaded, arm):
    panel = copy.deepcopy(next(p for p in loaded[0] if p["arm"] == arm and p["scope"] == "primary"))
    before = v35.evaluate_target(panel, "math")
    for c in panel["losses"]:
        if c != v35.CALIBRATION[arm]:
            panel["losses"][c]["math"] += 1000*c
    after = v35.evaluate_target(panel, "math")
    for a, b in zip(before["records"], after["records"]):
        assert a["predictions"]["K0"] == b["predictions"]["K0"]
        assert a["predictions"]["K1"] == b["predictions"]["K1"]
    assert before["oracle"]["amplitude"] != after["oracle"]["amplitude"]
    assert before["calibration"] == after["calibration"]
    # Change ONLY the one permitted compressed point. K0 still cannot change.
    panel["losses"][v35.CALIBRATION[arm]]["math"] += .5
    third = v35.evaluate_target(panel, "math")
    assert [r["predictions"]["K0"] for r in third["records"]] == [r["predictions"]["K0"] for r in after["records"]]
    assert [r["predictions"]["K1"] for r in third["records"]] != [r["predictions"]["K1"] for r in after["records"]]


@pytest.mark.parametrize("arm", v35.ARMS)
def test_information_contract_rejects_wrong_or_extra_calibration_and_outcomes(loaded, arm):
    p = next(p for p in loaded[0] if p["arm"] == arm)
    inputs, fit = basic(p), p["fits"]["math"]
    coords = [c for c in p["losses"] if c != v35.CALIBRATION[arm]]
    with pytest.raises(ValueError, match="K0"):
        v35.predict_k0(arm, fit, {**inputs, "observed": 999}, "math", coords)
    for point in ({"coordinate": -1, "loss": 1},
                  {"coordinate": v35.CALIBRATION[arm], "loss": 1, "second_point": 2}):
        with pytest.raises(ValueError, match="fixed calibration"):
            v35.predict_k1(arm, fit, inputs, "math", coords, point)
    with pytest.raises(ValueError, match="exclude calibration"):
        v35.predict_k1(arm, fit, inputs, "math", [v35.CALIBRATION[arm]],
                       {"coordinate": v35.CALIBRATION[arm], "loss": 1})


@pytest.mark.parametrize("response", [0., -1e-8, 1e-8, -2., 10.])
def test_signed_zero_nearzero_and_collapse_calibration_never_censored(loaded, response):
    p = next(p for p in loaded[0] if p["arm"] == "distillation")
    b = basic(p)
    values, status = v35.predict_k1("distillation", p["fits"]["math"], b, "math", [150, 600],
                                   {"coordinate": 75, "loss": b["dense_loss"]+response})
    assert values == pytest.approx([response, response])
    assert status["target_configurations_used"] == 1
    assert status["near_zero_response"] == (abs(response) <= v35.NEAR_ZERO)
    assert status["fallback"] is None


def test_numerically_zero_shape_is_explicit_k0_fallback(loaded):
    p = copy.deepcopy(next(p for p in loaded[0] if p["arm"] == "pruning"))
    p["fits"]["math"]["gamma"] = 1000.
    coords, b = [.7], basic(p)
    values, status = v35.predict_k1("pruning", p["fits"]["math"], b, "math", coords,
                                   {"coordinate": .9, "loss": b["dense_loss"]+.001})
    assert values == v35.predict_k0("pruning", p["fits"]["math"], b, "math", coords)
    assert status["fallback"] == "numerically_zero_shape_use_K0"
    assert status["target_configurations_used"] == 1


def test_oracle_is_signed_mae_optimum_not_least_squares():
    x, y = np.array([1., 2., 3., 0.]), np.array([-2., -4., 90., 100.])
    amplitude = v35.oracle_scalar(x, y)
    objective = lambda a: np.mean(np.abs(a*x-y))
    candidates = list(y[x != 0]/x[x != 0]) + [0., float(x@y/(x@x))]
    assert objective(amplitude) == pytest.approx(min(map(objective, candidates)))
    assert amplitude == 14.  # midpoint of the entire [-2,30] minimizing interval
    assert v35.oracle_scalar([1, 1, 1], [-2, -2, 50]) == -2
    assert v35.oracle_scalar([0, 0], [-3, 7]) == 0
    assert v35.oracle_scalar([1, 1], [-4, 2]) == -1
    with pytest.raises(ValueError):
        v35.oracle_scalar([1], [float("nan")])


def test_oracle_all_target_bound_is_not_claimed_for_remainder_subset():
    # Full-curve constant fit is [0,2]'s midpoint, while the one test point
    # alone permits a trivial zero error. Do not quietly call that shape transfer.
    p = {"arm": "distillation", "model": "new", "family": "fixture", "N0": 1e9,
         "dense": {"math": 1}, "losses": {75: {"math": 1}, 600: {"math": 3}},
         "fits": {"math": {"coefficient": 2., "train_row_ids": ["dev|75|math"]}}, "fit_source": "fixture"}
    target = v35.evaluate_target(p, "math")
    test, = [r for r in target["records"] if not r["is_calibration"]]
    assert target["oracle"]["fit_coordinates"] == [75, 600]
    assert target["oracle"]["all_target_mae"] == 1
    assert target["oracle"]["remainder_only_mae_lower_bound"] == 0
    assert abs(test["predictions"]["Oracle"]-test["observed"]) > abs(test["predictions"]["K0"]-test["observed"])


def test_freeze_training_excludes_each_target_and_full_curve_bound_holds(summary):
    for target in summary["targets"]:
        v35.assert_held_out(target["frozen_train_models"], target["model"])
        rows = target["records"]
        assert sum(r["is_calibration"] for r in rows) == 1
        assert target["calibration"]["coordinate"] == v35.CALIBRATION[target["arm"]]
        assert target["oracle"]["n_target_configurations_used"] == len(rows)
        errors = {b: np.mean([abs(r["observed"]-r["predictions"][b]) for r in rows]) for b in v35.BUDGETS}
        assert errors["Oracle"] <= min(errors[b] for b in ("zero", "K0", "K1")) + 1e-10


def test_paired_bootstrap_resamples_models_with_unequal_cell_counts():
    rows = []
    for model, value, size in (("a", 1., 1), ("b", 9., 3)):
        for coordinate in range(size):
            rows.append({"model": model, "coordinate": coordinate, "observed": value,
                         "predictions": {"zero": 0., "K0": .2*value, "K1": .5*value, "Oracle": value},
                         "near_zero_response": False, "negative_response": False, "collapse": True})
    result = v35.summarize(rows, 10000)
    assert result["metrics"]["zero"]["mae"] == 7.  # cell weight, not model mean 5
    assert result["metrics"]["zero"]["mae_ci95"] == pytest.approx([1., 9.])
    assert result["metrics"]["K1"]["improvement_ci95"] == pytest.approx([.5, 4.5])
    assert result["contrasts"]["K1_minus_K0"]["ci95"] == pytest.approx([-2.7, -.3])
    assert result == v35.summarize(rows, 10000)


def test_real_rosters_missing_cells_recipe_and_degenerate_cis_are_explicit(summary):
    assert summary["calibration"]["distillation"]["coordinate"] == min(v35.v25.BUDGETS) == 75
    expected = {"pruning": ["Qwen3-14B", "Qwen3-8B"],
                "quantization": ["Qwen3-14B", "Qwen3-8B"], "distillation": ["Qwen3-4B"]}
    for arm, models in expected.items():
        result = summary["results"][arm]
        assert result["prospective_models"] == models
        assert result["development_lomo"]["math"]["remainder"]["n_models"] == (3 if arm == "distillation" else 12)
        for model in models:
            for cap in v35.CAPS:
                r = result["per_target"][model][cap]["remainder"]
                assert r["n_models"] == 1
                assert "degenerate" in r["ci_note"]
                assert v35.CALIBRATION[arm] not in r["coordinates"]
                for metric in r["metrics"].values():
                    assert metric["mae_ci95"] == pytest.approx([metric["mae"]]*2)
                assert not result["per_target"][model][cap]["diagnosis"]["a_ci_support"]
    q14 = summary["results"]["quantization"]["per_target"]["Qwen3-14B"]["math"]
    assert q14["regions"]["measurable_bits"]["status"] == "no_measured_score_cells"
    assert summary["results"]["distillation"]["per_target"]["Qwen3-4B"]["math"]["remainder"]["coordinates"] == [600]
    assert len(summary["excluded_recipe_cells"]) == 1
    assert summary["excluded_recipe_cells"][0]["model"] == "gemma3-4b"
    assert summary["excluded_recipe_cells"][0]["coordinate"] == 600


def test_losses_are_own_dense_deltas_and_no_collapse_or_negative_filter(loaded, summary):
    for target in summary["targets"]:
        for row in target["records"]:
            assert row["observed"] == pytest.approx(row["loss"]-row["dense_loss"])
    for panel in loaded[0]:
        for cap in v35.CAPS:
            target, = [t for t in summary["targets"] if (t["model"], t["arm"], t["scope"], t["capability"]) ==
                       (panel["model"], panel["arm"], panel["scope"], cap)]
            assert {r["coordinate"] for r in target["records"]} == set(panel["losses"])
    q14, = [t for t in summary["targets"] if (t["arm"], t["model"], t["capability"]) ==
             ("quantization", "Qwen3-14B", "qa")]
    assert any(r["negative_response"] for r in q14["records"])
    assert any(r["collapse"] and r["coordinate"] == 3 for r in q14["records"])
    gemma, = [t for t in summary["targets"] if (t["arm"], t["model"], t["capability"]) ==
              ("distillation", "gemma3-270m", "math")]
    assert len({r["dense_loss"] for r in gemma["records"]}) == 2  # preserve V25 drift


def test_full_dev_fit_cannot_be_used_for_development_target(loaded):
    panel = next(p for p in loaded[0] if p["arm"] == "pruning" and p["cohort"] == "development_lomo")
    frozen = v35.audit.read_json(v35.V28)
    fit = frozen["methods"]["pruning"]["by_capability"]["math"]["fit"]
    with pytest.raises(ValueError, match="leaked"):
        v35.predict_k0("pruning", fit, basic(panel), "math", [.8])
    with pytest.raises(ValueError, match="true dense"):
        v35.parse_table({"8": dict.fromkeys(v35.CAPS, 1.)}, "quantization")
    with pytest.raises(ValueError, match="duplicate"):
        v35.parse_table({"dense": dict.fromkeys(v35.CAPS, 1.), "4": dict.fromkeys(v35.CAPS, 1.),
                         "4.0": dict.fromkeys(v35.CAPS, 1.)}, "quantization")


def test_json_report_are_finite_provenanced_and_claims_are_qualified(summary):
    json.dumps(summary, allow_nan=False)
    assert str(v35.V28.relative_to(v35.ROOT)) in summary["input_sha256"]
    assert str(v35.V30B.relative_to(v35.ROOT)) in summary["input_sha256"]
    assert v35.architecture_n0(v35.QWEN14_ARCH) == 13212057600
    report = v35.render(summary)
    for text in ("K0 is the main", "INCONCLUSIVE", "no shape-transfer claim", "degenerate", "post-hoc diagnostic",
                 "small historical drift", "not a newly timestamped preregistration", "remainder-only"):
        assert text in report


def test_dry_run_does_not_write(monkeypatch, summary, capsys):
    monkeypatch.setattr(v35, "build_summary", lambda n_boot: summary)
    def forbidden(*args, **kwargs):
        raise AssertionError("dry-run attempted a write")
    monkeypatch.setattr(v35.audit, "write_outputs", forbidden)
    monkeypatch.setattr(v35.Path, "write_text", forbidden)
    v35.main(["--dry-run", "--n-boot", "200"])
    assert "Unified information budgets" in capsys.readouterr().out


def test_input_changes_prevent_publishing(tmp_path, monkeypatch, summary):
    # Check the actual shared provenance guard on a task-owned temporary source.
    source = tmp_path / "source.json"
    source.write_text("{}")
    monkeypatch.setattr(v35.audit, "ROOT", tmp_path)
    hashes = v35.audit.provenance([source])
    source.write_text('{"changed":true}')
    with pytest.raises(RuntimeError, match="Input changed"):
        v35.audit.write_outputs({"input_sha256": hashes}, "report", tmp_path / "out")
    assert not (tmp_path / "out").exists()
