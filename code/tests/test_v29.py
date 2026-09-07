"""CPU sanity checks; synthetic fixtures never enter the report."""
import copy
import subprocess
import sys

import pytest

from analysis import v29_prune_failure_diagnosis as v29


def fixture_rows():
    rows = []
    for i in range(5):
        for cap in v29.CAPS:
            for d in v29.GRID:
                y = (.1+i*.1)*((1-d)/.3)**2
                rows.append({"model": f"fixture-{i}", "family": "a" if i < 3 else "b",
                             "N0": (i+1)*1e9, "dense_loss": 1.5-i*.1,
                             "capability": cap, "density": d, "observed": y,
                             "loss": 1.5-i*.1+y, "precliff": y <= 1.,
                             "row_id": f"fixture-{i}|{cap}|{d}"})
    return rows


def test_matched_range_excludes_deep_and_collapse_without_changing_rows():
    rows = [{"density": .8, "precliff": True}, {"density": .6, "precliff": False},
            {"density": .5, "precliff": True}, {"density": .6, "precliff": True}]
    assert v29.restrict_range(rows) == [rows[0], rows[1], rows[3]]
    assert v29.restrict_range(rows, precliff=True) == [rows[0], rows[3]]


def test_shrinkage_direct_limit_population_limit_and_noise_gain():
    y, x, mean = -.01, .003, .2
    assert v29.calibrated_amplitude(y, x, mean, 0) == pytest.approx(y/x)
    assert v29.calibrated_amplitude(y, x, mean, 1e8) == pytest.approx(mean)
    lam = .01
    a = v29.calibrated_amplitude(y, x, mean, lam)
    b = v29.calibrated_amplitude(y+.001, x, mean, lam)
    assert b-a == pytest.approx(.001*x/(x*x+lam))
    assert abs(b-a) < .001/x
    with pytest.raises(ValueError):
        v29.calibrated_amplitude(y, x, mean, -1)


def test_amplitude_projection_is_orthogonal_and_preserves_sign_diagnosis():
    p = [1., 4., 9.]
    result = v29.amplitude_decomposition(p, [-.1, -.4, -.9])
    assert result["signed_scale"] == pytest.approx(-.1)
    assert result["nonnegative_scale"] == 0
    assert result["sign_reversal_required"]
    assert result["scaled_mae"] < 1e-12
    assert result["raw_mse"] == pytest.approx(result["amplitude_or_sign_mse"]+result["shape_mse"])
    result = v29.amplitude_decomposition(p, [.3, .8, 2.])
    assert result["raw_mse"] == pytest.approx(result["amplitude_or_sign_mse"]+result["shape_mse"])


def test_threshold_is_censored_and_first_crossing_survives_later_recovery():
    d = [.9, .8, .7, .6]
    assert v29.threshold_crossing(d, [.01, .03, .1, .2])["status"] == "not_observed"
    assert v29.threshold_crossing(d, [.01, 1.2, .9, .2])["density_bracket"] == [.8, .9]
    assert v29.threshold_crossing(d, [-.01, -.2, -.5, -1.2])["status"] == "not_observed"
    assert v29.threshold_crossing(d, [-.01, -.2, -.5, -1.2], absolute=True)["density_bracket"] == [.6, .7]


def test_whole_target_outcomes_cannot_change_mode_A_or_shrinkage_selection():
    rows = fixture_rows()
    before = v29.evaluate_lomo(rows, shrinkage=True)
    changed = copy.deepcopy(rows)
    for r in changed:
        if r["model"] == "fixture-0":
            r["observed"] += 100
            r["loss"] += 100
    after = v29.evaluate_lomo(changed, shrinkage=True)
    before_folds = [f for f in before["folds"] if f["held_out"] == "fixture-0"]
    after_folds = [f for f in after["folds"] if f["held_out"] == "fixture-0"]
    for a, b in zip(before_folds, after_folds):
        for key in ("v28_fit", "old_fit", "old_cap_fit", "regularization_selection", "mode_A_inputs"):
            assert a[key] == b[key]
        assert all(not row.startswith("fixture-0|") for row in a["train_row_ids"])
        assert a["mode_B_calibration"] != b["mode_B_calibration"]
    for result in (before, after):
        assert all(r["density"] != .9 for r in result["records"])
        assert all("v28_B80" not in r["predictions"] for r in result["records"] if r["density"] == .8)
    for name in ("old_A", "old_cap_A", "v28_mean_A", "v28_A"):
        assert [r["predictions"][name] for r in before["records"] if r["model"] == "fixture-0"] == [
            r["predictions"][name] for r in after["records"] if r["model"] == "fixture-0"]


def test_json_parser_uses_dense_and_ignores_infill_metadata():
    losses = {"1.0": dict.fromkeys(v29.CAPS, 1.), ".9": dict.fromkeys(v29.CAPS, 1.1),
              "_infill_meta": {"ignored": "not a loss"}}
    assert set(v29.parse_losses(losses)) == {1., .9}
    with pytest.raises(ValueError, match="dense"):
        v29.parse_losses({".9": losses[".9"]})


def test_paired_cross_range_draws_and_degenerate_single_model_ci():
    rows = [{"model": m, "observed": y, "predictions": {"a": 0., "b": 0.}}
            for m, y in [("m1", 1.), ("m1", 2.), ("m2", 10.)]]
    result = v29.paired_estimands({"a": (rows, "a"), "b": (rows, "b")},
                                  {"difference": {"a": 1, "b": -1}}, 100)
    assert result["difference"]["ci95"] == [0., 0.]
    one = v29.estimate([.1, .2, .3], ["one"]*3, 100)
    assert one["ci95"] == pytest.approx([.2, .2])


def test_cpu_import_never_loads_model_libraries():
    result = subprocess.run([sys.executable, "-c",
                             "import sys; from analysis import v29_prune_failure_diagnosis; "
                             "assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules"],
                            cwd=v29.ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_existing_json_reproduces_both_historical_density_only_scores():
    rows, frozen, _, _ = v29.load_rows()
    dev = [r for r in rows if r["model"] != "Qwen3-8B"]
    result = v29.historical_audit(dev, 100)
    assert (result["n_precliff"], result["n_train"], result["n_test"]) == (174, 146, 28)
    assert result["summaries"]["pooled"]["metrics"]["historical_own_points"]["mae"] == pytest.approx(.3328818952736116)
    assert result["family_summaries"]["pooled"]["metrics"]["old_family_A"]["mae"] == pytest.approx(.339276430787577)
    assert len(frozen["metadata"]["models"]) == 12
