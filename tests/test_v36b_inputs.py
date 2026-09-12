"""Four input budgets, direct signed targets and paired whole-step Pythia folds."""
from __future__ import annotations

import copy
import itertools
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from analysis import v36_pythia_controlled_fit as v36
from analysis import v36b_input_comparison as v36b


def synthetic_rows(arm="pruning", *, l0_effect=True):
    rows = []
    for i, size in enumerate(v36b.SIZES):
        for j, step in enumerate(v36.STEPS):
            n0, d0 = v36.matrix_n0(size), v36.training_tokens(step)
            l0 = 1.8 - .2*i - .13*j + .07*i*j + .04*j*j
            cell = f"pythia-{size}--step{step}"
            for k, q in enumerate(v36.CONFIGS[arm]):
                y = (.1*(k+1) + (.07-.11*k)*np.log(n0/1e9) +
                     (.3*k-.2)*l0*l0_effect + (.1+.23*k)*np.log(d0/1e9))
                rows.append({"row_id": f"{arm}|{cell}|math|{q:g}", "cell": cell,
                             "arm": arm, "capability": "math", "size": size, "step": step,
                             "N0": n0, "D0": d0, "L0": l0, "config": q,
                             "observed": float(y), "loss": float(l0+y),
                             **v36.response_flags(arm, q, y)})
    return rows


@pytest.fixture(scope="module")
def summary():
    paths = [v36b.ROOT / f"results/{v36.FILES[a][0]}/pythia-{s}--step{st}/{v36.FILES[a][1]}"
             for a, s, st in itertools.product(v36.ARMS, v36b.SIZES, v36.STEPS)]
    if not all(p.exists() for p in paths):
        pytest.skip("Available clean Pythia loss JSON panel is not on disk; no model runs allowed")
    return v36b.build_summary()


def test_exactly_four_input_sets_and_no_outcome_or_unadvertised_input():
    expected = {"simple": set(), "N0_D0": {"N0", "D0"},
                "N0_L0": {"N0", "L0"}, "N0_D0_L0": {"N0", "D0", "L0"}}
    assert {m: set(fields) for m, fields in v36b.INPUT_SETS.items()} == expected
    row = synthetic_rows()[0]
    for model, fields in expected.items():
        inputs = v36b.basic_input(row, model)
        assert set(inputs) == fields | {"config"}
        with pytest.raises(ValueError, match="no outcomes"):
            v36.covariates([{**inputs, "observed": 10}], input_fields=v36b.INPUT_SETS[model])
        raw = v36.covariates([inputs], input_fields=v36b.INPUT_SETS[model])
        want = [row[f] if f == "L0" else np.log(row[f]/1e9) for f in v36b.INPUT_SETS[model]]
        np.testing.assert_allclose(raw, [want])
    for bad in (("N0", "N0"), ("observed",)):
        with pytest.raises(ValueError, match="distinct subset"):
            v36.predictor_fields(input_fields=bad)


@pytest.mark.parametrize("model", v36b.INPUT_SETS)
def test_single_direct_solver_call_receives_raw_signed_config_targets(monkeypatch, model):
    train = [r for r in synthetic_rows() if r["step"] != 143000]
    for row, value in zip(train, np.random.default_rng(36).normal(size=len(train))):
        row["observed"] = float(value)
        row["loss"] = 1000.  # Absolute loss is not a regression label.
        row["a_c"] = -999.  # Nor is a per-source amplitude.
    original, calls = np.linalg.lstsq, []

    def spy(x, y, **kwargs):
        calls.append((x.copy(), y.copy()))
        return original(x, y, **kwargs)

    monkeypatch.setattr(np.linalg, "lstsq", spy)
    fit = v36b.fit_direct(train, model)
    assert len(calls) == 1
    x, y = calls[0]
    assert x.shape == (16, 4*(1+len(v36b.INPUT_SETS[model])))
    np.testing.assert_array_equal(y, [r["observed"] for r in train])
    np.testing.assert_allclose(fit["coefficients"], original(x, y, rcond=None)[0])
    assert fit["target"] == "observed signed config-level Delta L"
    assert fit["n_cells"] == 4 and fit["rank"] == x.shape[1]
    inputs = [v36b.basic_input(r, model) for r in train]
    raw = v36.covariates(inputs, input_fields=v36b.INPUT_SETS[model])
    np.testing.assert_allclose(fit["center"], raw.mean(axis=0))
    np.testing.assert_allclose(fit["scale"], raw.std(axis=0))


def test_standalone_training_scale_fit_does_not_read_dense_measurements():
    train = [r for r in synthetic_rows() if r["step"] != 143000]
    expected = v36b.fit_direct(train, "N0_D0")
    for r in train:
        del r["L0"]
        del r["loss"]
    assert v36b.fit_direct(train, "N0_D0") == expected
    # Conversely the dense-input standalone fit needs no D0.
    train = [r for r in synthetic_rows() if r["step"] != 143000]
    expected = v36b.fit_direct(train, "N0_L0")
    for r in train:
        del r["D0"]
    assert v36b.fit_direct(train, "N0_L0") == expected


@pytest.mark.parametrize("arm", v36.ARMS)
def test_all_models_share_whole_step_folds_and_recover_direct_family(arm):
    rows = synthetic_rows(arm)
    result = v36b.cross_validate(rows)
    scored = []
    for fold in result["folds"]:
        train = [rows[i] for i in fold["train_indices"]]
        test = [rows[i] for i in fold["test_indices"]]
        assert fold["n_train"] == 16 and fold["n_test"] == 8
        assert len({r["cell"] for r in train}) == 4
        assert len({r["cell"] for r in test}) == 2
        assert not {r["cell"] for r in train} & {r["cell"] for r in test}
        assert {r["step"] for r in test} == {fold["held_out"]}
        assert all(r["step"] != fold["held_out"] for r in train)
        assert {r["size"] for r in test} == set(v36b.SIZES)
        assert {r["config"] for r in test} == set(v36.CONFIGS[arm])
        assert all(fit["train_row_ids"] == fold["train_row_ids"] for fit in fold["fits"].values())
        scored.extend(fold["test_indices"])
    assert sorted(scored) == list(range(24))
    assert len({r["row_id"] for r in result["records"]}) == 24
    for r in result["records"]:
        assert set(r["predictions"]) == set(v36b.INPUT_SETS) | set(v36b.BASELINES)
    assert result["core_table"]["N0_D0_L0"]["mae"] < 1e-11
    nd = v36b.cross_validate(synthetic_rows(arm, l0_effect=False))
    assert nd["core_table"]["N0_D0"]["mae"] < 1e-11
    with pytest.raises(ValueError, match="complete clean"):
        v36b.cross_validate(rows[:-1])
    with pytest.raises(ValueError, match="complete clean"):
        v36b.cross_validate(rows[:-1] + rows[:1])


def test_held_out_compressed_outcomes_do_not_change_any_fit_or_prediction():
    rows = synthetic_rows()
    before = v36b.cross_validate(rows)
    changed = copy.deepcopy(rows)
    for i, r in enumerate(changed):
        if r["step"] == 16000:
            r["observed"] += 100+i
            r["loss"] += 100+i
    after = v36b.cross_validate(changed)
    assert before["folds"][0]["fits"] == after["folds"][0]["fits"]
    # The empirical reference may change after scoring; candidate predictions may not.
    predictions = lambda c: [{k: v for k, v in r["predictions"].items() if k != "simple"}
                             for r in c["records"] if r["step"] == 16000]
    assert predictions(before) == predictions(after)
    assert before["folds"][0]["mae"]["N0_D0"] != after["folds"][0]["mae"]["N0_D0"]


def test_baselines_fit_training_only_and_reference_is_one_global_candidate():
    rows = synthetic_rows()
    result = v36b.cross_validate(rows)
    strongest = min(v36b.BASELINES, key=lambda m: result["baseline_candidates"][m]["mae"])
    assert result["strongest_simple_baseline"] == strongest
    for fold in result["folds"]:
        train = [rows[i] for i in fold["train_indices"]]
        for r in result["records"]:
            if r["step"] != fold["held_out"]:
                continue
            y = [t["observed"] for t in train if t["config"] == r["config"]]
            assert r["predictions"]["zero"] == 0
            assert r["predictions"]["config_mean"] == pytest.approx(np.mean(y))
            assert r["predictions"]["config_median"] == pytest.approx(np.median(y))
            assert r["predictions"]["simple"] == r["predictions"][strongest]
    for model, m in result["core_table"].items():
        direct_mae = np.mean([abs(r["observed"]-r["predictions"][model]) for r in result["records"]])
        assert m["mae"] == pytest.approx(direct_mae)
        assert m["improvement"] == pytest.approx(result["baseline_candidates"][strongest]["mae"]-direct_mae)
        assert m["n"] == 24 and m["reference"] == strongest


def test_paired_interval_reuses_whole_steps_not_independent_model_or_row_draws():
    rows = synthetic_rows()
    for r in rows:
        error = 2 + 3*v36.STEPS.index(r["step"])
        r["predictions"] = {"ref": r["observed"]+error, "new": r["observed"]+error-1}
    m = v36b.paired_comparison(rows, "ref", "new")
    assert m["reference_mae"] == pytest.approx(5)
    assert m["mae"] == pytest.approx(4)
    assert m["mae_ci95"] == pytest.approx([1, 7])
    assert m["improvement_ci95"] == pytest.approx([1, 1])
    assert m["leave_one_group_out_range"] == pytest.approx([1, 1])
    assert m["n_groups"] == 3


def test_three_curves_and_correlations_preserve_native_units_and_unique_cells():
    rows = synthetic_rows()
    for r in rows:
        j = v36.STEPS.index(r["step"])
        r["L0"] = [4., 3., 2.][j]
        r["observed"] = [-.1, .1, .2][j]
        r["loss"] = r["L0"] + r["observed"]
    c = v36b.curves(rows)
    assert len(c["dense"]) == 6 and len(c["compressed"]) == 24
    assert c["n_increment_rises_but_absolute_loss_falls"] == 8
    for t in c["trajectories"]:
        assert t["D0"] == [v36.training_tokens(st) for st in v36.STEPS]
        assert t["dense_loss"]["monotonicity"] == "decreasing"
        assert t["delta_loss"]["monotonicity"] == "increasing"
        assert t["compressed_loss"]["values"] == [3.9, 3.1, 2.2]
        np.testing.assert_allclose(np.array(t["dense_loss"]["values"])+t["delta_loss"]["values"],
                                   t["compressed_loss"]["values"])
    corr = v36b.correlations(rows)
    assert corr["all_cells"]["n_cells"] == 6
    assert all(c["n_cells"] == 3 for c in corr["by_size"].values())
    assert corr["all_cells"]["pearson_D0_L0"] == pytest.approx(
        np.corrcoef([v36.training_tokens(st) for st in v36.STEPS], [4, 3, 2])[0, 1])
    assert v36b.trajectory([1., 3., 2.])["monotonicity"] == "nonmonotonic"
    assert v36b.trajectory([1., 1., 1.])["monotonicity"] == "flat"


def test_clean_artifact_matches_v36_ab_fits_and_scores(summary):
    original_sizes = v36.SIZES
    rows, hashes = v36b.load_grid()
    assert v36.SIZES == original_sizes  # No import-time or loader global mutation.
    assert len(hashes) == 12 and len(rows) == 144
    assert {r["size"] for r in rows} == {"410m", "1.4b"}
    assert all(r["observed"] == r["loss"]-r["L0"] for r in rows)
    assert summary["n_cells_per_arm"] == 6 and summary["new_model_runs"] == 0
    assert summary["integrity"]["headline_ready"]
    for arm, cap in itertools.product(v36.ARMS, v36.CAPS):
        panel = [r for r in rows if (r["arm"], r["capability"]) == (arm, cap)]
        original = v36.cross_validate(panel, "step")
        current = summary["results"][arm][cap]["comparison"]
        assert current["core_table"]["N0_L0"]["mae"] == original["metrics"]["A_mae"]
        assert current["core_table"]["N0_D0_L0"]["mae"] == original["metrics"]["B_mae"]
        assert current["input_pairs"]["combined_vs_L0"]["improvement_ci95"] == original["metrics"]["improvement_ci95"]
        for old, new in zip(original["folds"], current["folds"]):
            assert old["test_indices"] == new["test_indices"]
            for a, b in (("A", "N0_L0"), ("B", "N0_D0_L0")):
                np.testing.assert_array_equal(old["fits"][a]["coefficients"], new["fits"][b]["coefficients"])
    assert v36b.build_summary() == summary
    assert json.loads(json.dumps(summary, allow_nan=False)) == summary
    report = v36b.render(summary)
    for phrase in ("KEY four-input table: pruning", "KEY four-input table: quantization",
                   "{N0, D0}", "{N0, L0}", "{N0, D0, L0}", "Absolute compressed",
                   "both predictors failing", "saturated", "24 observations", "2.8b is excluded"):
        assert phrase in report


@pytest.mark.parametrize("sizes", (None, "410m,1.4b"))
def test_cpu_only_and_clean_default_or_explicit_environment(sizes, summary):
    env = dict(os.environ)
    env.pop("SDL_V36_SIZES", None)
    if sizes is not None:
        env["SDL_V36_SIZES"] = sizes
    result = subprocess.run([sys.executable, "-c", (
        "import sys; from analysis import v36_pythia_controlled_fit as v; "
        "original = v.SIZES; from analysis import v36b_input_comparison as b; "
        "s = b.build_summary(); assert v.SIZES == original; "
        "assert s['inputs']['sizes'] == ['410m', '1.4b']; "
        "assert not {'torch', 'transformers', 'matplotlib'} & set(sys.modules)")],
        cwd=v36b.ROOT, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_dry_run_does_not_write_or_plot(monkeypatch, capsys, summary):
    monkeypatch.setattr(v36b, "build_summary", lambda: summary)

    def forbidden(*args, **kwargs):
        raise AssertionError("Dry run must not write or plot")

    monkeypatch.setattr(v36b, "write_outputs", forbidden)
    monkeypatch.setattr(v36b, "plot_curves", forbidden)
    v36b.main(["--dry-run"])
    assert "KEY four-input" in capsys.readouterr().out


def test_input_hash_change_fails_before_writing(tmp_path, summary):
    source = tmp_path / "changed.json"
    source.write_text("{}")
    changed = {**summary, "input_sha256": {"changed.json": "incorrect"}}
    with pytest.raises(RuntimeError, match="Input changed"):
        v36b.write_outputs(changed, v36b.render(changed), root=tmp_path)
    assert not (tmp_path / "results").exists()
