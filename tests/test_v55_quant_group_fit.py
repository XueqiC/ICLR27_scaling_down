"""Numerical and holdout-integrity checks for the CPU-only V55 workflow."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from analysis import v55_quant_group_fit as fit


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def panel(tmp_path):
    directory = tmp_path / "data"
    configs = (*fit.DEV_CONFIGS, "b4_g64", "b4_g256", "b3_g128", "b4_g128", "b5_g128")
    for i, tag in enumerate((*fit.DEV_TAGS, fit.JOINT_TAG)):
        dense = {cap: 2.8 + 0.3 * c - 0.07 * i + 0.013 * i ** 2 * (c + 1)
                 for c, cap in enumerate(fit.CAPS)}
        table = {"dense": dense, "_meta": {"model_tag": tag}}
        for config in configs:
            x, v = fit.coordinates(config)
            table[config] = {
                cap: dense[cap] + (-0.2 + 0.04 * i + 0.03 * c) * 2 ** (-1.3 * x + 0.4 * v)
                + 0.002 * (i + 1) * x * v for c, cap in enumerate(fit.CAPS)}
        write_json(fit.measurement_path(directory, tag), table)
    return directory, tmp_path / "out", tmp_path / "fallback.json"


def independent_ridge(x, y):
    return np.linalg.lstsq(np.vstack((x, np.sqrt(1e-3) * np.eye(x.shape[1]))),
                           np.r_[y, np.zeros(x.shape[1])], rcond=None)[0]


def test_exact_panel_signed_targets_and_counts(panel):
    directory, _, _ = panel
    states, rows, hashes = fit.load_dev(directory)
    assert len(states) == 6 and len(rows) == 72 and len(hashes) == 6
    assert {r["config"] for r in rows} == set(fit.DEV_CONFIGS)
    assert min(r["dL"] for r in rows) < 0
    assert [fit.matrix_n0(s) for s in ("160m", "410m", "1b", "1.4b")] == [
        84_934_656, 301_989_888, 805_306_368, 1_207_959_552]
    assert states[0]["D0"] == 33_554_432_000
    for state in states:
        source = Path(state["path"])
        assert hashes[str(source)] == hashlib.sha256(source.read_bytes()).hexdigest()
        table = json.loads(source.read_text())
        for row in (r for r in rows if r["state"] == state["tag"]):
            cap = row["capability"]
            assert row["dL"] == table[row["config"]][cap] - table["dense"][cap]
    # Poison every non-dev response. Loading/standardization/fitting must not
    # inspect their contents, even though V54 stores them beside the dev data.
    for tag in fit.DEV_TAGS:
        path = fit.measurement_path(directory, tag)
        table = json.loads(path.read_text())
        for key in list(table):
            if key not in (*fit.DEV_CONFIGS, "dense", "_meta"):
                table[key] = "not a loss object"
        write_json(path, table)
    after_states, after_rows, _ = fit.load_dev(directory)
    assert after_states == states and after_rows == rows


def test_all_fits_match_independent_raw_response_regressions(panel):
    states, rows, _ = fit.load_dev(panel[0])
    stats, models = fit.fit_all(rows)
    raw = np.array([r["phi_raw"] for r in rows])
    np.testing.assert_allclose(stats["center"], raw.mean(axis=0))
    np.testing.assert_allclose(stats["scale"], raw.std(axis=0))
    assert len(fit.P_GRID) == 56 and len(fit.Q_GRID) == 41
    for cap in fit.CAPS:
        cr = [r for r in rows if r["capability"] == cap]
        z = np.column_stack((np.ones(len(cr)),
                             (np.array([r["phi_raw"] for r in cr]) - raw.mean(0)) / raw.std(0)))
        y = np.array([r["dL"] for r in cr])
        xy = np.array([fit.coordinates(r["config"]) for r in cr])
        best = None
        for p in fit.P_GRID:
            for q in fit.Q_GRID:
                design = z * np.exp2(-p * xy[:, 0] + q * xy[:, 1])[:, None]
                beta = independent_ridge(design, y)
                sse = float(np.sum((design @ beta - y) ** 2))
                if best is None or sse < best[0]:
                    best = sse, p, q, beta
        sep = models[cap]["separable"]
        assert (sep["p"], sep["q"]) == best[1:3]
        np.testing.assert_allclose(sep["beta"], best[3], rtol=1e-9, atol=1e-11)
        assert sep["dev_sse"] == pytest.approx(best[0], abs=1e-12)
        u, v = xy[:, 0] - xy[:, 0].mean(), xy[:, 1]
        terms = np.column_stack((np.ones(len(u)), u, v, u * v, u * u))
        design = np.concatenate([z * terms[:, j, None] for j in range(5)], axis=1)
        np.testing.assert_allclose(np.array(models[cap]["low_order_2d"]["coefficients"]).ravel(),
                                   independent_ridge(design, y), rtol=1e-8, atol=1e-10)
        assert models[cap]["low_order_2d"]["design_rank"] <= 16  # Two dev bit levels.
        for config in fit.DEV_CONFIGS:
            mask = np.array([r["config"] == config for r in cr])
            beta = independent_ridge(z[mask], y[mask])
            np.testing.assert_allclose(models[cap]["same_input_interpolation"]["anchors"][config], beta,
                                       rtol=1e-8, atol=1e-10)
            assert models[cap]["mean"]["anchors"][config] == np.mean(y[mask])
            assert models[cap]["median"]["anchors"][config] == np.median(y[mask])
    # A zero target gives exact ties throughout the grid, independent of folds.
    zero_rows = [{**r, "dL": 0.0} for r in rows]
    model = fit.fit_capability(zero_rows, "math", stats)["separable"]
    assert model["p"] == 0.5 and model["q"] == -1.0


@pytest.mark.parametrize("config", (*fit.DEV_CONFIGS, "b4_g128", "b2_g16", "b7_g1024"))
def test_bilinear_interpolation_and_linear_extension(config):
    def surface(key):
        x, v = fit.coordinates(key)
        return 0.8 + 1.2 * x - 2 * v + 0.6 * x * v
    anchors = {key: surface(key) for key in fit.DEV_CONFIGS}
    assert fit.interpolate(anchors, config) == pytest.approx(surface(config), abs=1e-12)


def test_loso_holds_out_whole_state_and_refits_scaler(panel):
    states, rows, _ = fit.load_dev(panel[0])
    table, folds, predictions = fit.loso(states, rows)
    assert len(folds) == 6 and len(predictions) == 72 and len(table) == 6
    assert all(row["n"] == dict.fromkeys(fit.CAPS, 24) for row in table)
    for fold in folds:
        assert fold["held_out"] not in fold["train_states"]
        assert fold["n_train_cells"] == 20 and fold["n_held_out_cells"] == 4
        train = [r for r in rows if r["state"] != fold["held_out"]]
        assert fold["standardization"] == fit.standardization(train)
    changed = copy.deepcopy(rows)
    for row in changed:
        if row["state"] == states[0]["tag"]:
            row["dL"] += 100
            row["phi_raw"][1] += 200
    _, changed_folds, _ = fit.loso(states, changed)
    assert folds[0]["models"] == changed_folds[0]["models"]
    assert folds[0]["standardization"] == changed_folds[0]["standardization"]


def test_workflow_determinism_errors_hashes_and_no_overwrites(panel):
    directory, out, fallback = panel
    inputs = {p: p.read_bytes() for p in directory.rglob("*.json")}
    reg = fit.develop(directory, out)
    assert reg["precommitted_rule"] == fit.RULE and "selected_candidate" not in reg
    frozen = fit.freeze(directory, out, fallback)
    assert len(frozen["predictions"]) == 99 and not frozen["missing"]
    assert all(set(r["predictions"]) == set(fit.METHODS) for r in frozen["predictions"])
    result = fit.compare(directory, out)
    assert len(result["rows"]) == 99
    for name, n in (("bit_test", 12), ("granularity_test", 18), ("joint_test", 3)):
        summary = result["test_sets"][name]
        assert summary["n_scored_cells"] == summary["n_expected_cells"] == n
        assert summary["response_flags"]["complete"]
        for metric in summary["mae_table"]:
            for cap in fit.CAPS:
                rs = [r for r in result["rows"] if r["test_set"] == name and r["capability"] == cap]
                expected = np.mean([abs(r["dL"] - r["predictions"][metric["candidate"]]) for r in rs])
                assert metric["mae"][cap] == expected
    for row in result["rows"]:
        assert row["absolute_errors"]["zero"] == abs(row["dL"])
    another = out.parent / "another"
    fit.develop(directory, another)
    fit.freeze(directory, another, fallback)
    fit.compare(directory, another)
    for name in ("register.json", "predictions.json", "compare.json"):
        assert (out / name).read_bytes() == (another / name).read_bytes()
    outputs = {p: p.read_bytes() for p in out.iterdir()}
    for operation in (fit.develop, fit.freeze, fit.compare):
        with pytest.raises(FileExistsError, match="Refusing to overwrite"):
            operation(directory, out)
    assert all(p.read_bytes() == raw for p, raw in {**inputs, **outputs}.items())


def test_freeze_ignores_test_outcomes_and_prefers_same_file_dense(panel):
    directory, out, fallback = panel
    fit.develop(directory, out)
    write_json(fallback, {"tag": fit.JOINT_TAG, "dense": dict.fromkeys(fit.CAPS, 100)})
    first = fit.freeze(directory, out, fallback)
    joint = next(s for s in first["states"] if s["tag"] == fit.JOINT_TAG)
    assert joint["path"] == str(fit.measurement_path(directory, fit.JOINT_TAG))
    # Mutate only held-out outcomes, retaining the dense input.
    for tag in (*fit.DEV_TAGS, fit.JOINT_TAG):
        path = fit.measurement_path(directory, tag)
        table = json.loads(path.read_text())
        for key in list(table):
            if key not in ("dense", "_meta") and (tag == fit.JOINT_TAG or key not in fit.DEV_CONFIGS):
                table[key] = "test response must never be consumed by freeze"
        write_json(path, table)
    another = out.parent / "another"
    another.mkdir()
    (another / "register.json").write_bytes((out / "register.json").read_bytes())
    second = fit.freeze(directory, another, fallback)
    assert first["predictions"] == second["predictions"]


def test_joint_fallback_missing_joint_and_partial_compare(panel, capsys):
    directory, out, fallback = panel
    fit.develop(directory, out)
    joint_path = fit.measurement_path(directory, fit.JOINT_TAG)
    joint = json.loads(joint_path.read_text())
    joint_path.unlink()
    write_json(fallback, {"tag": fit.JOINT_TAG, "dense": joint["dense"]})
    state, sha = fit.joint_input(directory, fallback)
    assert state["L0"] == joint["dense"] and state["path"] == str(fallback)
    assert sha == hashlib.sha256(fallback.read_bytes()).hexdigest()
    fallback.unlink()
    frozen = fit.freeze(directory, out, fallback)
    assert len(frozen["predictions"]) == 90 and len(frozen["missing"]) == 1
    assert "JOINT TEST OMITTED" in capsys.readouterr().out
    # Measure the omitted joint test later: annotate its responses, but do not
    # manufacture a retroactive prediction or an error for any method.
    joint["b3_g128"]["math"] = joint["dense"]["math"] + 5
    write_json(joint_path, joint)
    for tag in fit.DEV_TAGS:
        path = fit.measurement_path(directory, tag)
        table = json.loads(path.read_text())
        del table["b4_g64"]
        del table["b4_g256"]
        write_json(path, table)
    result = fit.compare(directory, out)
    bit = result["test_sets"]["bit_test"]
    assert bit["n_measured_cells"] == 0
    assert all(m["mae"] == dict.fromkeys(fit.CAPS, None) for m in bit["mae_table"])
    assert bit["response_flags"]["near_zero_all_cells"] is None
    jt = result["test_sets"]["joint_test"]
    assert jt["n_measured_cells"] == 3 and jt["n_scored_cells"] == 0
    assert jt["response_flags"]["collapse_any_measured"]
    assert not jt["response_flags"]["collapse_all_measured"]
    assert all(r["absolute_errors"] is None for r in result["rows"] if r["test_set"] == "joint_test")


def test_flags_have_strict_thresholds_and_preserve_negatives():
    assert fit.response_flags([-0.029, 0.029], 2)["near_zero_all_cells"] is True
    assert fit.response_flags([-0.03, 0.01], 2)["near_zero_all_cells"] is False
    assert fit.response_flags([0.03], 2)["near_zero_all_cells"] is False
    assert fit.response_flags([-0.01], 2)["near_zero_all_cells"] is None
    assert fit.response_flags([4.0], 1)["collapse_any_measured"] is False
    assert fit.response_flags([4.0001], 1)["collapse_any_measured"] is True
    assert fit.response_flags([-5.0], 1)["collapse_any_measured"] is False


def test_cli_missing_dev_writes_nothing_and_import_is_numpy_only(tmp_path):
    command = [sys.executable, "-B", str(Path(fit.__file__)), "develop",
               "--data-root", str(tmp_path / "missing"), "--out", str(tmp_path / "out")]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 1 and "requires all 24 cells" in result.stderr
    assert not (tmp_path / "out").exists()
    code = ("import sys; from analysis import v55_quant_group_fit; "
            "assert not {'torch', 'scipy', 'transformers'} & sys.modules.keys()")
    subprocess.run([sys.executable, "-B", "-c", code], cwd=fit.ROOT, check=True)
