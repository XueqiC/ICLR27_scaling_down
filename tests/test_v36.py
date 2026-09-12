"""V36 whole-axis holdouts, token accounting, direct targets and integrity checks."""
from __future__ import annotations

import copy
import itertools
import json
import subprocess
import sys

import numpy as np
import pytest

from analysis import v36_pythia_controlled_fit as v36


def synthetic_rows(arm="pruning"):
    """A full-rank response family with distinct config slopes, including D0."""
    rows = []
    for i, size in enumerate(v36.SIZES):
        for j, step in enumerate(v36.STEPS):
            n0, d0 = v36.matrix_n0(size), v36.training_tokens(step)
            l0 = 1.8 - .2*i - .13*j + .07*i*j + .04*j*j
            cell = f"pythia-{size}--step{step}"
            for k, q in enumerate(v36.CONFIGS[arm]):
                y = (.1*(k+1) + (.07-.11*k)*np.log(n0/1e9) +
                     (.3*k-.2)*l0 + (.1+.23*k)*np.log(d0/1e9))
                rows.append({"row_id": f"{arm}|{cell}|math|{q:g}", "cell": cell,
                             "arm": arm, "capability": "math", "size": size,
                             "step": step, "N0": n0, "D0": d0, "L0": l0,
                             "config": q, "observed": float(y), "loss": float(l0+y),
                             **v36.response_flags(arm, q, y)})
    return rows


def _full_grid_on_disk():
    from pathlib import Path
    return all((v36.ROOT / f"results/v6-capability-geometry/pythia-{s}--step{st}/prune_losses.json").exists()
               and (v36.ROOT / f"results/v10-quantization/pythia-{s}--step{st}/quant_losses.json").exists()
               for s in v36.SIZES for st in v36.STEPS)


_NO_GRID_REASON = ("full Pythia panel not on disk: pythia-2.8b step-revisions collapse to identical "
                   "weights upstream (verified after clean re-download); the available-panel analysis "
                   "runs via SDL_V36_SIZES=410m,1.4b (see PYTHIA_CONTROLLED.md)")


@pytest.fixture(scope="module")
def summary():
    if not _full_grid_on_disk():
        pytest.skip(_NO_GRID_REASON)
    return v36.build_summary()


def test_d0_exact_mapping_and_units():
    assert v36.TOKENS_PER_STEP == 2_097_152
    expected = {16000: 33_554_432_000, 64000: 134_217_728_000, 143000: 299_892_736_000}
    for step, tokens in expected.items():
        assert v36.training_tokens(step) == tokens
    with pytest.raises(ValueError, match="Unknown"):
        v36.training_tokens(15000)
    assert [v36.matrix_n0(s) for s in v36.SIZES] == [301_989_888, 1_207_959_552, 2_516_582_400]


@pytest.mark.parametrize("key", ("size", "step"))
def test_folds_hold_entire_axis_and_score_every_row_once(key):
    rows = synthetic_rows()
    folds = v36.folds(rows, key)
    assert len(folds) == 3
    scored = []
    for f in folds:
        train, test = [rows[i] for i in f["train_indices"]], [rows[i] for i in f["test_indices"]]
        assert len(train) == 24 and len(test) == 12
        assert len({r["cell"] for r in train}) == 6
        assert len({r["cell"] for r in test}) == 3
        assert not {r["cell"] for r in train} & {r["cell"] for r in test}
        assert {r[key] for r in test} == {f["held_out"]}
        assert f["held_out"] not in {r[key] for r in train}
        other = "step" if key == "size" else "size"
        assert len({r[other] for r in test}) == 3
        assert {r["config"] for r in train} == {r["config"] for r in test} == set(v36.CONFIGS["pruning"])
        scored.extend(f["test_indices"])
    assert sorted(scored) == list(range(36))
    with pytest.raises(ValueError, match="all three"):
        v36.folds([r for r in rows if r[key] != rows[0][key]], key)
    with pytest.raises(ValueError, match="Holdout key"):
        v36.folds(rows, "cell")


@pytest.mark.parametrize("arm,key", itertools.product(v36.ARMS, ("size", "step")))
def test_direct_model_recovers_known_history_response_out_of_sample(arm, key):
    result = v36.cross_validate(synthetic_rows(arm), key)
    assert result["metrics"]["B_mae"] < 1e-11
    assert result["metrics"]["A_mae"] > .01
    for fold in result["folds"]:
        assert fold["fits"]["A"]["n_parameters"] == 12
        assert fold["fits"]["B"]["n_parameters"] == 16
        assert fold["fits"]["B"]["training_mse"] < 1e-20


@pytest.mark.parametrize("with_d0", (False, True))
def test_solver_receives_raw_config_level_targets_not_source_labels(monkeypatch, with_d0):
    rows = [r for r in synthetic_rows() if r["size"] != "410m"]
    # Arbitrary signed, nonseparable responses prevent a noiseless-amplitude fixture.
    for r, y in zip(rows, np.random.default_rng(36).normal(size=len(rows))):
        r["observed"] = float(y)
    original, calls = np.linalg.lstsq, []

    def spy(x, y, **kwargs):
        calls.append((x.copy(), y.copy()))
        return original(x, y, **kwargs)

    monkeypatch.setattr(np.linalg, "lstsq", spy)
    fit = v36.fit_direct(rows, with_d0)
    assert len(calls) == 1  # No preliminary per-source fitting stage.
    x, y = calls[0]
    assert x.shape == (24, 16 if with_d0 else 12)
    np.testing.assert_array_equal(y, [r["observed"] for r in rows])
    manual = original(x, y, rcond=None)[0]
    np.testing.assert_allclose(fit["coefficients"], manual)
    assert fit["n_observations"] == 24 and fit["n_cells"] == 6
    assert fit["target"] == "observed signed config-level Delta L"


def test_nested_design_training_only_scalers_and_a_ignores_d0():
    rows = [r for r in synthetic_rows() if r["step"] != 143000]
    fits = {m: v36.fit_direct(rows, m == "B") for m in ("A", "B")}
    designs = {}
    for m, fit in fits.items():
        inputs = [v36.basic_input(r, m == "B") for r in rows]
        raw = v36.covariates(inputs, m == "B")
        np.testing.assert_allclose(fit["center"], raw.mean(axis=0))
        np.testing.assert_allclose(fit["scale"], raw.std(axis=0))
        designs[m] = v36.design_matrix(inputs, "pruning", m == "B", fit["center"], fit["scale"])
    np.testing.assert_allclose(designs["A"], designs["B"][:, :12])
    altered = copy.deepcopy(rows)
    for i, row in enumerate(altered):
        row["D0"] *= 1 + i
    np.testing.assert_array_equal(v36.fit_direct(altered, False)["coefficients"], fits["A"]["coefficients"])
    inputs = [v36.basic_input(rows[0], True)]
    inputs[0]["observed"] = 100
    with pytest.raises(ValueError, match="no outcomes"):
        v36.predict(fits["B"], inputs)
    with pytest.raises(ValueError, match="measured configurations"):
        v36.predict(fits["A"], [{**v36.basic_input(rows[0], False), "config": .5}])


@pytest.mark.parametrize("key", ("size", "step"))
def test_held_out_compressed_outcomes_cannot_change_fits_or_predictions(key):
    rows = synthetic_rows()
    before = v36.cross_validate(rows, key)
    target = before["folds"][0]["held_out"]
    altered = copy.deepcopy(rows)
    for i, r in enumerate(altered):
        if r[key] == target:
            r["observed"] += 100+i
            r["loss"] += 100+i
    after = v36.cross_validate(altered, key)
    assert before["folds"][0]["fits"] == after["folds"][0]["fits"]
    p = lambda result: [r["predictions"] for r in result["records"] if r[key] == target]
    assert p(before) == p(after)
    assert before["folds"][0]["A_mae"] != after["folds"][0]["A_mae"]


def test_different_config_shapes_are_fitted_directly_without_shared_source_amplitude():
    rows = synthetic_rows()
    fit = v36.fit_direct(rows, True)
    changed = copy.deepcopy(rows)
    for row in changed:
        if row["config"] == .6:
            row["observed"] += 10
    after = v36.fit_direct(changed, True)
    inputs = [v36.basic_input(r, True) for r in rows]
    difference = v36.predict(after, inputs)-v36.predict(fit, inputs)
    np.testing.assert_allclose(difference, [10 if r["config"] == .6 else 0 for r in rows], atol=1e-11)


def test_paired_uncertainty_resamples_whole_groups_and_keeps_pairing():
    rows = synthetic_rows()
    for r in rows:
        error = 2 + v36.SIZES.index(r["size"])*3
        r["predictions"] = {"A": r["observed"]+error, "B": r["observed"]+error-1}
    m = v36.paired_metrics(rows, "size")
    assert m["A_mae"] == pytest.approx(5)
    assert m["B_mae"] == pytest.approx(4)
    assert m["improvement_ci95"] == pytest.approx([1, 1])
    assert m["A_mae_ci95"] == pytest.approx([2, 8])
    assert m["leave_one_group_out_range"] == pytest.approx([1, 1])
    assert m["n_groups"] == 3 and m["n"] == 36


def test_raw_trends_check_both_transitions_and_keep_unstable_ratios():
    rows = synthetic_rows()
    trajectories = {.9: [0., .001, -.001], .8: [-1., -2., -3.],
                    .7: [.1, .2, .3], .6: [1., 3., 2.]}
    for r in rows:
        r["observed"] = trajectories[r["config"]][v36.STEPS.index(r["step"])]
        r.update(v36.response_flags("pruning", r["config"], r["observed"]))
    result = v36.raw_trends(rows)
    assert result["n_trajectories"] == 12
    assert result["monotonicity_counts"] == {"increasing": 3, "decreasing": 3, "flat": 0, "nonmonotonic": 6}
    for t in result["trajectories"]:
        assert t["observed"] == trajectories[t["config"]]
        if t["config"] in (.9, .8):
            assert t["late_over_early_positive_damage"] is None
        if t["config"] == .6:
            assert t["late_minus_early"] > 0 and t["monotonicity"] == "nonmonotonic"


@pytest.mark.skipif(not _full_grid_on_disk(), reason=_NO_GRID_REASON)
def test_own_dense_references_complete_grid_and_no_censoring(summary):
    rows, hashes = v36.load_grid()
    assert len(rows) == 216 and len(hashes) == 18
    assert len({r["row_id"] for r in rows}) == 216
    for row in rows:
        assert row["observed"] == row["loss"]-row["L0"]
        assert row["D0"] == v36.training_tokens(row["step"])
    assert any(r["near_zero"] for r in rows)
    assert any(r["negative"] for r in rows)
    assert sum(r["int3"] for r in rows) == 27
    assert any(r["large_damage"] for r in rows)
    for arm, cap in itertools.product(v36.ARMS, v36.CAPS):
        expected = {r["row_id"] for r in rows if (r["arm"], r["capability"]) == (arm, cap)}
        for h in summary["results"][arm][cap]["holdouts"].values():
            assert h["metrics"]["n"] == 36
            assert {r["row_id"] for r in h["records"]} == expected


@pytest.mark.parametrize("arm", v36.ARMS)
def test_input_validation_no_missing_anchor_config_or_nonfinite_values(arm):
    anchor = v36.FILES[arm][2]
    table = {anchor: dict.fromkeys(v36.CAPS, 2.),
             **{f"{q:g}": dict.fromkeys(v36.CAPS, 1.5 if i < 3 else 12.)
                for i, q in enumerate(v36.CONFIGS[arm])}}
    dense, compressed = v36.parse_losses(table, arm)
    assert dense["math"] == 2. and len(compressed) == 4
    no_anchor = {k: v for k, v in table.items() if k != anchor}
    with pytest.raises(ValueError, match="true dense"):
        v36.parse_losses(no_anchor, arm)
    bad = copy.deepcopy(table)
    bad.pop(f"{v36.CONFIGS[arm][0]:g}")
    with pytest.raises(ValueError, match="Incomplete"):
        v36.parse_losses(bad, arm)
    bad = copy.deepcopy(table)
    bad[f"{v36.CONFIGS[arm][0]:g}"]["math"] = float("nan")
    with pytest.raises(ValueError, match="Nonfinite"):
        v36.parse_losses(bad, arm)
    bad = copy.deepcopy(table)
    bad[f"{v36.CONFIGS[arm][0]:.2f}"] = dict.fromkeys(v36.CAPS, 2.)
    with pytest.raises(ValueError, match="duplicate"):
        v36.parse_losses(bad, arm)


def test_duplicate_detection_compares_full_payload_not_just_bytes_or_deltas():
    rows = []
    for arm, cap in itertools.product(v36.ARMS, v36.CAPS):
        for r in synthetic_rows(arm):
            r['capability'] = cap
            r['row_id'] = r['row_id'].replace('|math|', f'|{cap}|')
            r['source'] = f"{arm}/{r['cell']}.json"
            rows.append(r)
    hashes = {r['source']: r['source'] for r in rows}
    assert not v36.checkpoint_integrity(rows, hashes)['duplicate_step_payloads']
    for arm, cap in itertools.product(v36.ARMS, v36.CAPS):
        for config in v36.CONFIGS[arm]:
            matched = [r for r in rows if (r['arm'], r['capability'], r['size'], r['config']) ==
                       (arm, cap, '2.8b', config)]
            for r in matched:
                r['L0'], r['loss'] = matched[0]['L0'], matched[0]['loss']
                hashes[r['source']] = arm + '-duplicate'
    integrity = v36.checkpoint_integrity(rows, hashes)
    # Simulate the provenance issue observed in the supplied 2.8B grid.
    assert not integrity["headline_ready"]
    assert integrity["unique_size_step_payloads_per_arm"] == dict.fromkeys(v36.ARMS, 7)
    assert len(integrity["duplicate_step_payloads"]) == 2
    for g in integrity["duplicate_step_payloads"]:
        assert g["size"] == "2.8b" and g["steps"] == list(v36.STEPS) and g["byte_identical"]
    # Different serialization hashes still cannot hide equal numeric panels.
    fake_hashes = {path: str(i) for i, path in enumerate(hashes)}
    assert len(v36.checkpoint_integrity(rows, fake_hashes)["duplicate_step_payloads"]) == 2
    changed = copy.deepcopy(rows)
    for r in changed:
        if r["size"] == "2.8b":
            # Keep all deltas equal but alter the full measured loss panels.
            r["L0"] += r["step"]*.0001
            r["loss"] += r["step"]*.0001
    assert not v36.checkpoint_integrity(changed, fake_hashes)["duplicate_step_payloads"]


def test_rank_deficiency_and_arm_pooling_are_rejected():
    rows = synthetic_rows()
    with pytest.raises(ValueError, match="one arm"):
        v36.fit_direct(rows+synthetic_rows("quantization"), True)
    for r in rows:
        r["L0"] = 2 + np.log(r["D0"])
    with pytest.raises(ValueError, match="Rank-deficient"):
        v36.fit_direct(rows, True)


def test_artifact_is_finite_deterministic_and_qualified(summary):
    serialized = json.dumps(summary, allow_nan=False)
    assert json.loads(serialized) == summary
    assert v36.build_summary() == summary
    report = v36.render(summary)
    for phrase in ("fixed-training-recipe SERIES", "architectures and hyperparameters still vary",
                   "Distillation is PENDING", "flaky GPUs", "raw configuration-level",
                   "leave-one-SIZE-out", "leave-one-STEP-out", "retrospective",
                   "no censoring"):
        assert phrase.lower() in report.lower()
    if summary['integrity']['duplicate_step_payloads']:
        assert 'PROVISIONAL' in report and 'not yet a validated headline' in report
    assert summary["new_model_runs"] == 0


def test_dry_run_does_not_write(monkeypatch, summary, capsys):
    monkeypatch.setattr(v36, "build_summary", lambda: summary)

    def forbidden(*args, **kwargs):
        raise AssertionError("dry run attempted a write")

    monkeypatch.setattr(v36, "write_outputs", forbidden)
    monkeypatch.setattr(v36.Path, "write_text", forbidden)
    v36.main(["--dry-run"])
    assert "Pythia controlled training-history" in capsys.readouterr().out


def test_changed_input_prevents_output(tmp_path, summary):
    changed = copy.deepcopy(summary)
    source = tmp_path / "source.json"
    source.write_text("{}")
    changed["input_sha256"] = {"source.json": "incorrect"}
    with pytest.raises(RuntimeError, match="Input changed"):
        v36.write_outputs(changed, v36.render(changed), root=tmp_path)
    assert not (tmp_path / "results").exists()


@pytest.mark.skipif(not _full_grid_on_disk(), reason=_NO_GRID_REASON)
def test_cpu_only_import_and_run_without_torch_or_transformers():
    result = subprocess.run([sys.executable, "-c", (
        "import sys; from analysis import v36_pythia_controlled_fit as v; "
        "v.build_summary(); assert 'torch' not in sys.modules; "
        "assert 'transformers' not in sys.modules")], cwd=v36.ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
