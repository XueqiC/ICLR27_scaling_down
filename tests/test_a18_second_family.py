"""A18 second-family confirmation: the ridge estimator, the leave-one-state-out weight, the forms on a synthetic
development grid, the cost accounting and the freeze guard (CPU only, no network)."""
import json
import math
import re
from statistics import mean

import numpy as np
import pytest

from analysis import a18_second_family as a18
from analysis import v53_prune_dev as p53
from analysis.paper_artifacts import Artifacts
from analysis.paper_table_layout import PROFILES


def test_ridge_at_zero_is_least_squares_and_survives_a_constant_column():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(12, 3)); beta = np.array([1.0, -2.0, 0.5]); y = x @ beta
    assert np.allclose(a18.ridge(x, y, 0.0), beta)
    x0 = np.hstack([x, np.zeros((12, 1))])                 # a standardized constant input is all zeros
    b0 = a18.ridge(x0, y, 0.0)
    assert np.allclose(b0[:3], beta) and b0[3] == 0.0       # minimum-norm solution, no exception
    assert np.linalg.norm(a18.ridge(x, y, 10.0)) < np.linalg.norm(beta)


def synthetic_states(n0s=(1e9, 7e9), d0s=(4e11, 2e12, 3.6e12), amp=0.1, gamma=3.0, noise=0.0, seed=1):
    """Development records whose response is an exact compact power form: y = A(z) * shape(d, gamma)."""
    rng = np.random.default_rng(seed)
    states = {}
    for n0 in n0s:
        for d0 in d0s:
            tag = f"s{n0:.0e}@{d0:.0e}"
            dense = {"probes": {c: {"mean_loss": 1.0 + 0.1 * (c == "code") + 0.05 * math.log10(d0), "tokens": 1000, "items": []} for c in a18.CAPS},
                     "new": {c: {"mean_loss": 1.1, "tokens": 500, "items": []} for c in a18.CAPS}, "seconds": 10.0}
            meas = {"1.0": dense}
            for d in a18.DEV_DENSITIES:
                a = amp * (1 + 0.2 * math.log10(n0 / 1e9)) * (1 + 0.1 * math.log10(d0 / 1e12))
                meas[str(d)] = {"probes": {c: {"mean_loss": dense["probes"][c]["mean_loss"] + a * p53.shape(d, gamma) + noise * rng.normal(), "tokens": 1000, "items": []} for c in a18.CAPS},
                                "new": {c: {"mean_loss": 1.1 + a * p53.shape(d, gamma), "tokens": 500, "items": []} for c in a18.CAPS}, "seconds": 5.0}
            states[tag] = {"tag": tag, "role": "dev", "D0": d0, "N0_block_matrices": int(n0), "load_seconds": 2.0, "measurements": meas, "densities": list(a18.DEV_DENSITIES),
                           "total_seconds": 32.0, "tokens_evaluated": 7500}
    return states


def test_forms_fit_and_predict_an_exact_power_response():
    states = synthetic_states()
    rows = a18.dev_rows(states)
    assert len(rows) == 6 * 4 * 3
    stats = p53.zstats(rows)
    fits = a18.fit_forms(rows, "math", stats)
    assert set(fits) == set(a18.FORMS)
    assert math.isclose(fits["power"]["gamma"], 3.0, abs_tol=0.051)      # the gamma grid has 0.05 steps
    for form in a18.FORMS:
        assert all(math.isfinite(v) for v in np.ravel(fits[form].get("beta", [0.0])))
    z = p53.standardize(p53.raw_features(3e9, 1.2e12, 1.0 + 0.05 * math.log10(1.2e12)), stats)
    truth = 0.1 * (1 + 0.2 * math.log10(3)) * (1 + 0.1 * math.log10(1.2)) * p53.shape(0.75, 3.0)
    assert abs(a18.predict("power", fits["power"], z, 0.75) - truth) < 0.02
    assert abs(a18.predict("per_density", fits["per_density"], z, 0.75) - truth) < 0.05
    assert a18.predict("median", fits["median"], z, 0.75) > 0


def test_leave_one_state_out_uses_the_same_grid_for_every_form():
    states = synthetic_states(noise=0.02)
    rows = a18.dev_rows(states); stats = p53.zstats(rows)
    fits = a18.fit_forms(rows, "code", stats)
    for form in ("power", "quadratic", "per_density"):
        assert fits[form]["lambda"] in a18.LAMBDA_GRID
        assert fits[form]["loso_mae"] >= 0


def test_cost_accounting_counts_anchors_once(tmp_path, monkeypatch):
    states = synthetic_states()
    out = tmp_path / "a18"; (out / "measurements").mkdir(parents=True)
    monkeypatch.setattr(a18, "OUT", out)
    monkeypatch.setattr(a18, "STATES", {t: ("1B", "rev", "dev") for t in states})
    for t, rec in states.items():
        (out / "measurements" / f"{t.replace('@', '__')}.json").write_text(json.dumps(rec))
    c = a18.cost()
    full, red = c["totals"]["full"], c["totals"]["reduced"]
    assert full["configurations"] == 24 and red["configurations"] == 12
    assert math.isclose(full["seconds"], 6 * (2.0 + 10.0 + 4 * 5.0)) and math.isclose(red["seconds"], 6 * (2.0 + 10.0 + 2 * 5.0))
    assert 0.5 < c["totals"]["reduced_over_full"]["seconds"] < 1.0    # the shared dense anchor keeps the saving below half
    assert math.isclose(c["totals"]["reduced_over_full"]["configurations"], 0.5)


def test_frozen_artifacts_are_never_overwritten(tmp_path):
    p = tmp_path / "freeze.json"
    a18.write_new(p, {"a": 1})
    with pytest.raises(SystemExit):
        a18.write_new(p, {"a": 2})
    assert json.loads(p.read_text()) == {"a": 1}


def test_body_table_has_three_evaluations_and_registered_full_grid(tmp_path):
    target = tmp_path / "efficiency_body.tex"
    a18.render_body(target)
    text = target.read_text()
    body = text.split("\\midrule\n", 1)[1].split(r"\bottomrule", 1)[0]
    rows = [line.removesuffix(r" \\").split(" & ") for line in body.splitlines()
            if line != r"\midrule"]
    assert len(rows) == 6 and all(len(row) == 8 for row in rows)
    assert all(cell.strip() and cell not in ("/", "--") for row in rows for cell in row)
    assert [row[:2] for row in rows] == [[word, "Half"] for _, word in a18.BODY_FORMS] + [
        ["Per-density regression", "Full"]]
    assert body.count(r"\midrule") == 1
    assert "\\midrule\nPer-density regression & Full" in body

    a11 = a18.read(a18.ROOT / a18.E11)["by_capability"]
    a14 = a18.read(a18.A14)["budgets"]
    olmo = a18.read(a18.OUT / "score.json")["summary"]
    wanda = a18.read(a18.ROOT / "results/a19-wanda-efficiency/score.json")["summary"]
    for i, row in enumerate(rows):
        form = a18.BODY_FORMS[i][0] if i < 5 else "per_density"
        grid = "reduced" if i < 5 else "full"
        for j, cap in enumerate(("math", "code")):
            if i == 0:
                pythia = a11[cap]["mae"]["power_18"]
            elif i == 5:
                pythia = a11[cap]["mae"]["A2_36"]
            else:
                pythia = a14["18"]["mae"][form][cap]
            assert row[2 + j] == f"{pythia:.3f}"
            assert row[4 + j] == f"{olmo[cap][f'{form}_{grid}']['probes']:.3f}"
            assert row[6 + j] == f"{wanda[cap][f'{form}_{grid}']['probes']:.3f}"
        assert all(re.fullmatch(r"\d+\.\d{3}", cell) for cell in row[2:])
    assert [" ".join(row[2:]) for row in rows] == [
        "0.073 0.104 0.017 0.020 0.023 0.026",
        "0.115 0.142 0.026 0.045 0.026 0.048",
        "0.098 0.166 0.018 0.029 0.031 0.046",
        "0.145 0.212 0.026 0.039 0.029 0.046",
        "0.108 0.136 0.025 0.035 0.024 0.032",
        "0.060 0.084 0.019 0.027 0.025 0.027",
    ]

    caption = "Measurement efficiency of the compact pruning form in three evaluations, mean absolute error in nats per token on four test states in no fit. The half grid keeps every development state at half the densities, 18 of 36 configurations per capability on Pythia and 12 of 24 on OLMo-2; the full-grid per-density regression is the pre-specified comparator. On OLMo-2 and under Wanda every form was pre-specified before the test; on the Pythia magnitude panel the forms other than the compact form are retrospective refits."
    assert f"\\caption{{{caption}}}" in text
    assert text.index(r"\caption{") < text.index(r"\begin{tabular*}")
    assert text.startswith("% Generated by analysis/a18_second_family.py --body; do not edit.\n")
    assert r"\begin{table}[tb]" in text and "\\centering\n\\small\n" in text
    assert r"\label{tab:efficiency-body}" in text
    assert text.count(r"\multicolumn{2}") == 3
    for heading in ("Pythia, magnitude", "OLMo-2, magnitude", "Pythia, Wanda"):
        assert "{" + heading + "}" in text
    assert r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}" in text
    assert r"Form & Grid & Math & Code & Math & Code & Math & Code \\" in text
    weights, = PROFILES["tab:efficiency-body"]
    assert len(weights) == 8 and sum(weights) == pytest.approx(100)


def test_efficiency_confirmation_uses_registered_cells_and_budgets():
    summary = a18.read_efficiency_confirmation()
    assert summary == a18.read(a18.ROOT / a18.E11)
    states = a18.read(a18.ROOT / a18.E11_STATE)
    plan = a18.read(a18.ROOT / "results/a11-efficiency-confirmation/plan.json")
    assert states["state_order"] == ["pythia-160m@step80000", "pythia-410m@step112000",
                                     "pythia-1.4b@step48000", "pythia-1b@step48000"]
    assert not set(states["state_order"]) & set(plan["exclusion_audit"]["a9_states"])
    assert not set(states["state_order"]) & set(plan["exclusion_audit"]["selection_rule_states"])
    assert len(states["records"]) == 12
    assert len(summary["cells"]) == len({(r["state"], r["density"], r["capability"])
                                       for r in summary["cells"]}) == 72
    budgets = states["development_measurements_per_capability"]
    assert budgets["power_18"] == 18
    assert budgets["A2_36"] == budgets["median_curve_36"] == 36
    for cap in a18.CAPS:
        records = [r for r in states["records"] if r["capability"] == cap]
        assert len(records) == 4
        for method in ("power_18", "A2_36", "median_curve_36"):
            for record in records:
                assert record["densities"] == [0.9, 0.85, 0.8, 0.75, 0.7, 0.65]
                assert record["n_cells"] == 6
                assert record["mae"][method] == pytest.approx(mean(record["absolute_errors"][method]))
            assert summary["by_capability"][cap]["mae"][method] == pytest.approx(
                mean(r["mae"][method] for r in records))


@pytest.mark.parametrize("corruption, message", [
    ("mean", "per-state errors"), ("duplicate", "missing or duplicate"),
    ("missing", "missing or duplicate"), ("budget", "half-budget"),
    ("full_budget", "half-budget"), ("digest", "digest mismatch"),
    ("aggregate", "aggregate error disagrees with per-state means"),
])
def test_efficiency_rejects_inconsistent_records(monkeypatch, corruption, message):
    read = Artifacts.read

    def changed_read(audit, path):
        data = read(audit, path)
        if path == a18.E11_STATE:
            if corruption == "mean":
                data["records"][0]["mae"]["power_18"] += 0.1
            elif corruption == "duplicate":
                data["records"].append(data["records"][0])
            elif corruption == "missing":
                data["records"].pop()
            elif corruption == "budget":
                data["development_measurements_per_capability"]["power_18"] += 1
            elif corruption == "full_budget":
                data["development_measurements_per_capability"]["A2_36"] += 1
            elif corruption == "digest":
                data["input_sha256"][a18.E11] = "0" * 64
        if corruption == "aggregate" and path in (a18.E11, a18.E11_STATE):
            data["by_capability"]["math"]["mae"]["power_18"] += 0.1
        return data

    monkeypatch.setattr(Artifacts, "read", changed_read)
    with pytest.raises(ValueError, match=message):
        a18.read_efficiency_confirmation()


@pytest.mark.parametrize("budget, form", [("18", "power"), ("36", "per_density")])
@pytest.mark.parametrize("cap", ["math", "code"])
def test_body_rejects_a14_disagreement(tmp_path, monkeypatch, budget, form, cap):
    read = a18.read

    def changed_read(path):
        data = read(path)
        if path == a18.A14:
            data["budgets"][budget]["mae"][form][cap] += 2e-6
        return data

    monkeypatch.setattr(a18, "read", changed_read)
    target = tmp_path / "efficiency_body.tex"
    with pytest.raises(ValueError, match="A14 .* disagrees with registered A11"):
        a18.render_body(target)
    assert not target.exists()
