"""A18 second-family confirmation: the ridge estimator, the leave-one-state-out weight, the forms on a synthetic
development grid, the cost accounting and the freeze guard (CPU only, no network)."""
import json
import math

import numpy as np
import pytest

from analysis import a18_second_family as a18
from analysis import v53_prune_dev as p53


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
