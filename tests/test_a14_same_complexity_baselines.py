"""A14 refits the same-complexity forms on A11's rows and reproduces A11's own predictions first."""
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/a14-same-complexity/summary.json"
A11 = ROOT / "results/a11-efficiency-confirmation"
TABLES = ROOT / "paper/paper/tables"
if not TABLES.exists():
    TABLES = ROOT / "paper/tables"


@pytest.fixture(scope="module")
def results():
    if not OUT.exists():
        pytest.skip("A14 not run in this checkout")
    return json.loads(OUT.read_text())


def test_stored_a11_errors_are_reproduced(results):
    a11 = json.loads((A11 / "summary.json").read_text())
    for budget, stored in (("18", "power_18"), ("36", "power_36")):
        for cap in ("math", "code", "qa"):
            assert results["budgets"][budget]["mae"]["power"][cap] == pytest.approx(a11["by_capability"][cap]["mae"][stored])
            assert results["budgets"][budget]["mae"]["median"][cap] == pytest.approx(a11["by_capability"][cap]["mae"][f"median_curve_{budget}"])
    for cap in ("math", "code", "qa"):
        assert results["budgets"]["36"]["mae"]["per_density"][cap] == pytest.approx(a11["by_capability"][cap]["mae"]["A2_36"])
    # The reduced grid keeps all nine states at each of its two densities, so the per-density regression is
    # determined at 18 measurements as well: two anchors of four coefficients each, scored on the same cells.
    per18 = results["budgets"]["18"]["mae"]["per_density"]
    for cap in ("math", "code", "qa"):
        anchors = results["budgets"]["18"]["fits"][cap]["per_density"]["anchors"]
        assert sorted(anchors) == ["0.7", "0.9"] and all(len(b) == 4 for b in anchors.values())
    for cap in ("math", "code"):   # the paper's claim: the compact form leads at 18, the regression at 36
        assert per18[cap] > results["budgets"]["18"]["mae"]["power"][cap]
        assert results["budgets"]["36"]["mae"]["per_density"][cap] < results["budgets"]["36"]["mae"]["power"][cap]


def test_forms_parameters_and_status(results):
    assert results["status"] == "RETROSPECTIVE"
    assert set(results["forms"]) == {"power", "linear", "quadratic", "strength", "median", "per_density", "pruning_law"}
    assert results["parameters"]["power"] == 5 and results["parameters"]["linear"] == 4 and results["parameters"]["pruning_law"] == 2
    for budget in ("18", "36"):
        fits = results["budgets"][budget]["fits"]
        for cap in ("math", "code", "qa"):
            assert fits[cap]["linear"]["gamma"] == 1.0
            assert fits[cap]["pruning_law"]["P0"] == pytest.approx(pow(2.718281828459045, fits[cap]["pruning_law"]["log_P0"]))
        assert results["budgets"][budget]["n_cells_per_capability"] == 24
        assert results["k1"][budget]["calibration_density"] == 0.9 and results["k1"][budget]["n_cells_per_capability"] == 20


def test_compact_form_is_most_accurate_on_math_and_code(results):
    for budget in ("18", "36"):
        mae = results["budgets"][budget]["mae"]
        for cap in ("math", "code"):
            rivals = [mae[f][cap] for f in ("linear", "quadratic", "strength", "median", "pruning_law")]
            assert mae["power"][cap] < min(rivals)


def test_table_matches_summary(results):
    from analysis import a14_same_complexity_baselines as gen
    table = (TABLES / "a14_baselines.tex").read_text()
    assert table == gen.render(results)
    assert r"\label{tab:a14-baselines}" in table and "sengupta2025compression" in table
