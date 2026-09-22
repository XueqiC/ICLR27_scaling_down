"""P2 Wanda panel: the fit helpers, the summary consistency and the table renderer (CPU only)."""
import json
import math
from pathlib import Path

import pytest

from analysis import p2_wanda_panel as p2

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "results/p2-wanda-panel/summary.json"


def test_power_fit_recovers_scale_and_exponent():
    ds = [0.9, 0.8, 0.7, 0.6, 0.5]
    ys = [2.0 * (1 - d) ** 3.0 for d in ds]
    A, alpha = p2.fit_power(ds, ys)
    assert math.isclose(A, 2.0, rel_tol=1e-9) and math.isclose(alpha, 3.0, rel_tol=1e-9)
    A_fixed, alpha_fixed = p2.fit_power(ds, ys, alpha=3.0)
    assert math.isclose(A_fixed, 2.0, rel_tol=1e-9) and alpha_fixed == 3.0


@pytest.mark.skipif(not SUMMARY.exists(), reason="Wanda panel not measured in this checkout")
def test_summary_cells_match_the_loss_records_and_magnitude_panel():
    summary = json.loads(SUMMARY.read_text())
    for tag, entry in summary["models"].items():
        rec = json.loads((p2.OUT / tag / "wanda_losses.json").read_text())
        mag = json.loads((ROOT / "results/v6-capability-geometry" / p2.V6_DIRS[tag] / "prune_losses.json").read_text())
        assert rec["scope"] == "block linear layers" and rec["scope_parameters"] < rec["matrix_parameters_total"]
        for d, cells in entry["cells"].items():
            for c in p2.CAPS:
                assert math.isclose(cells[c]["wanda"], rec["losses"][d][c] - rec["losses"]["1.0"][c], abs_tol=1e-12)
                if cells[c]["magnitude"] is not None:
                    assert math.isclose(cells[c]["magnitude"], mag[d][c] - mag["1.0"][c], abs_tol=1e-12)


@pytest.mark.skipif(not SUMMARY.exists(), reason="Wanda panel not measured in this checkout")
def test_rendered_table_has_one_row_per_measured_density_and_no_dashes():
    summary = json.loads(SUMMARY.read_text())
    tex = p2.render(summary)
    n_rows = sum(len(e["cells"]) for e in summary["models"].values())
    assert tex.count(r"\\") - 2 == n_rows  # two header rows
    assert "& -- " not in tex and "---" not in tex
    assert tex.index(r"\caption{") < tex.index(r"\begin{tabular")
