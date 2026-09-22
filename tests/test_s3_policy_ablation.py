"""The retrospective S3 policy ablation reuses the sealed outcomes and never rewrites them."""
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/s3-policy-ablation/summary.json"
S3 = ROOT / "results/s3-selection-validation"
TABLES = ROOT / "paper/paper/tables"
if not TABLES.exists():  # public checkout: the LaTeX tree sits one level up
    TABLES = ROOT / "paper/tables"


@pytest.fixture(scope="module")
def summary():
    if not OUT.exists():
        pytest.skip("ablation not run in this checkout")
    return json.loads(OUT.read_text())


def test_sealed_policies_are_reproduced_cell_for_cell(summary):
    score = json.loads((S3 / "score_v2.json").read_text())
    sealed = {(r["reference"], r["objective"], r["budget"]): r for r in score["rows"]}
    assert summary["status"] == "RETROSPECTIVE"
    assert summary["sealed_checks"] == 2 * len(score["rows"]) == 544
    for row in summary["rows"]:
        s = sealed[row["reference"], row["objective"], row["budget"]]
        for name in ("rule", "quant-only"):
            assert row["policies"][name]["selected"] == s["policies"][name]["selected"]
            assert row["policies"][name]["regret"] == pytest.approx(s["policies"][name]["regret"]) if s["policies"][name]["regret"] is not None else row["policies"][name]["regret"] is None
    for ref in score["references"]:
        for objective, o in ref["objectives"].items():
            e = summary["per_reference"][f"{ref['reference']}|{objective}"]
            assert e["n_paired"] == o["n_paired"]
            assert e["rule"]["mean_regret"] == pytest.approx(o["policies"]["rule"]["mean_paired_regret"])
            assert e["quant-only"]["mean_regret"] == pytest.approx(o["policies"]["quant-only"]["mean_paired_regret"])
            assert e["mean_opportunity"] == pytest.approx(o["mean_opportunity"])


def test_decomposition_identity_and_residual(summary):
    for row in summary["rows"]:
        if not row["paired"]:
            continue
        q = row["policies"]["quant-only"]
        assert q["regret"] == pytest.approx((q["actual_loss"] - row["quant_oracle_loss"]) + row["opportunity"])
        for name, p in row["policies"].items():
            if p["actual_loss"] is not None:
                assert p["quant_residual"] == pytest.approx(row["quant_oracle_loss"] - p["actual_loss"])
    for key, e in summary["per_reference"].items():
        d = e["decomposition"]
        assert d["quant_gap"] == pytest.approx(d["quant_internal"] + d["opportunity"])


def test_priority_orders_come_from_the_development_panel(summary):
    v64 = json.loads((ROOT / "results/v64-selection-feasible/summary.json").read_text())
    for key, entry in summary["development_priority"].items():
        objective, budget = key.split("@")
        cells = [c for c in v64["cells"][objective] if abs(c["budget"] - float(budget)) < 1e-9]
        assert sum(entry["wins"].values()) == len(cells) == 17
        assert entry["order"][0] == max(entry["rate"], key=lambda m: (entry["rate"][m], entry["wins"][m], -["prune", "quant", "distill", "dense"].index(m)))
    # Question answering prefers the student wherever it was a development candidate.
    assert summary["development_priority"]["qa@0.50"]["order"][0] == "distill"
    assert summary["development_priority"]["math@0.50"]["order"][0] == "quant"


def test_table_matches_summary_and_names_retrospective_status(summary):
    from analysis import s3_policy_ablation as gen
    table = (TABLES / "s3_ablation.tex").read_text()
    assert table == gen.render(summary)
    assert "defined after the round" in table and r"\label{tab:s3-ablation}" in table
    assert "-0.000" not in table
    assert summary["references_without_D0"] == 2
