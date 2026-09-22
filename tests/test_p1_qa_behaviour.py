"""P1 evaluates fresh question-answering items on the models the selection policies chose."""
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/p1-qa-behaviour"


@pytest.fixture(scope="module")
def plan():
    if not (OUT / "plan.json").exists():
        pytest.skip("P1 not planned in this checkout")
    return json.loads((OUT / "plan.json").read_text())


def test_items_are_fresh_and_unique(plan):
    items = plan["items"]
    assert len(items) == 384 and len({it["index"] for it in items}) == 384
    register = json.loads((ROOT / "results/v71-qa-scope/register.json").read_text())["sets"]["2wiki_new"]
    used = set(register["indices"]) | set(register["excluded_indices"])
    assert not used.intersection(it["index"] for it in items)
    assert plan["items_meta"]["excluded_teacher_questions"] > 0 and plan["items_meta"]["excluded_exemplar_questions"] == 4
    assert all(it["prompt"].endswith("\nAnswer:") and it["completion"].startswith(" ") for it in items)


def test_roster_covers_every_policy_choice_and_the_controls(plan):
    ablation = json.loads((ROOT / "results/s3-policy-ablation/summary.json").read_text())
    by_ref = {}
    for m in plan["models"]:
        by_ref.setdefault(m["reference"], {})[m["candidate"]] = set(m["roles"])
    for row in ablation["rows"]:
        if row["objective"] != "qa" or not row["paired"]:
            continue
        r = by_ref[row["reference"]]
        for name in ("rule", "priority", "quant-only"):
            assert name in r[row["policies"][name]["selected"]]
        assert "oracle" in r[row["oracle_id"]] and "quant_oracle" in r[row["quant_oracle_id"]]
    for ref, r in by_ref.items():
        assert "dense" in r["dense:source"] and "pristine_student" in r["pristine:student"]
    assert len(plan["models"]) == 22


def test_results_and_summary_are_consistent_when_present(plan):
    if not (OUT / "results.json").exists():
        pytest.skip("P1 not run")
    res = json.loads((OUT / "results.json").read_text())
    keys = {f"{m['reference']}|{m['candidate']}" for m in plan["models"]}
    assert set(res["evaluations"]) <= keys
    for e in res["evaluations"].values():
        assert len(e["items"]) == len(plan["items"])
        assert 0 <= e["exact_match"] <= 1 and 0 <= e["token_f1"] <= 1 and e["mean_loss"] > 0
        assert abs(e["mean_loss"] - sum(i["loss_sum"] for i in e["items"]) / sum(i["tokens"] for i in e["items"])) < 1e-9
    if res.get("status") == "COMPLETE":
        assert set(res["evaluations"]) == keys
        summary = json.loads((OUT / "summary.json").read_text())
        for ref, entry in summary["references"].items():
            for cid, row in entry["models"].items():
                for cmp in ("vs_dense", "vs_quant_only"):
                    if cmp in row:
                        for metric in ("loss", "exact_match", "token_f1"):
                            lo, hi = row[cmp][metric]["ci95"]
                            assert lo <= row[cmp][metric]["mean"] <= hi
