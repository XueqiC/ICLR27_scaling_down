"""Tables 24 and 28 must score the same stored generations with the same extraction rule.

Regression test for the reviewer-found inconsistency (ICLR 2027 review, D8L2): the "Original" column of the
end-marker control table read 96-token scores stored at run time under an earlier extraction rule, while the
fresh-item table rescored the same generations with the current rule. Both analyses now go through
``p1_long_generation.rescore_long``; their shared rows must agree, and the paired 96-token contrasts must be
reproducible from one per-item source.
"""
import json
import statistics
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
P1 = ROOT / "results/p1-qa-behaviour"
P3 = ROOT / "results/p3-eos-control"


def _read(path):
    return json.loads(path.read_text())


@pytest.mark.skipif(not (P3 / "summary.json").exists() or not (P1 / "summary_long.json").exists(),
                    reason="recorded results not present")
def test_original_student_rows_match_fresh_item_table():
    long = _read(P1 / "summary_long.json")["references"]
    p3 = _read(P3 / "summary.json")["references"]
    assert p3, "end-marker control summary is empty"
    for ref, entry in p3.items():
        row = long[ref]["models"]["distill:new_s3"]
        assert entry["s3_student"]["em_96"] == pytest.approx(row["exact_match_96_span"], abs=1e-12), ref
        assert entry["s3_student"]["f1_96"] == pytest.approx(row["token_f1_96_span"], abs=1e-12), ref
        assert entry["s3_student"]["truncated_96"] == pytest.approx(row["truncated_rate_96"], abs=1e-12), ref


@pytest.mark.skipif(not (P3 / "results.json").exists() or not (P1 / "results_long.json").exists(),
                    reason="recorded generations not present")
def test_paired_96_token_contrast_uses_one_rule():
    from analysis import p1_long_generation as p1l
    from analysis import p1_qa_behaviour as p1
    answers = [it["answer"] for it in p1.read(p1.OUT / "plan.json")["items"]]
    long_all = p1l.rescore_long(_read(P1 / "results_long.json")["evaluations"], answers)
    control = _read(P3 / "results.json")["evaluations"]
    p1l.rescore_long({k: e["long"] for k, e in control.items()}, answers)
    p3 = _read(P3 / "summary.json")["references"]
    for e in control.values():
        ref = e["reference"]
        orig = long_all[f"{ref}|distill:new_s3"]["items"]
        diff = statistics.mean(float(a["em_span"]) - float(b["em_span"]) for a, b in zip(e["long"]["items"], orig))
        assert p3[ref]["paired_control_minus_s3"]["em_96"]["mean"] == pytest.approx(diff, abs=1e-12), ref
