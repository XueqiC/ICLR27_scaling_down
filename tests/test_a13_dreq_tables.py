"""Saved-evidence, publication-language and row-coverage checks for both A13 tables."""
import copy
import hashlib
import json
import os
from pathlib import Path
import re

import pytest

from analysis.a13_dreq_table import (
    ACCOUNTING, FULL_HEADER, OUTCOMES, READOUTS, distance_cell, generate, recommendation_cell,
    render_main_table, render_table,
)


@pytest.fixture(scope="module")
def accounting():
    root = Path(os.environ.get("A13_EVIDENCE_ROOT", Path(__file__).resolve().parents[1]))
    return json.loads((root / ACCOUNTING).read_bytes())


def data_rows(text, main=False):
    return [line.split(" & ") for line in text.splitlines()
            if line.startswith(tuple(READOUTS.values()) if main else ("1 billion &", "4 billion &"))
            and line.endswith(r" \\")]


@pytest.mark.parametrize("renderer,label,count", [
    (render_main_table, "tab:dreq-main", 36),
    (render_table, "tab:dreq-full", 144),
])
def test_complete_unique_rows_plain_language_and_captions(accounting, renderer, label, count):
    text = renderer(accounting["requests"])
    main = renderer is render_main_table
    rows = data_rows(text, main)
    assert len(rows) == count
    assert all(len(row) == 8 for row in rows)
    assert len({tuple(row[:3]) for row in rows}) == (count if main else 24)
    assert text.count(r"\caption{") == text.count(r"\label{") == 1
    assert r"\label{" + label + "}" in text
    assert "largest pool varies the training seed; smaller pools vary the pool draw" in text
    assert "Intervals are observed brackets, not confidence intervals." in text
    assert all(outcome in text for outcome in OUTCOMES.values())
    visible = "\n".join(line for line in text.splitlines() if not line.startswith("%"))
    for forbidden in ("not tested", "no rec.", "n/a", "loglinear", "student monotone",
                      "replicate_ambiguous", "non_monotone", "lower_bound", "upper_bound"):
        assert forbidden not in visible.lower()
    assert not re.search(r"\b(?:1B|4B|MBPP|MATH-500|QA|NA)\b", visible)
    assert "/" not in visible
    assert not re.search(r"tab:dreq-full-\d", text)
    assert r"\resizebox" not in text and r"\shortstack" not in text
    assert "none" in text and "unresolved" in text and r"\geq" in text


def test_primary_table_uses_boundary_recommendations_only(accounting):
    text = render_main_table(accounting["requests"])
    assert r"\begin{table}[!htbp]" in text
    assert text.count(r"\begin{tabular*}{\textwidth}") == 1
    assert r"\scriptsize" in text or r"\footnotesize" in text
    assert r"\tau=0.25" in text
    rows = data_rows(text, main=True)
    for readout in READOUTS.values():
        assert len([r for r in rows if r[0] == readout]) == 6
    assert {r[2] for r in rows} == {"50", "100", "200"}
    # Non-primary evidence and the companion loss rule cannot leak into this table.
    changed = copy.deepcopy(accounting["requests"])
    for row in changed:
        if row["tau"] != .25:
            row["measured_crossing"] = {"status": "not a primary row", "interval": None}
        row["recommendations"]["delta"] = {}
    assert render_main_table(changed) == text
    example = next(r for r in rows if r[:3] == ["Math", "4 billion", "50"])
    assert example[3:5] == ["upper bound at the smallest pool", r"$\leq 9.105$"]
    assert example[5] == r"38.133\newline ($\geq 1.432$)"
    assert example[6] == r"9.004\newline (inside)"


def test_full_table_is_one_continuous_longtable(accounting):
    text = render_table(accounting["requests"])
    assert text.count(r"\begin{longtable}") == text.count(r"\end{longtable}") == 1
    assert r"\begin{table}" not in text and r"\begin{tabular" not in text
    assert text.count(FULL_HEADER) == 2
    assert text.index(r"\caption{") < text.index(r"\endfirsthead") < text.index(r"\endhead")
    assert text.count(r"\endfirsthead") == text.count(r"\endhead") == 1
    assert not any(command in text for command in (r"\clearpage", r"\newpage", r"\pagebreak"))
    for i, name in enumerate(READOUTS.values()):
        heading = r"\multicolumn{8}{@{}l@{}}{\textbf{" + name + r"}} \\*"
        assert text.count(heading) == 1
        block = text.split(heading)[1].split(r"\multicolumn{8}")[0]
        rows = data_rows(block)
        assert len(rows) == len({tuple(r[:3]) for r in rows}) == 24
        assert {r[2] for r in rows} == {"0", "0.1", "0.25", "0.5"}
        assert all("boundary: " in cell and "loss: " in cell for row in rows for cell in row[5:])


@pytest.mark.parametrize("size,distance,inside,status,expected", [
    (None, None, None, "no_recommendation", "none"),
    (32000, None, None, "unresolved_interval", r"32.000\newline (unresolved)"),
    (32000, 0., True, "interval_distance", r"32.000\newline (inside)"),
    (32000, 0., True, "censored_distance_lower_bound", r"32.000\newline (inside)"),
    (32000, 1.25, False, "interval_distance", r"32.000\newline ($1.250$)"),
    (32000, 1.25, False, "censored_distance_lower_bound", r"32.000\newline ($\geq 1.250$)"),
])
def test_recommendations_preserve_abstention_unresolved_and_censoring(size, distance, inside, status, expected):
    assert recommendation_cell(dict(recommended_D_U=size, log_distance=distance,
                                    inside_interval=inside, distance_status=status)) == expected


def test_full_accounting_keeps_censored_zero_distinct_from_finite_zero():
    row = dict(recommended_D_U=32000, log_distance=0., inside_interval=True,
               distance_status="censored_distance_lower_bound")
    assert distance_cell(row) == r"$\geq 0.000$"
    row["distance_status"] = "interval_distance"
    assert distance_cell(row) == "$0.000$"


def test_generate_writes_both_outputs_with_exact_evidence_and_provenance(accounting, tmp_path):
    # A12 sources are deliberately absent: rendering must use only saved A13 evidence.
    source = tmp_path / ACCOUNTING
    source.parent.mkdir(parents=True)
    original = (json.dumps(accounting) + "\n").encode()
    source.write_bytes(original)
    paths = generate(tmp_path)
    assert source.read_bytes() == original
    assert len(paths) == 6
    for stem, count in (("dreq_main", 36), ("dreq_full", 144)):
        table = next(p for p in paths if p.name == stem + ".tex")
        sidecar = json.loads(table.with_suffix(".json").read_text())
        assert sidecar["n_requests"] == len(sidecar["requests"]) == count
        assert sidecar["label"] == "tab:" + stem.replace("_", "-")
        assert sidecar["table_sha256"] == hashlib.sha256(table.read_bytes()).hexdigest()
        assert sidecar["accounting_input_sha256"] == {str(ACCOUNTING): hashlib.sha256(original).hexdigest()}
        assert all(sidecar["input_sha256"][p] == h for p, h in accounting["input_sha256"].items())
        selected = [r for r in accounting["requests"] if stem == "dreq_full" or r["tau"] == .25]
        assert sidecar["requests"] == selected
        assert sidecar["definitions"] == accounting["definitions"]
        assert table.with_name(stem + "_sources.md") in paths
    before = {p: p.read_bytes() for p in paths}
    assert generate(tmp_path) == paths
    assert all(p.read_bytes() == before[p] for p in paths)


def test_duplicate_or_missing_requests_are_rejected(accounting):
    requests = accounting["requests"]
    for bad in (requests[:-1], requests[:-1] + requests[:1]):
        with pytest.raises(ValueError, match="144 distinct"):
            render_main_table(bad)
        with pytest.raises(ValueError, match="144 distinct"):
            render_table(bad)
