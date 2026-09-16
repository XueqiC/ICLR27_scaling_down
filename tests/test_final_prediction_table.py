import json
import re
from statistics import mean

import pytest

from analysis import final_prediction_table as gen
from analysis.paper_artifacts import ROOT
from paper_generator_checks import check_access, refuses_symlink, refuses_external_io, refuses_output_file_symlink


TASKS = [
    "Pruning, new source state", "Quantization, unseen bit-width",
    "Quantization, unseen group size", "Quantization, new state", "Distillation, new pool",
]


def raw_source(ref):
    path, pointer = ref.split("#", 1)
    assert path.startswith(("results/", "docs/"))
    data = json.loads((ROOT/path).read_text())
    for key in pointer.lstrip("/").split("/") if pointer else []:
        key = key.replace("~1", "/").replace("~0", "~")
        data = data[int(key)] if isinstance(data, list) else data[key]
    return data


@pytest.fixture(scope="module")
def generated():
    tex_path = ROOT/"generated/tables/main_prediction_v2.tex"
    side_path = ROOT/"generated/tables/main_prediction_v2_sources.md"
    original_tex, original_side = tex_path.read_bytes(), side_path.read_bytes()
    rows, audit, access = gen.generate(write_tex=False)
    check_access(audit, access)
    tex = gen.render_table(rows, audit, sidecar="main_prediction_v2_preview_sources.md")
    assert tex_path.read_bytes() == original_tex
    assert side_path.read_bytes() == original_side
    assert all(not p.endswith(".tex") for p in access[1])
    side = (ROOT/"generated/tables/main_prediction_v2_preview_sources.md").read_text()
    blocks = re.findall(r"```json\n(.*?)\n```", side, re.S)
    return rows, audit, tex, json.loads(blocks[0]), json.loads(blocks[1])


def test_table_every_cell_matches_sidecar_and_frozen_json(generated):
    rows, audit, tex, recipes, caption = generated
    assert len(rows) == 5 and len(recipes) == 25
    body = tex.split("\\midrule\n", 1)[1].split("\\bottomrule", 1)[0]
    table_rows = [line.removesuffix(r" \\").split(" & ") for line in body.splitlines()]
    assert len(table_rows) == 5 and all(len(row) == 5 for row in table_rows)
    assert {(r["row"], r["column"]) for r in recipes} == {
        (i, j) for i in range(5) for j in range(5)}
    for record in [*recipes, caption]:
        # Independently resolve every JSON pointer and operation, then compare
        # with the actual TeX cell at the sidecar's physical row/column.
        parts = []
        for p in record["parts"]:
            if isinstance(p, str):
                assert not re.search(r"\d", p), "Printed numeric literals need JSON provenance"
                parts.append(p)
                continue
            assert p["sources"]
            vals = [raw_source(s) for s in p["sources"]]
            op = p["op"]
            if op == "identity": v = vals[0]
            elif op == "mean": v = mean(vals)
            elif op == "length": v = len(vals[0])
            elif op == "keys": v = ", ".join(vals[0])
            elif op == "unique": v = ", ".join(str(v) for v in sorted(set(vals)))
            elif op == "join": v = ", ".join(map(str, vals[0]))
            elif op == "shared":
                assert vals and all(v == vals[0] for v in vals)
                v = vals[0]
            elif op == "label":
                assert vals == p["expected"]
                v = p["label"]
            else: pytest.fail(f"Unhandled source operation {op}")
            assert v is not None
            parts.append(format(v, p["format"]) if p["format"] else str(v))
        assert record["context"] or any(isinstance(p, dict) for p in record["parts"])
        for ref in record["context"]:
            raw_source(ref)
        expected = r"\newline ".join("".join(
            token if token.startswith("$") else gen.tex_escape(token)
            for token in re.split(r"(\$[^$]+\$)", line)) for line in "".join(parts).split("\n"))
        if "row" in record:
            actual = table_rows[record["row"]][record["column"]]
        else:
            actual = next(line for line in tex.splitlines() if line.startswith(r"\caption{"))[9:-1].removeprefix(r"\footnotesize ")
        assert actual == expected == record["rendered"]
    sources = [s for r in recipes for p in r["parts"] if isinstance(p, dict) for s in p["sources"]]
    assert not any(forbidden in s for s in sources for forbidden in (
        "strongest_observed_baseline", "strongest_baseline_mae", "v72-prune-repeat", "paired_difference",
        "a2-curvature-interaction"))
    # Each task retains its freeze/selection provenance after the status column is removed.
    assert all(row[0].context for row in rows)
    assert any("frozen_at_utc" in ref for row in rows for ref in row[0].context)


def test_only_complete_frozen_tasks_and_compact_layout(generated):
    rows, audit, tex, recipes, caption = generated
    assert [row[0].plain(audit).replace("\n", " ") for row in rows] == TASKS
    assert all(len(row) == 5 for row in rows)
    assert all(c.plain(audit).strip() for row in rows for c in row)
    assert not re.search(r"not tested|n/a", tex, re.I)
    assert "unseen density" not in tex and "budget response" not in tex and "data-reuse response" not in tex
    assert gen.HEADERS == ("Task", "Relation (parameters)", "Tested range", "Error (nats)", "Baseline (nats)")
    assert " & ".join(gen.HEADERS) + r" \\" in tex
    assert "Status" not in tex
    assert r"\centering\footnotesize" in tex and r"\tiny" not in tex
    assert tex.count(r"\begin{table*}[!htbp]") == 1
    assert tex.count(r"\begin{tabular*}{\textwidth}") == 1
    assert r"\extracolsep{\fill}" in tex
    assert tex.count(r">{\raggedright\arraybackslash\hspace{0pt}}p{") == 5
    assert sum(map(float, gen.COLUMN_WIDTHS)) == pytest.approx(1)
    assert not re.search(r"\\(?:resizebox|scalebox|rotatebox|multicolumn)", tex)
    assert r"\label{tab:main-prediction-v2}" in tex
    assert caption["rendered"] == gen.CAPTION
    assert len(gen.CAPTION.split()) <= 60
    assert "Math / Code / QA" in gen.CAPTION
    assert gen.CAPTION.endswith("All rows are predictions frozen before measurement.")
    assert r"$A_c(\mathbf x)\,((1-d)/0.3)^{\gamma_c}$ (5)" in tex
    assert "per-bit source regression (20)" in tex
    assert r"piecewise interpolation over the $(b,g)$ grid (9)" in tex
    assert "QA: zero change (0)" in tex
    assert r"Math, Code: $a_c\log(1+E)$ (1); QA: $u(a+bu+qv)$ (3)" in tex
    assert r"\phi" not in tex and "$m_c" not in tex
    for row in rows:
        assert len(row[2].plain(audit).split()) <= 16
        assert "@step" not in row[2].plain(audit)
        assert len(row[3].plain(audit).split(" / ")) == 3
        assert "[" not in row[3].plain(audit)  # No gain intervals relabelled as MAE intervals.


def test_frozen_errors_and_development_baselines_unchanged(generated):
    rows, audit, _, _, _ = generated
    assert [row[3].plain(audit) for row in rows] == [
        "0.24 / 0.24 / 0.68", "0.19 / 0.21 / 0.44", "0.21 / 0.56 / 0.46",
        "0.30 / 0.15 / 0.22", "0.07,0.06 / 0.02,0.05 / 0.51,0.46",
    ]
    assert [row[4].plain(audit) for row in rows] == [
        "per-density / per-density / development median\n0.23 / 0.22 / 0.22",
        "development median\n0.55 / 0.73 / 0.54",
        "bilinear / zero change / development median\n0.33 / 0.68 / 0.46",
        "bilinear / zero change / development median\n0.35 / 0.56 / 0.14",
        "budget, loss surface / reuse / reuse, size surface\n0.07,0.03 / 0.02,0.05 / 0.61,0.45",
    ]
    assert "270M, 1B pairs" in rows[-1][2].plain(audit)
    assert "no student averaging" in rows[-1][3].note


@pytest.mark.parametrize("task_index", [0, 1])
@pytest.mark.parametrize("column", [3, 4])
def test_absent_score_omits_whole_task(generated, monkeypatch, task_index, column):
    rows, _, _, _, _ = generated
    ref = next(p["sources"][0] for p in rows[task_index][column].parts
               if isinstance(p, dict) and p["format"])
    resolve = gen.resolve
    monkeypatch.setattr(gen, "resolve", lambda audit, source: None if source == ref else resolve(audit, source))
    audit = gen.Artifacts()
    complete = gen.build(audit)
    assert [row[0].plain(audit).replace("\n", " ") for row in complete] == [task for i, task in enumerate(TASKS) if i != task_index]
    assert not re.search(r"not tested|n/a", gen.render_table(complete, audit), re.I)
    assert any("Task omitted" in note for note in audit.notes)


@pytest.mark.parametrize("component", ["parent", "directory"])
def test_table_refuses_symlinked_output(tmp_path, component):
    refuses_symlink(gen, tmp_path, "tables", component)


def test_table_confines_io(tmp_path):
    refuses_external_io(tmp_path)
    refuses_output_file_symlink(tmp_path, "tables", "main_prediction_v2.tex")
