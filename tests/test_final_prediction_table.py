import json
import re
from statistics import mean

import pytest

from analysis import final_prediction_table as gen
from analysis.paper_artifacts import ROOT
from paper_generator_checks import check_access, refuses_symlink, refuses_external_io, refuses_output_file_symlink


TASKS = [
    "Pruning, new state", "Quantization, unseen bit-width",
    "Quantization, unseen group size", "Quantization, new state", "Distillation, new pool",
]


def raw_source(ref):
    path, pointer = ref.split("#", 1)
    assert path.startswith(("results/", "paper/docs/"))
    data = json.loads((ROOT/path).read_text())
    for key in pointer.lstrip("/").split("/") if pointer else []:
        key = key.replace("~1", "/").replace("~0", "~")
        data = data[int(key)] if isinstance(data, list) else data[key]
    return data


@pytest.fixture(scope="module")
def generated():
    tex_path = ROOT/"paper/paper/tables/main_prediction_v2.tex"
    side_path = ROOT/"paper/paper/tables/main_prediction_v2_sources.md"
    original_tex, original_side = tex_path.read_bytes(), side_path.read_bytes()
    rows, audit, access = gen.generate(write_tex=False)
    check_access(audit, access)
    tex = gen.render_table(rows, audit, sidecar="main_prediction_v2_preview_sources.md")
    assert tex_path.read_bytes() == original_tex
    assert side_path.read_bytes() == original_side
    assert all(not p.endswith(".tex") for p in access[1])
    side = (ROOT/"paper/paper/tables/main_prediction_v2_preview_sources.md").read_text()
    blocks = re.findall(r"```json\n(.*?)\n```", side, re.S)
    return rows, audit, tex, json.loads(blocks[0]), json.loads(blocks[1])


def test_table_every_cell_matches_sidecar_and_frozen_json(generated):
    rows, audit, tex, recipes, caption = generated
    assert len(rows) == 5 and len(recipes) == 30
    body = tex.split("\\midrule\n", 1)[1].split("\\bottomrule", 1)[0]
    table_rows = [line.removesuffix(r" \\").split(" & ") for line in body.splitlines()]
    assert len(table_rows) == 5 and all(len(row) == 6 for row in table_rows)
    assert {(r["row"], r["column"]) for r in recipes} == {
        (i, j) for i in range(5) for j in range(6)}
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
            elif op == "weighted_mean": v = sum(vals[i] * vals[i+1] for i in range(0,len(vals),2)) / sum(vals[1::2])
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
        "strongest_observed_baseline", "strongest_baseline_mae", "v72-prune-repeat",
        "a2-curvature-interaction"))
    # Each task retains its freeze/selection provenance after the status column is removed.
    assert all(row[0].context for row in rows)
    assert any("frozen_at_utc" in ref for row in rows for ref in row[0].context)


def test_only_complete_frozen_tasks_and_compact_layout(generated):
    rows, audit, tex, recipes, caption = generated
    assert [row[0].plain(audit).split("\n")[0] for row in rows] == TASKS
    assert all(len(row) == 6 for row in rows)
    assert all(c.plain(audit).strip() for row in rows for c in row)
    assert all(c.strip() for line in tex.split("\\midrule\n", 1)[1].split("\\bottomrule", 1)[0].splitlines()
               for c in line.removesuffix(r" \\").split(" & "))
    assert not re.search(r"not tested|n/a", tex, re.I)
    assert "unseen density" not in tex
    assert gen.HEADERS == (
        "Task (test set)", "Frozen candidate", "Candidate error (Math / Code / QA)",
        "Delivered relation", "Delivered error (Math / Code / QA)",
        "Baseline error (Math / Code / QA)")
    assert " & ".join(gen.HEADERS) + r" \\" in tex
    assert r"\centering\footnotesize" in tex and r"\tiny" not in tex
    assert tex.count(r"\begin{table*}[!htbp]") == 1
    assert tex.count(r"\begin{tabular*}{\textwidth}") == 1
    assert r"\extracolsep{\fill}" in tex
    assert tex.count(r">{\raggedright\arraybackslash\hspace{0pt}}p{") == 6
    assert sum(map(float, gen.COLUMN_WIDTHS)) == pytest.approx(1)
    assert not re.search(r"\\(?:resizebox|scalebox|rotatebox|multicolumn)", tex)
    assert r"\label{tab:main-prediction-v2}" in tex
    assert caption["rendered"] == gen.CAPTION
    assert len(gen.CAPTION.split()) <= 60
    assert "Math / Code / QA" in gen.CAPTION
    assert gen.CAPTION.startswith("Frozen prediction tasks by method.")
    assert "mean absolute errors in native-token nats" in gen.CAPTION
    assert "selected within development folds" in gen.CAPTION
    assert "on the same cells." in gen.CAPTION
    assert "Daggers mark delivered rules fixed after the test." in gen.CAPTION
    assert "Brackets give stored paired intervals of baseline minus candidate." in gen.CAPTION
    assert "Distillation entries list the 270M and 1B students in that order." in gen.CAPTION
    assert "blank" not in gen.CAPTION.lower()
    assert "All rows are predictions frozen" not in gen.CAPTION
    assert "three checkpoints outside every fit; includes 6.9B" in tex
    assert "Power form" in tex and "Source-free median density curve" in tex
    assert "Source surface" in tex and "joint budget/pool" in tex
    assert r"$u(a+bu+qv)$" not in tex and r"$A_c" not in tex
    for i, row in enumerate(rows):
        assert len(row[2].plain(audit).split(" / ")) == 3
        assert len(row[5].plain(audit).split(" / ")) == 3
        if i < 4:
            assert "[" not in row[2].plain(audit)
        if i in (0, 2, 3):
            assert r"$^{\dagger}$" in row[3].plain(audit)
            assert row[4].plain(audit).endswith(r"$^{\dagger}$")
        else:
            assert all(r"\dagger" not in c.plain(audit) for c in row)


def test_frozen_errors_and_development_baselines_unchanged(generated):
    rows, audit, _, _, _ = generated
    candidate = [re.sub(r" \[[^]]+\]", "", row[2].plain(audit)).replace(",\n", ", ") for row in rows]
    assert candidate == [
        "0.24 / 0.24 / 0.68", "0.19 / 0.21 / 0.44", "0.21 / 0.56 / 0.46",
        "0.30 / 0.15 / 0.22", "0.07, 0.06 / 0.02, 0.05 / 0.51, 0.46",
    ]
    assert [row[5].plain(audit) for row in rows] == [
        "0.23 / 0.22 / 0.22", "0.55 / 0.73 / 0.54", "0.33 / 0.68 / 0.46",
        "0.35 / 0.56 / 0.14", "0.07, 0.03 / 0.02, 0.05 / 0.61, 0.45",
    ]
    assert [row[4].plain(audit).removesuffix(r"$^{\dagger}$") for row in rows] == [
        "0.28 / 0.21 / 0.22", "0.19 / 0.21 / 0.44", "0.07 / 0.12 / 0.46", "0.09 / 0.15 / 0.14",
        "0.07, 0.06 / 0.02, 0.05 / 0.51, 0.46",
    ]
    assert "270M, 1B pairs" in rows[-1][0].plain(audit)
    assert "no student averaging" in rows[-1][2].note
    assert "per-density / per-density / development median" in rows[0][5].note
    assert "bilinear / zero change / development median" in rows[2][5].note
    assert "budget, loss surface / reuse / reuse, size surface" in rows[-1][5].note


def test_delivery_identities_and_bit_candidate_as_tested(generated):
    rows, audit, _, recipes, _ = generated
    summary = raw_source(gen.S86 + "#/main_rows")
    expected = {"C35": ["median_curve"] * 3,
                "C44": ["same_input_interpolation", "same_input_interpolation", "median"],
                "C46": ["median"] * 3, "C47": ["E", "E", "joint"], "C48": ["E", "E", "joint"]}
    for row in summary:
        if row["row"] in expected:
            assert [row["capabilities"][c]["delivered"] for c in gen.CAPS] == expected[row["row"]]
    # V55's old interpolation is not the later delivered model: its bit-test
    # configurations were added to development before V69 was frozen.
    old_test = raw_source(gen.Q55 + "#/test_sets/bit_test/configs")
    later_dev = raw_source(gen.Q69 + "#/dev_configs")
    assert set(old_test) <= set(later_dev)
    assert rows[1][3].plain(audit) == "Source surface (as tested)"
    assert rows[1][4].plain(audit) == rows[1][2].plain(audit)
    candidate = next(r for r in recipes if (r["row"], r["column"]) == (1, 2))
    delivered = next(r for r in recipes if (r["row"], r["column"]) == (1, 4))
    assert candidate["parts"] == delivered["parts"]
    assert all(r"\dagger" not in c.plain(audit) for c in rows[1])
    assert "different models and boundary rules" in rows[1][4].note
    assert rows[-1][1].plain(audit) == rows[-1][3].plain(audit)


def test_composite_relations_name_each_capability(generated):
    rows, audit, tex, _, _ = generated
    for i in (2, 3):
        assert rows[i][1].plain(audit) == "Math: source surface;\nCode: median;\nQA: zero change"
    assert rows[2][3].plain(audit) == "Math: interpolation;\nCode: interpolation;\nQA: median" + r"$^{\dagger}$"
    for column in (1, 3):
        assert rows[-1][column].plain(audit) == "Math: reuse;\nCode: reuse;\nQA: joint budget/pool"
    assert "Surface / median / zero" not in tex
    assert "Math, Code:" not in tex


def test_test_sets_match_frozen_membership(generated):
    rows, audit, _, _, _ = generated
    dev = {s["tag"] for s in raw_source(gen.P53 + "#/dev_states")}
    tags = [raw_source(p + "#/tag") for p in audit.data if p.startswith("results/v53-prune-dev/compare_")]
    assert len(tags) == 3 and not set(tags) & dev
    assert "pythia-6.9b@step80000" in tags
    bits = raw_source(gen.Q55 + "#/test_sets/bit_test")
    assert bits["configs"] == ["b4_g64", "b4_g256"]
    assert {s.split("@")[0] for s in bits["states"]} == {"pythia-160m", "pythia-410m", "pythia-1.4b"}
    quant = raw_source("results/v69-quant-confirm/compare.json#/rows")
    for i, new in [(2, False), (3, True)]:
        rr = [r for r in quant if (r["test_set"] != "development_state_boundary") == new]
        assert {r["state"] for r in rr} == ({"pythia-1.4b@step112000"} if new else
               {"pythia-410m@step143000", "pythia-1.4b@step16000"})
        assert {r["config"] for r in rr} == {f"b{b}_g{g}" for b in (3,4,5)
                                             for g in ((32,128,512) if new else (32,512))}
    assert "410M at step 143k, 1.4B at step 16k" in rows[2][0].plain(audit)
    assert "1.4B at step 112k" in rows[3][0].plain(audit)
    assert "groups 64 and 256" in rows[1][0].plain(audit)
    assert "groups 32 and 512" in rows[2][0].plain(audit)
    assert "groups 32, 128, and 512" in rows[3][0].plain(audit)
    for i in (2, 3):
        assert "bits 3 to 5" in rows[i][0].plain(audit)
    assert all("@" not in row[0].plain(audit) for row in rows)
    assert all(not re.search(r"\b[bg] \d", row[0].plain(audit)) for row in rows)


def test_paired_intervals_match_candidate_and_baseline(generated):
    rows, audit, _, recipes, _ = generated
    groups = raw_source(gen.D70 + "#/groups")
    freeze = raw_source("results/v70-distill-confirm/freeze.json#")
    displayed = re.findall(r"\[([^]]+)\]", rows[-1][2].plain(audit))
    assert len(displayed) == 6
    for text, index in zip(displayed, (0,3,1,4,2,5)):
        g = groups[index]
        assert g["selected"] == freeze["selected"][g["capability"]]["method"]
        assert g["strongest_baseline"] == freeze["strongest_baseline"][g["student"]][g["capability"]]["method"]
        assert g["paired_difference"]["estimate"] == pytest.approx(g["baseline_mae"] - g["candidate_mae"])
        assert text == ", ".join(format(v, ".4f") for v in g["paired_difference"]["ci95"])
    for record in recipes:
        for part in record["parts"]:
            if isinstance(part, dict) and any("paired_difference" in s for s in part["sources"]):
                assert (record["row"],record["column"]) == (4,2)
                assert all("/ci95/" in s for s in part["sources"])
    assert gen.paired_interval(audit, gen.D70, "bootstrap") == []


def test_every_numeric_occurrence_is_in_inventory(generated):
    rows, audit, tex, recipes, caption = generated
    numbers = gen.numeric_inventory([*recipes, caption], audit)
    body = tex.split("\\midrule\n")[1].split("\\bottomrule")[0]
    assert [n["number"] for n in numbers] == re.findall(r"[-+]?\d+(?:\.\d+)?|\b(?:three|six)\b", body + caption["rendered"])
    assert all(n["sources"] for n in numbers)
    caption_numbers = [n for n in numbers if n.get("location") == "caption"]
    assert [n["number"] for n in caption_numbers] == ["270", "1"]
    assert all(n["sources"] == ["results/v70-distill-confirm/freeze.json#/confirmation_register/students"]
               for n in caption_numbers)
    side = gen.output_path(ROOT, "tables", "main_prediction_v2_preview_sources.md").read_text()
    blocks = re.findall(r"```json\n(.*?)\n```", side, re.S)
    assert json.loads(blocks[2]) == numbers


@pytest.mark.parametrize("task_index", [0, 1])
@pytest.mark.parametrize("column", [2, 5])
def test_absent_score_omits_whole_task(generated, monkeypatch, task_index, column):
    rows, _, _, _, _ = generated
    ref = next(p["sources"][0] for p in rows[task_index][column].parts
               if isinstance(p, dict) and p["format"])
    resolve = gen.resolve
    monkeypatch.setattr(gen, "resolve", lambda audit, source: None if source == ref else resolve(audit, source))
    audit = gen.Artifacts()
    complete = gen.build(audit)
    assert [row[0].plain(audit).split("\n")[0] for row in complete] == [task for i, task in enumerate(TASKS) if i != task_index]
    assert not re.search(r"not tested|n/a", gen.render_table(complete, audit), re.I)
    assert any("Task omitted" in note for note in audit.notes)


def test_missing_delivered_score_omits_incomplete_task(generated, monkeypatch):
    rows, _, _, _, _ = generated
    ref = rows[0][4].parts[0]["sources"][0]
    resolve = gen.resolve
    monkeypatch.setattr(gen, "resolve", lambda audit, source: None if source == ref else resolve(audit, source))
    audit = gen.Artifacts()
    complete = gen.build(audit)
    assert [row[0].plain(audit).split("\n")[0] for row in complete] == TASKS[1:]
    assert all(c.plain(audit).strip() for row in complete for c in row)
    assert any("Task omitted" in note for note in audit.notes)


@pytest.mark.parametrize("component", ["parent", "directory"])
def test_table_refuses_symlinked_output(tmp_path, component):
    refuses_symlink(gen, tmp_path, "tables", component)


def test_table_confines_io(tmp_path):
    refuses_external_io(tmp_path)
    refuses_output_file_symlink(tmp_path, "tables", "main_prediction_v2.tex")
