import json
import hashlib
import re
from pathlib import Path
from statistics import mean

import pytest

from analysis import final_prediction_table as gen
from analysis.paper_artifacts import ROOT
from analysis.paper_table_layout import group
from paper_generator_checks import check_access, refuses_symlink, refuses_external_io, refuses_output_file_symlink


TASKS = [
    "Three unseen checkpoints pruned to densities 0.575, 0.675 and 0.85",
    "Pythia 410 million and 1.4 billion at unseen group sizes 32 and 512, bit widths 3 to 5",
    "An unseen 1.4 billion stage at bit widths 3 to 5, group sizes 32 to 512",
    "Gemma 270 million distilled on six new pools at 50 to 200 thousand tokens",
    "Gemma 1 billion distilled on six new pools at 50 to 200 thousand tokens",
]


def plain_signs(text):
    """Scores print their sign as math ($-$0.047) so plus and minus share one width."""
    return re.sub(r"\$([+-])\$(?=\d)", r"\1", text)


def physical_rows(tex):
    body = plain_signs(tex.split("\\midrule\n", 1)[1].split("\\bottomrule", 1)[0])
    return [line.removesuffix(r" \\").split(" & ") for line in body.splitlines()
            if line.strip() != r"\midrule"]


def group_text(rendered):
    count, pos = group(rendered, len(r"\multicolumn"))
    spec, pos = group(rendered, pos)
    contents, end = group(rendered, pos)
    assert count == "6" and end == len(rendered)
    assert spec == r"@{}>{\raggedright\arraybackslash}p{\dimexpr\textwidth-0pt\relax}@{}"
    assert contents.startswith(r"\textit{")
    text, end = group(contents, len(r"\textit"))
    assert end == len(contents)
    return text


def raw_source(ref):
    path, pointer = ref.split("#", 1)
    assert path.startswith(("results/", "paper/docs/", "docs/"))
    data = json.loads((ROOT/path).read_text())
    for key in pointer.lstrip("/").split("/") if pointer else []:
        key = key.replace("~1", "/").replace("~0", "~")
        data = data[int(key)] if isinstance(data, list) else data[key]
    return data


@pytest.fixture(scope="module")
def generated():
    tex_path = gen.output_path(ROOT, "tables", "main_prediction_v2.tex")
    side_path = gen.output_path(ROOT, "tables", "main_prediction_v2_sources.md")
    existing = lambda path: path.read_bytes() if path.exists() else None
    original_tex, original_side = existing(tex_path), existing(side_path)
    rows, audit, access = gen.generate(write_tex=False)
    check_access(audit, access)
    tex = gen.render_table(rows, audit, sidecar="main_prediction_v2_preview_sources.md")
    assert existing(tex_path) == original_tex
    assert existing(side_path) == original_side
    assert all(not p.endswith(".tex") for p in access[1])
    side = (gen.output_path(ROOT, "tables", "main_prediction_v2_preview_sources.md")).read_text()
    blocks = re.findall(r"```json\n(.*?)\n```", side, re.S)
    return rows, audit, tex, json.loads(blocks[0]), json.loads(blocks[1])


def test_table_every_cell_matches_sidecar_and_frozen_json(generated):
    rows, audit, tex, recipes, caption = generated
    assert len(rows) == 5 and len(recipes) == 95
    table_rows = physical_rows(tex)
    assert len(table_rows) == 20
    assert [len(row) for row in table_rows] == [1, 6, 6, 6] * 5
    assert {(r["row"], r["column"]) for r in recipes} == {
        (i, j) for i in range(20) for j in range(1 if i % 4 == 0 else 6)}
    for record in [*recipes, caption]:
        # Independently resolve every JSON pointer and operation, then compare
        # with the actual TeX cell at the sidecar's physical row/column.
        parts = []

        def independent(p):
            assert p["sources"]
            vals = [raw_source(s) for s in p["sources"]]
            op = p["op"]
            if op == "identity": return vals[0]
            if op == "mean": return mean(vals)
            if op == "weighted_mean": return sum(vals[i] * vals[i+1] for i in range(0,len(vals),2)) / sum(vals[1::2])
            if op == "difference":
                b, d = p["parts"]
                assert list(p["sources"]) == list(d["sources"]) + list(b["sources"])
                return independent(b) - independent(d)
            pytest.fail(f"Unhandled numeric operation {op}")

        for p in record["parts"]:
            if isinstance(p, str):
                assert not re.search(r"\d", p), "Printed numeric literals need JSON provenance"
                parts.append(p)
                continue
            assert p["sources"]
            vals = [raw_source(s) for s in p["sources"]]
            op = p["op"]
            if op == "difference": v = independent(p)
            elif op == "identity": v = vals[0]
            elif op == "mean": v = mean(vals)
            elif op == "weighted_mean": v = sum(vals[i] * vals[i+1] for i in range(0,len(vals),2)) / sum(vals[1::2])
            elif op == "length": v = len(vals[0])
            elif op == "per_capability":
                assert vals[0] % len(vals[1]) == 0
                v = vals[0] // len(vals[1])
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
        plain = "".join(parts)
        lines = plain.split("\n")
        prefix = ""
        if record["compact_scores"] == "comparison":
            prefix = gen.tex_escape(lines.pop(0)) + r"\newline "
        # Mathematics and cross-references pass through the generator's escaping.
        rendered_lines = [gen.render_text(line) for line in lines]
        if record["compact_scores"]:
            assert len(rendered_lines) == 3
            expected = prefix + r"\TableOneErrors" + "".join("{" + line + "}" for line in rendered_lines)
        else:
            expected = r"\newline ".join(rendered_lines)
        if "row" in record:
            actual = table_rows[record["row"]][record["column"]]
            assert record["table"] == "tab:main-prediction-v2"
            assert not record["compact_scores"]
            if record["row"] % 4 == 0:
                assert record["column_span"] == 6
                assert group_text(actual) == expected
                expected = (r"\multicolumn{6}{@{}>{\raggedright\arraybackslash}p{\dimexpr\textwidth-0pt\relax}@{}}{\textit{"
                            + expected + "}}")
            else:
                assert "column_span" not in record
        else:
            actual = next(line for line in tex.splitlines() if line.startswith(r"\caption{"))[9:-1].removeprefix(gen.CAPTION_FONT + " ")
        # physical_rows reads signs as plain text; the recorded rendering keeps the math signs.
        assert actual == plain_signs(expected) and expected == record["rendered"]
    sources = [s for r in recipes for p in r["parts"] if isinstance(p, dict) for s in p["sources"]]
    assert not any(forbidden in s for s in sources for forbidden in (
        "strongest_observed_baseline", "strongest_baseline_mae", "v72-prune-repeat",
        "a2-curvature-interaction"))
    # Each task retains its freeze/selection provenance after the status column is removed.
    assert all(row[0].context for row in rows)
    assert any("frozen_at_utc" in ref for row in rows for ref in row[0].context)


def test_only_complete_frozen_tasks_and_capability_layout(generated):
    rows, audit, tex, recipes, caption = generated
    assert [row[0].plain(audit).split("\n")[0] for row in rows] == TASKS
    assert all(len(row) == 8 for row in rows)
    assert all(c.plain(audit).strip() for row in rows for c in row)
    assert all(c.strip() for line in tex.split("\\midrule\n", 1)[1].split("\\bottomrule", 1)[0].splitlines() if line.strip() != r"\midrule"
               for c in line.removesuffix(r" \\").split(" & "))
    assert not re.search(r"not tested|n/a", tex, re.I)
    assert "unseen density" not in tex
    assert gen.HEADERS == (
        "Capability", "Delivered relation", "Error",
        "Strongest development baseline", "Error", "Improvement")
    assert gen.CANDIDATE_COLUMNS == (0, 1, 2, 3)
    assert " & ".join(gen.render_text(h) for h in gen.HEADERS) + r" \\" in tex
    assert r"\centering\footnotesize" in tex and r"\tiny" not in tex
    assert tex.count(r"\begin{table*}[t]\normalfont") == 1
    assert tex.count(r"\begin{tabular*}{\textwidth}") == 1
    assert r"\extracolsep{\fill}" in tex
    # One-line entries in natural-width columns: text left, scores right.
    assert (r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}ll" + gen.ERROR_COLUMN + "l"
            + gen.ERROR_COLUMN + "c@{}}") in tex
    assert sum(map(float, gen.CANDIDATE_WIDTHS)) == pytest.approx(1)
    assert r"\setlength{\tabcolsep}{1.5pt}" in tex
    assert gen.TABLE_FONT == r"\footnotesize\fontsize{8}{9.5}\selectfont"
    assert gen.CAPTION_FONT == r"\footnotesize\fontsize{8.5}{10}\selectfont"
    assert r"\centering" + gen.TABLE_FONT in tex
    assert r"\caption{" + gen.CAPTION_FONT in tex
    assert r"\TableOneErrors" not in tex and r"\newline" not in tex and "/" not in tex
    assert tex.count(r"\multicolumn{6}") == 5
    assert not re.search(r"\\(?:resizebox|scalebox|rotatebox|dagger)", tex)
    assert r"\label{tab:main-prediction-v2}" in tex
    assert tex.index(r"\caption{") < tex.index(r"\label{") < tex.index(r"\begin{tabular*}")
    assert caption["rendered"] == gen.caption_cell(audit).render(audit)
    assert len(caption["rendered"].split()) <= 100
    assert "at equal budget" in caption["rendered"]
    assert "mean absolute errors in nats per token" in caption["rendered"]
    assert "improvement is baseline minus delivered" in caption["rendered"]
    assert "Group lines give the task, development configuration measurements per capability" in caption["rendered"]
    assert "The baseline was chosen inside the development folds and scored on the same cells" in caption["rendered"]
    assert "chosen after seeing the result" in caption["rendered"]
    assert "compares each row with its baseline" in caption["rendered"]
    assert r"Table~\ref{tab:main-prediction-candidates}" in caption["rendered"]
    assert r"\textbackslash" not in caption["rendered"] and r"\textasciitilde" not in caption["rendered"]
    assert "development configuration measurements" in caption["rendered"]
    assert tex.count(r"\midrule") == len(rows)
    assert not re.search(r"\b(?:QA|MAE|A2|[0-9.]+[MB])\b", tex)
    assert "[" not in caption["rendered"]
    assert "Source-free median density curve" in tex
    assert "Strongest development baseline" in tex
    assert not any("Development baseline for" in n for n in audit.notes)
    assert not re.search(r"\b(?:bit-width|bits|groups|step|[bg] [0-9]|[0-9]+k)\b|\$", plain_signs(tex))
    assert all("\n" not in row[0].plain(audit) for row in rows)
    assert "No stored error" not in tex
    actual = physical_rows(tex)
    for i, row in enumerate(rows):
        expected = TASKS[i] + f"; {row[6].plain(audit)} measurements"
        if i < 3:
            expected += "; Retrospective"
        assert group_text(actual[4 * i][0]) == expected
        assert [r[0] for r in actual[4 * i + 1:4 * i + 4]] == [
            "Mathematics", "Code", "Question answering"]
        for r in actual[4 * i + 1:4 * i + 4]:
            assert all(re.fullmatch(r"\d+\.\d{3}", r[j]) for j in (2, 4))
            assert re.fullmatch(r"[+-]\d+\.\d{3}", r[5])
    for i, row in enumerate(rows):
        for j in (2, 4):
            lines = row[j].plain(audit).splitlines()
            assert len(lines) == 3
            for cap, line in zip(("Math", "Code", "QA"), lines):
                prefix = ""
                assert line.startswith(prefix)
                if i == 3:
                    assert re.fullmatch(prefix + r"[+-]?\d+\.\d{3}", line)
                else:
                    assert re.fullmatch(prefix + r"[+-]?\d+\.\d{3}", line)
            assert row[j].compact_scores == "ordered"
        assert "[" not in row[2].plain(audit)
        if i == 3:
            assert row[3].plain(audit) == "The same predictor"
        assert all("chosen after this test" not in c.plain(audit) for c in row)


def test_frozen_errors_and_development_baselines_unchanged(generated):
    rows, audit, tex, recipes, _ = generated
    def scores(row, column):
        return " / ".join(row[column].plain(audit).splitlines())
    assert [scores(row, 2) for row in rows] == [
        "0.243 / 0.236 / 0.678", "0.215 / 0.557 / 0.457",
        "0.303 / 0.153 / 0.222", "0.074 / 0.019 / 0.515", "0.057 / 0.049 / 0.463",
    ]
    assert [scores(row, 4) for row in rows] == [
        "0.277 / 0.214 / 0.221", "0.065 / 0.116 / 0.458", "0.088 / 0.153 / 0.141",
        "0.074 / 0.019 / 0.515", "0.057 / 0.049 / 0.463",
    ]
    assert [scores(row, 7) for row in rows] == [
        "-0.047 / +0.008 / +0.000", "+0.268 / +0.562 / +0.000",
        "+0.262 / +0.410 / +0.000", "-0.009 / +0.004 / +0.095", "-0.023 / +0.005 / -0.013",
    ]
    assert "270 million and 1 billion" in gen.caption_cell(audit).plain(audit)
    assert "no student averaging" in rows[3][2].note
    baselines = [row[5].plain(audit).splitlines() for row in rows]
    assert [b[0] for b in baselines] == [
        "Per-density regression; question answering: median",
        "Bilinear regression, no change and the median",
        "Bilinear regression, no change and the median",
        "Budget regression; reuse regression for code and question answering", "Regressions on loss, reuse and size"]
    assert all(len(b) == 4 for b in baselines)
    assert [b[1:] for b in baselines] == [
        ["0.230", "0.222", "0.221"], ["0.333", "0.678", "0.458"],
        ["0.350", "0.563", "0.141"], ["0.065", "0.023", "0.610"],
        ["0.033", "0.053", "0.450"],
    ]
    actual = physical_rows(tex)
    lookup = {(r["row"], r["column"]): r for r in recipes}
    for i, row in enumerate(rows):
        for column, original in ((2, 4), (4, 5), (5, 7)):
            original_parts = [p for p in row[original].parts if isinstance(p, dict) and p.get("format")]
            assert len(original_parts) == 3
            for c, part in enumerate(original_parts):
                assert lookup[4 * i + c + 1, column]["parts"] == [part]
                assert actual[4 * i + c + 1][column] == gen.evaluate(audit, part)


def test_per_capability_identities_come_from_frozen_selections(generated):
    rows, audit, tex, recipes, _ = generated
    actual = physical_rows(tex)
    delivered = [
        ["Source-free median density curve"] * 3,
        ["Interpolation", "Interpolation", "Development median"],
        ["Development median"] * 3,
        ["Reuse form", "Reuse form", "Budget and pool form"],
        ["Reuse form", "Reuse form", "Budget and pool form"],
    ]
    baselines = [
        ["Per-density regression", "Per-density regression", "Development median"],
        ["Bilinear regression", "No change", "Development median"],
        ["Bilinear regression", "No change", "Development median"],
        ["Budget regression", "Reuse regression", "Reuse regression"],
        ["Loss regression", "Reuse regression", "Size regression"],
    ]
    summary = raw_source(gen.S86 + "#/main_rows")
    lookup = {(r["row"], r["column"]): r for r in recipes}
    for i, task in enumerate(("C35", "C44", "C46", "C47", "C48")):
        physical = actual[4 * i + 1:4 * i + 4]
        assert [r[1] for r in physical] == delivered[i]
        assert [r[3] for r in physical] == baselines[i]
        frozen = next(r for r in summary if r["row"] == task)
        for j, cap in enumerate(gen.CAPS):
            name = lookup[4 * i + j + 1, 1]["parts"][0]
            assert len(name["sources"]) == 1
            assert raw_source(name["sources"][0]) == frozen["capabilities"][cap]["delivered"]
            if i >= 3:
                assert name["sources"] == [f"results/v70-distill-confirm/freeze.json#/selected/{cap}/method"]
            else:
                assert name["sources"][0].endswith(f"/capabilities/{cap}/delivered")
            baseline = lookup[4 * i + j + 1, 3]["parts"][0]
            # The summary cell's original entries supply both identity and score.
            assert baseline == rows[i][5].capabilities[cap].parts[0]


def test_body_entries_fit_paragraph_columns_at_paper_font_size(generated):
    # Read Times metrics without building the paper or rendering a new artifact.
    from matplotlib import get_data_path
    from matplotlib._afm import AFM

    with (Path(get_data_path()) / "fonts/pdfcorefonts/Times-Roman.afm").open("rb") as f:
        font = AFM(f)
    _, _, tex, _, _ = generated
    # Natural-width columns: the widest entry of each column, plus the ten interior
    # tabcolseps, must fit the 5.5-inch text width at 8pt so no row can overflow.
    widest = [0.0] * 6
    for row in [list(gen.HEADERS), *physical_rows(tex)]:
        if len(row) == 1:
            continue  # Group headings are p-columns and may wrap.
        for i, text in enumerate(row):
            widest[i] = max(widest[i], font.string_width_height(text)[0] * 8 / 1000)
    widest[2] = widest[4] = .075 * 5.5 * 72.27  # the fixed-width error columns
    assert sum(widest) + 10 * 1.5 <= 5.5 * 72.27


def test_appendix_candidates_table_is_byte_for_byte_unchanged(generated):
    rows, audit, _, _, _ = generated
    tex = gen.render_candidates(rows, audit)
    written = gen.output_path(ROOT, "tables", "main_prediction_candidates.tex")
    if written.exists():  # the public checkout carries no generated LaTeX tables
        assert tex == written.read_text()
    # Frozen before the main-table-only layout change, including caption and macro.
    assert hashlib.sha256(tex.encode()).hexdigest() == "32f31720d1fd7c0786df4d1f61c4fc588344d0bc096db602c0456d915a20934a"


def test_development_measurements_count_configurations_per_capability(generated):
    rows, audit, _, recipes, _ = generated
    assert [int(row[6].plain(audit).split()[0]) for row in rows] == [84, 54, 54, 100, 100]
    prune = raw_source(gen.P53 + "#")
    assert int(rows[0][6].plain(audit)) == sum(len(s["densities"]) for s in prune["dev_states"])
    assert prune["n_dev_rows"] == 3 * int(rows[0][6].plain(audit))
    for index, path in ((1, gen.Q69), (2, gen.Q69)):
        frozen = raw_source(path + "#")
        count = int(rows[index][6].plain(audit))
        assert count == len(frozen["dev_states"]) * len(frozen["dev_configs"])
        for cap in gen.CAPS:
            actual = {(r["state"], r["config"]) for r in frozen["dev_rows"] if r["capability"] == cap}
            assert len(actual) == count
    distill = raw_source("results/v70-distill-confirm/develop.json#")
    assert len(distill["points"]) == len({(p["cluster"], p["Tc"]) for p in distill["points"]}) == 100
    assert all(set(p["delta"]) == set(gen.CAPS) for p in distill["points"])
    assert "selection of QA's zero rule" in rows[1][6].note
    assert "not 100 per test student" in rows[3][6].note
    assert all(r["parts"][0]["sources"] for r in recipes if r["column"] == 4)
    for i, row in enumerate(rows):
        heading = next(r for r in recipes if r["row"] == 4 * i)
        assert row[6].parts[0] in heading["parts"]
        assert all(ref in heading["context"] for ref in row[6].context)


def test_delivery_identities_and_bit_test_moved_to_the_appendix(generated):
    rows, audit, tex, recipes, _ = generated
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
    assert rows[2][3].plain(audit).splitlines()[0] == "Development median"
    assert "unseen bit width" not in tex and "No stored error" not in tex
    earlier = raw_source(gen.Q55 + "#/candidate_definitions/same_input_interpolation")
    later = raw_source(gen.Q69 + "#/boundary_rule")
    assert "Bilinear" in earlier and earlier != later
    bit_cells = {(s, cfg) for s in raw_source(gen.Q55 + "#/test_sets/bit_test/states") for cfg in old_test}
    later_cells = {(r["state"], r["config"]) for r in raw_source("results/v69-quant-confirm/compare.json#/rows")}
    assert not bit_cells.intersection(later_cells)
    # The appendix prints these numbers; the audit note is where they come from.
    note = next(n for n in audit.notes if "earlier unseen-bit-width test" in n)
    assert "0.193 / 0.214 / 0.436" in note and "0.553 / 0.726 / 0.537" in note
    assert "24 development configuration measurements per capability" in note
    assert rows[3][3].plain(audit) == "The same predictor"


def test_delivered_relations_carry_their_timing(generated):
    rows, audit, tex, recipes, _ = generated
    summary = {r["row"]: r["delivered_timing"] for r in raw_source(gen.S86 + "#/main_rows")}
    marked = [i for i, row in enumerate(rows) if row[3].plain(audit).endswith("\nRetrospective")]
    assert marked == [0, 1, 2]
    assert [summary[k] for k in ("C35", "C44", "C46")] == [
        "fixed after test; reused frozen in Sec. 5", "fixed after test", "fixed after test"]
    assert summary["C47"] == summary["C48"] == "fixed before test"
    assert tex.count("; Retrospective") == 3
    assert "Retrospective for a relation chosen after seeing the result" in gen.caption_cell(audit).plain(audit)
    for i in marked:
        part = next(p for p in rows[i][3].parts if isinstance(p, dict) and p.get("label") == "Retrospective")
        assert part["sources"] == [gen.pointer(gen.S86, "main_rows", list(summary).index(
            {0: "C35", 1: "C44", 2: "C46"}[i]), "delivered_timing")]
        heading = next(r for r in recipes if r["row"] == 4 * i)
        assert part in heading["parts"]


def test_composite_relations_follow_caption_capability_order(generated):
    rows, audit, tex, _, _ = generated
    for i in (1, 2):
        assert rows[i][1].plain(audit) == "Source regression, the median and no change"
    assert rows[1][3].plain(audit).splitlines()[0] == (
        "Interpolate math and code; use the median for question answering")
    assert rows[3][1].plain(audit) == (
        "Reuse forms, with a budget and pool form for question answering")
    assert rows[0][1].plain(audit) == "Five-parameter power form"
    assert raw_source(gen.P53 + "#/n_params_per_capability/power") == 5
    assert raw_source(gen.P53 + "#/n_dev_states") == len(raw_source(gen.P53 + "#/dev_states")) == 17
    for c, count in (("math", 1), ("code", 1), ("qa", 3)):
        assert raw_source("results/v70-distill-confirm/freeze.json#/selected/" + c + "/n_params") == count


def test_test_sets_match_frozen_membership(generated):
    rows, audit, _, _, _ = generated
    dev = {s["tag"] for s in raw_source(gen.P53 + "#/dev_states")}
    tags = [raw_source(p + "#/tag") for p in audit.data if p.startswith("results/v53-prune-dev/compare_")]
    assert len(tags) == 3 and not set(tags) & dev
    assert "pythia-6.9b@step80000" in tags
    for path in audit.data:
        if path.startswith("results/v53-prune-dev/compare_"):
            assert raw_source(path + "#/densities") == [0.85, 0.675, 0.575]
    # The 6.9B checkpoint is new; its model size was already in development.
    assert any(t.startswith("pythia-6.9b@") for t in dev)
    assert "outside the size range" not in rows[0][0].plain(audit)
    bits = raw_source(gen.Q55 + "#/test_sets/bit_test")
    assert bits["configs"] == ["b4_g64", "b4_g256"]
    assert {s.split("@")[0] for s in bits["states"]} == {"pythia-160m", "pythia-410m", "pythia-1.4b"}
    quant = raw_source("results/v69-quant-confirm/compare.json#/rows")
    for i, new in [(1, False), (2, True)]:
        rr = [r for r in quant if (r["test_set"] != "development_state_boundary") == new]
        assert {r["state"] for r in rr} == ({"pythia-1.4b@step112000"} if new else
               {"pythia-410m@step143000", "pythia-1.4b@step16000"})
        assert {r["config"] for r in rr} == {f"b{b}_g{g}" for b in (3,4,5)
                                             for g in ((32,128,512) if new else (32,512))}
    assert all("group sizes 64 and 256" not in row[0].plain(audit) for row in rows)
    assert "group sizes 32 and 512" in rows[1][0].plain(audit)
    assert "group sizes 32 to 512" in rows[2][0].plain(audit)
    for i in (1, 2):
        assert "bit widths 3 to 5" in rows[i][0].plain(audit)
    assert all("@" not in row[0].plain(audit) for row in rows)
    assert all(not re.search(r"\b[bg] \d", row[0].plain(audit)) for row in rows)


def test_paired_intervals_remain_frozen_and_are_omitted_from_table(generated):
    rows, audit, _, recipes, caption = generated
    groups = raw_source(gen.D70 + "#/groups")
    freeze = raw_source("results/v70-distill-confirm/freeze.json#")
    assert "[" not in caption["rendered"]
    assert all("[" not in c.plain(audit) for row in rows for c in row)
    for index in (0,3,1,4,2,5):
        g = groups[index]
        assert g["selected"] == freeze["selected"][g["capability"]]["method"]
        assert g["strongest_baseline"] == freeze["strongest_baseline"][g["student"]][g["capability"]]["method"]
        assert g["paired_difference"]["estimate"] == pytest.approx(g["baseline_mae"] - g["candidate_mae"])
        stored = gen.cell(*gen.paired_interval(audit, gen.D70, "groups", index)).plain(audit)
        assert stored == " [" + ", ".join(format(v, ".4f") for v in g["paired_difference"]["ci95"]) + "]"
    for record in [*recipes, caption]:
        for part in record["parts"]:
            if isinstance(part, dict):
                assert not any("paired_difference" in s for s in part["sources"])
    assert gen.paired_interval(audit, gen.D70, "bootstrap") == []


def test_every_numeric_occurrence_is_in_inventory(generated):
    rows, audit, tex, recipes, caption = generated
    numbers = gen.numeric_inventory([*recipes, caption], audit)
    body = "\n".join(group_text(row[0]) if len(row) == 1 else " & ".join(row)
                     for row in physical_rows(tex))
    assert [n["number"] for n in numbers] == re.findall(r"[-+]?\d+(?:\.\d+)?|\b(?:one|three|four|five|six|seventeen|twenty|half)\b", body + caption["rendered"], re.I)
    assert all(n["sources"] for n in numbers)
    lookup = {(r["row"], r["column"]): r for r in recipes}
    for n in numbers:
        assert n["table"] == "tab:main-prediction-v2"
        record = lookup[n["row"], n["column"]] if "row" in n else caption
        part = record["parts"][n["part"]]
        assert n["op"] == part["op"] and n["sources"] == part["sources"]
        assert n["displayed"] == gen.evaluate(audit, part)
    assert {(n["row"], n["column"]) for n in numbers if "row" in n} == {
        (i, j) for i in range(20) for j in ((0,) if i % 4 == 0 else (2, 4, 5))}
    caption_numbers = [n for n in numbers if n.get("location") == "caption"]
    assert [n["number"] for n in caption_numbers[-2:]] == ["270", "1"]
    assert len(caption_numbers) == 2
    assert all(n["sources"] == ["results/v70-distill-confirm/freeze.json#/confirmation_register/students"]
               for n in caption_numbers[:2])
    side = gen.output_path(ROOT, "tables", "main_prediction_v2_preview_sources.md").read_text()
    blocks = re.findall(r"```json\n(.*?)\n```", side, re.S)
    # The sidecar inventory lists the main table first, then the candidates table.
    assert json.loads(blocks[4])[:len(numbers)] == numbers


@pytest.mark.parametrize("task_index", [0, 3])
@pytest.mark.parametrize("column", [2, 4])
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
    ref = next(p["sources"][0] for p in rows[0][4].parts if isinstance(p, dict))
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
