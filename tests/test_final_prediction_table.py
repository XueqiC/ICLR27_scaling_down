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
SHORT_TASKS = [
    "Three new checkpoints",
    "Group sizes 32 and 512",
    "A new 1.4 billion stage",
    "Six new pools, 270 million student",
    "Six new pools, 1 billion student",
]
TASK_ROWS = (1, 3, 4, 6, 7)
GROUP_ROWS = (0, 2, 5)
BASELINE_LABELS = [
    "Per-density regression; median for QA",
    "Bilinear regression; no change; median",
    "Bilinear regression; no change; median",
    "Budget regression; reuse regression for Code and QA",
    "Loss regression; reuse regression; size regression",
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
    assert count == "8" and end == len(rendered)
    assert spec == r"@{}l@{}"
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
    assert len(rows) == 5 and len(recipes) == 43
    table_rows = physical_rows(tex)
    assert len(table_rows) == 8
    assert [len(row) for row in table_rows] == [1, 8, 1, 8, 8, 1, 8, 8]
    assert {(r["row"], r["column"]) for r in recipes} == {
        (i, j) for i in range(8) for j in range(1 if i in GROUP_ROWS else 8)}
    side = gen.output_path(ROOT, "tables", "main_prediction_v2_preview_sources.md").read_text()
    blocks = re.findall(r"```json\n(.*?)\n```", side, re.S)
    candidate_recipes, candidate_caption = map(json.loads, blocks[2:4])
    assert {(r["row"], r["column"]) for r in candidate_recipes} == {
        (i, j) for i in range(5) for j in range(5)}
    candidates_tex = gen.render_candidates(rows, audit)
    for record in [*recipes, caption, *candidate_recipes, candidate_caption]:
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
                assert not re.search(r"\d", re.sub(r"\\ref\{[^}]*\}", "", p)), "Printed numeric literals need JSON provenance"
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
            if record["table"] == "tab:main-prediction-v2":
                actual = table_rows[record["row"]][record["column"]]
                assert not record["compact_scores"]
                if record["row"] in GROUP_ROWS:
                    assert record["column_span"] == 8
                    assert group_text(actual) == expected
                    expected = (r"\multicolumn{8}{@{}l@{}}{\textit{"
                                + expected + "}}")
                else:
                    assert "column_span" not in record
            else:
                assert record["table"] == "tab:main-prediction-candidates"
                actual = physical_rows(candidates_tex)[record["row"]][record["column"]]
                assert "column_span" not in record
        else:
            table_tex = tex if record["table"] == "tab:main-prediction-v2" else candidates_tex
            actual = next(line for line in table_tex.splitlines() if line.startswith(r"\caption{"))[9:-1].removeprefix(gen.CAPTION_FONT + " ")
        # physical_rows reads signs as plain text; the recorded rendering keeps the math signs.
        assert actual == plain_signs(expected) and expected == record["rendered"]
    sources = [s for r in recipes for p in r["parts"] if isinstance(p, dict) for s in p["sources"]]
    assert not any(forbidden in s for s in sources for forbidden in (
        "strongest_observed_baseline", "strongest_baseline_mae", "v72-prune-repeat",
        "a2-curvature-interaction"))
    # Each task retains its freeze/selection provenance after the status column is removed.
    assert all(row[0].context for row in rows)
    assert any("frozen_at_utc" in ref for row in rows for ref in row[0].context)


def test_only_complete_frozen_tasks_and_method_layout(generated):
    rows, audit, tex, recipes, caption = generated
    assert [row[0].plain(audit).split("\n")[0] for row in rows] == TASKS
    assert all(len(row) == 8 for row in rows)
    assert all(c.plain(audit).strip() for row in rows for c in row)
    actual = physical_rows(tex)
    assert all(c.strip() for row in actual for c in row)
    assert not re.search(r"not tested|n/a", tex, re.I)
    assert "unseen density" not in tex
    assert gen.HEADERS == (
        "Test", "Delivered relation", "Math", "Code", "QA", "Math", "Code", "QA")
    assert gen.CANDIDATE_COLUMNS == (0, 1, 2, 3, 5)
    assert r"& & \multicolumn{3}{c}{Error} & \multicolumn{3}{c}{Gain over baseline} \\" in tex
    assert r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}" in tex
    assert " & ".join(gen.render_text(h) for h in gen.HEADERS) + r" \\" in tex
    assert r"\centering\footnotesize" in tex and r"\tiny" not in tex
    assert tex.count(r"\begin{table*}[t]\normalfont") == 1
    assert tex.count(r"\begin{tabular*}{\textwidth}") == 1
    column_spec = "ll" + "r" * 6  # one-line labels in natural-width text columns
    assert r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}" + column_spec + "@{}}" in tex
    assert sum(map(float, gen.CANDIDATE_WIDTHS)) == pytest.approx(1)
    assert r"\setlength{\tabcolsep}{1.5pt}" in tex
    assert gen.TABLE_FONT == r"\footnotesize\fontsize{8}{9.5}\selectfont"
    assert gen.CAPTION_FONT == r"\footnotesize\fontsize{8.5}{10}\selectfont"
    assert r"\centering" + gen.TABLE_FONT in tex
    assert r"\caption{" + gen.CAPTION_FONT in tex
    assert r"\TableOneErrors" not in tex and r"\newline" not in tex and "/" not in tex
    assert tex.count(r"\multicolumn{8}") == 3
    assert not re.search(r"\\(?:resizebox|scalebox|rotatebox)", tex)
    assert r"\label{tab:main-prediction-v2}" in tex
    assert tex.index(r"\caption{") < tex.index(r"\label{") < tex.index(r"\begin{tabular*}")
    assert caption["rendered"] == gen.caption_cell(audit).render(audit)
    assert len(caption["rendered"].split()) <= 100
    for definition in (
        "Prediction at equal development budget",
        "Test: configurations or states unseen by the relation",
        "Delivered relation: predictor recommended on all evidence",
        "a dagger marks selection after seeing test results",
        "Error: test mean absolute error in nats per token, per capability",
        "QA: question answering",
        "Gain: strongest development baseline error minus delivered error",
        "positive favours the delivered relation",
        "chosen inside the development folds and scored on the same cells",
        "$n$, on each method line: development configuration measurements per capability",
        "names baselines and pre-specified candidates",
        "compares each row with its baseline",
    ):
        assert definition in caption["rendered"]
    assert r"Table~\ref{tab:main-prediction-candidates}" in caption["rendered"]
    assert r"\textbackslash" not in caption["rendered"] and r"\textasciitilde" not in caption["rendered"]
    assert tex.count(r"\midrule") == 3
    assert not re.search(r"\b(?:MAE|A2|[0-9.]+[MB])\b", tex)
    assert "[" not in caption["rendered"]
    assert "Strongest development baseline" not in tex
    assert not any("Development baseline for" in n for n in audit.notes)
    assert not re.search(r"\b(?:bit-width|bits|groups|step|[bg] [0-9]|[0-9]+k)\b", tex)
    assert all("\n" not in row[0].plain(audit) for row in rows)
    assert "No stored error" not in tex
    assert [group_text(actual[i][0]) for i in GROUP_ROWS] == ["Pruning, $n=84$", "Quantization, $n=54$", "Distillation, $n=100$"]
    assert [actual[i][0] for i in TASK_ROWS] == SHORT_TASKS
    assert [actual[i][1] for i in TASK_ROWS] == [
        r"Median density curve$^{\dagger}$",
        r"Interpolation; median for QA$^{\dagger}$",
        r"Development median$^{\dagger}$",
        "Reuse; budget and pool for QA",
        "Reuse; budget and pool for QA",
    ]
    for i, row in enumerate(rows):
        physical = actual[TASK_ROWS[i]]
        assert all(re.fullmatch(r"\d+\.\d{3}", physical[j]) for j in (2, 3, 4))
        assert all(re.fullmatch(r"[+-]\d+\.\d{3}", physical[j]) for j in (5, 6, 7))
        for j in (2, 4):
            lines = row[j].plain(audit).splitlines()
            assert len(lines) == 3
            assert all(re.fullmatch(r"[+-]?\d+\.\d{3}", line) for line in lines)
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
    assert "270 million and 1 billion" in gen.caption_cell(audit, gen.CANDIDATE_CAPTION).plain(audit)
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
        for column, original in ((2, 4), (5, 7)):
            original_parts = [p for p in row[original].parts if isinstance(p, dict) and p.get("format")]
            assert len(original_parts) == 3
            for c, part in enumerate(original_parts):
                assert lookup[TASK_ROWS[i], column + c]["parts"] == [part]
                assert actual[TASK_ROWS[i]][column + c] == gen.evaluate(audit, part)
        baseline_parts = [p for p in row[5].parts if isinstance(p, dict) and p.get("format")]
        delivered_parts = [p for p in row[4].parts if isinstance(p, dict)]
        for c in range(3):
            gain = lookup[TASK_ROWS[i], 5 + c]["parts"][0]
            assert gain["parts"] == [baseline_parts[c], delivered_parts[c]]


def test_per_capability_identities_come_from_frozen_selections(generated):
    rows, audit, tex, recipes, _ = generated
    candidates = gen.candidate_rows(rows, audit)
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
        assert [rows[i][3].capabilities[c].plain(audit) for c in gen.CAPS] == delivered[i]
        assert [gen.evaluate(audit, rows[i][5].capabilities[c].parts[0]) for c in gen.CAPS] == baselines[i]
        identity = lookup[TASK_ROWS[i], 1]["parts"][0]
        assert len(identity["sources"]) == 3
        assert candidates[i][4].plain(audit) == BASELINE_LABELS[i]
        baseline_summary = candidates[i][4].parts[0]
        assert len(baseline_summary["sources"]) == 3
        frozen = next(r for r in summary if r["row"] == task)
        for j, cap in enumerate(gen.CAPS):
            name = rows[i][3].capabilities[cap].parts[0]
            assert len(name["sources"]) == 1
            assert identity["sources"][j] == name["sources"][0]
            assert raw_source(name["sources"][0]) == frozen["capabilities"][cap]["delivered"]
            if i >= 3:
                assert name["sources"] == [f"results/v70-distill-confirm/freeze.json#/selected/{cap}/method"]
            else:
                assert name["sources"][0].endswith(f"/capabilities/{cap}/delivered")
            # The appendix retains each selection's identity source, while main
            # gain recipes retain the original baseline score operands.
            baseline = rows[i][5].capabilities[cap].parts[0]
            assert baseline_summary["sources"][j:j+1] == baseline["sources"]
        assert identity["expected"] == [frozen["capabilities"][c]["delivered"] for c in gen.CAPS]


def test_body_entries_fit_paragraph_columns_at_paper_font_size(generated):
    # Read Times metrics without building the paper or rendering a new artifact.
    from matplotlib import get_data_path
    from matplotlib._afm import AFM

    with (Path(get_data_path()) / "fonts/pdfcorefonts/Times-Roman.afm").open("rb") as f:
        font = AFM(f)
    _, _, tex, _, _ = generated
    # Text wraps in two paragraph columns; the seven numeric columns and sixteen
    # interior tabcolseps must still fit the paper's 5.5-inch text width at 8pt.
    widest = [0.0] * 9
    for row in [list(gen.HEADERS), *physical_rows(tex)]:
        if len(row) == 1:
            continue  # Group headings are p-columns and may wrap.
        for i, text in enumerate(row):
            # Reserve an em for the math-mode gain sign (wider than the text hyphen).
            sign = 8 if re.match(r"[+-]\d", text) else 0
            if sign:
                text = text[1:]
            dagger = 4 if r"$^{\dagger}$" in text else 0  # a superscript dagger is about 4pt wide
            text = text.replace(r"$^{\dagger}$", "")
            width = font.string_width_height(text.replace("$", ""))[0] * 8 / 1000 + sign + dagger
            widest[i] = max(widest[i], width)
    assert sum(widest) + 16 * 1.5 <= 5.5 * 72.27
    # Spanning headers must fit their three numeric columns and interior gaps.
    for heading, columns in (("Error", slice(2, 5)), ("Gain over baseline", slice(5, 8))):
        assert font.string_width_height(heading)[0] * 8 / 1000 <= sum(widest[columns]) + 4 * 1.5


def test_appendix_candidates_adds_baselines_preserving_original_cells(generated):
    rows, audit, _, _, _ = generated
    tex = gen.render_candidates(rows, audit)
    written = gen.output_path(ROOT, "tables", "main_prediction_candidates.tex")
    if written.exists():  # the public checkout carries no generated LaTeX tables
        assert tex == written.read_text()
    assert gen.CANDIDATE_HEADERS == ("Prediction task", "Pre-specified candidate", "Candidate error (nats)",
                                     "Delivered relation", "Strongest development baseline")
    assert " & ".join(gen.CANDIDATE_HEADERS) + r" \\" in tex
    actual = physical_rows(tex)
    assert len(actual) == 5 and all(len(row) == 5 for row in actual)
    assert [row[0] for row in actual] == TASKS
    assert [row[4] for row in actual] == BASELINE_LABELS
    assert [row[:4] for row in actual] == [
        [c.render(audit) for c in row[:4]] for row in rows]
    # Separately pin the old four columns, then the complete new table.
    original_cells = "\n".join(" & ".join(row[:4]) for row in actual)
    assert hashlib.sha256(original_cells.encode()).hexdigest() == "b94099633c0c15a0890fafa7c972f5f2b6a8ad69e2e8d23e305f8537f730d764"
    assert hashlib.sha256(tex.encode()).hexdigest() == "aa6b85abe2ed5fc931bc53b7756ff9451ba37dd15652772c520aef9b25ffe4fe"
    caption = gen.caption_cell(audit, gen.CANDIDATE_CAPTION).plain(audit)
    assert len(caption.split()) <= 60
    assert "strongest development baselines" in caption
    assert "math, code and question answering order unless qualified" in caption


def test_development_measurements_count_configurations_per_capability(generated):
    rows, audit, tex, recipes, _ = generated
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
    # The development count sits on each method line, shared by that method's tasks.
    assert [group_text(physical_rows(tex)[i][0]) for i in GROUP_ROWS] == [
        "Pruning, $n=84$", "Quantization, $n=54$", "Distillation, $n=100$"]
    assert [row[6].plain(audit) for row in rows] == ["84", "54", "54", "100", "100"]
    for group_row, task in zip(GROUP_ROWS, (0, 1, 3)):
        record = next(r for r in recipes if (r["row"], r["column"]) == (group_row, 0))
        count_parts = [p for p in rows[task][6].parts if isinstance(p, dict)]
        assert [p for p in record["parts"] if isinstance(p, dict)][-len(count_parts):] == count_parts
        assert set(rows[task][6].context) <= set(record["context"])


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
    assert "Retrospective" not in tex
    assert tex.count(r"$^{\dagger}$") == 3
    assert gen.render_candidates(rows, audit).count(r"\newline Retrospective") == 3
    assert "a dagger marks selection after seeing test results" in gen.caption_cell(audit).plain(audit)
    for i in marked:
        part = next(p for p in rows[i][3].parts if isinstance(p, dict) and p.get("label") == "Retrospective")
        assert part["sources"] == [gen.pointer(gen.S86, "main_rows", list(summary).index(
            {0: "C35", 1: "C44", 2: "C46"}[i]), "delivered_timing")]
        identity = next(r for r in recipes if (r["row"], r["column"]) == (TASK_ROWS[i], 1))
        dagger = identity["parts"][-1]
        assert dagger == {**part, "label": r"$^{\dagger}$"}
        assert raw_source(dagger["sources"][0]).startswith("fixed after test")
    for i in (3, 4):
        identity = next(r for r in recipes if (r["row"], r["column"]) == (TASK_ROWS[i], 1))
        assert len(identity["parts"]) == 1


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
        (i, j) for i in TASK_ROWS for j in (0, 2, 3, 4, 5, 6, 7)} | {(i, 0) for i in GROUP_ROWS}
    caption_numbers = [n for n in numbers if n.get("location") == "caption"]
    assert not caption_numbers
    for i, row in enumerate(rows):
        test = lookup[TASK_ROWS[i], 0]["parts"][0]
        assert test["sources"] == row[0].parts[0]["sources"]
        assert test["expected"] == row[0].parts[0]["expected"]
    side = gen.output_path(ROOT, "tables", "main_prediction_v2_preview_sources.md").read_text()
    blocks = re.findall(r"```json\n(.*?)\n```", side, re.S)
    # The sidecar inventory lists the main table first, then the candidates table.
    candidate_recipes, candidate_caption = map(json.loads, blocks[2:4])
    candidate_numbers = gen.numeric_inventory([*candidate_recipes, candidate_caption], audit)
    assert json.loads(blocks[4]) == numbers + candidate_numbers
    candidate_plain = "\n".join(" & ".join(row) for row in physical_rows(gen.render_candidates(rows, audit)))
    candidate_plain += "\n" + candidate_caption["rendered"]
    candidate_plain = re.sub(r"\\ref\{[^}]*\}", "", candidate_plain)
    assert [n["number"] for n in candidate_numbers] == re.findall(gen.NUMBER_PATTERN, candidate_plain, re.I)
    assert all(n["sources"] and n["table"] == "tab:main-prediction-candidates" for n in candidate_numbers)
    candidate_lookup = {(r["row"], r["column"]): r for r in candidate_recipes}
    for n in candidate_numbers:
        record = candidate_lookup[n["row"], n["column"]] if "row" in n else candidate_caption
        part = record["parts"][n["part"]]
        assert n["op"] == part["op"] and n["sources"] == part["sources"]
        assert n["displayed"] == gen.evaluate(audit, part)
    candidate_caption_numbers = [n for n in candidate_numbers if n.get("location") == "caption"]
    assert [n["number"] for n in candidate_caption_numbers] == ["270", "1"]
    assert all(n["sources"] == ["results/v70-distill-confirm/freeze.json#/confirmation_register/students"]
               for n in candidate_caption_numbers)
    assert "method headings occupy rows 0, 2 and 5" in side
    assert "task rows occupy rows 1, 3, 4, 6 and 7" in side


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
