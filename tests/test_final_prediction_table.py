import json
import re
from statistics import mean

import pytest

from analysis import final_prediction_table as gen
from analysis.paper_artifacts import ROOT
from paper_generator_checks import check_access, refuses_symlink, refuses_external_io, refuses_output_file_symlink


def raw_source(ref):
    path,pointer=ref.split("#",1)
    assert path.startswith(("results/","docs/"))
    data=json.loads((ROOT/path).read_text())
    for key in pointer.lstrip("/").split("/") if pointer else []:
        key=key.replace("~1","/").replace("~0","~")
        data=data[int(key)] if isinstance(data,list) else data[key]
    return data


def test_table_every_cell_matches_sidecar_and_frozen_json():
    tex_path = ROOT/"generated/tables/main_prediction_v2.tex"
    original_tex = tex_path.read_bytes()
    side_path = ROOT/"generated/tables/main_prediction_v2_sources.md"
    original_side = side_path.read_bytes()
    rows,audit,access=gen.generate(write_tex=False)
    check_access(audit,access)
    tex=gen.render_table(rows,audit,sidecar="main_prediction_v2_preview_sources.md")
    assert tex_path.read_bytes() == original_tex
    assert side_path.read_bytes() == original_side
    assert all(not p.endswith(".tex") for p in access[1])
    side=(ROOT/"generated/tables/main_prediction_v2_preview_sources.md").read_text()
    blocks=re.findall(r"```json\n(.*?)\n```",side,re.S)
    recipes=json.loads(blocks[0])
    caption=json.loads(blocks[1])
    assert len(rows)==8 and len(recipes)==48
    body=tex.split("\\toprule\n",1)[1].split("\\bottomrule",1)[0]
    table_rows=[block.strip().splitlines() for block in body.split("\\midrule\n")]
    assert len(table_rows)==8 and all(len(block)==5 for block in table_rows)
    for record in [*recipes,caption]:
        # Independent JSON access, aggregation and formatting, including numeric
        # strings in formulas, identifiers, configuration grids and intervals.
        parts=[]
        for p in record["parts"]:
            if isinstance(p,str):
                parts.append(p);continue
            vals=[raw_source(s) for s in p["sources"]]
            op=p["op"]
            if op=="identity": v=vals[0]
            elif op=="mean": v=mean(vals)
            elif op=="length": v=len(vals[0])
            elif op=="keys": v=", ".join(vals[0])
            elif op=="unique": v=", ".join(str(v) for v in sorted(set(vals)))
            elif op=="join": v=", ".join(map(str,vals[0]))
            elif op=="shared":
                assert vals and all(v==vals[0] for v in vals)
                v=vals[0]
            elif op=="label":
                assert vals==p["expected"]
                v=p["label"]
            else: pytest.fail(f"Unhandled source operation {op}")
            parts.append("not tested" if v is None else format(v,p["format"]) if p["format"] else str(v))
        for ref in record["context"]:
            raw_source(ref)
        # Rendering itself is presentation only. Verify the exact emitted TeX
        # cell, not just the returned in-memory records.
        expected=r"\newline ".join("".join(token if token.startswith("$") else gen.tex_escape(token)
                                            for token in re.split(r"(\$[^$]+\$)", line))
                                    for line in "".join(parts).split("\n"))
        if "".join(parts) in gen.STATUSES:
            expected=r"\mbox{"+expected+"}"
        if "row" in record:
            block=table_rows[record["row"]]
            column=record["column"]
            if column==0:
                actual=block[0].split(r"\mbox{\textbf{",1)[1].split(r"}}\hfill ",1)[0]
            elif column==5:
                actual=block[0].split(r"\hfill ",1)[1].removesuffix(r"} \\")
            else:
                actual=block[column].removesuffix(r" \\").split(" & ",1)[1]
        else:
            actual=next(line for line in tex.splitlines() if line.startswith(r"\caption{"))[9:-1].removeprefix(r"\footnotesize ")
        assert actual==expected==record["rendered"]
        assert re.findall(r"[-+]?\d+(?:\.\d+)?",actual)==re.findall(r"[-+]?\d+(?:\.\d+)?",expected)
    assert [row[-1].plain(audit) for row in rows]==[
        "development", *["frozen prediction"]*4, "development", "development", "frozen prediction"]
    assert all("strongest_observed_baseline" not in source and "strongest_baseline_mae" not in source
               for r in recipes for p in r["parts"] if isinstance(p,dict) for source in p["sources"])
    assert "not tested" in tex
    # Key values differ from the post-hoc oracle: catch accidental reuse of
    # V86 Row.baselines or A2 strongest_baseline_mae.
    assert rows[7][4].plain(audit)=="budget only and response surface with initial loss 0.07,0.03; reuse only 0.02,0.05; reuse only and response surface with student size 0.61,0.45"
    assert rows[7][3].plain(audit)=="Math 0.07,0.06; Code 0.02,0.05; QA 0.51,0.46"
    assert rows[5][4].plain(audit)=="constant 0.09; constant 0.11; constant 1.44"
    assert rows[2][4].plain(audit)=="development median 0.55; development median 0.73; development median 0.54"
    assert rows[6][4].plain(audit)=="response surface and reuse only 0.08; zero change, response surface and reuse only 0.07; response surface 0.76"

    # The local artifact is LOSO over 17 states, not the requested density
    # holdout. Keep that evidence gap visible; never rebrand V72 or LOSO scores.
    assert raw_source(gen.P53+"#/n_dev_states")==17
    assert all(isinstance(f["held_out"], str) and "@step" in f["held_out"]
               for f in raw_source(gen.P53+"#/loso_folds"))
    assert rows[0][-1].plain(audit)=="development"
    assert rows[0][3].plain(audit)==rows[0][4].plain(audit)=="not tested"
    assert "density-holdout" in rows[0][3].note
    assert not any(p.get("format") for r in recipes if r["row"]==0 and r["column"] in (3,4)
                   for p in r["parts"] if isinstance(p,dict))
    assert not any("v72-prune-repeat" in s for r in recipes
                   for p in r["parts"] if isinstance(p,dict) for s in p["sources"])
    assert rows[1][-1].plain(audit)=="frozen prediction"
    assert rows[1][3].plain(audit)=="Math 0.24; Code 0.24; QA 0.68"
    assert rows[1][2].plain(audit)=="new stages of seen sizes; a 6.9B source outside the size range"
    assert "budgets of at least 150000 tokens held out" in rows[5][2].plain(audit)

    assert r"\centering\footnotesize" in tex and r"\tiny" not in tex
    assert tex.count(r"\begin{table*}[t]")==1 and r"\rotatebox" not in tex
    assert r"\setlength{\tabcolsep}{3pt}" in tex
    assert all(header+" & " in tex for header in gen.HEADERS[1:5])
    assert sum(map(float,gen.COLUMN_WIDTHS))==pytest.approx(1)
    assert r"\textwidth-2\tabcolsep\relax" in tex
    assert "Inputs" not in gen.HEADERS
    assert caption["rendered"]==gen.CAPTION
    assert r"$A_c(\mathbf x)r^{\gamma_c}$ (5)" in tex
    assert r"$\phi^\top Q_c$ (20)" in tex
    assert r"$m_c(b,g)$ (9)" in tex and r"$0$ (0)" in tex
    assert "additive logarithmic form (4); curved form (5); interaction form (5)" in tex
    assert r"Math and Code: $a_c\log(1+E)$ (1); QA: $u(a+bu+qv)$ (3)" in tex
    assert "training probes" in caption["rendered"]
    for i,row in enumerate(rows):
        assert len(row)==6
        assert len(row[2].plain(audit).split())<=16
        assert "/" not in row[2].plain(audit)
        assert all("\n" not in c.plain(audit) for c in row)
        assert "@step" not in row[2].plain(audit) and "training_probe:" not in row[2].plain(audit)
        if i:
            assert re.match(r"Math \d.*; Code \d.*; QA \d",row[3].plain(audit))
            assert len(row[4].plain(audit).split("; "))==3
        # Only A2 stores intervals for the printed MAE estimand. V70 gain
        # intervals cannot become MAE intervals merely to fill a table cell.
        if i not in (5,6):
            assert "[" not in row[3].plain(audit)
        else:
            assert len(re.findall(r"\d+\.\d{2} \[\d+\.\d{2},\d+\.\d{2}\]",row[3].plain(audit)))==3
    assert "paired_difference" not in " ".join(s for r in recipes for p in r["parts"]
                                                 if isinstance(p,dict) for s in p["sources"])

    assert [row[0].plain(audit) for row in rows]==[
        "Pruning, unseen density", "Pruning, new source state",
        "Quantization, unseen bit-width", "Quantization, unseen group size",
        "Quantization, new state", "Distillation, budget response",
        "Distillation, data-reuse response", "Distillation, new pool"]
    assert r"\-" not in tex


@pytest.mark.parametrize("status", sorted(gen.STATUSES))
def test_status_cannot_hyphenate(status):
    assert gen.render_text(status)==r"\mbox{"+status+"}"


@pytest.mark.parametrize("component",["parent","directory"])
def test_table_refuses_symlinked_output(tmp_path,component):
    refuses_symlink(gen,tmp_path,"tables",component)


def test_table_confines_io(tmp_path):
    refuses_external_io(tmp_path)
    refuses_output_file_symlink(tmp_path,"tables","main_prediction_v2.tex")
