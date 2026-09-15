import json
import re
from statistics import mean

import pytest

from analysis import final_prediction_table as gen
from analysis.paper_artifacts import ROOT
from paper_generator_checks import check_access, refuses_symlink, refuses_external_io, refuses_output_file_symlink


def raw_source(ref):
    path,pointer=ref.split("#",1)
    assert path.startswith(("results/","paper/docs/"))
    data=json.loads((ROOT/path).read_text())
    for key in pointer.lstrip("/").split("/") if pointer else []:
        key=key.replace("~1","/").replace("~0","~")
        data=data[int(key)] if isinstance(data,list) else data[key]
    return data


def test_table_every_cell_matches_sidecar_and_frozen_json():
    rows,audit,access=gen.generate()
    check_access(audit,access)
    tex=(ROOT/"paper/paper/tables/main_prediction_v2.tex").read_text()
    side=(ROOT/"paper/paper/tables/main_prediction_v2_sources.md").read_text()
    blocks=re.findall(r"```json\n(.*?)\n```",side,re.S)
    recipes=json.loads(blocks[0])
    caption=json.loads(blocks[1])
    assert len(rows)==8 and len(recipes)==48
    table_rows=[line for block in re.findall(r"\\midrule\n(.*?)\\bottomrule",tex,re.S)
                for line in block.splitlines()]
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
                                    for line in "".join(parts).split("\n")).replace(" / ","/ ")
        actual=(table_rows[record["row"]].removesuffix(r" \\").split(" & ")[record["column"]]
                if "row" in record else next(line for line in tex.splitlines() if line.startswith(r"\caption{"))[9:-1].removeprefix(r"\footnotesize "))
        assert actual==expected==record["rendered"]
        assert re.findall(r"[-+]?\d+(?:\.\d+)?",actual)==re.findall(r"[-+]?\d+(?:\.\d+)?",expected)
    assert [row[-1].plain(audit) for row in rows]==[
        "development", *["frozen prediction"]*4, "development", "development", "frozen prediction"]
    assert all("strongest_observed_baseline" not in source and "strongest_baseline_mae" not in source
               for r in recipes for p in r["parts"] if isinstance(p,dict) for source in p["sources"])
    assert "not tested" in tex
    # Key values differ from the post-hoc oracle: catch accidental reuse of
    # V86 Row.baselines or A2 strongest_baseline_mae.
    assert rows[7][4].plain(audit)=="T/L 0.07,0.03 / E 0.02,0.05 / E/N 0.61,0.45"
    assert rows[7][3].plain(audit)=="math 0.07,0.06 / code 0.02,0.05 / QA 0.51,0.46"
    assert rows[5][4].plain(audit)=="const 0.09 / const 0.11 / const 1.44"
    assert rows[2][4].plain(audit)=="med 0.55 / med 0.73 / med 0.54"

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
    assert rows[1][3].plain(audit)=="math 0.24 / code 0.24 / QA 0.68"
    assert rows[1][2].plain(audit)=="seen-size stages; 6.9B outside"
    assert r"$T\ge150$k held out" in rows[5][2].plain(audit)

    assert r"\centering\footnotesize" in tex and r"\tiny" not in tex
    assert tex.count(r"\begin{table*}[t]")==1 and r"\rotatebox" not in tex
    assert r"\setlength{\tabcolsep}{3pt}" in tex
    assert " & ".join(gen.HEADERS) in tex
    assert "Inputs" not in gen.HEADERS
    assert ("Inputs: source size, initial loss and pretraining tokens for source-conditioned forms; "
            "the configuration for all; distillation forms take supervised budget, pool size and reuse") in tex
    assert r"$A_c(\mathbf x)r^{\gamma_c}$ (5)" in tex
    assert r"$\phi^\top Q_c$ (20)" in tex
    assert r"$m_c(b,g)$ (9)" in tex and r"$0$ (0)" in tex
    assert r"$Au+Bv$ (4); $Au+Bh_p(E)$ (5); $Au+Bv+kuv$ (5)" in tex
    assert r"$a_c\ell_E$ (1)" in tex and r"$u(a+bu+qv)$ (3)" in tex
    assert "Not tested: density holdout, exact fixed-budget reuse" in tex
    assert "training probes" in caption["rendered"]
    for i,row in enumerate(rows):
        assert len(row)==6
        assert len(row[2].plain(audit).split())<=9
        assert all("\n" not in c.plain(audit) for c in row)
        assert "@step" not in row[2].plain(audit) and "training_probe:" not in row[2].plain(audit)
        if i:
            assert re.match(r"math \d.* / code \d.* / QA \d",row[3].plain(audit))
            assert len(row[4].plain(audit).split(" / "))==3
        # Only A2 stores intervals for the printed MAE estimand. V70 gain
        # intervals cannot become MAE intervals merely to fill a table cell.
        if i not in (5,6):
            assert "[" not in row[3].plain(audit)
        else:
            assert len(re.findall(r"\d+\.\d{2} \[\d+\.\d{2},\d+\.\d{2}\]",row[3].plain(audit)))==3
    assert "paired_difference" not in " ".join(s for r in recipes for p in r["parts"]
                                                 if isinstance(p,dict) for s in p["sources"])


@pytest.mark.parametrize("component",["parent","directory"])
def test_table_refuses_symlinked_output(tmp_path,component):
    refuses_symlink(gen,tmp_path,"tables",component)


def test_table_confines_io(tmp_path):
    refuses_external_io(tmp_path)
    refuses_output_file_symlink(tmp_path,"tables","main_prediction_v2.tex")
