import json
from statistics import median

import pytest

from analysis import plot_fig_explanation as gen
from analysis.paper_artifacts import ROOT
from paper_generator_checks import check_access, check_figures, refuses_symlink, refuses_external_io, refuses_output_file_symlink


def test_explanation_frozen_profiles_corners_and_three_families():
    data,audit,access=gen.generate();check_access(audit,access);check_figures("explanation")
    a2=json.loads((ROOT/gen.A2).read_text());a5=json.loads((ROOT/gen.A5).read_text())
    assert gen.CAPTION=="failed to reject"
    assert len(data["profiles"])==3
    for p in data["profiles"]:
        full=next(r for r in a2["parameter_intervals"] if r["capability"]==p["capability"] and r["distribution"].startswith("training_probe:"))
        assert p["p"]==full["fits"]["F_curv"]["fit"]["p"]
        assert p["interval"]==full["fits"]["F_curv"]["p_interval"]
        assert any(f["boundary"] for f in p["folds"])
    assert len(data["corners"])==6
    for r in data["corners"]:
        source=a5["students"][r["student"]]["readouts"][r["capability"]]
        assert r["interval"]==source["interval"] and r["noise"]==source["noise_on_I"]
        assert r["additive"]==0
        assert r["F_int"]!=source["predicted_disagreement"]
    assert len({r["family"] for r in data["displacement"]})==3
    for group in data["displacement"]:
        damage=[];errors=[]
        for r in group["records"]:
            path,pointer=r["source"].split("#/")
            raw=json.loads((ROOT/path).read_text())["per_capability"][pointer.split("/")[-1]]
            damage.append(abs(raw["measured_delta"]))
            errors.append(abs(raw["second_order_prediction"]-raw["measured_delta"])/abs(raw["measured_delta"]))
        assert group["damage"]==median(damage)
        assert group["median_relative_error"]==median(errors)
        assert all("cluster" not in r["source"] and "uncentred" not in r["source"] for r in group["records"])


@pytest.mark.parametrize("component",["parent","directory"])
def test_explanation_refuses_symlinked_output(tmp_path,component):
    refuses_symlink(gen,tmp_path,"figs",component)


def test_explanation_confines_io(tmp_path):
    refuses_external_io(tmp_path)
    refuses_output_file_symlink(tmp_path,"figs","explanation.png")
