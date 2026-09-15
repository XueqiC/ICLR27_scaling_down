from collections import defaultdict

import pytest

from analysis import plot_fig_responses_v2 as gen
from analysis.paper_artifacts import Artifacts, development_rows
from paper_generator_checks import check_access, check_figures, refuses_symlink, refuses_external_io, refuses_output_file_symlink


def test_responses_produces_files_from_nearest_budget_and_separate_seeds():
    rows,audit,access=gen.generate();check_access(audit,access);check_figures("responses_v2")
    raw=development_rows(Artifacts())
    groups=defaultdict(list)
    for r in raw:
        if r["T_actual"]>0:
            groups[r["run_id"],r["capability"],r["distribution"]].append(r)
    assert len(rows)==132
    for r in rows:
        cap=r["series"] if r["panel"]=="left" else "qa"
        candidates=groups[r["run_id"],cap,r["distribution"]]
        expected=min(candidates,key=lambda p:(abs(p["T_actual"]-200000),p["checkpoint_id"]))
        assert r["checkpoint_id"]==expected["checkpoint_id"]
        assert r["delta"]==expected["delta"]
        assert r["reuse"]==expected["T_actual"]/expected["D_U_pool"]
    for panel in ("left","right"):
        for student in gen.STUDENTS:
            for series in (gen.CAPS if panel=="left" else gen.SCOPES):
                subset=[r for r in rows if (r["panel"],r["student"],r["series"])==(panel,student,series)]
                by_rung=defaultdict(set)
                for r in subset: by_rung[r["U"]].add(r["pool_seed"])
                assert all(len(seeds)==2 for seeds in by_rung.values())
    assert all(r["status"]=="development" for r in rows)


@pytest.mark.parametrize("component",["parent","directory"])
def test_responses_refuses_symlinked_output(tmp_path,component):
    refuses_symlink(gen,tmp_path,"figs",component)


def test_responses_confines_io(tmp_path):
    refuses_external_io(tmp_path)
    refuses_output_file_symlink(tmp_path,"figs","responses_v2.pdf")
