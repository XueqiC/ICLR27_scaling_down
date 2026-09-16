from collections import defaultdict
from statistics import mean
import json

import pytest

from analysis import plot_fig_responses_v2 as gen
from analysis.paper_artifacts import Artifacts, development_rows, pyplot, output_path, ROOT
from analysis import paper_figure_style as style
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
    for letter, panel in gen.PANELS.items():
        sidecar = json.loads(output_path(ROOT, "figs", f"fig1_{letter}_data.json").read_text())
        assert sidecar["records"] == gen.panel_records(rows, panel)


@pytest.mark.parametrize("panel", gen.PANELS.values())
def test_one_mean_line_per_student_readout_and_raw_seed_markers(panel):
    all_rows = gen.build(Artifacts())
    rows = gen.panel_records(all_rows, panel)
    plt = pyplot()
    style.apply_style(gen.PANEL_KIND)
    fig = plt.figure(figsize=gen.PANEL_SIZE)
    try:
        ax = gen.draw_panel(fig, rows, panel)
        lines = [line for line in ax.lines if line.get_color() in (gen.QA_COLORS if panel == "distributions" else gen.COLORS).values()
                 and line.get_linestyle() != "None"]
        points = [line for line in ax.lines if line.get_marker() in ("o", "s")]
        assert len(lines) == (9 if panel == "distributions" else 3)
        assert len(points) == 2*len(lines)
        assert all(line.get_linewidth() == 1.1 and line.get_marker() == "None" for line in lines)
        assert all(line.get_linestyle() == "None" and line.get_markersize() == 3.8 for line in points)
        assert ax.get_legend() is None
        for series in (gen.SCOPES if panel == "distributions" else (panel,)):
            color = gen.QA_COLORS[series] if panel == "distributions" else gen.COLORS[series]
            for student, ls in zip(gen.STUDENTS, gen.STYLES):
                subset = [r for r in rows if (r["series"], r["student"]) == (series, student)]
                rungs = defaultdict(list)
                for r in subset:
                    rungs[r["U"]].append(r)
                expected = sorted((mean(r["reuse"] for r in part), mean(r["delta"] for r in part))
                                  for part in rungs.values())
                line = next(line for line in lines if line.get_color() == color and line.get_linestyle() == ls)
                assert list(zip(line.get_xdata(), line.get_ydata())) == expected
        from collections import Counter
        for parity, marker in ((1, "o"), (0, "s")):
            assert Counter((x, y) for line in points if line.get_marker() == marker
                           for x, y in zip(line.get_xdata(), line.get_ydata())) == Counter(
                               (r["reuse"], r["delta"]) for r in rows
                               if r["pool_seed"] % 2 == parity)
    finally:
        plt.close(fig)


@pytest.mark.parametrize("component",["parent","directory"])
def test_responses_refuses_symlinked_output(tmp_path,component):
    refuses_symlink(gen,tmp_path,"figs",component)


def test_responses_confines_io(tmp_path):
    refuses_external_io(tmp_path)
    refuses_output_file_symlink(tmp_path,"figs","responses_v2.pdf")
