import json
from statistics import median

import pytest

from analysis import plot_fig_explanation as gen
from analysis.paper_artifacts import ROOT, Artifacts, pyplot
from analysis import paper_figure_style as style
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


@pytest.mark.parametrize("panel", "ab")
def test_explanation_line_marker_whisker_and_jitter_sizes(panel):
    from matplotlib.container import ErrorbarContainer
    from matplotlib.collections import PolyCollection
    audit = Artifacts()
    data = dict(profiles=gen.curvature(audit), corners=gen.corners(audit), displacement=gen.displacement(audit))
    plt = pyplot()
    style.apply_style("double")
    fig = plt.figure(figsize=gen.PANEL_SIZES[panel])
    try:
        ax = gen.draw_panel(fig, data, panel)
        assert tuple(fig.get_size_inches()) == (2.7, 1.45)
        assert ax.get_legend() is None
        for line in ax.lines:
            if line.get_marker() in ("o", "s", "^", "D", "x"):
                assert line.get_markersize() == 3.8
        for container in ax.containers:
            if isinstance(container, ErrorbarContainer):
                assert all(list(bars.get_linewidths()) == [1.1] for bars in container.lines[2])
                assert all(cap.get_markersize() == 4 and cap.get_markeredgewidth() == .6
                           for cap in container.lines[1])
        if panel == "a":
            references = [line for line in ax.lines if line.get_linestyle() in (":", "--")
                          and list(line.get_ydata()) in ([0, 0], [1, 1])]
            assert len(references) == 2
            assert all(line.get_color() == style.PALETTE["reference"] for line in references)
            for i, profile in enumerate(data["profiles"]):
                folds = [line for line in ax.lines if line.get_color() == gen.COLORS[profile["capability"]]
                         and line.get_marker() in ("o", "x")]
                assert len(folds) == len(profile["folds"])
                xs = [line.get_xdata()[0] for line in folds]
                assert xs == [i] * len(xs)
                assert [line.get_ydata()[0] for line in folds] == [r["p"] for r in profile["folds"]]
        else:
            assert len(ax.lines) == 3 and all(line.get_linewidth() == 1.1 for line in ax.lines)
    finally:
        plt.close(fig)
