"""Drift summaries must preserve frozen ratios and separate tolerance sets."""
import copy
from collections import Counter

import pytest

from analysis import paper_figure_style as style
from analysis import plot_fig_drift as drift
from analysis.paper_artifacts import Artifacts, pyplot
from test_paper_figure_style import check_artists


def test_medians_use_saved_bands_and_individual_ratios_without_mutating_rows():
    rows = []
    # Actual budgets differ within a saved band; an extreme ratio distinguishes
    # the median from a mean. Tight-tolerance records must remain separate.
    for band, tolerance, xs, ratios in (
        ("25k", .01, [24., 25., 26.], [1., 2., 90.]),
        ("50k", .01, [49., 51.], [2., 4.]),
        ("25k", .003, [24.], [1.]),
        ("50k", .003, [51.], [4.]),
    ):
        for i, (x, ratio) in enumerate(zip(xs, ratios)):
            rows.append(dict(panel="a", capability="math", tolerance=tolerance,
                             band=band, trajectory_clusters=[f"pool-{i}", "pool-b"],
                             x=x, delta=ratio))
    rows += [dict(rows[0], panel="b", delta=-999),
             dict(rows[0], capability="code", delta=-999)]
    original = copy.deepcopy(rows)
    plt = pyplot()
    style.apply_style("double")
    fig = plt.figure(figsize=drift.PANEL_SIZE)
    try:
        ax = drift.draw_panel(fig, rows, "a")
        lines = {line.get_marker(): line for line in ax.lines
                 if line.get_color() == drift.COLORS["math"] and line.get_linewidth() == 1.8}
        assert list(lines["o"].get_xdata()) == [25., 50.]
        assert list(lines["o"].get_ydata()) == [2., 3.]
        assert list(lines["^"].get_xdata()) == [24., 51.]
        assert list(lines["^"].get_ydata()) == [1., 4.]
        assert rows == original
    finally:
        plt.close(fig)


@pytest.mark.parametrize("panel", "ab")
def test_print_size_layers_and_inside_legend(panel):
    rows, unavailable = drift.build(Artifacts())
    assert not unavailable
    plt = pyplot()
    style.apply_style("double")
    fig = plt.figure(figsize=drift.PANEL_SIZE)
    try:
        ax = drift.draw_panel(fig, rows, panel)
        check_artists(fig)
        assert tuple(fig.get_size_inches()) == (2.7, 1.45)
        assert ax.get_xlim() == (0, 220) and ax.get_ylim() == (-1, 8)
        thin = [line for line in ax.lines if line.get_linewidth() == .6]
        heavy = [line for line in ax.lines if line.get_linewidth() == 1.8]
        expected = Counter()
        for cap in drift.CAPS:
            for tolerance in drift.TOLERANCES:
                part = [r for r in rows if (r["panel"], r["capability"], r["tolerance"])
                        == (panel, cap, tolerance)]
                for dyad in {tuple(r["trajectory_clusters"]) for r in part}:
                    points = sorted((r["x"], r["delta"]) for r in part
                                    if tuple(r["trajectory_clusters"]) == dyad)
                    expected[drift.COLORS[cap], tuple(points)] += 1
        assert Counter((line.get_color(), tuple(zip(line.get_xdata(), line.get_ydata())))
                       for line in thin) == expected
        assert all(line.get_alpha() == .35 and line.get_marker() in ("", "None") for line in thin)
        assert Counter((line.get_color(), line.get_marker()) for line in heavy) == Counter(
            (drift.COLORS[cap], marker) for cap in drift.CAPS for marker in ("o", "^"))
        assert all(line.get_alpha() == 1 and line.get_markersize() == 5
                   and line.get_markerfacecolor() == line.get_color()
                   and line.get_markeredgecolor() == "white"
                   and line.get_markeredgewidth() > 0 for line in heavy)
        assert min(line.get_zorder() for line in heavy) > max(line.get_zorder() for line in thin)
        assert all(len(line.get_xdata()) == (4 if line.get_marker() == "o" else 3) for line in heavy)
        reference = [line for line in ax.lines if line.get_linewidth() == 1.1]
        assert len(reference) == 1 and list(reference[0].get_ydata()) == [0, 0]
        legend = ax.get_legend()
        assert [t.get_text() for t in legend.get_texts()] == ["Math", "Code", "QA"]
        assert all(t.get_fontsize() == 8.5 for t in legend.get_texts())
        box = legend.get_window_extent(fig.canvas.get_renderer())
        assert ax.bbox.contains(box.x0, box.y0) and ax.bbox.contains(box.x1, box.y1)
    finally:
        plt.close(fig)
