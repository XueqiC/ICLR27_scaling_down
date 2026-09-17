"""Pooled drift ranges preserve all frozen pair/tolerance records."""
import copy
from statistics import median

import pytest

from analysis import paper_figure_style as style
from analysis import plot_fig_drift as drift
from analysis.paper_artifacts import Artifacts, pyplot
from test_paper_figure_style import check_artists


def test_pooled_range_and_median_count_nested_pairs_once_without_mutating_rows():
    rows = []
    for band, xs, ratios in (("25k", [24., 25., 26.], [1., 2., 90.]),
                             ("50k", [49., 51.], [2., 4.])):
        for i, (x, ratio) in enumerate(zip(xs, ratios)):
            rows.append(dict(panel="a", capability="math", tolerance=.01, band=band,
                             pair_id=f"{band}-{i}", x=x, delta=ratio))
    rows += [dict(rows[0], tolerance=.003), dict(rows[-1], tolerance=.003),
             dict(rows[0], panel="b", delta=-999), dict(rows[0], capability="code", delta=-999)]
    original = copy.deepcopy(rows)
    assert drift.pooled_summaries(rows, "a", "math") == [
        dict(band="25k", x=25., median=2., min=1., max=90.),
        dict(band="50k", x=50., median=3., min=2., max=4.)]
    assert rows == original
    # Pool the union of tolerances, including a pair seen only in the tighter set.
    rows.append(dict(rows[0], tolerance=.003, pair_id="tight-only", x=27., delta=4.))
    assert drift.pooled_summaries(rows, "a", "math")[0] == dict(
        band="25k", x=25.5, median=3., min=1., max=90.)


def test_conflicting_nested_pair_values_are_rejected():
    row = dict(panel="a", capability="math", tolerance=.01, band="25k", pair_id="p", x=25., delta=1.)
    with pytest.raises(ValueError, match="Inconsistent"):
        drift.pooled_summaries([row, dict(row, tolerance=.003, delta=2.)], "a", "math")


@pytest.mark.parametrize("panel", "ab")
def test_three_bands_three_median_lines_and_no_inside_legend(panel):
    rows, unavailable = drift.build(Artifacts())
    assert not unavailable
    original = copy.deepcopy(rows)
    plt = pyplot()
    style.apply_style("double")
    fig = plt.figure(figsize=drift.PANEL_SIZE)
    try:
        ax = drift.draw_panel(fig, rows, panel)
        check_artists(fig)
        assert tuple(fig.get_size_inches()) == (2.7, 1.15)
        assert ax.get_xlim() == (0, 220) and ax.get_ylim() == (-1, 8)
        assert list(ax.get_xticks()) == [0, 100, 200]
        assert list(ax.get_yticks()) == [0, 3, 6]
        assert len(ax.lines) == 4  # Three medians and the zero reference only.
        assert len(ax.collections) == 3
        heavy = [line for line in ax.lines if line.get_marker() == "o"]
        assert len(heavy) == 3
        assert all(line.get_marker() == "o" and line.get_markersize() == 3.8
                   and line.get_linestyle() == "-" and line.get_alpha() == 1
                   and line.get_markerfacecolor() == line.get_color()
                   and line.get_markeredgecolor() == style.PALETTE["white"] for line in heavy)
        for cap, line, band in zip(drift.CAPS, heavy, ax.collections):
            assert line.get_color() == drift.COLORS[cap]
            pairs = {(r["band"], r["pair_id"]): r for r in rows
                     if r["panel"] == panel and r["capability"] == cap}
            expected = []
            for name in {key[0] for key in pairs}:
                part = [r for (b, _), r in pairs.items() if b == name]
                expected.append((median(r["x"] for r in part), median(r["delta"] for r in part),
                                 min(r["delta"] for r in part), max(r["delta"] for r in part)))
            expected.sort()
            assert list(zip(line.get_xdata(), line.get_ydata())) == [(x, y) for x, y, _, _ in expected]
            vertices = band.get_paths()[0].vertices
            for x, _, lo, hi in expected:
                ys = vertices[vertices[:, 0] == x, 1]
                assert min(ys) == lo and max(ys) == hi
            assert band.get_alpha() == .15 and list(band.get_linewidths()) == [0]
            assert band.get_zorder() < line.get_zorder()
        assert ax.get_legend() is None and not fig.legends
        assert rows == original
    finally:
        plt.close(fig)


def test_drift_legend_strip_has_capabilities_and_range_entry():
    plt = pyplot()
    style.apply_style("legend")
    fig = plt.figure(figsize=drift.LEGEND_SIZE)
    try:
        legend = drift.draw_legend(fig)
        check_artists(fig)
        assert tuple(fig.get_size_inches()) == (5.5, .3)
        assert [text.get_text() for text in legend.get_texts()] == [
            "Math", "Code", "QA", "Range across pool pairs"]
    finally:
        plt.close(fig)


def test_drift_export_preserves_legend_and_aligns_short_panels(monkeypatch):
    import hashlib
    import json
    from analysis import paper_panel_exports as exports
    from analysis.paper_artifacts import ROOT, output_path
    from paper_generator_checks import check_access

    legend = output_path(ROOT, "figs", "fig5_legend.pdf")
    # Arrange the preexisting export here, independent of test ordering or a
    # developer's generated/ directory. The subsequent call exercises refresh.
    drift.generate()
    before = hashlib.sha256(legend.read_bytes()).hexdigest()
    save, combine = exports.save_panel, exports.combine_panels
    rectangles = []

    def inspect(fig, stem, kind, audit, records, **kwargs):
        check_artists(fig)
        if fig.axes:
            assert tuple(fig.get_size_inches()) == (2.7, 1.15)
            rectangles.append(tuple(fig.axes[0].get_position().bounds))
        save(fig, stem, kind, audit, records, **kwargs)

    def inspect_combined(plt, specs, size):
        fig = combine(plt, specs, size)
        check_artists(fig)
        assert tuple(size) == (5.5, 1.45)
        assert fig.axes[0].get_position().bounds == fig.axes[1].get_position().bounds
        return fig

    monkeypatch.setattr(exports, "save_panel", inspect)
    monkeypatch.setattr(exports, "combine_panels", inspect_combined)
    _, audit, access = drift.generate()
    check_access(audit, access)
    assert len(rectangles) == 2 and rectangles[0] == rectangles[1]
    assert legend.read_bytes().startswith(b"%PDF-")
    assert str(legend) in access[1]  # Palette changes must refresh the legend too.
    for panel in "ab":
        sidecar = json.loads(output_path(ROOT, "figs", f"fig5_{panel}_data.json").read_text())
        assert sidecar["axes"][0]["rectangle"] == list(rectangles[0])
