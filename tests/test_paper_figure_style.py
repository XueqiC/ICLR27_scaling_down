"""Publication contract: physical typography, uncropped panels and frozen output."""
import re

import pytest

from analysis import paper_figure_style as style
from analysis import plot_fig_responses_v2 as responses
from analysis import plot_fig_explanation as explanation
from analysis import plot_fig_generalization as generalization
from analysis.paper_artifacts import ROOT, pyplot, output_path

EXPECTED_PANEL_SIZES = {
    **{f"fig1_{p}": (1.35, 1.25) for p in "abcd"}, "fig1_legend": (5.5, .42),
    **{f"fig2_{p}": (2.7, 1.45) for p in "ab"}, "fig2_legend": (5.5, .42),
    **{f"fig3_{p}": (2.7, 1.6) for p in "ab"}, "fig3_legend": (5.5, .42),
    "lr_pilot_a": (2.7, 1.45), "lr_pilot_legend": (5.5, .42),
    **{p: (2.7, 1.6) for p in ("corner_test_a", "corner_test_b", "corner_contrasts_a")},
    "corner_legend": (5.5, .42),
    **{f"fig4_{p}": (1.8, 1.35) for p in "abc"}, "fig4_legend": (5.5, .3),
    **{f"fig5_{p}": (2.7, 1.15) for p in "ab"},
    "fig5_legend": (5.5, .3),
    "fig7_a": (1.75, 1.5), **{f"fig7_{p}": (1.25, 1.5) for p in "bcd"},
    "fig7_legend": (5.5, .3),
    "fig8_a": (2.7, 2.0), "fig8_b": (2.7, 2.0), "fig8_legend": (5.5, .3),
}


@pytest.fixture(scope="module", autouse=True)
def scratch_root(tmp_path_factory):
    # pyplot deliberately confines tempfile probes to its Matplotlib cache.
    # Allocate pytest's unrelated scratch files before that global is changed.
    tmp_path_factory.getbasetemp()


def all_figures(fig):
    yield fig
    for sub in fig.subfigs:
        yield from all_figures(sub)


def check_artists(fig):
    fig.set_dpi(220)  # Validate the same glyph metrics as the exported preview.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for canvas in all_figures(fig):
        assert not canvas.texts, "Captions/notes belong in sidecars"
        for i, legend in enumerate(canvas.legends):
            assert all(not legend.get_window_extent(renderer).overlaps(other.get_window_extent(renderer))
                       for other in canvas.legends[i+1:]), "Shared legend groups must not overlap"
        for legend in canvas.legends:
            box = legend.get_window_extent(renderer)
            # SubFigure transforms can differ at an edge by ~1e-13 pixels.
            assert box.x0 >= canvas.bbox.x0-.5 and box.y0 >= canvas.bbox.y0-.5
            assert box.x1 <= canvas.bbox.x1+.5 and box.y1 <= canvas.bbox.y1+.5
            assert box.y0 == pytest.approx(canvas.bbox.y0, abs=.5), "Strip has no bottom padding"
            assert canvas.bbox.y1-box.y1 >= .02*fig.dpi-.5, "Keep a small top margin"
            ys = sorted(t.get_window_extent(renderer).y0 for t in legend.get_texts())
            # Math subscripts can move a glyph bounding box by 1-2 pixels on the same baseline.
            row_count = 1 + sum(b-a > 2 for a, b in zip(ys, ys[1:]))
            assert row_count <= 2, "Shared keys use at most two readable rows"
            assert legend.handlelength == 1.6 and legend.columnspacing == 1.4
            from matplotlib.transforms import Bbox
            entries = [Bbox.union([handle.get_window_extent(renderer), text.get_window_extent(renderer)])
                       for handle, text in zip(legend.legend_handles, legend.get_texts())]
            for i, entry in enumerate(entries):
                assert all(not entry.overlaps(other) for other in entries[i+1:]), "Legend entries touch"
            for ax in canvas.get_axes():
                assert not box.overlaps(ax.xaxis.label.get_window_extent(renderer))
    for ax in fig.axes:
        assert ax.get_legend() is None, "All legends belong in strips above the panels"
        assert not any(ax.get_title(loc) for loc in ("left", "center", "right"))
        assert not ax.texts, "Titles and explanatory text belong in captions"
        assert ax.get_xlabel() or ([t.get_text() for t in ax.get_xticklabels()] == ["Math", "Code", "QA"]
                                  and all(t.get_rotation() == 0 for t in ax.get_xticklabels())) or (len(ax.get_xticklabels()) == 12 and
                                  all(t.get_rotation() == 45 for t in ax.get_xticklabels()))
        assert "\n" not in ax.get_xlabel() + ax.get_ylabel(), "Axis labels must stay on one line"
        canvas = ax.get_figure()
        for label in (ax.xaxis.label, ax.yaxis.label):
            if label.get_text():
                bounds = label.get_window_extent(renderer)
                assert bounds.x0 >= canvas.bbox.x0-.5 and bounds.y0 >= canvas.bbox.y0-.5
                assert bounds.x1 <= canvas.bbox.x1+.5 and bounds.y1 <= canvas.bbox.y1+.5
        assert ax.bbox.height / canvas.bbox.height >= .58, "Preserve usable plot height"
        assert (ax.get_ylabel() or any(t.get_text() for t in ax.get_yticklabels())
                or (len(ax.images) == 1 and ax.images[0].get_array().shape == (4, 17))), \
            "Only aligned selection maps may omit repeated state labels"
        text = ax.get_xticklabels(which="both") + ax.get_yticklabels(which="both")
        text += [ax.xaxis.label, ax.yaxis.label]
        if ax.get_legend():
            text += ax.get_legend().get_texts()
        assert all(t.get_weight() == "bold" and t.get_family() == ["serif"] for t in text)
    # A full-canvas PDF must not clip labels or row names at a page edge.
    box = fig.get_tightbbox(renderer).transformed(fig.dpi_scale_trans)
    assert box.x0 >= -.5 and box.y0 >= -.5, (box.bounds, fig.bbox.bounds)
    assert box.x1 <= fig.bbox.x1+.5 and box.y1 <= fig.bbox.y1+.5, (box.bounds, fig.bbox.bounds)


@pytest.mark.parametrize("kind,sizes", [("narrow", (7, 8, 7)), ("panel", (7.5, 8.5, 7.5)), ("double", (8.5, 9.5, 8.5)),
                                          ("full", (9, 10, 9)), ("legend", (8.5, 8.5, 8.5))])
def test_shared_serif_bold_rcparams(kind, sizes):
    plt = pyplot()
    rc = style.apply_style(kind)
    assert plt.rcParams["font.family"] == ["serif"]
    assert plt.rcParams["font.serif"] == style.SERIF
    assert plt.rcParams["mathtext.fontset"] == "stix"
    assert plt.rcParams["mathtext.default"] == "bf"
    for key in ("font.weight", "axes.labelweight", "axes.titleweight"):
        assert plt.rcParams[key] == "bold"
    assert [rc[k] for k in ("xtick.labelsize", "axes.labelsize", "legend.fontsize")] == list(sizes)
    assert rc["ytick.labelsize"] == sizes[0]
    assert rc["lines.linewidth"] == 1.1 and rc["lines.markersize"] == 3.8
    assert rc["errorbar.capsize"] == 2
    assert plt.rcParams["savefig.bbox"] is None


@pytest.mark.parametrize("gen,prefix,letters", [
    (responses, "fig1", "abcd"), (explanation, "fig2", "ab"), (generalization, "fig3", "ab"),
])
def test_panel_files_and_artist_contract(monkeypatch, gen, prefix, letters):
    saved = {}
    rectangles = {}
    save = gen.save_panel
    combined_save = gen.save_figure

    def inspect_panel(fig, stem, kind, audit, records):
        assert tuple(fig.get_size_inches()) == EXPECTED_PANEL_SIZES[stem]
        check_artists(fig)
        tick, label, legend = style.SIZES[kind]
        for ax in fig.axes:
            assert ax.xaxis.label.get_fontsize() == label
            assert all(t.get_fontsize() == tick for t in ax.get_xticklabels()+ax.get_yticklabels())
            if gen is generalization:
                assert ax.get_position().width >= .58
        for canvas in all_figures(fig):
            legends = canvas.legends + [a.get_legend() for a in canvas.get_axes() if a.get_legend()]
            for leg in legends:
                sizes = [t.get_fontsize() for t in leg.get_texts()]
                expected = [legend]*len(sizes)
                if stem in ("fig1_legend", "fig2_legend", "fig3_legend"):
                    ys = {round(t.get_window_extent(fig.canvas.get_renderer()).y0) for t in leg.get_texts()}
                    assert len(ys) == (2 if stem == "fig1_legend" else 1)
                assert sizes == expected
                assert all(t.get_weight() == "bold" for t in leg.get_texts())
        if fig.axes:
            rectangles[stem] = tuple(fig.axes[0].get_position().bounds)
        saved[stem] = tuple(fig.get_size_inches())
        save(fig, stem, kind, audit, records)

    def inspect_combined(fig, stem, audit):
        check_artists(fig)
        assert fig.get_size_inches()[0] == 5.5
        panels = [sub for sub in fig.subfigs if sub.axes]
        assert len(panels) == (4 if gen is responses else 2)
        row_positions = {round(s.bbox.y0) for s in panels}
        assert len(row_positions) == 1
        if gen is responses:
            assert tuple(fig.get_size_inches()) == pytest.approx((5.5, 1.67))
            for panel in panels:
                assert panel.bbox.size/fig.dpi == pytest.approx((1.35, 1.25))
            for left, right in zip(panels, panels[1:]):
                assert (right.bbox.x0-left.bbox.x1)/fig.dpi == pytest.approx(.03)
        strips = [sub for sub in fig.subfigs if sub.legends]
        assert len(strips) == 1
        assert strips[0].bbox.y0 >= max(sub.bbox.y1 for sub in panels)-.5
        combined_save(fig, stem, audit)

    monkeypatch.setattr(gen, "save_panel", inspect_panel)
    monkeypatch.setattr(gen, "save_figure", inspect_combined)
    rows, _, _ = gen.generate()
    if gen in (responses, explanation):
        assert len(set(rectangles.values())) == 1
    for letter in letters:
        stem = f"{prefix}_{letter}"
        path = output_path(ROOT, "figs", f"{stem}.pdf")
        if gen in (responses, explanation):
            import json
            sidecar = json.loads(path.with_name(f"{stem}_data.json").read_text())
            assert sidecar["axes"][0]["rectangle"] == list(rectangles[stem])
        raw = path.read_bytes()
        assert raw.startswith(b"%PDF-") and len(raw) > 1000
        # Matplotlib's PDF page box is in physical points, independent of DPI.
        bounds = re.search(rb"/MediaBox\s*\[\s*0\s+0\s+([\d.]+)\s+([\d.]+)\s*\]", raw)
        assert bounds
        assert [float(v)/72 for v in bounds.groups()] == pytest.approx(saved[stem])
        assert (path.parent / f"{stem}_sources.md").stat().st_size > 100
    if gen is responses:
        assert "fig1_legend" in saved
        assert output_path(ROOT, "figs", "fig1_legend.pdf").stat().st_size > 1000
        import json
        legend = json.loads(output_path(ROOT, "figs", "fig1_legend_data.json").read_text())
        assert set(legend["legend_entries"]) == {
            "Math", "Code", "QA", "2Wiki", "MuSiQue", "TriviaQA", "270M", "1B", "4B", "S1", "S2"}
    assert set(saved) == {f"{prefix}_{p}" for p in letters} | {f"{prefix}_legend"}
    for stem in ("fig2_c", "fig3_c", "fig3_d"):
        assert not output_path(ROOT, "figs", stem+".pdf").exists()


def test_explanations_moved_to_caption_files():
    directory = output_path(ROOT, "figs")
    expected = {
        "responses_v2": ("Positive = worse", "Development endpoints nearest 200k"),
        "explanation": ("Curvature intervals are conditional on development", "post-hoc",
                        "Fold estimate, Boundary", "Group, Channel and Prune"),
        "generalization": ("Paired gain CIs are translated",
                           "Prune in, frozen", "Prune out, frozen", "Baseline, Relation and Lower of pair"),
    }
    for stem, phrases in expected.items():
        text = (directory / f"{stem}_caption.txt").read_text()
        assert all(phrase in text for phrase in phrases)


@pytest.mark.parametrize("gen,labels", [
    (generalization, {"Math", "Code", "QA", "Baseline", "Relation", "Lower of pair"}),
    (explanation, {"Fold estimate", "Boundary", "Full", "Group", "Channel", "Prune"}),
])
def test_expanded_legend_labels_fit_at_unchanged_font_size(gen, labels):
    plt = pyplot()
    style.apply_style("legend")
    fig = plt.figure(figsize=gen.LEGEND_SIZE)
    try:
        legend = gen.draw_legend(fig)
        check_artists(fig)
        assert {t.get_text() for t in legend.get_texts()} == labels
        assert all(t.get_fontsize() == 8.5 for t in legend.get_texts())
    finally:
        plt.close(fig)


@pytest.mark.parametrize("gen", [responses, explanation])
def test_harmonised_rows_have_explicit_matching_rectangles_and_sparse_ticks(gen):
    from analysis.paper_artifacts import Artifacts
    from analysis import plot_fig_lr_pilot as pilot
    plt = pyplot()
    audit = Artifacts()
    if gen is responses:
        rows = gen.build(audit)
        rectangle = gen.row_rectangle(rows, plt)
        fig = gen.plot(rows, plt)
    else:
        data = dict(profiles=gen.curvature(audit), corners=gen.corners(audit),
                    displacement=gen.displacement(audit))
        rectangle = gen.row_rectangle(data, plt)
        fig = gen.plot(data, plt)
    try:
        check_artists(fig)
        assert fig.get_layout_engine() is None
        for ax in fig.axes:
            assert ax.get_position().bounds == pytest.approx(rectangle, abs=1e-14)
            assert ax.get_position().bounds == fig.axes[0].get_position().bounds
            assert ax.bbox.y0 == pytest.approx(fig.axes[0].bbox.y0)
            assert ax.bbox.height == pytest.approx(fig.axes[0].bbox.height)
            max_ticks = 3 if gen is responses else 4
            assert len(ax.get_xticks()) <= max_ticks and len(ax.get_yticks()) <= max_ticks
            if gen is responses:
                assert ax.get_xlabel() == "Reuse ratio E"
                assert ax.get_ylabel() == ("Loss change (nats)" if ax is fig.axes[0] else "")
                assert ax.xaxis.label.get_fontsize() == ax.yaxis.label.get_fontsize() == 8
                assert all(t.get_fontsize() == 7 for t in ax.get_xticklabels()+ax.get_yticklabels())
            else:
                assert ax.get_ylabel()
            assert not ax.spines["top"].get_visible() and not ax.spines["right"].get_visible()
            assert ax.spines["left"].get_linewidth() == .6
            for axis in (ax.xaxis, ax.yaxis):
                if len(axis.get_majorticklocs()) and not (gen is explanation and ax is fig.axes[0] and axis is ax.xaxis):
                    assert axis.get_major_formatter().func is style.compact_number
                for tick in axis.get_major_ticks():
                    assert tick.tick1line.get_markersize() == 2
                    assert tick.tick1line.get_markeredgewidth() == .6
                for grid in axis.get_gridlines():
                    assert grid.get_visible() and grid.get_alpha() == .15 and grid.get_linewidth() == .4
            # Horizontal tick labels must fit without overlapping each other.
            renderer = fig.canvas.get_renderer()
            boxes = [t.get_window_extent(renderer) for t in ax.get_xticklabels()]
            assert all(not a.overlaps(b) for i, a in enumerate(boxes) for b in boxes[i+1:])
        if gen is responses:
            assert len({ax.get_ylim() for ax in fig.axes}) == 4
            for ax, panel in zip(fig.axes, gen.PANELS.values()):
                assert all(ax.get_ylim()[0] < r["delta"] < ax.get_ylim()[1]
                           for r in gen.panel_records(rows, panel))
            assert fig.axes[0].get_ylim()[1] < 1 and fig.axes[1].get_ylim()[1] < 1
            assert fig.axes[2].get_ylim()[1] > 5
        else:
            a, b = fig.axes
            assert a.get_xlabel() == ""
            assert [t.get_text() for t in a.get_xticklabels()] == ["Math", "Code", "QA"]
            assert b.get_xlabel() == "Absolute damage (nats)"
    finally:
        plt.close(fig)
