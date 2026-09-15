"""Publication contract: physical typography, uncropped panels and frozen output."""
import re

import pytest

from analysis import paper_figure_style as style
from analysis import plot_fig_responses_v2 as responses
from analysis import plot_fig_explanation as explanation
from analysis import plot_fig_generalization as generalization
from analysis.paper_artifacts import ROOT, pyplot


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
        for legend in canvas.legends:
            box = legend.get_window_extent(renderer)
            assert canvas.bbox.contains(box.x0, box.y0)
            assert canvas.bbox.contains(box.x1, box.y1)
            ys = [t.get_window_extent(renderer).y0 for t in legend.get_texts()]
            assert max(ys) - min(ys) < 3, "Outside legends must be a single row"
            for ax in canvas.get_axes():
                assert not box.overlaps(ax.xaxis.label.get_window_extent(renderer))
    for ax in fig.axes:
        assert not any(ax.get_title(loc) for loc in ("left", "center", "right"))
        assert not ax.texts, "Only axis/row labels and legends remain in the figure"
        assert ax.get_xlabel(), "Each panel needs its own x label"
        assert ax.get_ylabel() or any(t.get_text() for t in ax.get_yticklabels())
        text = ax.get_xticklabels(which="both") + ax.get_yticklabels(which="both")
        text += [ax.xaxis.label, ax.yaxis.label]
        if ax.get_legend():
            text += ax.get_legend().get_texts()
        assert all(t.get_weight() == "bold" and t.get_family() == ["serif"] for t in text)
    # A full-canvas PDF must not clip labels or row names at a page edge.
    box = fig.get_tightbbox(renderer).transformed(fig.dpi_scale_trans)
    assert box.x0 >= -.5 and box.y0 >= -.5, (box.bounds, fig.bbox.bounds)
    assert box.x1 <= fig.bbox.x1+.5 and box.y1 <= fig.bbox.y1+.5, (box.bounds, fig.bbox.bounds)


@pytest.mark.parametrize("kind,sizes", [("panel", (14, 15, 13)), ("full", (13, 14, 12))])
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
    assert rc["lines.linewidth"] == 2 and rc["lines.markersize"] == 7
    assert plt.rcParams["savefig.bbox"] is None


@pytest.mark.parametrize("gen,prefix,letters", [
    (responses, "fig1", "ab"), (explanation, "fig2", "abc"), (generalization, "fig3", "abcd"),
])
def test_panel_files_and_artist_contract(monkeypatch, gen, prefix, letters):
    saved = {}
    save = gen.save_panel
    combined_save = gen.save_figure

    def inspect_panel(fig, stem, kind, audit, records):
        check_artists(fig)
        tick, label, legend = style.SIZES[kind]
        for ax in fig.axes:
            assert ax.xaxis.label.get_fontsize() == label
            assert all(t.get_fontsize() == tick for t in ax.get_xticklabels()+ax.get_yticklabels())
        for canvas in all_figures(fig):
            legends = canvas.legends + [a.get_legend() for a in canvas.get_axes() if a.get_legend()]
            assert all(t.get_fontsize() == legend and t.get_weight() == "bold"
                       for leg in legends for t in leg.get_texts())
        saved[stem] = tuple(fig.get_size_inches())
        save(fig, stem, kind, audit, records)

    def inspect_combined(fig, stem, audit):
        check_artists(fig)
        combined_save(fig, stem, audit)

    monkeypatch.setattr(gen, "save_panel", inspect_panel)
    monkeypatch.setattr(gen, "save_figure", inspect_combined)
    rows, _, _ = gen.generate()
    for letter in letters:
        stem = f"{prefix}_{letter}"
        path = ROOT / f"paper/paper/figs/{stem}.pdf"
        raw = path.read_bytes()
        assert raw.startswith(b"%PDF-") and len(raw) > 1000
        # Matplotlib's PDF page box is in physical points, independent of DPI.
        bounds = re.search(rb"/MediaBox\s*\[\s*0\s+0\s+([\d.]+)\s+([\d.]+)\s*\]", raw)
        assert bounds
        assert [float(v)/72 for v in bounds.groups()] == pytest.approx(saved[stem])
        assert (path.parent / f"{stem}_sources.md").stat().st_size > 100
    if gen is responses:
        assert "fig1_legend" in saved
        assert (ROOT / "paper/paper/figs/fig1_legend.pdf").stat().st_size > 1000
    if gen is generalization:
        # Both students must be in c, and every 4B record keeps its triangle.
        plt = pyplot()
        style.apply_style("panel")
        fig = plt.figure(figsize=gen.PANEL_SIZES["c"])
        ax = gen.draw_panel(fig, rows, "c")
        points = [line for line in ax.lines if line.get_marker() in ("o", "^")]
        assert [t.get_text() for t in ax.get_yticklabels()] == ["1B", "4B dev."]
        assert sum(line.get_marker() == "^" for line in points) == 3
        assert sum(line.get_marker() == "o" for line in points) == 3
        assert ax.get_xlim() == (-2.5, 2.5)
        plt.close(fig)


def test_explanations_moved_to_caption_files():
    directory = ROOT / "paper/paper/figs"
    expected = {
        "responses_v2": ("Positive = worse", "Development endpoints nearest 200k"),
        "explanation": ("failed to reject", "Curvature intervals are conditional on development", "post-hoc"),
        "generalization": ("No frozen response-law predictions", "Paired gain CIs are translated", "failed to reject"),
    }
    for stem, phrases in expected.items():
        text = (directory / f"{stem}_caption.txt").read_text()
        assert all(phrase in text for phrase in phrases)
