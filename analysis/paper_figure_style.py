"""Final-size serif styling and panel exports for artifact-only paper figures.

Choose typography from each panel's physical width.
Export the complete canvas so PDF placement preserves point sizes.
"""
from __future__ import annotations

if __package__:
    from .paper_artifacts import FONT_ROOTS, output_path, write_notes
else:
    from paper_artifacts import FONT_ROOTS, output_path, write_notes

SERIF = ["Times New Roman", "Nimbus Roman", "TeX Gyre Termes", "DejaVu Serif"]
SIZES = {"panel": (7.5, 8.5, 7.5), "double": (8.5, 9.5, 8.5),
         "full": (9, 10, 9), "legend": (8.5, 8.5, 8.5)}
_fonts_registered = False


def kind_for_width(width):
    """Narrow panels use the small tier; Figure 3a uses the middle tier."""
    return "panel" if width <= 2 else "double" if width <= 3 else "full"


def apply_style(kind):
    """Apply bold serif rcParams; return the exact settings for audit/testing."""
    import matplotlib as mpl
    from matplotlib import font_manager

    global _fonts_registered
    tick, label, legend = SIZES[kind]
    if not _fonts_registered:
        # Avoid fontconfig subprocesses under the frozen-input guard. Old caches
        # may predate the system's Times/Nimbus installation.
        for root in FONT_ROOTS:
            for path in sorted(root.rglob("*")):
                name = path.name.lower().replace("_", "").replace("-", "")
                if path.suffix.lower() in (".ttf", ".otf") and any(
                        family in name for family in ("times", "nimbusroman", "texgyretermes")):
                    font_manager.fontManager.addfont(str(path))
        _fonts_registered = True
    rc = {
        "font.family": "serif", "font.serif": SERIF, "mathtext.fontset": "stix",
        "mathtext.default": "bf", "font.weight": "bold", "font.size": tick,
        "axes.labelweight": "bold", "axes.titleweight": "bold",
        "axes.labelsize": label, "axes.titlesize": label,
        "xtick.labelsize": tick, "ytick.labelsize": tick, "legend.fontsize": legend,
        "axes.linewidth": .6, "lines.linewidth": 1.1, "lines.markersize": 3.8,
        "lines.markeredgewidth": .6, "patch.linewidth": .6,
        "errorbar.capsize": 2,
        "xtick.major.width": .6, "ytick.major.width": .6,
        "xtick.minor.width": .4, "ytick.minor.width": .4,
        "xtick.major.size": 2, "ytick.major.size": 2,
        "xtick.minor.size": 1.2, "ytick.minor.size": 1.2,
        "xtick.major.pad": 2, "ytick.major.pad": 2, "axes.labelpad": 2,
        "axes.spines.top": False, "axes.spines.right": False,
        "grid.linewidth": .4, "legend.frameon": False,
        "legend.borderpad": .2, "legend.labelspacing": .15,
        "legend.handlelength": 1.0, "legend.handletextpad": .35,
        "legend.columnspacing": .65, "legend.borderaxespad": .25,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.bbox": None, "savefig.pad_inches": .02, "savefig.dpi": 220,
    }
    mpl.rcParams.update(rc)
    return rc


def panel_axes(fig, size, *, left=.36, bottom=.30, right=.06, top=.06):
    """Margins in inches; all labels belong to this panel's own canvas."""
    w, h = size
    # add_axes also works identically on SubFigure, without shared subplotpars.
    return fig.add_axes((left/w, bottom/h, (w-left-right)/w, (h-bottom-top)/h))


def finish_panel(ax):
    """Leave headroom for long vertical labels on flat, uncropped canvases."""
    # Set this after set_ylabel, which resets the label's vertical position.
    ax.yaxis.label.set_y(.42)
    return ax


def legend_row(fig, handles, **kwargs):
    """A single compact row on the panel's own canvas, below its x label."""
    options = dict(loc="lower center", ncol=len(handles), bbox_to_anchor=(.5, 0),
                   borderaxespad=0)
    options.update(kwargs)
    return fig.legend(handles=handles, **options)


def legend_strip(fig, handles):
    """Measure generous 8.5-pt entries and wrap to two rows when needed.

    Use the same renderer and physical-width check for Figures and SubFigures.
    Reorder Matplotlib's column-major input so the visible key reads row-wise.
    """
    from math import ceil

    owner = fig
    while not hasattr(owner, "canvas"):
        owner = owner.figure
    for rows in (1, 2):
        columns = ceil(len(handles) / rows)
        ordered = [handles[r*columns+c] for c in range(columns) for r in range(rows)
                   if r*columns+c < len(handles)]
        legend = legend_row(fig, ordered, loc="center", bbox_to_anchor=(.5, .5),
                            ncol=columns, fontsize=8.5, handlelength=1.6,
                            handletextpad=.5, columnspacing=1.4,
                            labelspacing=.5, borderpad=0)
        owner.canvas.draw()
        box = legend.get_window_extent(owner.canvas.get_renderer())
        if box.width <= fig.bbox.width - .08*owner.dpi and box.height <= fig.bbox.height:
            return legend
        legend.remove()
    raise ValueError("Legend needs more canvas space for two readable rows")


def save_panel(fig, stem, kind, audit, records):
    """Save one vector PDF, provenance, and physical typography metadata."""
    import json
    from matplotlib import font_manager
    from matplotlib.legend import Legend

    fig.canvas.draw()
    for ax in fig.axes:
        for text in ax.get_xticklabels(which="both") + ax.get_yticklabels(which="both"):
            text.set_weight("bold")
    path = output_path(audit.root, "figs", f"{stem}.pdf")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, metadata={"Creator": "frozen paper generator"})
    tick, label, legend = SIZES[kind]
    w, h = map(float, fig.get_size_inches())
    font = font_manager.FontProperties(family=SERIF, weight="bold")
    selected = font_manager.FontProperties(fname=font_manager.findfont(font)).get_name()
    audit.rule(f"{stem}.pdf: {w:g} x {h:g} in; bold {selected}; ticks {tick} pt, "
               f"axis labels {label} pt, legend {legend} pt; default lines 1.1 pt, markers 3.8 pt; caps 2 pt. "
               "Full canvas retained; no titles, panel letters, or explanatory annotations.")
    write_notes(stem, audit, records)
    output_path(audit.root, "figs", f"{stem}_data.json").write_text(
        json.dumps({"size_inches": [w, h], "records": records,
                    "input_sha256": audit.inputs,
                    "axes": [{"xlim": list(ax.get_xlim()), "ylim": list(ax.get_ylim()),
                              "xscale": ax.get_xscale(), "yscale": ax.get_yscale(),
                              "row_labels": [t.get_text() for t in ax.get_yticklabels()]}
                             for ax in fig.axes],
                    "legend_entries": [t.get_text() for leg in fig.findobj(Legend)
                                       for t in leg.get_texts()],
                    "legend_font_sizes_pt": [t.get_fontsize() for leg in fig.findobj(Legend)
                                             for t in leg.get_texts()],
                    "font": {"family": selected, "weight": "bold", "ticks_pt": tick,
                             "labels_pt": label, "legend_pt": legend},
                    "style_defaults": {"line_width_pt": 1.1, "marker_size_pt": 3.8,
                              "errorbar_capsize_pt": 2},
                    "axis_font_sizes_pt": [
                        {"x_ticks": [t.get_fontsize() for t in ax.get_xticklabels()],
                         "y_ticks": [t.get_fontsize() for t in ax.get_yticklabels()],
                         "x_label": ax.xaxis.label.get_fontsize(),
                         "y_label": ax.yaxis.label.get_fontsize()} for ax in fig.axes]},
                   indent=2, allow_nan=False) + "\n")
    print(f"{stem}: {w:g} x {h:g} in; {selected} bold; "
          f"ticks/labels/legend = {tick}/{label}/{legend} pt; default lines/markers/caps = 1.1/3.8/2 pt")


def write_caption(stem, audit, text):
    output_path(audit.root, "figs", f"{stem}_caption.txt").write_text(text.strip() + "\n")


def combine_panels(plt, panels, size):
    """Recreate vector panels at their exact inch sizes on a combined canvas.

    panels contains (kind, (left, bottom, width, height), draw) tuples. Subfigures
    retain their own axis labels and legends; no rasterization or rescaling.
    """
    fig = plt.figure(figsize=size)
    xs = sorted({round(v, 8) for _, (x, _, w, _), _ in panels for v in (x, x+w)})
    ys = sorted({round(v, 8) for _, (_, y, _, h), _ in panels for v in (y, y+h)}, reverse=True)
    grid = fig.add_gridspec(len(ys)-1, len(xs)-1, left=0, right=1, bottom=0, top=1,
                           wspace=0, hspace=0,
                           width_ratios=[b-a for a, b in zip(xs, xs[1:])],
                           height_ratios=[a-b for a, b in zip(ys, ys[1:])])
    for kind, (x, y, w, h), draw in panels:
        apply_style(kind)
        sub = fig.add_subfigure(grid[ys.index(round(y+h, 8)):ys.index(round(y, 8)),
                                    xs.index(round(x, 8)):xs.index(round(x+w, 8))])
        draw(sub)
    return fig
