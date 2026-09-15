"""Final-size serif styling and panel exports for artifact-only paper figures.

Keep each panel's original style kind when changing its physical dimensions.
Export the complete canvas so PDF placement preserves point sizes.
"""
from __future__ import annotations

if __package__:
    from .paper_artifacts import FONT_ROOTS, output_path, write_notes
else:
    from paper_artifacts import FONT_ROOTS, output_path, write_notes

SERIF = ["Times New Roman", "Nimbus Roman", "TeX Gyre Termes", "DejaVu Serif"]
SIZES = {"panel": (14, 15, 13), "full": (13, 14, 12)}
_fonts_registered = False


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
        "axes.linewidth": 1.2, "lines.linewidth": 2.0, "lines.markersize": 7,
        "lines.markeredgewidth": 1.2, "patch.linewidth": 1.2,
        "xtick.major.width": 1.2, "ytick.major.width": 1.2,
        "xtick.minor.width": 1.0, "ytick.minor.width": 1.0,
        "xtick.major.pad": 2, "ytick.major.pad": 2, "axes.labelpad": 2,
        "axes.spines.top": False, "axes.spines.right": False,
        "grid.linewidth": .6, "legend.frameon": False,
        "legend.borderpad": .2, "legend.labelspacing": .15,
        "legend.handlelength": 1.0, "legend.handletextpad": .35,
        "legend.columnspacing": .65, "legend.borderaxespad": .25,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.bbox": None, "savefig.pad_inches": .02, "savefig.dpi": 220,
    }
    mpl.rcParams.update(rc)
    return rc


def panel_axes(fig, size, *, left=.57, bottom=.48, right=.06, top=.04):
    """Margins in inches; all labels belong to this panel's own canvas."""
    w, h = size
    # add_axes also works identically on SubFigure, without shared subplotpars.
    return fig.add_axes((left/w, bottom/h, (w-left-right)/w, (h-bottom-top)/h))


def legend_row(fig, handles, **kwargs):
    """A single compact row on the panel's own canvas, below its x label."""
    options = dict(loc="lower center", ncol=len(handles), bbox_to_anchor=(.5, 0),
                   borderaxespad=0)
    options.update(kwargs)
    return fig.legend(handles=handles, **options)


def legend_strip(fig, handles):
    """Fit a single row by tightening handle spacing, never scaling the fonts."""
    return legend_row(fig, handles, loc="center", bbox_to_anchor=(.5, .5),
                      handlelength=.65, handletextpad=.15, columnspacing=.35,
                      borderpad=0)


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
               f"axis labels {label} pt, legend {legend} pt; lines 2 pt, markers 7 pt. "
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
                             "labels_pt": label, "legend_pt": legend}},
                   indent=2, allow_nan=False) + "\n")
    print(f"{stem}: {w:g} x {h:g} in; {selected} bold; "
          f"ticks/labels/legend = {tick}/{label}/{legend} pt; lines/markers = 2/7 pt")


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
