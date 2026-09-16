"""Final-size serif styling and panel exports for artifact-only paper figures.

Choose typography from each panel's physical width.
Export the complete canvas so PDF placement preserves point sizes.
"""
from __future__ import annotations

from pathlib import Path

# The only colour literals in paper generators live here. Semantic aliases
# below deliberately share a hue; data values never select arbitrary colours.
PALETTE = {
    "math": "#2b6cb0", "code": "#d97706", "qa": "#2f855a",
    "musique": "#68b36b", "triviaqa": "#a8d5a2",
    "pruning": "#6b46c1", "quantization": "#0e7490",
    "per_channel": "#9a6b2f", "distillation": "#b83280", "dense": "#9ca3af",
    "teal_dark": "#155e75", "teal_light": "#67c3d9",
    "purple_dark": "#443075", "purple_light": "#b49add",
    "reference": "#4b5563", "black": "#000000", "white": "#ffffff",
    "background": "#f3f4f6", "grid": "#d1d5db", "transparent": "none",
}
CAPABILITY_COLORS = {c: PALETTE[c] for c in ("math", "code", "qa")}
QA_COLORS = {"2wiki_new": PALETTE["qa"], "musique": PALETTE["musique"],
             "triviaqa": PALETTE["triviaqa"]}
METHOD_COLORS = {k: PALETTE[v] for k, v in {
    "prune": "pruning", "pruning": "pruning", "quant": "quantization",
    "quantization": "quantization", "grouped_rtn": "quantization",
    "per_channel_rtn": "per_channel", "distill": "distillation",
    "distillation": "distillation", "dense": "dense"}.items()}
BIT_COLORS = dict(zip((3, 4, 5), (PALETTE[k] for k in ("teal_dark", "quantization", "teal_light"))))
STUDENT_STYLES = {"270M": ":", "1B": "--", "4B": "-"}
SEED_MARKERS = ("o", "s")
HATCHES = {"math": "///", "code": "...", "qa": "xx",
           "prediction": "///", "oracle_differs": "xx", "ambiguous": "///",
           "infeasible": "xx"}
PANEL_BANDS = {"three": (1.8, 1.35), "two": (2.7, 1.45), "full": (5.5, 1.5)}
FONT_ROOTS = (Path("/usr/share/fonts"), Path("/usr/local/share/fonts"))


def darker(color):
    """Hatch ink stays in the fill's hue, with 55% of its RGB intensity."""
    from matplotlib.colors import to_rgb
    return tuple(.55*v for v in to_rgb(color))


def neutral_cmap():
    """Continuous diagnostics outside the capability/method colour vocabulary."""
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("neutral", [PALETTE["background"], PALETTE["reference"]])


def method_ramp(method, count):
    """Ordered knob levels, dark to light, within the method hue."""
    from matplotlib.colors import LinearSegmentedColormap
    keys = ("purple_dark", "pruning", "purple_light") if method in ("prune", "pruning") else ("teal_dark", "quantization", "teal_light")
    ramp = LinearSegmentedColormap.from_list(method, [PALETTE[k] for k in keys])
    return [ramp(i/max(1, count-1)) for i in range(count)]

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
        "grid.linewidth": .4, "grid.color": PALETTE["grid"],
        "axes.prop_cycle": mpl.cycler(color=list(CAPABILITY_COLORS.values())),
        "text.color": PALETTE["black"], "axes.labelcolor": PALETTE["black"],
        "axes.edgecolor": PALETTE["reference"], "hatch.linewidth": .4, "legend.frameon": False,
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


def compact_number(value, _position=None):
    """One plain-number format for every numeric axis in a harmonised row."""
    from numpy import format_float_positional
    return format_float_positional(value, precision=6, unique=False, fractional=False,
                                   trim="-").replace("-", "−")


def row_axis_style(ax):
    """Use identical grids and ticks, without changing any rc sizes."""
    import matplotlib as mpl
    from matplotlib.ticker import FuncFormatter, NullLocator
    # Categorical minor labels use the same visible tick strokes as major ticks.
    for direction in ("x", "y"):
        ax.tick_params(axis=direction, which="both",
                       width=mpl.rcParams[f"{direction}tick.major.width"],
                       length=mpl.rcParams[f"{direction}tick.major.size"],
                       pad=mpl.rcParams[f"{direction}tick.major.pad"])
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(FuncFormatter(compact_number))
        axis.set_minor_locator(NullLocator())
    ax.grid(visible=False, which="both")
    ax.grid(visible=True, which="both", alpha=.15)


def measure_row_rectangle(plt, size, draws, kind="panel"):
    """Measure row-wide tick/label margins in inches at export resolution.

    Probe every panel, take the widest y tick and tallest x tick, and reserve
    the largest axis-label thickness. Return one explicit figure-fraction box;
    never use an automatic layout engine or measure panels independently.
    """
    from math import ceil
    import matplotlib as mpl

    apply_style(kind)
    widths, heights, ylabels, xlabels = [], [], [], []
    for draw in draws:
        fig = plt.figure(figsize=size, dpi=220)
        try:
            ax = draw(fig)
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            widths.extend(t.get_window_extent(renderer).width/fig.dpi
                          for t in ax.get_yticklabels(which="both") if t.get_text())
            heights.extend(t.get_window_extent(renderer).height/fig.dpi
                           for t in ax.get_xticklabels(which="both") if t.get_text())
            if ax.get_ylabel():
                ylabels.append(ax.yaxis.label.get_window_extent(renderer).width/fig.dpi)
            if ax.get_xlabel():
                xlabels.append(ax.xaxis.label.get_window_extent(renderer).height/fig.dpi)
        finally:
            plt.close(fig)
    pad = .025
    left = pad + max(widths, default=0) + max(ylabels, default=0)
    left += (mpl.rcParams["ytick.major.size"] + mpl.rcParams["ytick.major.pad"]
             + mpl.rcParams["axes.labelpad"])/72
    bottom = pad + max(heights, default=0) + max(xlabels, default=0)
    bottom += (mpl.rcParams["xtick.major.size"] + mpl.rcParams["xtick.major.pad"]
               + mpl.rcParams["axes.labelpad"])/72
    # Round upwards to 0.01 inch to leave room for PDF/Agg metric differences.
    left, bottom = (ceil(v*100)/100 for v in (left, bottom))
    width, height = size
    return (left/width, bottom/height, (width-left-.08)/width,
            (height-bottom-.06)/height)


def position_row_axes(ax, rectangle):
    """Apply the measured box; centre long x labels on the panel canvas."""
    ax.set_position(rectangle)
    fig = ax.get_figure()
    transform = fig.transSubfigure if hasattr(fig, "transSubfigure") else fig.transFigure
    ax.xaxis.set_label_coords(.5, .025/(fig.bbox.height/fig.dpi), transform=transform)
    ax.xaxis.label.set_verticalalignment("bottom")
    return ax


def finish_panel(ax):
    """Leave headroom for long vertical labels on flat, uncropped canvases."""
    # Set this after set_ylabel, which resets the label's vertical position.
    ax.yaxis.label.set_y(.42)
    return ax


def legend_row(fig, handles, **kwargs):
    """Anchor a compact key at the bottom of its own canvas."""
    options = dict(loc="lower center", ncol=len(handles), bbox_to_anchor=(.5, 0),
                   borderaxespad=0)
    options.update(kwargs)
    return fig.legend(handles=handles, **options)


def legend_strip(fig, handles):
    """Measure generous 8.5-pt entries and wrap to two rows when needed.

    Use the same renderer and physical-width check for Figures and SubFigures.
    Reorder Matplotlib's column-major input so the visible key reads row-wise.
    The strip sits above the panels: remove bottom padding and leave spare
    canvas height above the entries. Reserve at least .02 inches at the top.
    """
    from math import ceil
    from matplotlib.lines import Line2D

    # Proxies may be created before apply_style (e.g. a standalone appendix
    # command). Their sizes must not depend on the previous plot's rcParams.
    for handle in handles:
        if isinstance(handle, Line2D):
            handle.set_linewidth(1.1)
            handle.set_markersize(3.8)
            handle.set_markeredgewidth(.6)
        elif hasattr(handle, "set_linewidth"):
            handle.set_linewidth(.6)

    owner = fig
    while not hasattr(owner, "canvas"):
        owner = owner.figure
    for rows in (1, 2):
        columns = ceil(len(handles) / rows)
        ordered = [handles[r*columns+c] for c in range(columns) for r in range(rows)
                   if r*columns+c < len(handles)]
        legend = legend_row(fig, ordered,
                            ncol=columns, fontsize=8.5, handlelength=1.6,
                            handletextpad=.5, columnspacing=1.4,
                            labelspacing=.5, borderpad=0)
        owner.canvas.draw()
        box = legend.get_window_extent(owner.canvas.get_renderer())
        if box.width <= fig.bbox.width - .08*owner.dpi and box.height <= fig.bbox.height - .02*owner.dpi:
            return legend
        legend.remove()
    raise ValueError("Legend needs more canvas space for two readable rows")


def artist_sizes(fig):
    """Record rendered overrides, rather than reporting defaults as data sizes."""
    from matplotlib.container import ErrorbarContainer
    from matplotlib.legend import Legend
    from matplotlib.lines import Line2D

    lines = [line for ax in fig.axes for line in ax.lines]
    lines += [line for legend in fig.findobj(Legend) for line in legend.legend_handles
              if isinstance(line, Line2D)]
    errorbars = [c for ax in fig.axes for c in ax.containers if isinstance(c, ErrorbarContainer)]
    return {
        "line_widths_pt": sorted({line.get_linewidth() for line in lines
                                  if line.get_linestyle() not in ("", "None", "none")}),
        "marker_sizes_pt": sorted({line.get_markersize() for line in lines
                                   if line.get_marker() not in ("", "None", "none", None)}),
        "errorbar_widths_pt": sorted({float(w) for c in errorbars for bars in c.lines[2]
                                      for w in bars.get_linewidths()}),
        "cap_sizes_pt": sorted({cap.get_markersize()/2 for c in errorbars for cap in c.lines[1]}),
        "cap_widths_pt": sorted({cap.get_markeredgewidth() for c in errorbars for cap in c.lines[1]}),
    }


MAX_MARKER_DISPLACEMENT = .015
MAX_CLUSTER_SPREAD = .06
DODGE_CAPTION = ("Marker displacement is at most 1.5% of each axis range (also on "
                 "the displayed scale for nonlinear axes); cluster spread is at most 6%. "
                 "When bounded dodging cannot retain partial visibility, the cluster "
                 "stays at its true coordinates with thin white edges and smaller "
                 "glyphs above larger ones. Coincident neighbours and any hidden "
                 "markers are documented in the sidecar.")


def _update_marker_position(point, dx):
    """Apply a display offset and record both visual and native-coordinate error."""
    import numpy as np
    from matplotlib.transforms import ScaledTranslation
    fig, ax = point.figure, point.axes
    record = point._marker_record
    point.set_transform(ax.transData + ScaledTranslation(dx/72, 0, fig.dpi_scale_trans))
    true = np.array([record["x"], record["y"]])
    center = point.get_transform().transform([true])[0]
    drawn = ax.transData.inverted().transform(center)
    spans = np.abs([np.diff(ax.get_xlim())[0], np.diff(ax.get_ylim())[0]])
    record.update(dx_pt=float(dx), dy_pt=0., center_pt=(center*72/fig.dpi).tolist(),
                  drawn_coordinate=drawn.tolist(),
                  axis_displacement_fraction=[abs(dx)/record["axis_range_pt"][0], 0.],
                  data_displacement_fraction=(np.abs(drawn-true)/spans).tolist())
    connector = getattr(point, "_dodge_connector", None)
    if connector is not None:
        connector.set_visible(bool(dx))


def prepare_figure(fig):
    """Bound all display dodging; keep crowded clusters at their true positions.

    Both physical axis span and native coordinate range enforce the 1.5% cap,
    including log/symlog axes. No categorical pre-offsets belong in generators.
    Source lines and intervals are unchanged. A cluster that cannot fit falls
    back as a whole, rather than pushing later markers progressively farther.
    """
    import math
    import numpy as np
    from matplotlib.lines import Line2D
    from matplotlib.markers import MarkerStyle
    import matplotlib.patheffects as effects
    if getattr(fig, "_palette_prepared", False):
        return fig._marker_audit
    fig.canvas.draw()
    report, all_points = [], []
    for axis_index, ax in enumerate(fig.axes):
        pending = []
        span_pt = ax.bbox.size*72/fig.dpi
        cap = MAX_MARKER_DISPLACEMENT*span_pt[0]
        for line in list(ax.lines):
            marker = line.get_marker()
            if line.get_linestyle() not in ("", "None", "none"):
                line.set_linewidth(1.1)
            if marker in ("", "None", "none", None, "|", "_"):
                continue
            xy = np.asarray(line.get_xydata())
            if len(xy) == 0:
                continue
            display = line.get_transform().transform(xy)*72/fig.dpi
            glyph = MarkerStyle(marker)
            path = glyph.get_path().transformed(glyph.get_transform())
            size = 3.8  # Existing palette-pass size; already below the 4-pt allowance.
            bounds = path.get_extents().size*size + .6
            polygons = path.to_polygons()
            area = sum(abs(np.dot(v[:, 0], np.roll(v[:, 1], 1)) -
                           np.dot(v[:, 1], np.roll(v[:, 0], 1)))/2 for v in polygons)*size**2
            for (x, y), (px, py) in zip(xy, display):
                if math.isfinite(px) and math.isfinite(py):
                    pending.append(dict(line=line, x=x, y=y, px=px, py=py,
                                        bounds=bounds, area=area, marker=marker))
            line.set_marker("")
        pending.sort(key=lambda p: (p["px"], p["py"]))
        # Components include any neighbours that could touch under bounded
        # offsets, preventing a reset from disturbing a separate cluster.
        neighbours = [[] for _ in pending]
        true_neighbours = [[] for _ in pending]
        for i, p in enumerate(pending):
            for j in range(i):
                q = pending[j]
                width, height = (p["bounds"]+q["bounds"])/2
                if abs(p["py"]-q["py"]) <= height:
                    if abs(p["px"]-q["px"]) <= width+2*cap:
                        neighbours[i].append(j); neighbours[j].append(i)
                    if abs(p["px"]-q["px"]) <= width:
                        true_neighbours[i].append(j); true_neighbours[j].append(i)
        offsets, groups, fallback = {}, {}, set()
        remaining = set(range(len(pending)))
        while remaining:
            first = min(remaining); component = {first}; queue = [first]
            while queue:
                for j in neighbours[queue.pop()]:
                    if j not in component:
                        component.add(j); queue.append(j)
            remaining -= component
            placed = []
            candidates = [0.] + [sign*k*.55 for k in range(1, int(cap/.55)+1) for sign in (-1, 1)]
            for i in sorted(component):
                p = pending[i]
                for dx in candidates:
                    drawn = ax.transData.inverted().transform(
                        [(p["px"]+dx)*fig.dpi/72, p["py"]*fig.dpi/72])
                    native_fraction = abs(drawn[0]-p["x"])/abs(np.diff(ax.get_xlim())[0])
                    if native_fraction > MAX_MARKER_DISPLACEMENT+1e-12:
                        continue
                    if all((p["px"]+dx-qx)**2+(p["py"]-qy)**2 >= 1.1**2-.001
                           for qx, qy in placed):
                        offsets[i] = dx; placed.append((p["px"]+dx, p["py"])); break
                else:
                    fallback.update(component)
                    break
            for i in component:
                groups[i] = f"{axis_index}:{first}"
                if i in fallback:
                    offsets[i] = 0.
        base_id = len(report)
        for index, p in enumerate(pending):
            line, x, y, px, py = (p[k] for k in ("line", "x", "y", "px", "py"))
            dx = offsets[index]
            point = Line2D([x], [y], linestyle="", marker=p["marker"])
            color, face = line.get_color(), line.get_markerfacecolor()
            point.set_color(color); point.set_markerfacecolor(face)
            point.set_markeredgecolor(color if face in ("none", PALETTE["white"]) else PALETTE["white"])
            point.set_markeredgewidth(.6); point.set_markersize(3.8); point.set_alpha(1)
            if point.get_marker() in ("x", "+"):
                point.set_markeredgecolor(color)
                point.set_path_effects([effects.Stroke(linewidth=1.0, foreground=PALETTE["white"]), effects.Normal()])
            point.set_clip_on(False)
            ax.add_line(point); point._palette_point = True
            point._glyph_area = p["area"]
            record = dict(id=base_id+index, axis=axis_index, x=float(x), y=float(y),
                          true_center_pt=[px, py], axis_range_pt=span_pt.tolist(),
                          marker=point.get_marker(), marker_size_pt=3.8, cluster=groups[index],
                          fallback_to_true=index in fallback,
                          coincident_neighbours=[base_id+j for j in true_neighbours[index]])
            point._marker_record = record
            _update_marker_position(point, dx)
            if dx:
                connector = Line2D([px/72, (px+dx)/72], [py/72, py/72],
                                   color=color, linewidth=.4, alpha=.45, zorder=10,
                                   transform=fig.dpi_scale_trans)
                ax.add_line(connector); point._dodge_connector = connector
            report.append(record); all_points.append(point)
    # Preserve sizes; the smaller glyph footprint draws above larger glyphs.
    for index, point in enumerate(sorted(all_points, key=lambda p: (-p._glyph_area, p._marker_record["id"]))):
        point.set_zorder(20+index*.0001)
        point._marker_record["zorder"] = point.get_zorder()
    fig._palette_prepared = True
    fig._marker_audit = report
    fig._marker_pixel_audit = verify_marker_pixels(fig)
    return report


def verify_marker_pixels(fig):
    """Accept visible markers OR bounded, documented coincident neighbours.

    Use the exported 220-dpi Agg glyph stack. If bounded offsets still hide a
    marker, restore its entire cluster to true coordinates and accept overlap.
    No repeated painter-order repair and no extra displacement are permitted.
    Counts are keyed by stable sidecar point IDs, independently of z-order.
    """
    import numpy as np
    from matplotlib.backends.backend_agg import RendererAgg
    fig.set_dpi(220)
    points = sorted((line for ax in fig.axes for line in ax.lines if getattr(line, "_palette_point", False)),
                    key=lambda line: line.get_zorder())
    while True:
        fig.canvas.draw()
        width, height = map(int, fig.bbox.size)
        renderer = RendererAgg(width, height, fig.dpi)
        covered = np.zeros((height, width), dtype=float)
        counts = {}
        for point in reversed(points):
            renderer.clear(); point.draw(renderer)
            rgba = np.asarray(renderer.buffer_rgba())
            alpha = rgba[:, :, 3]/255.
            ink = (rgba[:, :, :3].min(axis=2) < 245) & (alpha > .2)
            counts[point._marker_record["id"]] = int(np.count_nonzero(ink & ((1-covered)*alpha > .1)))
            covered = alpha + covered*(1-alpha)
        hidden_clusters = {p._marker_record["cluster"] for p in points if counts[p._marker_record["id"]] == 0}
        reset = {p._marker_record["cluster"] for p in points
                 if p._marker_record["cluster"] in hidden_clusters and p._marker_record["dx_pt"] != 0}
        if not reset:
            break
        for point in points:
            if point._marker_record["cluster"] in reset:
                _update_marker_position(point, 0.)
                point._marker_record["fallback_to_true"] = True
    for point in points:
        record = point._marker_record
        if max(*record["axis_displacement_fraction"], *record["data_displacement_fraction"]) > MAX_MARKER_DISPLACEMENT+1e-12:
            raise ValueError(f"Marker exceeds displacement bound: {record['id']}")
        record["visible_ink_pixels"] = counts[record["id"]]
        if counts[record["id"]] == 0 and not record["coincident_neighbours"]:
            raise ValueError(f"Hidden marker lacks coincident neighbours: {record['id']}")
    visible = [counts[i] for i in sorted(counts)]
    hidden = [i for i in sorted(counts) if counts[i] == 0]
    return {"dpi": 220, "count": len(visible), "point_ids": sorted(counts),
            "minimum_visible_ink_pixels": min(visible, default=0), "visible_ink_pixels": visible,
            "fully_hidden": len(hidden), "accepted_coincident_point_ids": hidden,
            "undocumented_hidden": 0}


def save_panel(fig, stem, kind, audit, records, *, preserve_pdf=False):
    """Save the panel and metadata; optionally retain an existing PDF verbatim."""
    import json
    if __package__:
        from .paper_artifacts import output_path, write_notes, save_canvas
    else:
        from paper_artifacts import output_path, write_notes, save_canvas
    from matplotlib import font_manager
    from matplotlib.legend import Legend

    marker_audit = prepare_figure(fig)
    pixel_audit = fig._marker_pixel_audit
    fig.canvas.draw()
    for ax in fig.axes:
        for text in ax.get_xticklabels(which="both") + ax.get_yticklabels(which="both"):
            text.set_weight("bold")
    path = output_path(audit.root, "figs", f"{stem}.pdf")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not (preserve_pdf and path.is_file()):
        save_canvas(fig, path, metadata={"Creator": "frozen paper generator"})
    save_canvas(fig, path.with_suffix(".png"), dpi=220)
    tick, label, legend = SIZES[kind]
    w, h = map(float, fig.get_size_inches())
    font = font_manager.FontProperties(family=SERIF, weight="bold")
    selected = font_manager.FontProperties(fname=font_manager.findfont(font)).get_name()
    rendered = artist_sizes(fig)
    audit.rule(f"{stem}.pdf: {w:g} x {h:g} in; bold {selected}; ticks {tick} pt, "
               f"axis labels {label} pt, legend {legend} pt; default lines 1.1 pt, markers 3.8 pt; caps 2 pt. "
               f"Rendered artist sizes: {rendered}. "
               "Full canvas retained; no titles, panel letters, or explanatory annotations.")
    audit.rule(DODGE_CAPTION)
    write_notes(stem, audit, records)
    output_path(audit.root, "figs", f"{stem}_data.json").write_text(
        json.dumps({"size_inches": [w, h], "records": records,
                    "marker_visibility": {"count": len(marker_audit), "maximum_displacement_fraction": MAX_MARKER_DISPLACEMENT,
                                          "maximum_cluster_spread_fraction": MAX_CLUSTER_SPREAD,
                                          "coincident_definition": "True glyph bounding boxes overlap (including 0.6-pt edges)",
                                          "acceptance": "visible OR bounded with documented coincident neighbours",
                                          "points": marker_audit, "png_pixel_check": pixel_audit},
                    "input_sha256": audit.inputs,
                    "axes": [{"rectangle": list(ax.get_position().bounds),
                              "xticks": [float(t) for t in ax.get_xticks()],
                              "yticks": [float(t) for t in ax.get_yticks()],
                              "x_tick_labels": [t.get_text() for t in ax.get_xticklabels()],
                              "minor_y_tick_labels": [t.get_text() for t in ax.get_yticklabels(minor=True)],
                              "xlim": list(ax.get_xlim()), "ylim": list(ax.get_ylim()),
                              "xscale": ax.get_xscale(), "yscale": ax.get_yscale(),
                              "row_labels": [t.get_text() for t in ax.get_yticklabels(which="both")]}
                             for ax in fig.axes],
                    "legend_entries": [t.get_text() for leg in fig.findobj(Legend)
                                       for t in leg.get_texts()],
                    "legend_font_sizes_pt": [t.get_fontsize() for leg in fig.findobj(Legend)
                                             for t in leg.get_texts()],
                    "font": {"family": selected, "weight": "bold", "ticks_pt": tick,
                             "labels_pt": label, "legend_pt": legend},
                    "style_defaults": {"line_width_pt": 1.1, "marker_size_pt": 3.8,
                              "errorbar_capsize_pt": 2},
                    "artist_sizes": rendered,
                    "legend_layout": {"placement": "above panels", "bottom_padding_inches": 0,
                                      "minimum_top_padding_inches": .02} if fig.legends else None,
                    "axis_font_sizes_pt": [
                        {"x_ticks": [t.get_fontsize() for t in ax.get_xticklabels()],
                         "y_ticks": [t.get_fontsize() for t in ax.get_yticklabels(which="both")],
                         "x_label": ax.xaxis.label.get_fontsize(),
                         "y_label": ax.yaxis.label.get_fontsize()} for ax in fig.axes]},
                   indent=2, allow_nan=False) + "\n")
    for ax in fig.axes:
        print(f"{stem}: axes rectangle (left, bottom, width, height) = {tuple(ax.get_position().bounds)}")
    print(f"{stem}: {w:g} x {h:g} in; {selected} bold; "
          f"ticks/labels/legend = {tick}/{label}/{legend} pt; rendered sizes = {rendered}")


def write_caption(stem, audit, text):
    if __package__:
        from .paper_artifacts import output_path
    else:
        from paper_artifacts import output_path
    output_path(audit.root, "figs", f"{stem}_caption.txt").write_text(text.strip() + "\n\n" + DODGE_CAPTION + "\n")


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
