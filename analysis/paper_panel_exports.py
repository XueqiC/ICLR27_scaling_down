"""Exports for the additional frozen main-text panels (PNG previews only)."""
from __future__ import annotations

if __package__:
    from .paper_figure_style import PALETTE, CAPABILITY_COLORS, QA_COLORS, METHOD_COLORS as SEMANTIC_METHOD_COLORS, BIT_COLORS, HATCHES, darker, method_ramp
else:
    from paper_figure_style import PALETTE, CAPABILITY_COLORS, QA_COLORS, METHOD_COLORS as SEMANTIC_METHOD_COLORS, BIT_COLORS, HATCHES, darker, method_ramp

import json

if __package__:
    from .paper_artifacts import CAPS, COLORS, output_path, write_notes, save_canvas
    from .paper_figure_style import apply_style, combine_panels, save_panel, write_caption, kind_for_width, prepare_figure
else:
    from paper_artifacts import CAPS, COLORS, output_path, write_notes, save_canvas
    from paper_figure_style import apply_style, combine_panels, save_panel, write_caption, kind_for_width, prepare_figure

LABELS = {"math": "Math", "code": "Code", "qa": "QA"}


def ref(path, *keys):
    """An RFC 6901 JSON pointer, including escaping for paths used as keys."""
    return path + "#/" + "/".join(str(k).replace("~", "~0").replace("/", "~1") for k in keys)


def capability_handles():
    from matplotlib.lines import Line2D
    return [Line2D([], [], color=COLORS[c], label=LABELS[c]) for c in CAPS]


def axes_defaults(ax):
    from matplotlib.ticker import MaxNLocator
    ax.axhline(0, color=PALETTE["reference"], lw=1.1, zorder=0)
    ax.grid(axis="y", alpha=.15)
    ax.yaxis.set_major_locator(MaxNLocator(4))


def export(plt, audit, stem, panels, caption, *, columns=None, width=None, legend=None, preserve_legend=False, center=False):
    """panels: (filename, size_inches, draw_callable, records, caption) tuples."""
    specs, manifest = [], []
    columns = columns or len(panels)
    gap = .10
    if width is not None and columns > 1:
        gap = (width - sum(p[1][0] for p in panels[:columns])) / (columns-1)
        if gap < -1e-10:
            raise ValueError("Panels exceed the requested row width")
    width = width or max(sum(p[1][0] for p in panels[i:i+columns]) + gap*(len(panels[i:i+columns])-1)
                        for i in range(0, len(panels), columns))
    heights = [max(p[1][1] for p in panels[i:i+columns]) for i in range(0, len(panels), columns)]
    height = sum(heights) + gap*(len(heights)-1)
    top = height
    if legend is not None:
        height += legend[1][1]
    for row, i in enumerate(range(0, len(panels), columns)):
        top -= heights[row]
        row_panels = panels[i:i+columns]
        row_width = sum(p[1][0] for p in row_panels) + gap*(len(row_panels)-1)
        x = (width-row_width)/2 if center else 0.
        for name, size, draw, records, panel_caption in panels[i:i+columns]:
            kind = kind_for_width(size[0])
            apply_style(kind)
            fig = plt.figure(figsize=size)
            draw(fig)
            save_panel(fig, name, kind, audit, records)
            write_caption(name, audit, panel_caption)
            plt.close(fig)
            specs.append((kind, (x, top, *size), draw))
            manifest.append({"file": f"{name}.pdf", "size_inches": list(size)})
            x += size[0] + gap
        top -= gap
    if legend is not None:
        name, size, draw, records, panel_caption = legend
        apply_style("legend")
        fig = plt.figure(figsize=size)
        draw(fig)
        save_panel(fig, name, "legend", audit, records, preserve_pdf=preserve_legend)
        write_caption(name, audit, panel_caption)
        plt.close(fig)
        specs.append(("legend", (0, height-size[1], *size), draw))
        manifest.append({"file": f"{name}.pdf", "size_inches": list(size)})
    fig = combine_panels(plt, specs, (width, height))
    path = output_path(audit.root, "figs", f"{stem}.png")
    prepare_figure(fig)
    save_canvas(fig, path, dpi=220)
    save_canvas(fig, path.with_suffix(".pdf"), metadata={"Creator": "frozen paper generator"})
    plt.close(fig)
    manifest.append({"file": path.name, "size_inches": [width, height], "dpi": 220})
    write_caption(stem, audit, caption)
    write_notes(stem, audit, [r for p in panels for r in p[3]])
    output_path(audit.root, "figs", f"{stem}_files.json").write_text(
        json.dumps(manifest, indent=2) + "\n")
    print(f"{path.name}: {width:g} x {height:g} in at 220 dpi")
    return manifest
