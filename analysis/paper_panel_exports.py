"""Exports for the additional frozen main-text panels (PNG previews only)."""
from __future__ import annotations

import json

if __package__:
    from .paper_artifacts import CAPS, COLORS, output_path, write_notes
    from .paper_figure_style import apply_style, combine_panels, save_panel, write_caption
else:
    from paper_artifacts import CAPS, COLORS, output_path, write_notes
    from paper_figure_style import apply_style, combine_panels, save_panel, write_caption

LABELS = {"math": "Math", "code": "Code", "qa": "QA"}


def ref(path, *keys):
    """An RFC 6901 JSON pointer, including escaping for paths used as keys."""
    return path + "#/" + "/".join(str(k).replace("~", "~0").replace("/", "~1") for k in keys)


def capability_handles():
    from matplotlib.lines import Line2D
    return [Line2D([], [], color=COLORS[c], label=LABELS[c]) for c in CAPS]


def axes_defaults(ax):
    from matplotlib.ticker import MaxNLocator
    ax.axhline(0, color=".55", lw=1.2, zorder=0)
    ax.grid(axis="y", alpha=.15)
    ax.yaxis.set_major_locator(MaxNLocator(4))


def export(plt, audit, stem, panels, caption, *, columns=None):
    """panels: (filename, size_inches, draw_callable, records, caption) tuples."""
    specs, manifest = [], []
    columns = columns or len(panels)
    gap = .10
    width = max(sum(p[1][0] for p in panels[i:i+columns]) + gap*(len(panels[i:i+columns])-1)
                for i in range(0, len(panels), columns))
    heights = [max(p[1][1] for p in panels[i:i+columns]) for i in range(0, len(panels), columns)]
    height = sum(heights) + gap*(len(heights)-1)
    top = height
    for row, i in enumerate(range(0, len(panels), columns)):
        top -= heights[row]
        x = 0.
        for name, size, draw, records, panel_caption in panels[i:i+columns]:
            apply_style("panel")
            fig = plt.figure(figsize=size)
            draw(fig)
            save_panel(fig, name, "panel", audit, records)
            write_caption(name, audit, panel_caption)
            output_path(audit.root, "figs", f"{name}_data.json").write_text(
                json.dumps({"size_inches": size, "records": records,
                            "input_sha256": audit.inputs}, indent=2, allow_nan=False) + "\n")
            plt.close(fig)
            specs.append(("panel", (x, top, *size), draw))
            manifest.append({"file": f"{name}.pdf", "size_inches": list(size)})
            x += size[0] + gap
        top -= gap
    fig = combine_panels(plt, specs, (width, height))
    path = output_path(audit.root, "figs", f"{stem}.png")
    fig.savefig(path, dpi=220)
    plt.close(fig)
    manifest.append({"file": path.name, "size_inches": [width, height], "dpi": 220})
    write_caption(stem, audit, caption)
    write_notes(stem, audit, [r for p in panels for r in p[3]])
    output_path(audit.root, "figs", f"{stem}_files.json").write_text(
        json.dumps(manifest, indent=2) + "\n")
    print(f"{path.name}: {width:g} x {height:g} in at 220 dpi")
    return manifest
