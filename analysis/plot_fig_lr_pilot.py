#!/usr/bin/env python3
"""Final positive LR-pilot checkpoints; frozen JSON only, CPU only."""
from __future__ import annotations

if __package__:
    from .paper_figure_style import PALETTE, CAPABILITY_COLORS, QA_COLORS, METHOD_COLORS as SEMANTIC_METHOD_COLORS, BIT_COLORS, HATCHES, darker, method_ramp
else:
    from paper_figure_style import PALETTE, CAPABILITY_COLORS, QA_COLORS, METHOD_COLORS as SEMANTIC_METHOD_COLORS, BIT_COLORS, HATCHES, darker, method_ramp

if __package__:
    from .paper_artifacts import ROOT, CAPS, COLORS, Artifacts, frozen_run, pyplot
    from .paper_figure_style import finish_panel, panel_axes, position_row_axes, row_axis_style, measure_row_rectangle, legend_strip
    from .paper_panel_exports import ref, axes_defaults, export
else:
    from paper_artifacts import ROOT, CAPS, COLORS, Artifacts, frozen_run, pyplot
    from paper_figure_style import finish_panel, panel_axes, position_row_axes, row_axis_style, measure_row_rectangle, legend_strip
    from paper_panel_exports import ref, axes_defaults, export

STUDENTS = ("gemma3-1b", "gemma3-4b")
RATES = ("5e-5", "1e-4", "2e-4")
PANEL_SIZE = (2.7, 1.45)
LEGEND_SIZE = (5.5, .42)
CAPTION = """Learning-rate pilot for Gemma-3 1B (dashed) and 4B (solid).
Colour identifies capability, using the Figure 3 palette. The x axis is the
configured peak learning rate encoded in the run name, not the decayed lr field
at the checkpoint. The y axis is the saved delta field at the final positive
update, with positive values indicating higher loss than the initial student.
All six pilots use full teacher data with 75 examples per domain and data seed
11; their final saved update is 8, at 37,903 processed input tokens. These are
development pilots, with one run per student/rate and no uncertainty estimate.
The standalone appendix panel lr_pilot_a.pdf is 2.7 x 1.45 inches.
Lines are 1.1 pt and markers are 3.8 pt. The shared capability and student key
lr_pilot_legend.pdf sits above; the LR ticks give the configured peak rates,
and markers denote the final checkpoints.
"""


def build(audit):
    rows = []
    for student in STUDENTS:
        for rate in RATES:
            run = f"results/v12-distill/{student}/gpt-5.6-luna_full_75_lrpilot_{rate}_lora_dseed11"
            candidates = []
            for path in sorted((audit.root / run / "trajectory").glob("*/eval.json")):
                relative = path.relative_to(audit.root).as_posix()
                data = audit.read(relative)
                if data["updates"] > 0 and data["processed_tokens"] > 0:
                    candidates.append((data["updates"], relative, data))
            if not candidates:
                raise ValueError(f"Missing positive checkpoint: {run}")
            _, path, data = max(candidates, key=lambda v: v[0])
            if data["student"] != student or data["output_suffix"] != f"lrpilot_{rate}":
                raise ValueError(f"Pilot identity mismatch: {path}")
            for cap in CAPS:
                rows.append({"student": student, "capability": cap, "learning_rate": float(rate),
                             "delta": data["delta"][cap], "updates": data["updates"],
                             "processed_tokens": data["processed_tokens"],
                             "delta_source": ref(path, "delta", cap),
                             "learning_rate_source": ref(path, "output_suffix"),
                             "updates_source": ref(path, "updates"),
                             "processed_tokens_source": ref(path, "processed_tokens")})
    audit.rule("Final checkpoint = maximum positive updates with positive processed_tokens. "
               "learning_rate = float(output_suffix after lrpilot_); delta used verbatim.")
    return rows


def draw_panel(fig, rows, rectangle=None):
    ax = panel_axes(fig, PANEL_SIZE, left=.35, bottom=.30, right=.07)
    for student, ls in zip(STUDENTS, ("--", "-")):
        for cap in CAPS:
            part = sorted((r for r in rows if r["student"] == student and r["capability"] == cap),
                          key=lambda r: r["learning_rate"])
            ax.plot([r["learning_rate"] for r in part], [r["delta"] for r in part],
                    color=COLORS[cap], ls=ls, marker="o", lw=1.1, ms=3.8)
    axes_defaults(ax)
    ax.set(xscale="log", xlabel="Peak learning rate", ylabel="Loss change (nats)",
           xlim=(4.3e-5, 2.35e-4), ylim=(-1.45, .18))
    ax.set_xticks([float(r) for r in RATES])
    ax.set_yticks([-1, 0])
    row_axis_style(ax)
    finish_panel(ax)
    return position_row_axes(ax, rectangle) if rectangle is not None else ax


def draw_legend(fig):
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=COLORS[c], label="QA" if c == "qa" else c.title()) for c in CAPS]
    handles += [Line2D([], [], color=PALETTE["reference"], ls=ls, label=name)
                for name, ls in (("1B", "--"), ("4B", "-"))]
    return legend_strip(fig, handles)


def generate(root=ROOT):
    with frozen_run(root) as access:
        audit, plt = Artifacts(root), pyplot(root)
        rows = build(audit)
        rectangle = measure_row_rectangle(plt, PANEL_SIZE, [lambda f: draw_panel(f, rows)], kind="double")
        export(plt, audit, "lr_pilot",
               [("lr_pilot_a", PANEL_SIZE, lambda f: draw_panel(f, rows, rectangle), rows, CAPTION)],
               CAPTION, width=5.5, center=True,
               legend=("lr_pilot_legend", LEGEND_SIZE, draw_legend, [], CAPTION))
        return rows, audit, access


if __name__ == "__main__":
    generate()
