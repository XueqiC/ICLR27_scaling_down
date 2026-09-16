#!/usr/bin/env python3
"""Matched-budget observed responses from the canonical, frozen A1 table."""
from __future__ import annotations

if __package__:
    from .paper_figure_style import PALETTE, CAPABILITY_COLORS, QA_COLORS, METHOD_COLORS as SEMANTIC_METHOD_COLORS, BIT_COLORS, HATCHES, darker, method_ramp
else:
    from paper_figure_style import PALETTE, CAPABILITY_COLORS, QA_COLORS, METHOD_COLORS as SEMANTIC_METHOD_COLORS, BIT_COLORS, HATCHES, darker, method_ramp

import sys
sys.dont_write_bytecode = True

from collections import defaultdict
from statistics import mean

if __package__:
    from .paper_artifacts import ROOT, CAPS, COLORS, Artifacts, frozen_run, pyplot, development_rows, save_figure, write_notes
else:
    from paper_artifacts import ROOT, CAPS, COLORS, Artifacts, frozen_run, pyplot, development_rows, save_figure, write_notes

STUDENTS = ("gemma3-270m", "gemma3-1b", "gemma3-4b")
STYLES = (":", "--", "-")
SCOPES = ("2wiki_new", "musique", "triviaqa")
PANEL_SIZE = (1.35, 1.25)
PANEL_KIND = "narrow"
PANEL_GAP = .03
LEGEND_SIZE = (5.5, .42)
FIGSIZE = (5.5, PANEL_SIZE[1] + LEGEND_SIZE[1])
PANELS = {"a": "math", "b": "code", "c": "qa", "d": "distributions"}
if __package__:
    from .paper_figure_style import apply_style, finish_panel, panel_axes, legend_strip, save_panel, write_caption, combine_panels, measure_row_rectangle, position_row_axes, row_axis_style
else:
    from paper_figure_style import apply_style, finish_panel, panel_axes, legend_strip, save_panel, write_caption, combine_panels, measure_row_rectangle, position_row_axes, row_axis_style

CAPTION = """Capability responses: (a) Math, (b) Code, (c) QA on the registered
2Wiki training probe, and (d) QA on fresh 2Wiki, MuSiQue and TriviaQA distributions.
Positive = worse: positive loss change means higher loss than the student's own
initial state. Development endpoints nearest 200k supervised tokens per trajectory;
reuse E = T/D_U in supervised-token passes, loss change in native-token nats.
All points, including 4B, are development. Each panel has its own loss-change range.
Colours identify capabilities in (a-c), and the green ramp identifies 2Wiki,
MuSiQue and TriviaQA in (d). Dotted, dashed and solid lines denote 270M, 1B and 4B.
Circles and squares identify the first and second registered pool seeds in each
rung (41/42 in the core, 51/52 at the critical rung). One line per student/readout
connects arithmetic mean reuse and loss change over both seeds at each pool-size
rung. Marker-only circles and squares retain each seed's observed coordinates;
all 132 underlying records remain unchanged in the sidecars.
Four 1.35 x 1.25-inch panels form one 5.5-inch row with 0.03-inch gaps,
with fig1_legend.pdf above. Only the first panel carries the shared y label.
The narrowest style band uses 7-pt ticks and 8-pt axis labels; lines are
1.1 pt and seed markers 3.8 pt. The learning-rate pilot is a separate appendix
figure, lr_pilot_a.pdf with lr_pilot_legend.pdf.
"""


def endpoints(rows):
    """One nearest-positive checkpoint per trajectory, capability and distribution.

    T_actual and D_U_pool are the canonical supervised-token accounting; never
    pool distinct seeds, interpolate checkpoints, or use nominal epoch counts.
    """
    groups = defaultdict(list)
    for row in rows:
        if row["T_actual"] > 0:
            groups[row["run_id"], row["capability"], row["distribution"]].append(row)
    return [min(part, key=lambda r: (abs(r["T_actual"] - 200000), r["checkpoint_id"]))
            for _, part in sorted(groups.items())]


def build(audit):
    rows = endpoints(development_rows(audit))
    result = []
    for row in rows:
        distribution = row["distribution"].split(":", 1)[0]
        if distribution == "training_probe":
            panel, series = "left", row["capability"]
        elif distribution in SCOPES:
            panel, series = "right", distribution
        else:
            continue
        if not row["D_U_pool"]:
            raise ValueError("Missing positive canonical pool-token count")
        result.append({"panel": panel, "series": series, "student": row["student_id"],
                       "reuse": row["T_actual"] / row["D_U_pool"], "delta": row["delta"],
                       "pool_seed": row["pool_seed"], "U": row["U"], "T_actual": row["T_actual"],
                       "D_U_pool": row["D_U_pool"], "run_id": row["run_id"],
                       "checkpoint_id": row["checkpoint_id"], "distribution": row["distribution"],
                       "status": "development"})
    audit.rule("Reused a1_development_table.load_development_table (schema, cohort, metadata and hash checks). "
               "A1 CSV fields: run_id, checkpoint_id, student_id, pool_seed, U, T_actual, D_U_pool, "
               "capability, distribution, delta. Select positive T_actual nearest 200000 separately per trajectory/readout. "
               "reuse=T_actual/D_U_pool; y=delta (own-initial loss subtracted by A1). Both pool seeds remain separate points.")
    audit.rule("The four-rung core includes pool seeds 41/42 and, for the critical rung, 51/52. "
               "Seed markers identify the first/second registered seed within each rung; "
               "one line connects arithmetic mean reuse/delta over the two seeds per rung. "
               "All individual seed records are retained unchanged. "
               "Scope rows in the same CSV originate in v99-scope. All points, including 4B, are development.")
    return result


def panel_records(rows, panel):
    if panel == "distributions":
        return [r for r in rows if r["panel"] == "right"]
    return [r for r in rows if r["panel"] == "left" and r["series"] == panel]


def draw_panel(fig, rows, panel, rectangle=None):
    from matplotlib.ticker import MaxNLocator
    ax = panel_axes(fig, PANEL_SIZE, left=.35, bottom=.30, right=.07)
    selected = panel_records(rows, panel)
    series = SCOPES if panel == "distributions" else (panel,)
    for c in series:
        color = QA_COLORS[c] if panel == "distributions" else COLORS[c]
        for student, style in zip(STUDENTS, STYLES):
            subset = [r for r in selected if r["series"] == c and r["student"] == student]
            rungs = defaultdict(list)
            for row in subset:
                rungs[row["U"]].append(row)
            if any(len(part) != 2 or len({r["pool_seed"] for r in part}) != 2
                   for part in rungs.values()):
                raise ValueError("Expected two distinct pool seeds per student/readout/rung")
            points = sorted((mean(r["reuse"] for r in part), mean(r["delta"] for r in part))
                            for part in rungs.values())
            if points:
                ax.plot([x for x, _ in points], [y for _, y in points],
                        linestyle=style, color=color, lw=1.1, alpha=.9)
            for parity, marker in ((1, "o"), (0, "s")):
                line = sorted((r for r in subset if r["pool_seed"] % 2 == parity), key=lambda r: r["reuse"])
                ax.plot([r["reuse"] for r in line], [r["delta"] for r in line],
                        linestyle="none", marker=marker, ms=3.8, color=color, alpha=.85)
    ax.axhline(0, color=PALETTE["reference"], lw=1.1)
    ax.set(xlabel="Reuse ratio E", ylabel="Loss change (nats)" if panel == "math" else "")
    ax.set_xticks([4, 8, 12])
    # Independent ranges retain all seed extrema and the zero reference.
    values = [0.] + [r["delta"] for r in selected]
    lo, hi = min(values), max(values)
    padding = .05*(hi-lo)
    ax.set_ylim(lo-padding, hi+padding)
    ticks = MaxNLocator(nbins=2, min_n_ticks=2).tick_values(*ax.get_ylim())
    ax.set_yticks([v for v in ticks if ax.get_ylim()[0] <= v <= ax.get_ylim()[1]])
    row_axis_style(ax)
    finish_panel(ax)
    return position_row_axes(ax, rectangle) if rectangle is not None else ax


def draw_legend(fig):
    from matplotlib.lines import Line2D
    colours = [Line2D([], [], color=color, lw=1.1, label=name)
               for names, colors in ((("Math", "Code", "QA"), COLORS.values()),
                                     (("2Wiki", "MuSiQue", "TriviaQA"), QA_COLORS.values()))
               for color, name in zip(colors, names)]
    handles = [Line2D([], [], color=PALETTE["reference"], ls=style, lw=1.1, label=student)
               for student, style in zip(("270M", "1B", "4B"), STYLES)]
    handles += [Line2D([], [], ls="", marker=m, ms=3.8, color=PALETTE["reference"], label=label)
                for m, label in (("o", "S1"), ("s", "S2"))]
    return legend_strip(fig, colours + handles)


def row_rectangle(rows, plt):
    return measure_row_rectangle(plt, PANEL_SIZE, [
        lambda f, p=p: draw_panel(f, rows, p) for p in PANELS.values()
    ], kind=PANEL_KIND)


def plot(rows, plt, rectangle=None):
    if rectangle is None:
        rectangle = row_rectangle(rows, plt)
    row_width = len(PANELS)*PANEL_SIZE[0] + (len(PANELS)-1)*PANEL_GAP
    left = (FIGSIZE[0]-row_width)/2
    specs = [(PANEL_KIND, (left + i*(PANEL_SIZE[0]+PANEL_GAP), 0, *PANEL_SIZE),
              lambda f, p=p: draw_panel(f, rows, p, rectangle)) for i, p in enumerate(PANELS.values())]
    specs.append(("legend", (0, PANEL_SIZE[1], *LEGEND_SIZE), draw_legend))
    return combine_panels(plt, specs, FIGSIZE)


def generate(root=ROOT):
    with frozen_run(root) as access:
        audit = Artifacts(root)
        plt = pyplot(root)
        rows = build(audit)
        rectangle = row_rectangle(rows, plt)
        for letter, panel in PANELS.items():
            apply_style(PANEL_KIND)
            fig = plt.figure(figsize=PANEL_SIZE)
            draw_panel(fig, rows, panel, rectangle)
            save_panel(fig, f"fig1_{letter}", PANEL_KIND, audit, panel_records(rows, panel))
            write_caption(f"fig1_{letter}", audit, CAPTION)
            plt.close(fig)
        apply_style("legend")
        fig = plt.figure(figsize=LEGEND_SIZE)
        draw_legend(fig)
        save_panel(fig, "fig1_legend", "legend", audit, [])
        write_caption("fig1_legend", audit, CAPTION)
        plt.close(fig)
        fig = plot(rows, plt, rectangle)
        save_figure(fig, "responses_v2", audit)
        write_notes("responses_v2", audit, rows)
        write_caption("responses_v2", audit, CAPTION)
        plt.close(fig)
        return rows, audit, access


if __name__ == "__main__":
    generate()
