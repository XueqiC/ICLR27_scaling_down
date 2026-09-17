#!/usr/bin/env python3
"""Appendix corner tests and contrasts from unchanged frozen A5/A7 records."""
from __future__ import annotations

if __package__:
    from .paper_artifacts import ROOT, CAPS, COLORS, Artifacts, frozen_run, pyplot
    from .paper_figure_style import (PALETTE, QA_COLORS, panel_axes, finish_panel,
        row_axis_style, position_row_axes, measure_row_rectangle, legend_strip)
    from .paper_panel_exports import export
else:
    from paper_artifacts import ROOT, CAPS, COLORS, Artifacts, frozen_run, pyplot
    from paper_figure_style import (PALETTE, QA_COLORS, panel_axes, finish_panel,
        row_axis_style, position_row_axes, measure_row_rectangle, legend_strip)
    from paper_panel_exports import export

PANEL_SIZE = (2.7, 1.6)
LEGEND_SIZE = (5.5, .42)
TEST_CAPTION = """Registered corner second differences, shown separately for QA (a)
and Math/Code (b) to retain their different scales. Each student/readout row shows
the measured second difference (capability-coloured circle), its registered
plus/minus twice-noise interval (whiskers), the additive prediction (zero tick),
and the frozen F_int prediction (hollow black diamond). The shaded reference band
is plus/minus one registered noise about zero; whiskers are not confidence intervals.
Both linear x axes include every prediction, measurement and full registered interval.
Dashed whiskers denote 1B; solid whiskers denote the 4B development student.
Primary QA additivity: failed to reject. This verdict applies only to the primary
QA test; Math is size-dependent and Code unresolved. All numbers are the stored
A7 predictions and matching A5 measurements and noise intervals; no fit is rerun.
Each panel is 2.7 x 1.6 inches; corner_legend.pdf supplies the shared key above.
Lines are 1.1 pt and markers 3.8 pt, with 2-pt whisker caps.
"""
CONTRAST_CAPTION = """Signed prediction minus measurement for corner second differences
(1B and 4B development) and registered additive corner contrasts on fresh 2Wiki,
MuSiQue and TriviaQA distributions. Hollow capability-coloured diamonds mark
frozen F_int residuals for the student rows. The green ramp identifies the fresh
QA distributions below the separator, whose registered additive predictions are
zero. Dashed whiskers denote 1B and solid whiskers 4B development. Whiskers reflect
the registered plus/minus twice-noise measurement bands about the frozen prediction;
they are not confidence intervals. The symmetric-log axis retains the original
-2.5 to 2.5 native-token-nat range. No frozen response-law predictions on fresh
evaluation distributions are available. Primary QA additivity: failed to reject;
Math is size-dependent and Code unresolved. Fresh secondary readouts are shown
individually without a shared verdict. All twelve records are unchanged from the
former Figure 6(c). The 2.7 x 1.6-inch panel uses corner_legend.pdf above.
"""


def contrast_rows(audit):
    if __package__:
        from .plot_fig_generalization_cells import build
    else:
        from plot_fig_generalization_cells import build
    return [{"kind": "corner", **r} for r in build(audit) if r["measurement_interval"] is not None]


def test_rows(rows, panel):
    return [r for r in rows if (r["capability"] == "qa") == (panel == "a")]


def test_limits(rows):
    values = [v for r in rows for v in (r["measured"], *r["interval"],
                                       r["additive"], r["F_int"], -r["noise"], r["noise"])]
    lo, hi = min(values), max(values)
    pad = .08 * (hi-lo)
    return lo-pad, hi+pad


def draw_test(fig, rows, panel, rectangle=None):
    from matplotlib.ticker import MaxNLocator
    rows = test_rows(rows, panel)
    ax = panel_axes(fig, PANEL_SIZE, left=.72, bottom=.34, right=.08)
    for y, r in enumerate(rows):
        color = COLORS[r["capability"]]
        ax.fill_betweenx([y-.36, y+.36], -r["noise"], r["noise"],
                         facecolor=PALETTE["background"], edgecolor=PALETTE["grid"], linewidth=.6, zorder=0)
        ax.plot(r["additive"], y, "|", color=PALETTE["reference"], ms=3.8, mew=.6)
        ax.plot(r["F_int"], y, "D", color=PALETTE["black"], mfc=PALETTE["transparent"], ms=3.8)
        lo, hi = r["interval"]
        bars = ax.errorbar(r["measured"], y, xerr=[[r["measured"]-lo], [hi-r["measured"]]],
                          fmt="o", color=color, ms=3.8, lw=1.1, elinewidth=1.1, capthick=.6, capsize=2)
        for bar in bars.lines[2]:
            bar.set_linestyle("-" if r["student"] == "gemma3-4b" else "--")
    row_axis_style(ax)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
    ax.set(yticks=range(len(rows)), yticklabels=[r["student"].replace("gemma3-", "").upper()+" "+
           ("QA" if r["capability"] == "qa" else r["capability"].title()) for r in rows],
           ylim=(len(rows)-.5, -.5), xlim=test_limits(rows), xlabel="Second difference (nats)")
    finish_panel(ax)
    return position_row_axes(ax, rectangle) if rectangle is not None else ax


def draw_corners(ax, rows):
    from matplotlib.ticker import NullFormatter
    rows = sorted(rows, key=lambda r: (r["panel"] == "C", r["panel"]))
    groups = list(dict.fromkeys(r["group"] for r in rows))
    for y, group in enumerate(groups):
        part = [r for r in rows if r["group"] == group]
        for k, r in enumerate(part):
            jitter = 0
            color = QA_COLORS.get(r["group"], COLORS[r["capability"]])
            dev = r["status"] == "development"
            lo, hi = r["residual_interval"]
            e = r["residual"]
            whisker = ax.errorbar(e, y+jitter, xerr=[[e-lo], [hi-e]], fmt="none", color=color, capsize=2, lw=1.1)
            for segment in whisker.lines[2]:
                segment.set_linestyle("-" if dev else "--")
            ax.plot(e, y+jitter, marker="D", ls="", color=color,
                    mfc=PALETTE["transparent"], alpha=.85)
    ax.axvline(0, color=PALETTE["reference"], lw=1.1)
    ax.axhline(1.5, color=PALETTE["grid"], lw=.6, zorder=0)
    ax.set_xscale("symlog", linthresh=.03)
    ax.set_xlim(-2.5, 2.5)
    ax.set_xticks([-1, 0, 1], labels=["−1", "0", "1"])
    ax.set_xticks([-.1, .1], minor=True)
    ax.xaxis.set_minor_formatter(NullFormatter())
    labels = {"Distillation: corner budgets (1B)": "1B",
              "4B DEVELOPMENT: corner budgets": "4B development",
              "2wiki_new": "2Wiki", "musique": "MuSiQue", "triviaqa": "TriviaQA"}
    ax.set(yticks=range(len(groups)), yticklabels=[labels[g] for g in groups],
           ylim=(len(groups)-.5, -.5), xlabel="Prediction error (nats)")
    ax.tick_params(axis="y", length=0, pad=3)
    ax.grid(axis="x", alpha=.15)
    ax.spines["left"].set_visible(False)


def draw_contrasts(fig, rows, rectangle=None):
    ax = panel_axes(fig, PANEL_SIZE, left=1.03, bottom=.34, right=.08)
    draw_corners(ax, rows)
    finish_panel(ax)
    return position_row_axes(ax, rectangle) if rectangle is not None else ax


def draw_legend(fig):
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=COLORS[c], label="QA / 2Wiki" if c == "qa" else c.title()) for c in CAPS]
    handles += [Line2D([], [], color=QA_COLORS[c], label=label)
                for c, label in (("musique", "MuSiQue"), ("triviaqa", "TriviaQA"))]
    handles += [Line2D([], [], color=PALETTE["reference"], marker="|", ls="", label="Additive"),
                Line2D([], [], color=PALETTE["black"], marker="D", mfc=PALETTE["transparent"], ls="", label="$F_{int}$ / contrast"),
                Line2D([], [], color=PALETTE["reference"], marker="o", ls="", label="Observed"),
                Line2D([], [], color=PALETTE["reference"], ls="--", label="1B"),
                Line2D([], [], color=PALETTE["reference"], ls="-", label="4B development")]
    return legend_strip(fig, handles)


def export_panels(rows, audit, plt):
    contrasts = contrast_rows(audit)
    rectangle = measure_row_rectangle(plt, PANEL_SIZE,
        [lambda f, p=p: draw_test(f, rows, p) for p in "ab"], kind="double")
    contrast_rectangle = measure_row_rectangle(plt, PANEL_SIZE,
        [lambda f: draw_contrasts(f, contrasts)], kind="double")
    legend = ("corner_legend", LEGEND_SIZE, draw_legend, [], TEST_CAPTION + CONTRAST_CAPTION)
    export(plt, audit, "corner_test", [
        (f"corner_test_{p}", PANEL_SIZE, lambda f, p=p: draw_test(f, rows, p, rectangle),
         test_rows(rows, p), TEST_CAPTION) for p in "ab"], TEST_CAPTION, width=5.5, center=True, legend=legend)
    export(plt, audit, "corner_contrasts", [
        ("corner_contrasts_a", PANEL_SIZE, lambda f: draw_contrasts(f, contrasts, contrast_rectangle),
         contrasts, CONTRAST_CAPTION)], CONTRAST_CAPTION, width=5.5, center=True, legend=legend)
    return contrasts


def generate(root=ROOT):
    if __package__:
        from .plot_fig_explanation import corners
    else:
        from plot_fig_explanation import corners
    with frozen_run(root) as access:
        audit, plt = Artifacts(root), pyplot(root)
        rows = corners(audit)
        contrasts = export_panels(rows, audit, plt)
        return {"tests": rows, "contrasts": contrasts}, audit, access


if __name__ == "__main__":
    generate()
