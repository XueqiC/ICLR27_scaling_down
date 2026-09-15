#!/usr/bin/env python3
"""Matched-budget observed responses from the canonical, frozen A1 table."""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

from collections import defaultdict

if __package__:
    from .paper_artifacts import ROOT, CAPS, COLORS, Artifacts, frozen_run, pyplot, development_rows, save_figure, write_notes
else:
    from paper_artifacts import ROOT, CAPS, COLORS, Artifacts, frozen_run, pyplot, development_rows, save_figure, write_notes

STUDENTS = ("gemma3-270m", "gemma3-1b", "gemma3-4b")
STYLES = (":", "--", "-")
SCOPES = ("2wiki_new", "musique", "triviaqa")
PANEL_SIZE = (2.7, 2.0)
LEGEND_SIZE = (5.5, .30)
FIGSIZE = (5.5, 2.32)
if __package__:
    from .paper_figure_style import apply_style, panel_axes, legend_row, save_panel, write_caption, combine_panels
else:
    from paper_figure_style import apply_style, panel_axes, legend_row, save_panel, write_caption, combine_panels

CAPTION = """Capability responses (a) and fresh QA distributions (b). Positive = worse:
positive loss change means higher loss than the student's own initial state.
Development endpoints nearest 200k supervised tokens per trajectory; every seed
shown separately. Reuse T/D_U is measured in supervised-token passes, and loss
change in native-token nats. All points, including 4B, are development.
Colours identify Math, Code and QA in (a), and 2Wiki, MuSiQue and TriviaQA in (b).
Dotted, dashed and solid lines denote 270M, 1B and 4B students. Circles and squares
identify the first and second registered pool seeds within each rung; the core
uses seeds 41/42 and the critical rung uses 51/52. No seed averaging is applied.
The shared line/marker key is supplied separately as fig1_legend.pdf.
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
               "Seed markers identify the first/second registered seed within each rung; no seed averaging. "
               "Scope rows in the same CSV originate in v99-scope. All points, including 4B, are development.")
    return result


def draw_panel(fig, rows, panel):
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator
    ax = panel_axes(fig, PANEL_SIZE, left=.48, bottom=.51)
    series = CAPS if panel == "left" else SCOPES
    names = {"math": "Math", "code": "Code", "qa": "QA", "2wiki_new": "2Wiki",
             "musique": "MuSiQue", "triviaqa": "TriviaQA"}
    for c, color in zip(series, COLORS.values()):
        for student, style in zip(STUDENTS, STYLES):
            subset = [r for r in rows if r["panel"] == panel and r["series"] == c and r["student"] == student]
            for parity, marker in ((1, "o"), (0, "s")):
                line = sorted((r for r in subset if r["pool_seed"] % 2 == parity), key=lambda r: r["reuse"])
                ax.plot([r["reuse"] for r in line], [r["delta"] for r in line],
                        linestyle=style, marker=marker, color=color, alpha=.85)
    ax.axhline(0, color=".5", lw=1.2)
    ax.set(xlabel="Reuse $T/D_U$", ylabel="Δ loss (nats)")
    ax.xaxis.set_major_locator(MaxNLocator(3))
    ax.yaxis.set_major_locator(MaxNLocator(4, integer=True))
    ax.grid(alpha=.15)
    ax.legend(handles=[Line2D([], [], color=color, label=names[c])
                       for c, color in zip(series, COLORS.values())], loc="upper left")
    return ax


def draw_legend(fig):
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=".25", ls=style, label=student)
               for student, style in zip(("270M", "1B", "4B"), STYLES)]
    handles += [Line2D([], [], ls="", marker=m, color=".25", label=label)
                for m, label in (("o", "Seed 1"), ("s", "Seed 2"))]
    legend_row(fig, handles)


def plot(rows, plt):
    return combine_panels(plt, [
        ("panel", (0, .32, *PANEL_SIZE), lambda f: draw_panel(f, rows, "left")),
        ("panel", (2.8, .32, *PANEL_SIZE), lambda f: draw_panel(f, rows, "right")),
        ("panel", (0, 0, *LEGEND_SIZE), draw_legend),
    ], FIGSIZE)


def generate(root=ROOT):
    with frozen_run(root) as access:
        audit = Artifacts(root)
        plt = pyplot(root)
        rows = build(audit)
        for letter, panel in zip("ab", ("left", "right")):
            apply_style("panel")
            fig = plt.figure(figsize=PANEL_SIZE)
            draw_panel(fig, rows, panel)
            save_panel(fig, f"fig1_{letter}", "panel", audit, [r for r in rows if r["panel"] == panel])
            plt.close(fig)
        fig = plt.figure(figsize=LEGEND_SIZE)
        draw_legend(fig)
        save_panel(fig, "fig1_legend", "panel", audit, [])
        plt.close(fig)
        fig = plot(rows, plt)
        save_figure(fig, "responses_v2", audit)
        write_notes("responses_v2", audit, rows)
        write_caption("responses_v2", audit, CAPTION)
        plt.close(fig)
        return rows, audit, access


if __name__ == "__main__":
    generate()
