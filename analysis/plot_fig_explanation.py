#!/usr/bin/env python3
"""Curvature, registered corner test, and post-hoc displacement account."""
from __future__ import annotations

if __package__:
    from .paper_figure_style import PALETTE, CAPABILITY_COLORS, QA_COLORS, METHOD_COLORS as SEMANTIC_METHOD_COLORS, BIT_COLORS, HATCHES, darker, method_ramp
else:
    from paper_figure_style import PALETTE, CAPABILITY_COLORS, QA_COLORS, METHOD_COLORS as SEMANTIC_METHOD_COLORS, BIT_COLORS, HATCHES, darker, method_ramp

import sys
sys.dont_write_bytecode = True

from collections import defaultdict
from statistics import median

if __package__:
    from .paper_artifacts import ROOT, CAPS, COLORS, Artifacts, frozen_run, pyplot, save_figure, write_notes
else:
    from paper_artifacts import ROOT, CAPS, COLORS, Artifacts, frozen_run, pyplot, save_figure, write_notes

A2="results/a2-curvature-interaction/summary.json"
A5="results/a5-corner-second-difference/summary.json"
A7="results/a7-closeout-audit/summary.json"
CAPTION="failed to reject"
PANEL_SIZES = {p: (1.8, 1.35) for p in "abc"}
KINDS = {"a": "panel", "b": "panel", "c": "panel"}
LEGEND_SIZE = (5.5, .42)
FIGSIZE = (5.5, 1.77)
if __package__:
    from .paper_figure_style import apply_style, finish_panel, panel_axes, legend_strip, save_panel, write_caption, combine_panels, measure_row_rectangle, position_row_axes, row_axis_style
else:
    from paper_figure_style import apply_style, finish_panel, panel_axes, legend_strip, save_panel, write_caption, combine_panels, measure_row_rectangle, position_row_axes, row_axis_style

CAPTION_TEXT = """Reuse-term curvature (a), registered corner test (b), and post-hoc
displacement account (c). Primary QA additivity: failed to reject. This applies
to the primary QA test only; math is size-dependent and code unresolved.
Curvature intervals are conditional on development. The exponent p is unitless.
Fold estimate circles and Boundary
crosses show the stored development fits; diamonds show the full-development fit
with its conditional interval. Reference lines mark p = 0 and p = 1.
Corner rows show each student/readout separately. The additive prediction is zero;
frozen F_int predictions are diamonds; measured second differences are circles.
Shading is plus/minus one registered noise, and whiskers are the registered
plus/minus twice-noise band, not a confidence interval. The second-difference
axis uses a symmetric-log scale in native-token nats.
Displacement is a post-hoc account, with both axes logarithmic. Each configuration
shows median absolute measured damage and median relative second-order prediction
error across its stored state/capability records. Zero-damage records are excluded.
The three families are pruning, grouped RTN, and per-channel RTN. There is no
distillation displacement measurement; the superseded uncentred run and the
separate cluster hardware run are not pooled.
The three 1.8 x 1.35-inch panels form one row at 5.5-inch text width, using
one axes rectangle measured from the widest y tick and tallest x tick in the row.
Numeric ticks share one compact format and each axis has at most four major ticks.
Panel (a) uses horizontal capability labels without a redundant x-axis title.
Panel (b) retains all six student/readout identities as minor categorical labels;
all panels retain their y-axis titles.
Lines are 1.1 pt and markers 3.8 pt; boundary crosses are 3.8 pt. Fold estimates
and full fits share their true capability coordinate; any display dodge is bounded.
Error bars and whiskers are 1.1 pt with 2 pt caps; (b)'s shaded band edges
and additive bars are 1.1 pt. The shared key fig2_legend.pdf sits above the panels:
Fold estimate, Boundary and Full apply to (a);
Additive, F_int and Observed to (b); Group, Channel and Prune to (c), denoting grouped
RTN, per-channel RTN and pruning. Panel (b) labels use student size/readout,
for example 1B QA. Damage and relative error in (c) are configuration medians.
"""


def curvature(audit):
    data=audit.read(A2); result=[]
    for i,row in enumerate(data["parameter_stability"]):
        # Training probes provide one comparable readout per capability.
        if not row["distribution"].startswith("training_probe:"):
            continue
        j,full=next((j,r) for j,r in enumerate(data["parameter_intervals"]) if r["distribution"]==row["distribution"])
        fit=full["fits"]["F_curv"]
        result.append({"capability":row["capability"],"p":fit["fit"]["p"],"interval":fit["p_interval"],
                       "boundary":fit["fit"]["boundary_hit"],"full_source":f"{A2}#/parameter_intervals/{j}/fits/F_curv",
                       "folds":[{"p":f["p"],"boundary":f["boundary"],"split":f["split"],"held":f["held"],
                                 "descriptor":f["descriptor"],"source":f"{A2}#/parameter_stability/{i}/folds/{k}"}
                                for k,f in enumerate(row["folds"]) if f["name"]=="F_curv"]})
    return sorted(result,key=lambda r:CAPS.index(r["capability"]))


def corners(audit):
    a7=audit.read(A7); a5=audit.read(A5); result=[]
    for i,row in enumerate(a7["checks"]["predictions"]["rows"]):
        measured=a5["students"][row["student"]]["readouts"][row["capability"]]
        if row["measured_I"]!=measured["I"] or row["registered_noise"]!=measured["noise_on_I"]:
            raise ValueError("A5/A7 second-difference or noise mismatch")
        result.append({"student":row["student"],"capability":row["capability"],"readout":row["readout"],
                       "measured":row["measured_I"],"interval":measured["interval"],"noise":row["registered_noise"],
                       "additive":row["additive_rectangle_prediction"],"F_int":row["predictions"]["F_int"]["I"],
                       "source":f"{A7}#/checks/predictions/rows/{i}",
                       "interval_source":f"{A5}#/students/{row['student']}/readouts/{row['capability']}/interval"})
    audit.rule("Corner caption: failed to reject. This wording applies to the primary QA additivity test. "
               "Math is size-dependent and code unresolved; students are never averaged. "
               "A7 check 4 rows supply additive_rectangle_prediction, predictions.F_int.I, measured_I and registered_noise; "
               "A5 supplies interval = measured I +/- twice registered noise, not a confidence interval. "
               "Shading is +/- one registered noise; whiskers are the registered +/- twice-noise band.")
    return result


def displacement(audit):
    findings="results/v88-displacement/FINDINGS.md"
    audit.read(findings)
    # The only JSON summary is the superseded, two-family uncentred run. Read it
    # for provenance, but never mix its measurements into the current run.
    audit.read("results/v88-displacement-uncentred-run/summary.json")
    groups=defaultdict(list)
    for path in sorted((audit.root/"results/v88-displacement").glob("*/*.json")):
        row=audit.read(path)
        if row.get("selftest") or row.get("experiment")!="v88-displacement":
            continue
        config=row["config"]
        for cap,metric in row["per_capability"].items():
            damage=abs(metric["measured_delta"])
            if damage==0:
                continue
            groups[config["kind"],config["id"]].append({"damage":damage,
                "relative_error":abs(metric["second_order_prediction"]-metric["measured_delta"])/damage,
                "source":f"{path.relative_to(audit.root).as_posix()}#/per_capability/{cap}","device":row["device_name"]})
    result=[{"family":family,"config":config,"damage":median(r["damage"] for r in rr),
             "median_relative_error":median(r["relative_error"] for r in rr),"n":len(rr),"records":rr}
            for (family,config),rr in sorted(groups.items())]
    if len({r["family"] for r in result})!=3:
        raise ValueError("Expected pruning, grouped RTN and per-channel RTN")
    audit.rule("V88 current narrative summary: results/v88-displacement/FINDINGS.md. "
               "No current three-family summary.json exists. The only JSON summary found is "
               "results/v88-displacement-uncentred-run/summary.json (superseded two-family run, not plotted). "
               "Panel c uses current results/v88-displacement/*/*.json per_capability.measured_delta and "
               "per_capability.second_order_prediction: per-configuration median |damage| and median |prediction-measured|/|measured|, "
               "equal state/capability records; zero damage excluded. This is aggregation, not fitting. "
               "The separate v88-displacement-cluster hardware run is not pooled. "
               "Three compression families here mean pruning, grouped RTN, per-channel RTN; there is no distillation displacement measurement.")
    return result


def fold_positions(folds):
    """Start at the true category; the shared renderer bounds display dodging."""
    return [0.] * len(folds)


def draw_panel(fig, data, panel, rectangle=None):
    from matplotlib.ticker import NullLocator
    size = PANEL_SIZES[panel]
    ax = panel_axes(fig, size, left=.35, bottom=.30, right=.08)
    row_axis_style(ax)
    if panel == "a":
        for i, r in enumerate(data["profiles"]):
            color = COLORS[r["capability"]]
            for offset, fold in zip(fold_positions(r["folds"]), r["folds"]):
                ax.plot(i+offset, fold["p"], ls="none", marker="x" if fold["boundary"] else "o",
                        ms=3.8, mew=.6,
                        color=color, alpha=.8)
            lo, hi = r["interval"]
            ax.errorbar(i, r["p"], yerr=[[r["p"]-lo], [hi-r["p"]]],
                        fmt="D", color=color, mfc=PALETTE["transparent"], ms=3.8, lw=1.1, elinewidth=1.1, capthick=.6, capsize=2)
        ax.axhline(0, color=PALETTE["reference"], lw=1.1, ls=":")
        ax.axhline(1, color=PALETTE["reference"], lw=1.1, ls="--")
        ax.set(xticks=range(3), xticklabels=["Math", "Code", "QA"],
               ylabel="Curvature $p$", xlim=(-.65, 2.75))
        ax.set_yticks([-1, 0, 1, 3])
    elif panel == "b":
        for i, r in enumerate(data["corners"]):
            color = COLORS[r["capability"]]
            ax.fill_betweenx([i-.36, i+.36], -r["noise"], r["noise"],
                             facecolor=PALETTE["background"], edgecolor=PALETTE["grid"], linewidth=.6, zorder=0)
            ax.plot(r["additive"], i, "|", color=PALETTE["reference"], ms=3.8, mew=.6)
            ax.plot(r["F_int"], i, "D", color=PALETTE["black"], mfc=PALETTE["transparent"], ms=3.8)
            lo, hi = r["interval"]
            ax.errorbar(r["measured"], i, xerr=[[r["measured"]-lo], [hi-r["measured"]]],
                        fmt="o", color=color, ms=3.8, lw=1.1, elinewidth=1.1, capthick=.6, capsize=2)
        ax.set_xscale("symlog", linthresh=.02)
        # Scales reset formatters/locators: reapply the row styling afterwards.
        row_axis_style(ax)
        ax.set_xticks([-.1, 0, .1])
        ax.yaxis.set_major_locator(NullLocator())
        ax.set_yticks(range(len(data["corners"])),
                     labels=[r["student"].replace("gemma3-", "").upper()+" "+
                             ("QA" if r["capability"] == "qa" else r["capability"].title())
                             for r in data["corners"]], minor=True)
        ax.set(ylim=(len(data["corners"])-.5, -.5),
               xlabel="Second difference (nats)", ylabel="Student/readout")
    else:
        labels = {"prune": "Pruning", "pruning": "Pruning", "grouped_rtn": "Grouped RTN",
                  "per_channel_rtn": "Per-channel RTN"}
        for family, marker in zip(sorted({r["family"] for r in data["displacement"]}), ("o", "s", "^")):
            color = SEMANTIC_METHOD_COLORS[family]
            part = sorted((r for r in data["displacement"] if r["family"] == family), key=lambda r: r["damage"])
            ax.plot([r["damage"] for r in part], [100*r["median_relative_error"] for r in part],
                    marker=marker, color=color, lw=1.1, ms=3.8, label=labels[family])
        ax.set(xscale="log", yscale="log", xlabel="Absolute damage (nats)",
               ylabel="Relative error (%)")
        row_axis_style(ax)
        ax.set_xticks([.01, 1])
        ax.set_yticks([1, 10])
    finish_panel(ax)
    return position_row_axes(ax, rectangle) if rectangle is not None else ax


def draw_legend(fig):
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker=m, ls="", color=c, mfc=PALETTE["transparent"] if m == "D" else c, ms=3.8,
                      mew=.6, label=label)
               for m, c, label in (("o", PALETTE["reference"], "Fold estimate"), ("x", PALETTE["reference"], "Boundary"),
                                   ("D", PALETTE["reference"], "Full"), ("|", PALETTE["reference"], "Additive"),
                                   ("D", PALETTE["black"], "$F_{int}$"), ("o", PALETTE["reference"], "Observed"))]
    handles += [Line2D([], [], marker=m, color=c, lw=1.1, ms=3.8, label=label)
                for m, c, label in zip(("o", "s", "^"), [SEMANTIC_METHOD_COLORS[f] for f in ("grouped_rtn", "per_channel_rtn", "prune")], ("Group", "Channel", "Prune"))]
    return legend_strip(fig, handles)


def row_rectangle(data, plt):
    return measure_row_rectangle(plt, PANEL_SIZES["a"],
                                 [lambda f, p=p: draw_panel(f, data, p) for p in "abc"])


def plot(data, plt, rectangle=None):
    if rectangle is None:
        rectangle = row_rectangle(data, plt)
    return combine_panels(plt, [
        ("panel", (0, 0, *PANEL_SIZES["a"]), lambda f: draw_panel(f, data, "a", rectangle)),
        ("panel", (1.85, 0, *PANEL_SIZES["b"]), lambda f: draw_panel(f, data, "b", rectangle)),
        ("panel", (3.7, 0, *PANEL_SIZES["c"]), lambda f: draw_panel(f, data, "c", rectangle)),
        ("legend", (0, PANEL_SIZES["a"][1], *LEGEND_SIZE), draw_legend),
    ], FIGSIZE)


def generate(root=ROOT):
    with frozen_run(root) as access:
        audit = Artifacts(root)
        plt = pyplot(root)
        data = {"profiles": curvature(audit), "corners": corners(audit), "displacement": displacement(audit)}
        audit.rule("A2 panel a uses training-probe parameter_stability[].folds[name=F_curv].p/boundary and "
                   "parameter_intervals[].fits.F_curv.fit.p / p_interval / fit.boundary_hit; no profile or bootstrap is rerun.")
        rectangle = row_rectangle(data, plt)
        for letter, key in zip("abc", ("profiles", "corners", "displacement")):
            apply_style(KINDS[letter])
            fig = plt.figure(figsize=PANEL_SIZES[letter])
            draw_panel(fig, data, letter, rectangle)
            save_panel(fig, f"fig2_{letter}", KINDS[letter], audit, data[key])
            write_caption(f"fig2_{letter}", audit, CAPTION_TEXT)
            plt.close(fig)
        apply_style("legend")
        fig = plt.figure(figsize=LEGEND_SIZE)
        draw_legend(fig)
        save_panel(fig, "fig2_legend", "legend", audit, [])
        write_caption("fig2_legend", audit, CAPTION_TEXT)
        plt.close(fig)
        fig = plot(data, plt, rectangle)
        save_figure(fig, "explanation", audit)
        write_notes("explanation", audit, data)
        write_caption("explanation", audit, CAPTION_TEXT)
        plt.close(fig)
        return data, audit, access


if __name__ == "__main__":
    generate()
