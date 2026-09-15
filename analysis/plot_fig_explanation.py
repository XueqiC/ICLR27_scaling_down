#!/usr/bin/env python3
"""Curvature, registered corner test, and post-hoc displacement account."""
from __future__ import annotations

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
PANEL_SIZES = {p: (1.8, 1.15) for p in "abc"}
KINDS = {"a": "panel", "b": "panel", "c": "panel"}
LEGEND_SIZE = (5.5, .42)
FIGSIZE = (5.5, 1.59)
if __package__:
    from .paper_figure_style import apply_style, finish_panel, panel_axes, legend_strip, save_panel, write_caption, combine_panels
else:
    from paper_figure_style import apply_style, finish_panel, panel_axes, legend_strip, save_panel, write_caption, combine_panels

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
The three 1.8 x 1.15-inch panels form one row at 5.5-inch text width.
The shared key is fig2_legend.pdf: Fold estimate, Boundary and Full apply to (a);
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


def draw_panel(fig, data, panel):
    from matplotlib.ticker import NullFormatter, MaxNLocator
    size = PANEL_SIZES[panel]
    if panel == "a":
        ax = panel_axes(fig, size, left=.35, bottom=.41, right=.06)
        for i, r in enumerate(data["profiles"]):
            color = COLORS[r["capability"]]
            for k, fold in enumerate(r["folds"]):
                x = i-.25+.38*k/max(1, len(r["folds"])-1)
                ax.plot(x, fold["p"], marker="x" if fold["boundary"] else "o", color=color, alpha=.65)
            lo, hi = r["interval"]
            ax.errorbar(i+.28, r["p"], yerr=[[r["p"]-lo], [hi-r["p"]]],
                        fmt="D", color=color, capsize=2)
        ax.axhline(0, color=".65", lw=1.1, ls=":")
        ax.axhline(1, color=".65", lw=1.1, ls="--")
        ax.set(xticks=range(3), xticklabels=["Math", "Code", "QA"],
               xlabel="Capability", ylabel="Curvature $p$")
        ax.yaxis.set_major_locator(MaxNLocator(4))
        for text in ax.get_xticklabels():
            text.set_rotation(30)
            text.set_ha("right")
            text.set_rotation_mode("anchor")
    elif panel == "b":
        ax = panel_axes(fig, size, left=.49, bottom=.30, right=.06)
        for i, r in enumerate(data["corners"]):
            color = COLORS[r["capability"]]
            ax.fill_betweenx([i-.36, i+.36], -r["noise"], r["noise"], color=".88", zorder=0)
            ax.plot(r["additive"], i-.18, "|", color=".2")
            ax.plot(r["F_int"], i+.18, "D", color="#7c4da1")
            lo, hi = r["interval"]
            ax.errorbar(r["measured"], i, xerr=[[r["measured"]-lo], [hi-r["measured"]]],
                        fmt="o", color=color, capsize=2)
        ax.set_xscale("symlog", linthresh=.02)
        ax.set_xticks([-.1, 0, .1], labels=["−0.1", "0", "0.1"])
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set(yticks=range(len(data["corners"])),
               yticklabels=[r["student"].replace("gemma3-", "").upper()+" "+
                            ("QA" if r["capability"] == "qa" else r["capability"].title())
                            for r in data["corners"]],
               ylim=(len(data["corners"])-.5, -.5),
               xlabel="Second difference (nats)")
        ax.xaxis.label.set_verticalalignment("bottom")
        ax.xaxis.set_label_coords(.5, .03, transform=fig.transSubfigure if hasattr(fig, "transSubfigure") else fig.transFigure)
    else:
        ax = panel_axes(fig, size, left=.40, bottom=.34, right=.08)
        labels = {"prune": "Pruning", "pruning": "Pruning", "grouped_rtn": "Grouped RTN",
                  "per_channel_rtn": "Per-channel RTN"}
        for family, color, marker in zip(sorted({r["family"] for r in data["displacement"]}),
                                          COLORS.values(), ("o", "s", "^")):
            part = sorted((r for r in data["displacement"] if r["family"] == family), key=lambda r: r["damage"])
            ax.plot([r["damage"] for r in part], [100*r["median_relative_error"] for r in part],
                    marker=marker, color=color, label=labels[family])
        ax.set(xscale="log", yscale="log", xlabel="Absolute damage (nats)",
               ylabel="Relative error (%)")
        ax.set_xticks([.01, 1])
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.grid(alpha=.15, which="both")
    return finish_panel(ax)


def draw_legend(fig):
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker=m, ls="", color=c, label=label)
               for m, c, label in (("o", ".3", "Fold estimate"), ("x", ".3", "Boundary"),
                                   ("D", ".3", "Full"), ("|", ".2", "Additive"),
                                   ("D", "#7c4da1", "$F_{int}$"), ("o", ".3", "Observed"))]
    handles += [Line2D([], [], marker=m, color=c, label=label)
                for m, c, label in zip(("o", "s", "^"), COLORS.values(), ("Group", "Channel", "Prune"))]
    return legend_strip(fig, handles)


def plot(data, plt):
    return combine_panels(plt, [
        ("panel", (0, .44, *PANEL_SIZES["a"]), lambda f: draw_panel(f, data, "a")),
        ("panel", (1.85, .44, *PANEL_SIZES["b"]), lambda f: draw_panel(f, data, "b")),
        ("panel", (3.7, .44, *PANEL_SIZES["c"]), lambda f: draw_panel(f, data, "c")),
        ("legend", (0, 0, *LEGEND_SIZE), draw_legend),
    ], FIGSIZE)


def generate(root=ROOT):
    with frozen_run(root) as access:
        audit = Artifacts(root)
        plt = pyplot(root)
        data = {"profiles": curvature(audit), "corners": corners(audit), "displacement": displacement(audit)}
        audit.rule("A2 panel a uses training-probe parameter_stability[].folds[name=F_curv].p/boundary and "
                   "parameter_intervals[].fits.F_curv.fit.p / p_interval / fit.boundary_hit; no profile or bootstrap is rerun.")
        for letter, key in zip("abc", ("profiles", "corners", "displacement")):
            apply_style(KINDS[letter])
            fig = plt.figure(figsize=PANEL_SIZES[letter])
            draw_panel(fig, data, letter)
            save_panel(fig, f"fig2_{letter}", KINDS[letter], audit, data[key])
            write_caption(f"fig2_{letter}", audit, CAPTION_TEXT)
            plt.close(fig)
        apply_style("legend")
        fig = plt.figure(figsize=LEGEND_SIZE)
        draw_legend(fig)
        save_panel(fig, "fig2_legend", "legend", audit, [])
        write_caption("fig2_legend", audit, CAPTION_TEXT)
        plt.close(fig)
        fig = plot(data, plt)
        save_figure(fig, "explanation", audit)
        write_notes("explanation", audit, data)
        write_caption("explanation", audit, CAPTION_TEXT)
        plt.close(fig)
        return data, audit, access


if __name__ == "__main__":
    generate()
