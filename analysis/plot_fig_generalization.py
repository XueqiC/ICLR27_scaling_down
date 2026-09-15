#!/usr/bin/env python3
"""Main-text paired MAEs from frozen predictions; CPU only, no fits/resampling.

Also regenerates the unchanged per-cell appendix as generalization_cells.
Paired gain intervals are reflected about the *fixed* baseline MAE for display;
they are never labelled as marginal MAE or per-cell prediction intervals.
"""
from __future__ import annotations

import math
import sys
from collections import defaultdict
from statistics import mean

sys.dont_write_bytecode = True

if __package__:
    from . import plot_fig_generalization_cells as cells
    from .paper_artifacts import ROOT, CAPS, COLORS, Artifacts, frozen_run, pyplot, save_figure, write_notes, output_path
else:
    import plot_fig_generalization_cells as cells
    from paper_artifacts import ROOT, CAPS, COLORS, Artifacts, frozen_run, pyplot, save_figure, write_notes, output_path

P53 = "results/v53-prune-dev/register.json"
Q69 = "results/v69-quant-confirm/develop.json"
D70 = "results/v70-distill-confirm/compare.json"
F70 = "results/v70-distill-confirm/freeze.json"
GREEN, GREY = "#24835b", "#777777"
MARKERS = {"math": "o", "code": "s", "qa": "D"}
SCOPE_NOTE = ("No frozen response-law predictions\non fresh evaluation distributions.\n"
              "Only registered corner contrasts\nare available below.")


def resolve(audit, reference):
    path, pointer = reference.split("#", 1)
    value = audit.data[path] if path in audit.data else audit.read(path)
    for key in pointer.removeprefix("/").split("/") if pointer else []:
        key = key.replace("~1", "/").replace("~0", "~")
        value = value[int(key)] if isinstance(value, list) else value[key]
    return value


def check_close(actual, expected, context):
    if not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError(f"Frozen score mismatch: {context}: {actual} != {expected}")


def gain_whisker(baseline_mae, interval):
    """Translate a stored baseline-minus-relation interval, without resampling."""
    if interval is None:
        return None
    lo, hi = interval
    if not all(math.isfinite(x) for x in (baseline_mae, lo, hi)) or lo > hi:
        raise ValueError("Invalid stored paired gain interval")
    return [baseline_mae - hi, baseline_mae - lo]


def baseline_choices(audit):
    # Same development-only selection as the main prediction table. Alternative
    # relation forms are eligible comparators; the plotted relation is excluded.
    prune = audit.data[P53]
    eligible = [(i, r) for i, r in enumerate(prune["loso_table"])
                if r["subset"] == "all" and r["candidate"] != "power"]
    pruning = {}
    for cap in CAPS:
        i, row = min(eligible, key=lambda ir: ir[1][f"{cap}_mae"])
        pruning[cap] = (row["candidate"], f"{P53}#/loso_table/{i}/{cap}_mae")
    quant = audit.read(Q69)
    quantization = {}
    for cap in CAPS:
        selected = quant["selected"][cap]["candidate"]
        baseline = min((m for m in quant["methods"] if m != selected),
                       key=lambda m: quant["loso"]["scores"][m][cap]["macro_mae"])
        quantization[cap] = (baseline, f"{Q69}#/loso/scores/{baseline}/{cap}/macro_mae")
    return pruning, quantization


def summarize(part):
    """Score both predictors on exactly the same cells, with equal cell weights."""
    first = part[0]
    baseline = first["baseline"]
    if any(r["baseline"] != baseline or r["relation"] != first["relation"] for r in part):
        raise ValueError("Cannot pool different predictor selections")
    relation_mae = mean(abs(r["predicted"] - r["measured"]) for r in part)
    baseline_mae = None if baseline is None else mean(
        abs(r["baseline_prediction"] - r["measured"]) for r in part)
    return {"kind": "mae", "panel": first["panel"], "group": first["group"],
            "stratum": first["stratum"], "capability": first["capability"],
            "status": first["status"], "relation": first["relation"], "baseline": baseline,
            "relation_mae": relation_mae, "baseline_mae": baseline_mae, "n": len(part),
            "below_baseline": None if baseline_mae is None else relation_mae < baseline_mae,
            "paired_gain_interval": None, "whisker": None, "interval_source": None,
            "baseline_note": first.get("baseline_note"), "cells": part}


def build(audit):
    source_cells = cells.build(audit)
    # Keep the cell figure's original disclosure in its own sidecar, and give
    # this encoding its own interval/selection notes.
    audit.notes = []
    pruning, quantization = baseline_choices(audit)
    groups = defaultdict(list)
    result = []
    for original in source_cells:
        if original["measurement_interval"] is not None:
            result.append({"kind": "corner", **original})
            continue
        r = {**original, "baseline": None, "baseline_prediction": None,
             "baseline_source": None, "selection_source": None, "stratum": ""}
        cap, source = r["capability"], r["source"]
        if source.startswith("results/v53-prune-dev/"):
            r["relation"] = "power"
            r["baseline"], r["selection_source"] = pruning[cap]
            r["baseline_source"] = source.rsplit("/", 1)[0] + "/" + r["baseline"]
        elif source.startswith("results/v72-prune-repeat/"):
            r["relation"], r["stratum"] = "power", "V72"
            r["baseline"], r["selection_source"] = pruning[cap]
            r["baseline_source"] = source + "/predictions/" + r["baseline"]
        elif source.startswith("results/v46-p1-newsource/"):
            r["relation"], r["stratum"] = "power", "V46"
            r["baseline_note"] = "Development-selected baseline unavailable in the V46 freeze."
        elif source.startswith("results/v69-quant-confirm/"):
            r["relation"] = r["candidate"]
            r["baseline"], r["selection_source"] = quantization[cap]
            if r["relation"] != audit.data[Q69]["selected"][cap]["candidate"]:
                raise ValueError("V69 development/freeze selection mismatch")
            r["baseline_source"] = source + "/predictions/" + r["baseline"]
        elif source.startswith(D70):
            stored = resolve(audit, source)
            student = stored["student"]
            r["stratum"], r["student"], r["relation"] = student, student, r["candidate"]
            r["selection_source"] = f"{F70}#/strongest_baseline/{student}/{cap}/method"
            r["baseline"] = resolve(audit, r["selection_source"])
            if stored["strongest_baseline"] != r["baseline"] or stored["selected"] != r["relation"]:
                raise ValueError("V70 development/freeze selection mismatch")
            r["baseline_source"] = source + "/predictions/" + r["baseline"]
        elif source.startswith("results/v78-rule-confirm/"):
            r["relation"] = "locked-rule"
            r["baseline_note"] = ("No development-selected prediction baseline stored for these cells; "
                                  "v64-law and policy regret are different comparisons.")
        else:
            raise ValueError(f"Unclassified frozen prediction: {source}")
        if r["baseline_source"] is not None:
            r["baseline_prediction"] = resolve(audit, r["baseline_source"])
        groups[r["panel"], r["group"], r["stratum"], cap].append(r)

    for part in groups.values():
        row = summarize(part)
        if part[0]["source"].startswith(D70):
            student, cap = row["stratum"], row["capability"]
            i, stored = next((i, g) for i, g in enumerate(audit.data[D70]["groups"])
                             if (g["student"], g["capability"]) == (student, cap))
            if (stored["selected"], stored["strongest_baseline"], stored["n_checkpoints"]) != (
                    row["relation"], row["baseline"], row["n"]):
                raise ValueError("V70 interval group/selection mismatch")
            for field, key in (("relation_mae", "candidate_mae"), ("baseline_mae", "baseline_mae")):
                check_close(row[field], stored[key], f"V70/{student}/{cap}/{key}")
            check_close(row["baseline_mae"] - row["relation_mae"],
                        stored["paired_difference"]["estimate"], "V70 paired difference")
            row["paired_gain_interval"] = stored["paired_difference"]["ci95"]
            row["whisker"] = gain_whisker(row["baseline_mae"], row["paired_gain_interval"])
            row["interval_source"] = f"{D70}#/groups/{i}/paired_difference/ci95"
        result.append(row)

    audit.rule("Same prediction cells and row families as generalization_cells. MAE is mean absolute "
               "prediction-minus-measurement error in native-token nats. Paired markers use identical cells "
               "and equal cell weights; n is the cell count per capability, not independent sample size.")
    audit.rule("Baselines use minimum development LOSO MAE excluding the selected relation: V53 "
               "loso_table[subset=all] for V53/V72; V69 loso.scores[*][cap].macro_mae for V69. "
               "V70 uses freeze.strongest_baseline[student][cap].method, never confirmation ranking.")
    audit.rule("V46 has no recorded development-selected baseline; V78 has no such baseline predictions "
               "on its plotted cells. Keep these relation MAEs grey and label baseline unavailable. "
               "Do not choose on test errors, transport the later V53 selection to V46, or call V78's "
               "v64-law a development-selected baseline. Split V46/V72 within the inside-range row to "
               "avoid pairing a subset baseline with a full-row relation MAE.")
    audit.rule("V70 remains split by student within the new-pool budgets row: paired_difference.ci95 "
               "covers one student/capability's 18 checkpoints (6 equally sized pools). Never average "
               "interval endpoints across students. Draw the stored baseline-minus-relation gain interval "
               "[lo, hi] as [baseline MAE - hi, baseline MAE - lo]. This is a translated paired gain "
               "interval at a fixed baseline reference, not a marginal MAE confidence interval.")
    audit.rule("A2 frozen_prediction_error_interval covers its development holdouts, not any of these "
               "V70 or A5/A7 corner cells. No A2 holdout row existed in the cell figure; none is added "
               "and no A2 interval is transplanted. No refitting, resampling, or invented interval.")
    audit.rule("Corner rows are unchanged: signed prediction-minus-measurement on a separate symlog "
               "axis, original capability colours, circles filled iff within the stored band, triangles "
               "for the 4B development student. Whiskers are A5 registered +/-2-noise bands, not CIs. "
               "Primary QA additivity failed to reject. Panel C shows only the registered additive "
               "corner predictions (zero); no frozen response-law predictions on fresh distributions exist.")
    audit.rule("V46 .55 is outside its original coarse .6--.9 range; .65 is inside. V72 repeats two "
               "revision labels with identical weights; both records retained, not independent states. "
               "V78 uses each frozen configuration once. V93 descriptors only verify state identity.")
    return result


def row_label(group, stratum):
    labels = {
        "Pruning: density inside range": "Pruning: inside range",
        "Pruning: density outside range": "Pruning: outside range",
        "Quantization: new group size": "Quantization:\nnew group size",
        "Distillation: new-pool budgets": "Distillation: new pools",
        "Pythia: new stages (power)": "Pythia: new stages\n(power)",
        "Pythia: new quantization state": "Pythia: new\nquantization state",
        "Pythia: locked rule on new states": "Pythia: locked rule\nbaseline unavailable",
    }
    label = labels[group]
    if stratum == "V46":
        label += "\nV46; baseline unavailable"
    elif stratum == "V72":
        label += "\nV72 repeat"
    elif stratum.startswith("gemma3-"):
        label += "\n" + {"gemma3-270m": "270M", "gemma3-1b": "1B"}[stratum]
    return label


def draw_maes(ax, rows, panel, limits):
    order = list(dict.fromkeys((r["group"], r["stratum"]) for r in rows if r["panel"] == panel))
    for y, key in enumerate(order):
        part = [r for r in rows if r["panel"] == panel and (r["group"], r["stratum"]) == key]
        for r in part:
            yy = y + (CAPS.index(r["capability"]) - 1) * .20
            marker = MARKERS[r["capability"]]
            color = GREEN if r["below_baseline"] else GREY
            candidate, baseline = r["relation_mae"], r["baseline_mae"]
            if baseline is not None:
                ax.plot([baseline, candidate], [yy, yy], color=".68", lw=.8, zorder=1)
                ax.plot(baseline, yy, marker=marker, mfc="white", mec=GREY, ms=5, ls="", zorder=3)
            if r["whisker"] is not None:
                lo, hi = r["whisker"]
                ax.hlines(yy, lo, hi, color=color, lw=1.0, zorder=2)
                ax.vlines([lo, hi], yy-.06, yy+.06, color=color, lw=.8, zorder=2)
            ax.plot(candidate, yy, marker=marker, color=color, ms=4, ls="", zorder=4)
        counts = {r["n"] for r in part}
        if len(counts) != 1:
            raise ValueError("A shared row count requires equal counts per capability")
        ax.text(1.025, y, str(counts.pop()), transform=ax.get_yaxis_transform(),
                va="center", fontsize=7, color=".35")
        if y < len(order)-1:
            ax.axhline(y+.5, color=".93", lw=.6, zorder=0)
    ax.text(1.025, 1.025, "n / cap.", transform=ax.transAxes, fontsize=7, color=".35")
    ax.set_xscale("log")
    ax.set_xlim(*limits)
    ax.set(yticks=range(len(order)), yticklabels=[row_label(*key) for key in order],
           ylim=(len(order)-.5, -.5), xlabel="MAE (native-token nats; log)")
    ax.tick_params(axis="y", labelsize=7.5, length=0, pad=7)
    ax.grid(axis="x", alpha=.15)
    ax.spines["left"].set_visible(False)


def draw_corners(ax, rows, panel):
    rows = [r for r in rows if r["panel"] == panel]
    groups = list(dict.fromkeys(r["group"] for r in rows))
    for y, group in enumerate(groups):
        part = [r for r in rows if r["group"] == group]
        for k, r in enumerate(part):
            # Deliberately identical to the original cell plot's corner grammar.
            jitter = 0 if len(part) == 1 else -.26 + .52*k/(len(part)-1)
            color = COLORS[r["capability"]]
            dev = r["status"] == "development"
            lo, hi = r["residual_interval"]
            e = r["residual"]
            ax.errorbar(e, y+jitter, xerr=[[e-lo], [hi-e]], fmt="none", color=color, capsize=2, lw=.85)
            ax.plot(e, y+jitter, marker="^" if dev else "o", ls="", color=color,
                    mfc=color if r["within"] else "white", ms=5 if dev else 4, alpha=.85)
    ax.axvline(0, color=".5", lw=.8)
    ax.set_xscale("symlog", linthresh=.03)
    ax.set_xlim(-2.5, 2.5)
    ax.set_xticks([-1, -.1, 0, .1, 1], labels=["−1", "−0.1", "0", "0.1", "1"])
    labels = {"Distillation: corner budgets (1B)": "Corner budgets\n1B",
              "4B DEVELOPMENT: corner budgets": "Corner budgets\n4B DEVELOPMENT",
              "2wiki_new": "2Wiki (fresh)", "musique": "MuSiQue", "triviaqa": "TriviaQA"}
    ax.set(yticks=range(len(groups)), yticklabels=[labels[g] for g in groups],
           ylim=(len(groups)-.5, -.5), xlabel="Prediction − measurement\n(native-token nats; symlog)")
    ax.tick_params(axis="y", labelsize=7.5, length=0, pad=7)
    ax.grid(axis="x", alpha=.15)
    ax.spines["left"].set_visible(False)


def plot(rows, plt):
    from matplotlib.lines import Line2D
    maes = [r for r in rows if r["kind"] == "mae"]
    corners = [r for r in rows if r["kind"] == "corner"]
    values = [v for r in maes for v in (r["relation_mae"], r["baseline_mae"], *(r["whisker"] or [])) if v is not None]
    if not all(math.isfinite(v) and v > 0 for v in values):
        raise ValueError("Log MAE axis requires positive values; never add a pseudocount or clip an interval")
    limits = min(values)/1.4, max(values)*1.4
    fig = plt.figure(figsize=(13.6, 5.7))
    grid = fig.add_gridspec(2, 3, left=.13, right=.975, top=.87, bottom=.19,
                           wspace=1.0, hspace=.85, height_ratios=[4, 1.05])
    titles = ("A  New configurations of seen states", "B  New sources or students",
              "C  New evaluation distributions")
    for i, panel in enumerate("AB"):
        ax = fig.add_subplot(grid[0, i])
        draw_maes(ax, maes, panel, limits)
        ax.set_title(titles[i], fontsize=9, pad=23)
        draw_corners(fig.add_subplot(grid[1, i]), corners, panel)
    # C has only three supported corner contrasts. No development QA holdout is
    # relabelled as a response-law prediction on a fresh distribution.
    scope = fig.add_subplot(grid[:, 2])
    scope.set_axis_off()
    scope.set_title(titles[2], fontsize=9, pad=23)
    scope.text(.5, .99, SCOPE_NOTE, transform=scope.transAxes, ha="center", va="top",
               fontsize=8, linespacing=1.5, color=".3")
    cx = scope.inset_axes([0, .20, 1, .48])
    draw_corners(cx, corners, "C")
    scope.text(.5, .03, "Primary QA additivity:\nfailed to reject", transform=scope.transAxes,
               ha="center", va="top", fontsize=8, linespacing=1.4)
    legend = [Line2D([], [], color=COLORS[c], marker=MARKERS[c], ls="",
                     label="QA" if c == "qa" else c.title()) for c in CAPS]
    legend += [Line2D([], [], color=GREY, mfc="white", marker="o", ls="", label="Baseline"),
               Line2D([], [], color=GREY, marker="o", ls="", label="Relation"),
               Line2D([], [], color=GREEN, marker="o", ls="", label="Lower MAE"),
               Line2D([], [], color=GREY, marker="|", lw=.8, label="Paired gain CI"),
               Line2D([], [], color=GREY, marker="^", ls="", mfc="white",
                      label="4B dev.; corners: filled in band / hollow outside")]
    fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(.5, .035), ncol=len(legend),
               frameon=False, fontsize=7.5, handlelength=1.2, columnspacing=1.3)
    fig.text(.5, .015, "Paired gain CIs are translated about the fixed baseline MAE; corner whiskers are registered ±2-noise bands.",
             ha="center", fontsize=7, color=".35")
    return fig


def format_pairs(rows):
    lines = ["| Panel / row | Capability | Relation | MAE | Development baseline | MAE | Cells |",
             "|---|---|---|---:|---|---:|---:|"]
    for r in sorted((r for r in rows if r["kind"] == "mae"), key=lambda r: (r["panel"], r["group"], r["stratum"], CAPS.index(r["capability"]))):
        label = r["group"] + (f" ({r['stratum']})" if r["stratum"] else "")
        baseline = "unavailable" if r["baseline_mae"] is None else f"{r['baseline_mae']:.8f}"
        lines.append(f"| {r['panel']} / {label} | {r['capability']} | {r['relation']} | "
                     f"{r['relation_mae']:.8f} | {r['baseline'] or 'unavailable'} | {baseline} | {r['n']} |")
    return "\n".join(lines)


def generate(root=ROOT):
    _, _, appendix_access = cells.generate(root)
    with frozen_run(root) as access:
        audit = Artifacts(root)
        plt = pyplot(root)
        rows = build(audit)
        fig = plot(rows, plt)
        save_figure(fig, "generalization", audit)
        write_notes("generalization", audit, rows)
        output_path(root, "figs", "generalization_mae_pairs.md").write_text(
            "# Generalization: per-row MAE pairs\n\n"
            "Native-token nats; cell counts are per capability. Baselines are selected on development "
            "evidence only. Unavailable baselines remain unscored.\n\n" + format_pairs(rows) + "\n")
        plt.close(fig)
        access[0].update(appendix_access[0])
        access[1].update(appendix_access[1])
        return rows, audit, access


if __name__ == "__main__":
    rows, _, _ = generate()
    print(format_pairs(rows))
