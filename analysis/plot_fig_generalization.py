#!/usr/bin/env python3
"""Main-text paired MAEs from frozen predictions; CPU only, no fits/resampling.

The legacy per-cell appendix remains independently reproducible.
Paired gain intervals are reflected about the *fixed* baseline MAE for display;
they are never labelled as marginal MAE or per-cell prediction intervals.
"""
from __future__ import annotations

if __package__:
    from .paper_figure_style import PALETTE, CAPABILITY_COLORS, QA_COLORS, METHOD_COLORS as SEMANTIC_METHOD_COLORS, BIT_COLORS, HATCHES, darker, method_ramp
else:
    from paper_figure_style import PALETTE, CAPABILITY_COLORS, QA_COLORS, METHOD_COLORS as SEMANTIC_METHOD_COLORS, BIT_COLORS, HATCHES, darker, method_ramp

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
GREY = PALETTE["reference"]
MARKERS = {c: "o" for c in CAPS}
PANEL_SIZES = {p: (2.7, 1.6) for p in "ab"}
KINDS = {p: "double" for p in "ab"}
LEGEND_SIZE = (5.5, .42)
FIGSIZE = (5.5, 2.02)
if __package__:
    from .paper_figure_style import apply_style, finish_panel, panel_axes, save_panel, write_caption, combine_panels
else:
    from paper_figure_style import apply_style, finish_panel, panel_axes, save_panel, write_caption, combine_panels


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
            continue
        if original["group"] == "Pythia: locked rule on new states":
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

    audit.rule("Same loss-prediction cells as generalization_cells, excluding locked-rule selection rows and corner contrasts. MAE is mean absolute "
               "prediction-minus-measurement error in native-token nats. Paired markers use identical cells "
               "and equal cell weights; n is the cell count per capability, not independent sample size.")
    audit.rule("Baselines use minimum development LOSO MAE excluding the selected relation: V53 "
               "loso_table[subset=all] for V53/V72; V69 loso.scores[*][cap].macro_mae for V69. "
               "V70 uses freeze.strongest_baseline[student][cap].method, never confirmation ranking.")
    audit.rule("V46 has no recorded development-selected baseline. Keep its relation MAEs "
               "capability-coloured and label baseline unavailable. Do not transport the later V53 "
               "selection to V46. Split V46/V72 within the inside-range row to keep paired cells identical. "
               "Locked-rule selection rows are excluded from this loss-prediction MAE figure.")
    audit.rule("V70 remains split by student within the new-pool budgets row: paired_difference.ci95 "
               "covers one student/capability's 18 checkpoints (6 equally sized pools). Never average "
               "interval endpoints across students. Draw the stored baseline-minus-relation gain interval "
               "[lo, hi] as [baseline MAE - hi, baseline MAE - lo]. This is a translated paired gain "
               "interval at a fixed baseline reference, not a marginal MAE confidence interval.")
    audit.rule("A2 frozen_prediction_error_interval covers its development holdouts, not any of these "
               "V70 or A5/A7 corner cells. No A2 holdout row existed in the cell figure; none is added "
               "and no A2 interval is transplanted. No refitting, resampling, or invented interval.")
    audit.rule("Corner rows retain their frozen values in the separate corner_contrasts appendix. "
               "V46 .55 is outside its original coarse .6--.9 range; .65 is inside. V72 repeats two "
               "revision labels with identical weights; both records retained, not independent states.")
    return result


def row_label(group, stratum):
    labels = {
        "Pruning: density inside range": "Prune in range",
        "Pruning: density outside range": "Prune out, frozen",
        "Quantization: new group size": "Quant group",
        "Distillation: new-pool budgets": "Distill",
        "Pythia: new stages (power)": "New stages",
        "Pythia: new quantization state": "New quant state",
    }
    label = labels[group]
    if stratum == "V46" and "inside" in group:
        label = "Prune in, frozen"
    elif stratum.startswith("gemma3-"):
        label += " " + {"gemma3-270m": "270M", "gemma3-1b": "1B"}[stratum]
    return label


def draw_maes(ax, rows, panel, limits):
    from matplotlib.ticker import NullFormatter
    order = list(dict.fromkeys((r["group"], r["stratum"]) for r in rows if r["panel"] == panel))
    labels = []
    for y, key in enumerate(order):
        part = [r for r in rows if r["panel"] == panel and (r["group"], r["stratum"]) == key]
        for r in part:
            # Keep the true row coordinate; bounded display dodging is shared.
            yy = y
            marker = MARKERS[r["capability"]]
            color = COLORS[r["capability"]]
            candidate, baseline = r["relation_mae"], r["baseline_mae"]
            if baseline is not None:
                ax.plot([baseline, candidate], [yy, yy], color=color if r["below_baseline"] else GREY, zorder=1)
                ax.plot(baseline, yy, marker=marker, mfc=PALETTE["white"], color=color, mec=color, ls="", zorder=3)
            if r["whisker"] is not None:
                lo, hi = r["whisker"]
                midpoint = (lo + hi) / 2
                ax.errorbar(midpoint, yy, xerr=[[midpoint-lo], [hi-midpoint]],
                            fmt="none", color=color, lw=1.1, capsize=2, zorder=2)
            ax.plot(candidate, yy, marker="D", mfc=PALETTE["transparent"], color=color, ls="", zorder=4)
        counts = {r["n"] for r in part}
        if len(counts) != 1:
            raise ValueError("A shared row count requires equal counts per capability")
        labels.append(row_label(*key))
        if y < len(order)-1:
            ax.axhline(y+.5, color=PALETTE["background"], lw=.6, zorder=0)
    ax.set_xscale("log")
    ax.set_xlim(*limits)
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set(yticks=range(len(order)), yticklabels=labels,
           ylim=(len(order)-.5, -.5), xlabel="MAE (nats)")
    ax.set_xticks([.1, 1])
    ax.tick_params(axis="y", length=0, pad=4)
    ax.grid(axis="x", alpha=.15)
    ax.spines["left"].set_visible(False)


def mae_limits(rows):
    maes = [r for r in rows if r["kind"] == "mae"]
    values = [v for r in maes for v in (r["relation_mae"], r["baseline_mae"], *(r["whisker"] or [])) if v is not None]
    if not all(math.isfinite(v) and v > 0 for v in values):
        raise ValueError("Log MAE axis requires positive values; never add a pseudocount or clip an interval")
    return min(values)/1.4, max(values)*1.4


def draw_panel(fig, rows, panel):
    ax = panel_axes(fig, PANEL_SIZES[panel], left=1.03, bottom=.40, right=.08, top=.06)
    draw_maes(ax, rows, panel.upper(), mae_limits(rows))
    return finish_panel(ax)


def draw_legend(fig):
    from matplotlib.lines import Line2D
    if __package__:
        from .paper_figure_style import legend_strip
    else:
        from paper_figure_style import legend_strip
    handles = [Line2D([], [], color=COLORS[c], marker=MARKERS[c], ls="",
                      label="QA" if c == "qa" else c.title()) for c in CAPS]
    handles += [Line2D([], [], color=GREY, mfc=PALETTE["white"], marker="o", ls="", label="Baseline"),
                Line2D([], [], color=GREY, marker="D", mfc=PALETTE["transparent"], ls="", label="Relation"),
                Line2D([], [], color=GREY, ls="-", label="Lower of pair")]
    return legend_strip(fig, handles)


def plot(rows, plt):
    return combine_panels(plt, [
        ("double", (0, 0, *PANEL_SIZES["a"]), lambda f: draw_panel(f, rows, "a")),
        ("double", (2.8, 0, *PANEL_SIZES["b"]), lambda f: draw_panel(f, rows, "b")),
        ("legend", (0, PANEL_SIZES["a"][1], *LEGEND_SIZE), draw_legend),
    ], FIGSIZE)


CAPTION_TEXT = """Generalization to new configurations of seen states (a) and new
sources or students (b). MAE axes are logarithmic in native-token nats and share
one range. Paired markers compare relation and development-selected baseline
on identical cells with equal cell weights. Math, Code and QA retain their
capability hues. Hollow circles denote baselines; hollow diamonds denote frozen
relation predictions. The connecting segment uses the capability colour when
the relation MAE is lower, and reference grey otherwise. Cell counts per capability
are in generalization_mae_pairs.md and record sidecars.
Prune in, frozen and Prune out, frozen are frozen new-state pruning predictions
inside and outside the fitted density range. These rows have no stored
development-selected baseline and retain unpaired capability-coloured markers.
Prune in range and the two frozen pruning rows remain separate, as do the
270M/1B new-pool budgets. Paired gain CIs are translated about the fixed baseline
MAE: [lo, hi] for baseline-minus-relation is drawn at
[baseline MAE - hi, baseline MAE - lo]. These are paired gain intervals, not
marginal MAE confidence intervals; no intervals are averaged across students.
No A2 development-holdout interval is transplanted to confirmation cells.
Two 2.7 x 1.6-inch panels form one 5.5-inch row with fig3_legend.pdf above:
Baseline, Relation and Lower of pair distinguish the paired MAEs. The locked-rule
selection row is excluded. Corner second differences and fresh-distribution
contrasts are shown separately in corner_contrasts_a.pdf.
"""


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
    with frozen_run(root) as access:
        audit = Artifacts(root)
        plt = pyplot(root)
        rows = build(audit)
        audit.rule("Display mapping: fig3_a = MAE A, fig3_b = MAE B. "
                   "Locked-rule selection rows and corner contrasts are excluded from MAEs.")
        for letter in "ab":
            apply_style(KINDS[letter])
            fig = plt.figure(figsize=PANEL_SIZES[letter])
            draw_panel(fig, rows, letter)
            panel_rows = [r for r in rows if r["panel"] == letter.upper()]
            save_panel(fig, f"fig3_{letter}", KINDS[letter], audit, panel_rows)
            write_caption(f"fig3_{letter}", audit, CAPTION_TEXT)
            plt.close(fig)
        apply_style("legend")
        fig = plt.figure(figsize=LEGEND_SIZE)
        draw_legend(fig)
        save_panel(fig, "fig3_legend", "legend", audit, [])
        write_caption("fig3_legend", audit, CAPTION_TEXT)
        plt.close(fig)
        if __package__:
            from .paper_figure_archive import archive_panels
        else:
            from paper_figure_archive import archive_panels
        archive_panels(root, ("fig3_c", "fig3_d"))
        fig = plot(rows, plt)
        save_figure(fig, "generalization", audit)
        write_notes("generalization", audit, rows)
        write_caption("generalization", audit, CAPTION_TEXT)
        output_path(root, "figs", "generalization_mae_pairs.md").write_text(
            "# Generalization: per-row MAE pairs\n\n"
            "Native-token nats; cell counts are per capability. Baselines are selected on development "
            "evidence only. Unavailable baselines remain unscored.\n\n" + format_pairs(rows) + "\n")
        plt.close(fig)
        return rows, audit, access


if __name__ == "__main__":
    rows, _, _ = generate()
    print(format_pairs(rows))
